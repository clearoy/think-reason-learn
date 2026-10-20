"""Policy Induction.

An interpretable ensemble framework for binary classification: an LLM writes
general guidance policies from labelled batches, Jev scores how each policy
moves the answer for every sample, and a strongly regularised logistic
regression keeps the few policies that matter.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Iterable,
    List,
    Literal,
    Self,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import numpy.typing as npt
import orjson
import pandas as pd
import sklearn
from joblib import dump as joblib_dump
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from think_reason_learn.core import TRLModel
from think_reason_learn.core.exceptions import (
    DataError,
    JevAuthError,
    JevCostCapError,
    LLMError,
)
from think_reason_learn.core.llms import (
    JevBudget,
    JevChoice,
    JevChoiceDict,
    JevQuestion,
    JevRequest,
    LLMChoice,
    NoulQuestion,
    TokenCounter,
    llm,
)
from think_reason_learn.core.llms._jev.client import (
    MAX_QUESTIONS_PER_REQUEST,
    loading_saved_model,
    require_typesafe_key,
)
from think_reason_learn.core.llms._jev.schemas import is_jev
from ._prompts import (
    POLICY_GEN_INSTRUCTIONS,
    POLICY_JEV_QUESTION,
    POLICY_JEV_STATE,
    POLICY_MERGE_INSTRUCTIONS,
)

logger = logging.getLogger(__name__)

# Compact terminal-style progress bar: "[TAG] ████░░░░ n/total · rate/s · eta MM:SS"
_BAR_FORMAT = "{desc} {bar} {n_fmt}/{total_fmt} · {rate_fmt} · eta {remaining}"
_BAR_ASCII = "░█"

# Generation is sampled for variety; set_task and the merge want a stable answer.
_GEN_TEMPERATURE = 1.0
_MERGE_TEMPERATURE = 0.0

# Samples sent to Jev per call, so a checkpoint is written every this many.
_SCORE_CHUNK = 50
_PREDICT_CHUNK = 50

_MANIFEST_VERSION = 4


def _penalty_kwargs(
    penalty: Literal["l1", "l2"], sklearn_version: str = sklearn.__version__
) -> Dict[str, Any]:
    """Arguments that give ``LogisticRegression`` an L1 or L2 penalty.

    scikit-learn 1.8 deprecated ``penalty`` (removed in 1.10) in favour of
    ``l1_ratio``: 1 for L1, 0 for L2. Before 1.8, ``l1_ratio`` is ignored unless
    ``penalty="elasticnet"``, so ``penalty`` must still be passed there.

    Raises:
        ValueError: ``penalty`` is not ``"l1"`` or ``"l2"``.
    """
    if penalty not in ("l1", "l2"):
        raise ValueError(f"penalty must be 'l1' or 'l2', got {penalty!r}")
    match = re.match(r"(\d+)\.(\d+)", sklearn_version)
    if match is None or (int(match.group(1)), int(match.group(2))) < (1, 8):
        return {"penalty": penalty}
    return {"l1_ratio": 1.0 if penalty == "l1" else 0.0}


# ── Schemas ────────────────────────────────────────────────────────────────────


class Policies(BaseModel):
    policies: List[str] = Field(..., description="The list of generated policies.")


# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass
class WeightTrainerConfig:
    """Configuration for training and optimizing ensemble weights.

    The defaults favour few final policies: an L1 penalty, a grid of strong
    regularisation strengths, and the one-standard-error rule, which picks the
    most regularised ``C`` whose CV score is within one standard error of the
    best.

    Args:
        beta: Beta for F-beta score (e.g. 0.5 weights precision more).
        penalty: Regularization type for logistic regression.
        cv_folds: Number of StratifiedKFold splits.
        Cs: Candidate regularization strengths. Smaller means stronger, so
            fewer policies keep a non-zero weight.
        one_se_rule: Pick the strongest ``C`` within one standard error of the
            best CV score instead of the best-scoring ``C``.
        threshold_grid: Decision thresholds to search over CV folds.
        class_weight_balanced: Use class_weight='balanced' in LR.
        random_state: Random seed for CV splits.
    """

    beta: float = 0.5
    penalty: Literal["l1", "l2"] = "l1"
    cv_folds: int = 5
    Cs: Iterable[float] = (0.01, 0.03, 0.1, 0.3, 1.0)
    one_se_rule: bool = True
    threshold_grid: Iterable[float] = tuple(np.linspace(0.01, 0.99, 99))
    class_weight_balanced: bool = False
    random_state: int = 0


# ── PolicyInduction ────────────────────────────────────────────────────────────


class PolicyInduction(TRLModel):
    """Interpretable ensemble binary classifier.

    ``fit`` runs five steps:

    1. Each generation batch independently writes general policies from its
       own labelled samples.
    2. The same LLM merges the policies and compresses them to at most
       ``max_raw_policies``.
    3. Jev scores every (sample, policy) pair: its probability of YES when
       guided by that policy.
    4. A strongly regularised logistic regression weights those probabilities
       and keeps few policies.
    5. A decision threshold is tuned on CV folds.

    Args:
        gen_llmc: LLMs that write and merge policies, in priority order.
        decision_llmc: Jev, the model that scores policies. Defaults to
            ``JevChoice()``, which needs ``TYPESAFE_API_KEY``. Jev is the only
            supported scorer.
        gen_batches: ``((a, b), n)``: ``n`` independent batches, each showing
            ``a`` YES and ``b`` NO samples. Batches share no samples, so the
            data needs at least ``a * n`` YES and ``b * n`` NO rows.
        max_raw_policies: The most policies the merge step keeps, which are
            all scored. Fewer policies end up in the final model.
        config: Weight training configuration.
        save_path: Directory for checkpoints and saved models.
        name: Instance name (alphanumeric + underscores only).
        random_state: Base random seed.
        confirm_requests: Before fit()/predict() make any LLM calls, print an
            estimated request count per model and require a y/n confirmation.
        _llm: LLM instance for testing (dependency injection). If None, uses
            the global llm.
    """

    _default_save_dir = "policy_induction"

    def __init__(
        self,
        gen_llmc: List[LLMChoice],
        decision_llmc: JevChoice | JevChoiceDict | None = None,
        gen_batches: Tuple[Tuple[int, int], int] = ((5, 5), 7),
        max_raw_policies: int = 30,
        config: WeightTrainerConfig | dict | None = None,
        save_path: str | PathLike[str] | None = None,
        name: str | None = None,
        random_state: int = 0,
        confirm_requests: bool = True,
        _llm: Any = None,
    ):
        self._validate_init(
            gen_llmc=gen_llmc,
            gen_batches=gen_batches,
            max_raw_policies=max_raw_policies,
            save_path=save_path,
            name=name,
        )
        self.gen_llmc = gen_llmc
        self.decision_llmc: JevChoice = self._parse_decision_llmc(decision_llmc)
        require_typesafe_key(
            [self.decision_llmc], method="PolicyInduction", param="decision_llmc"
        )
        self.gen_batches: Tuple[Tuple[int, int], int] = (
            (int(gen_batches[0][0]), int(gen_batches[0][1])),
            int(gen_batches[1]),
        )
        self.max_raw_policies = max_raw_policies
        self.config: WeightTrainerConfig = self._parse_config(config)
        self.random_state = random_state
        self.confirm_requests = confirm_requests
        self.name: str = self._get_name(name)
        self.save_path: Path = self._set_save_path(save_path)

        self._llm_instance: Any = _llm if _llm is not None else llm
        self._token_counter: TokenCounter = TokenCounter()
        self._pgen_instructions_template: str | None = None
        self._task_description: str | None = None

        self._X: pd.DataFrame | None = None
        self._y: npt.NDArray[np.str_] | None = None
        self._policies: List[str] = []
        # Samples x policies, Jev's P(YES) with NaN where not scored yet.
        self._scores: NDArray[np.float64] | None = None
        self._threshold: float = 0.0
        self._lr: LogisticRegression | None = None
        self._validation_result: dict | None = None
        self._fit_duration_seconds: float | None = None
        self._fit_completed_at: str | None = None

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not isinstance(value, (float, int)):
            raise TypeError("Threshold must be numeric.")
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError("Threshold must be between 0 and 1.")
        self._threshold = float(value)

    @property
    def lr(self) -> LogisticRegression:
        if self._lr is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._lr

    @property
    def validation_result(self) -> dict:
        if self._validation_result is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._validation_result

    @property
    def policy_gen_instructions_template(self) -> str | None:
        return self._pgen_instructions_template

    @property
    def policies(self) -> List[str]:
        """The merged policies that were scored (before the regression prunes)."""
        return list(self._policies)

    # ── Validation helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _validate_init(**kw: Any) -> None:
        if not kw["gen_llmc"]:
            raise ValueError("gen_llmc must be a non-empty list")
        batches = kw["gen_batches"]
        try:
            (pos, neg), n = batches
            ints = all(
                isinstance(v, int) and not isinstance(v, bool) for v in (pos, neg, n)
            )
        except (TypeError, ValueError):
            raise ValueError("gen_batches must look like ((a, b), n)") from None
        if not ints or pos < 0 or neg < 0 or pos + neg < 1 or n < 1:
            raise ValueError(
                "gen_batches ((a, b), n) needs integers a, b >= 0 with a + b >= 1 "
                "and n >= 1"
            )
        mrp = kw["max_raw_policies"]
        if not (isinstance(mrp, int) and not isinstance(mrp, bool) and 0 < mrp < 500):
            raise ValueError("max_raw_policies must be an int in [1, 499]")
        if not (kw["save_path"] is None or isinstance(kw["save_path"], (str, Path))):
            raise ValueError("save_path must be None, str, or Path")
        if not (kw["name"] is None or isinstance(kw["name"], str)):
            raise ValueError("name must be None or str")

    @staticmethod
    def _parse_decision_llmc(choice: JevChoice | JevChoiceDict | None) -> JevChoice:
        if choice is None:
            return JevChoice()
        if not is_jev(choice):
            raise ValueError(
                "decision_llmc must be a JevChoice: Jev is the only model that "
                "scores policies."
            )
        return choice if isinstance(choice, JevChoice) else JevChoice(**choice)

    def _parse_config(
        self, cfg: Union[None, WeightTrainerConfig, dict]
    ) -> WeightTrainerConfig:
        if cfg is None:
            return WeightTrainerConfig()
        if isinstance(cfg, WeightTrainerConfig):
            return cfg
        if isinstance(cfg, dict):
            try:
                result = WeightTrainerConfig(**cfg)
            except TypeError as e:
                raise ValueError(f"Invalid config keys: {e}")
            if result.penalty not in ("l1", "l2"):
                raise ValueError("penalty must be 'l1' or 'l2'")
            if result.beta <= 0:
                raise ValueError("beta must be positive")
            return result
        raise ValueError("config must be WeightTrainerConfig, dict, or None")

    # ── Data management ─────────────────────────────────────────────────────────

    def _set_data(self, X: pd.DataFrame, y: Sequence[str]) -> None:
        if not all(isinstance(v, str) for v in y):
            raise DataError("y must be a sequence of strings")
        if len(y) != X.shape[0]:
            raise DataError("X and y must have the same number of rows")
        upper_y = [v.upper() for v in y]
        if set(upper_y) != {"YES", "NO"}:
            raise DataError("y must contain only 'YES' or 'NO' (case insensitive)")
        y_arr = np.array(upper_y, dtype=np.str_)
        self._X = deepcopy(X).reset_index(drop=True)
        self._y = deepcopy(y_arr)

    def _batch_indices(self) -> List[NDArray[np.intp]]:
        """Row positions of each generation batch; batches never share a row.

        Raises:
            DataError: There are too few YES or NO rows for ``gen_batches``.
        """
        (pos, neg), n = self.gen_batches
        y = cast(NDArray[np.str_], self._y)
        yes_idx = np.where(y == "YES")[0]
        no_idx = np.where(y == "NO")[0]
        if len(yes_idx) < pos * n or len(no_idx) < neg * n:
            fit_n = min(
                len(yes_idx) // pos if pos else n, len(no_idx) // neg if neg else n
            )
            raise DataError(
                f"gen_batches (({pos}, {neg}), {n}) needs {pos * n} YES and "
                f"{neg * n} NO rows without reuse, but the data has "
                f"{len(yes_idx)} YES and {len(no_idx)} NO. Lower n (at most "
                f"{fit_n} fits) or a / b."
            )
        rng = np.random.default_rng(self.random_state)
        yes_idx = rng.permutation(yes_idx)
        no_idx = rng.permutation(no_idx)
        batches: List[NDArray[np.intp]] = []
        for i in range(n):
            batch = np.concatenate(
                [yes_idx[i * pos : (i + 1) * pos], no_idx[i * neg : (i + 1) * neg]]
            )
            rng.shuffle(batch)
            batches.append(batch)
        return batches

    @staticmethod
    def _render_sample(row: pd.Series) -> str:
        return "\n".join(f"{col}: {row[col]}" for col in row.index)

    # ── Task / instructions ─────────────────────────────────────────────────────

    async def set_task(self, task_description: str) -> str:
        """Set the task description and have the LLM write the generator prompt.

        Args:
            task_description: Description of the binary classification task.

        Returns:
            The policy generation instructions template string.
        """
        assert task_description, "task_description must be provided"
        self._task_description = task_description

        response = await self._llm_instance.respond(
            query=f"Generate policies for:\n{task_description}",
            llm_priority=self.gen_llmc,
            response_format=str,
            instructions=POLICY_GEN_INSTRUCTIONS,
            temperature=_GEN_TEMPERATURE,
        )
        await self._token_counter.append(
            provider=response.provider_model.provider,
            model=response.provider_model.model,
            value=response.total_tokens,
            caller="PolicyInduction.set_task",
        )
        if not response.response or not response.response.strip():
            raise ValueError(
                "Failed to generate a valid instructions template. "
                "Refine the task description or switch models."
            )
        self._pgen_instructions_template = response.response
        return response.response

    # ── Request confirmation ────────────────────────────────────────────────────

    def _confirm_requests(self, estimates: Dict[str, int]) -> None:
        """Print an estimated request count per model and require y/n to proceed.

        No-op (and no prompt) if the total estimate is 0 — nothing new to do,
        e.g. a fully-resumed run with everything already checkpointed.
        """
        if not self.confirm_requests or sum(estimates.values()) == 0:
            return
        lines = ["Estimated API requests:"]
        for label, count in estimates.items():
            lines.append(f"  {label}: ~{count} requests")
        lines.append("Proceed? [y/N]: ")
        answer = input("\n".join(lines)).strip().lower()
        if answer not in ("y", "yes"):
            raise RuntimeError("Aborted by user before making API requests.")

    @staticmethod
    def _llmc_label(llmc: LLMChoice | JevChoice) -> str:
        return llmc["model"] if isinstance(llmc, dict) else llmc.model

    @staticmethod
    def _jev_requests_per_sample(n_policies: int) -> int:
        return math.ceil(n_policies / MAX_QUESTIONS_PER_REQUEST)

    def _estimate_fit_requests(self) -> Dict[str, int]:
        """Estimate remaining LLM calls for fit(), accounting for any checkpoint."""
        ckpt = self._load_fit_ckpt()
        if self._policies or ckpt.get("merged"):
            gen_remaining = 0
        else:
            gen_remaining = self.gen_batches[1] - len(ckpt.get("batches", {})) + 1
        n_policies = len(self._policies or ckpt.get("merged") or []) or (
            self.max_raw_policies
        )
        n_samples = len(self._X) if self._X is not None else 0
        done = ckpt.get("scores", {})
        full_samples = n_samples
        if done and n_policies:
            full_samples = sum(
                1
                for s in range(n_samples)
                if any(str(s) not in done.get(str(p), {}) for p in range(n_policies))
            )
        return {
            f"{self._llmc_label(self.gen_llmc[0])} (generation)": gen_remaining,
            f"{self._llmc_label(self.decision_llmc)} (scoring)": full_samples
            * self._jev_requests_per_sample(n_policies),
        }

    def _estimate_predict_requests(self, samples: pd.DataFrame) -> Dict[str, int]:
        """Estimate remaining Jev calls for predict(), accounting for any checkpoint."""
        ckpt = self._read_predict_ckpt(self._predict_fingerprint(samples)) or {}
        done_ids = set(ckpt.get("completed", {}).keys())
        remaining = sum(1 for idx in samples.index if str(idx) not in done_ids)
        n_used = int(np.count_nonzero(self.lr.coef_[0]))
        return {
            f"{self._llmc_label(self.decision_llmc)} (predict)": remaining
            * self._jev_requests_per_sample(n_used)
        }

    # ── Checkpointing ───────────────────────────────────────────────────────────

    # One checkpoint file spans the whole fit() pipeline (the batches' policies,
    # the merged list and the scored cells), so resuming never mixes progress
    # from runs that describe different policies or data.
    _FIT_CKPT_NAME = "fit_checkpoint.json"
    _PREDICT_CKPT_NAME = "predict_checkpoint.json"

    def _ckpt_path(self, name: str) -> Path:
        return self.save_path / name

    def _write_ckpt(self, name: str, data: dict) -> None:
        """Atomically write a checkpoint file."""
        self.save_path.mkdir(parents=True, exist_ok=True)
        path = self._ckpt_path(name)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(orjson.dumps(data))
        tmp.replace(path)

    def _read_ckpt(self, name: str) -> dict | None:
        path = self._ckpt_path(name)
        return orjson.loads(path.read_bytes()) if path.exists() else None

    def _del_ckpt(self, *names: str) -> None:
        for name in names:
            p = self._ckpt_path(name)
            if p.exists():
                p.unlink()
        # Remove directories the checkpoint's mkdir(parents=True) created that
        # are now empty; rmdir raises OSError on the first non-empty ancestor,
        # so this never deletes a directory with content.
        d = self.save_path
        while d != d.parent:
            try:
                d.rmdir()
            except OSError:
                break
            d = d.parent

    def _fit_fingerprint(self) -> str:
        """Identifies a fit() run: the task, the generation setup and the data.

        A checkpoint written for anything else (other data, other batches, a
        different task) must not be resumed.
        """
        digest = hashlib.sha256()
        digest.update(
            repr(
                (
                    self._task_description,
                    self._pgen_instructions_template,
                    self.gen_batches,
                    self.max_raw_policies,
                    self.random_state,
                    self.decision_llmc.model,
                )
            ).encode()
        )
        if self._X is not None and self._y is not None:
            digest.update(repr(self._X.to_dict("split")).encode())
            digest.update("".join(self._y.tolist()).encode())
        return digest.hexdigest()

    def _load_fit_ckpt(self) -> dict:
        """The fit checkpoint, if this run wrote it; otherwise empty."""
        ckpt = self._read_ckpt(self._FIT_CKPT_NAME)
        if ckpt is None:
            return {}
        if ckpt.get("run") != self._fit_fingerprint():
            logger.info("Ignoring a fit checkpoint written for another run.")
            return {}
        return ckpt

    def _predict_fingerprint(self, samples: pd.DataFrame) -> str:
        """Identifies a predict() run: the model, and the samples' index and values.

        Checkpoint entries are keyed by index, so resuming one for other rows
        that share it, or after a refit or a threshold change, would replay
        answers that no longer apply.
        """
        lr = self._lr
        model = (
            self._policies,
            lr.coef_.tolist() if lr is not None else None,
            np.asarray(lr.intercept_).tolist() if lr is not None else None,
            self._threshold,
        )
        digest = hashlib.sha256(repr(model).encode())
        for idx, row in samples.iterrows():
            digest.update(f"{idx!r}\x1f{list(row.items())!r}\x1e".encode())
        return digest.hexdigest()

    def _read_predict_ckpt(self, fingerprint: str) -> dict | None:
        """The predict checkpoint, if this model wrote it for these samples."""
        ckpt = self._read_ckpt(self._PREDICT_CKPT_NAME)
        if ckpt is not None and ckpt.get("run") != fingerprint:
            logger.info(
                "Ignoring a predict checkpoint written for another model or "
                "other samples."
            )
            return None
        return ckpt

    # ── Policy generation ───────────────────────────────────────────────────────

    async def _generate_batch(self, rows: NDArray[np.intp]) -> List[str]:
        """Policies from one labelled batch, written without seeing any other."""
        X, y = cast(pd.DataFrame, self._X), cast(NDArray[np.str_], self._y)
        samples = "\n\n".join(
            f"[{y[i]}]\n{self._render_sample(X.iloc[i])}" for i in rows
        )
        query = (
            f"TASK DESCRIPTION:\n{self._task_description}\n\n"
            f"LABELLED SAMPLES:\n{samples}\n\n"
        )
        response = await self._llm_instance.respond(
            query=query,
            llm_priority=self.gen_llmc,
            response_format=Policies,
            instructions=self._pgen_instructions_template,
            temperature=_GEN_TEMPERATURE,
        )
        await self._token_counter.append(
            provider=response.provider_model.provider,
            model=response.provider_model.model,
            value=response.total_tokens,
            caller="PolicyInduction.generate",
        )
        if response.response is None:
            raise LLMError("No response from LLM for a generation batch.")
        return [p.strip() for p in response.response.policies if p.strip()]

    async def _merge_policies(self, raw: List[str]) -> List[str]:
        """Combine the batches' policies into at most ``max_raw_policies``."""
        unique = list(dict.fromkeys(raw))
        if not unique:
            raise LLMError("The generation batches produced no policies.")
        cap = self.max_raw_policies
        instructions = POLICY_MERGE_INSTRUCTIONS.format(max_policies=cap)
        query = (
            f"TASK DESCRIPTION:\n{self._task_description}\n\n"
            "POLICIES:\n" + "\n".join(f"- {p}" for p in unique)
        )
        merged: List[str] = []
        for _ in range(2):
            response = await self._llm_instance.respond(
                query=query,
                llm_priority=self.gen_llmc,
                response_format=Policies,
                instructions=instructions,
                temperature=_MERGE_TEMPERATURE,
            )
            await self._token_counter.append(
                provider=response.provider_model.provider,
                model=response.provider_model.model,
                value=response.total_tokens,
                caller="PolicyInduction.merge",
            )
            if response.response is None:
                raise LLMError("No response from LLM for the merge step.")
            merged = list(
                dict.fromkeys(
                    p.strip() for p in response.response.policies if p.strip()
                )
            )
            if 0 < len(merged) <= cap:
                return merged
            query += (
                f"\n\nYour previous answer had {len(merged)} policies. "
                f"Return at most {cap}."
            )
        logger.warning(
            "Merge returned %d policies, over the cap of %d; keeping the first %d.",
            len(merged),
            cap,
            cap,
        )
        return (merged or unique)[:cap]

    async def _generate_policies(self) -> None:
        """Generate batches independently, merge them, and checkpoint each step."""
        if self._policies:
            logger.info(f"Skipping generation: {len(self._policies)} policies held.")
            return
        if not self._pgen_instructions_template:
            raise ValueError("Call set_task() before fit().")

        ckpt = self._load_fit_ckpt()
        ckpt["run"] = self._fit_fingerprint()
        batches: Dict[str, List[str]] = ckpt.setdefault("batches", {})

        def flush() -> None:
            self._write_ckpt(self._FIT_CKPT_NAME, ckpt)

        if not ckpt.get("merged"):
            rows_by_batch = self._batch_indices()
            todo = [i for i in range(len(rows_by_batch)) if str(i) not in batches]

            async def run(i: int) -> None:
                batches[str(i)] = await self._generate_batch(rows_by_batch[i])
                logger.info(f"Batch {i + 1}: {len(batches[str(i)])} policies")
                flush()

            if todo:
                try:
                    async with asyncio.TaskGroup() as tg:
                        for i in todo:
                            tg.create_task(run(i))
                except ExceptionGroup as eg:
                    raise eg.exceptions[0] from None
            raw = [p for i in sorted(batches, key=int) for p in batches[i]]
            ckpt["merged"] = await self._merge_policies(raw)
            ckpt.setdefault("scores", {})
            flush()

        self._policies = list(ckpt["merged"])
        logger.info(f"Generation complete: {len(self._policies)} policies.")

    # ── Policy scoring ──────────────────────────────────────────────────────────

    @staticmethod
    def _jev_question(policy: str) -> NoulQuestion:
        return NoulQuestion(instructions=POLICY_JEV_QUESTION.format(policy=policy))

    def _jev_state(self, sample: str) -> str:
        return POLICY_JEV_STATE.format(
            task_description=self._task_description, sample=sample
        )

    async def _score_policies(self, budget: JevBudget) -> None:
        """Score every unscored (sample, policy) cell; checkpoint as it goes.

        Each sample is one Jev request carrying all of its unscored policies.
        """
        X = cast(pd.DataFrame, self._X)
        n, k = len(X), len(self._policies)
        scores = np.full((n, k), np.nan)

        ckpt = self._load_fit_ckpt()
        saved: Dict[str, Dict[str, float]] = (
            ckpt.get("scores", {}) if ckpt.get("merged") == self._policies else {}
        )
        for pid, cells in saved.items():
            for sample, value in cells.items():
                scores[int(sample), int(pid)] = value
        restored = int(np.count_nonzero(~np.isnan(scores)))
        if restored:
            logger.info(f"Restored scoring checkpoint: {restored} cells pre-scored.")
        self._scores = scores

        def flush() -> None:
            ckpt["run"] = self._fit_fingerprint()
            ckpt["merged"] = list(self._policies)
            ckpt["scores"] = {
                str(p): {
                    str(s): float(scores[s, p])
                    for s in range(n)
                    if not np.isnan(scores[s, p])
                }
                for p in range(k)
            }
            self._write_ckpt(self._FIT_CKPT_NAME, ckpt)

        pending = [s for s in range(n) if np.isnan(scores[s]).any()]
        logger.info(f"Scoring {len(pending)} samples against {k} policies.")
        pbar = tqdm(
            total=n,
            initial=n - len(pending),
            desc="[SCORE]",
            unit="",
            bar_format=_BAR_FORMAT,
            ascii=_BAR_ASCII,
        )
        try:
            for start in range(0, len(pending), _SCORE_CHUNK):
                chunk = pending[start : start + _SCORE_CHUNK]
                requests = [
                    JevRequest(
                        state=self._jev_state(self._render_sample(X.iloc[s])),
                        questions={
                            f"p{p}": self._jev_question(self._policies[p])
                            for p in range(k)
                            if np.isnan(scores[s, p])
                        },
                    )
                    for s in chunk
                ]
                try:
                    results = await self._llm_instance.answer_many(
                        [self.decision_llmc],
                        requests,
                        budget=budget,
                        token_counter=self._token_counter,
                        caller="PolicyInduction.score_policy",
                    )
                except (JevAuthError, JevCostCapError):
                    raise
                except LLMError as e:
                    logger.warning(
                        "Scoring failed for %d samples; they stay unscored: %s",
                        len(chunk),
                        e,
                    )
                    continue
                for s, result in zip(chunk, results):
                    for qid, value in result.answers.items():
                        if isinstance(value, float):
                            scores[s, int(qid[1:])] = value
                flush()
                pbar.update(len(chunk))
        finally:
            pbar.close()
            flush()

        missing = int(np.isnan(scores).sum())
        if missing == scores.size:
            raise LLMError("No (sample, policy) cell could be scored.")
        if missing:
            logger.warning(
                f"{missing} of {scores.size} (sample, policy) cells could not be "
                "scored and count as 0."
            )
        logger.info("Scoring complete.")

    # ── Weight fitting ──────────────────────────────────────────────────────────

    def _feature_matrix(self) -> Tuple[NDArray[np.float64], NDArray[np.int_]]:
        if self._scores is None or self._y is None:
            raise ValueError("Policies must be scored before weights are fitted.")
        X = np.nan_to_num(self._scores, nan=0.0)
        y = (self._y == "YES").astype(int)
        return X, y

    def _new_lr(self, C: float, max_iter: int) -> LogisticRegression:
        cfg = self.config
        return LogisticRegression(
            C=C,
            **_penalty_kwargs(cfg.penalty),
            solver="liblinear",
            max_iter=max_iter,
            class_weight="balanced" if cfg.class_weight_balanced else None,
            random_state=cfg.random_state,
        )

    def _fit_weights(self) -> None:
        """Pick the regularisation strength by CV and fit the final model.

        With ``one_se_rule`` the strongest ``C`` within one standard error of
        the best CV F-beta wins, so the model keeps as few policies as it can.
        A ``C`` that leaves every policy at zero weight is never chosen.

        Raises:
            ValueError: ``config.Cs`` is empty, or no candidate keeps a policy.
            DataError: A class has too few rows for cross-validation.
        """
        cfg = self.config
        X, y = self._feature_matrix()
        n_samples, n_policies = X.shape
        logger.info(f"Fitting weights: {n_samples} samples × {n_policies} policies.")

        Cs = sorted(float(c) for c in cfg.Cs)
        if not Cs:
            raise ValueError("config.Cs must contain at least one candidate C value.")
        folds = min(cfg.cv_folds, int(np.bincount(y).min()))
        if folds < 2:
            raise DataError("Each class needs at least 2 rows for cross-validation.")
        skf = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=cfg.random_state
        )

        means: Dict[float, float] = {}
        errors: Dict[float, float] = {}
        thresholds: Dict[float, List[float]] = {}
        for C in tqdm(
            Cs,
            desc="[FIT]",
            unit="",
            bar_format=_BAR_FORMAT,
            ascii=_BAR_ASCII,
        ):
            fold_scores, fold_thresholds = [], []
            for tr_idx, val_idx in skf.split(X, y):
                lr = self._new_lr(C, max_iter=500)
                lr.fit(X[tr_idx], y[tr_idx])
                probs = lr.predict_proba(X[val_idx])[:, 1]
                best_f, best_t = -1.0, 0.5
                for t in cfg.threshold_grid:
                    f = fbeta_score(
                        y[val_idx],
                        (probs >= t).astype(int),
                        beta=cfg.beta,
                        zero_division=0,  # type: ignore
                    )
                    if f > best_f:
                        best_f, best_t = f, t
                fold_scores.append(best_f)
                fold_thresholds.append(best_t)
            means[C] = float(np.mean(fold_scores))
            errors[C] = float(np.std(fold_scores, ddof=1) / math.sqrt(folds))
            thresholds[C] = fold_thresholds
            logger.info(f"C={C} → mean F{cfg.beta}={means[C]:.5f}")

        best_C = max(Cs, key=lambda c: means[c])
        eligible = (
            [c for c in Cs if means[c] >= means[best_C] - errors[best_C]]
            if cfg.one_se_rule
            else [best_C]
        )

        final_lr: LogisticRegression | None = None
        chosen_C = best_C
        for C in eligible:  # ascending, so strongest regularisation first
            lr = self._new_lr(C, max_iter=1000)
            lr.fit(X, y)
            if np.count_nonzero(lr.coef_[0]):
                final_lr, chosen_C = lr, C
                break
        if final_lr is None:
            raise ValueError(
                "Every policy has zero weight at the chosen regularisation. "
                "Add stronger-signal policies, or raise config.Cs."
            )

        self._lr = final_lr
        best_thresholds = thresholds[chosen_C]
        self.threshold = float(np.median(best_thresholds))
        used = int(np.count_nonzero(final_lr.coef_[0]))
        self._validation_result = {
            "best_C": chosen_C,
            "avg_cv_fbeta": means[chosen_C],
            "beta": cfg.beta,
            "thresholds_per_fold": best_thresholds,
            "recommended_threshold": self._threshold,
            "policies_used": used,
            "config": asdict(cfg),
        }
        logger.info(
            f"Fit complete. C={chosen_C}, F{cfg.beta}={means[chosen_C]:.4f}, "
            f"threshold={self._threshold:.3f}, {used}/{n_policies} policies used."
        )

    # ── Prediction ──────────────────────────────────────────────────────────────

    def _lr_predict(self, raw: NDArray) -> Literal["YES", "NO"]:
        if self._lr is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        vec = np.asarray(raw, dtype=float).reshape(1, -1)
        prob = self._lr.predict_proba(vec)[0, 1]
        return "YES" if prob >= self._threshold else "NO"

    async def _predict_chunk(
        self,
        samples: pd.DataFrame,
        used: List[int],
        budget: JevBudget,
        token_counter: TokenCounter,
    ) -> List[Tuple[Any, NDArray, Literal["YES", "NO"]]]:
        """Predict a few samples with one Jev request each, on weighted policies."""
        questions: Dict[str, JevQuestion] = {
            f"p{p}": self._jev_question(self._policies[p]) for p in used
        }
        rows = list(samples.iterrows())
        results = await self._llm_instance.answer_many(
            [self.decision_llmc],
            [
                JevRequest(
                    state=self._jev_state(self._render_sample(row)),
                    questions=questions,
                )
                for _, row in rows
            ],
            budget=budget,
            token_counter=token_counter,
            caller="PolicyInduction.predict_single",
        )
        records: List[Tuple[Any, NDArray, Literal["YES", "NO"]]] = []
        for (idx, _), result in zip(rows, results):
            vec = np.zeros(len(self._policies), dtype=float)
            answered = 0
            for qid, value in result.answers.items():
                if isinstance(value, float):
                    answered += 1
                    vec[int(qid[1:])] = value
            # No answers at all is no evidence, so no prediction; a partial
            # miss counts as 0 for the missing policies.
            if not answered:
                logger.warning(
                    "Predict failed for sample %s: no policy answers from Jev; "
                    "skipping.",
                    idx,
                )
                continue
            if answered < len(questions):
                logger.warning(
                    "Sample %s: %d/%d policy answers missing; counting them as 0.",
                    idx,
                    len(questions) - answered,
                    len(questions),
                )
            records.append((idx, vec, self._lr_predict(vec)))
        return records

    # ── Public API ──────────────────────────────────────────────────────────────

    def get_memory(self) -> pd.DataFrame:
        """Return each policy with its scores (Jev's P(YES)) on the training set."""
        index = pd.RangeIndex(len(self._policies))
        preds = [
            pd.Series(self._scores[:, p], index=self._X.index)  # type: ignore[union-attr]
            if self._scores is not None and self._X is not None
            else None
            for p in index
        ]
        return pd.DataFrame(
            {"policy": self._policies, "predictions": pd.Series(preds, dtype=object)},
            index=index,
        )

    async def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[str],
    ) -> Self:
        """Fit the PolicyInduction model.

        Runs generation, merge, scoring and weight fitting in sequence. One
        checkpoint file spans all of them, so resuming after an interruption
        never restores progress for different policies or data.

        Args:
            X: Feature DataFrame.
            y: Labels ('YES'/'NO').

        Returns:
            Self.

        Raises:
            DataError: The data has too few YES or NO rows for ``gen_batches``.
        """
        # A model loaded without the key must not spend on generation first.
        require_typesafe_key(
            [self.decision_llmc], method="PolicyInduction", param="decision_llmc"
        )
        if not self._pgen_instructions_template:
            raise ValueError("Call set_task() before fit().")
        self._set_data(X, y)
        if not self._policies:
            self._batch_indices()  # raises before any LLM spend
        self._confirm_requests(self._estimate_fit_requests())
        start = time.monotonic()

        await self._generate_policies()
        await self._score_policies(JevBudget(self.decision_llmc.max_cost_usd))
        self._fit_weights()
        self._fit_duration_seconds = time.monotonic() - start
        self._fit_completed_at = datetime.now(timezone.utc).isoformat()
        self._del_ckpt(self._FIT_CKPT_NAME)
        logger.info("PolicyInduction fit complete.")
        return self

    async def predict(
        self, samples: pd.DataFrame
    ) -> AsyncGenerator[Tuple[Any, NDArray, Literal["YES", "NO"], TokenCounter], None]:
        """Yield predictions for each sample in the DataFrame.

        Only policies with a non-zero weight are sent to Jev. Automatically
        checkpoints to self.save_path as it goes, resuming a checkpoint found
        there only if it was written by the same fitted model for the same
        samples (same index and values). The checkpoint is deleted once every
        sample has been predicted.

        May yield fewer records than len(samples): a sample for which no
        policy answer could be obtained at all is skipped rather than given a
        prediction built from an all-zero feature vector. Skipped samples are
        left out of the checkpoint, which is retained so a later run retries
        them.

        Args:
            samples: DataFrame of samples to classify.

        Yields:
            (sample_index, policy_vector, prediction, token_counter). The vector
            holds Jev's P(YES) for each policy, 0 for policies the model does
            not use.
        """
        if not self._policies or self._lr is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        require_typesafe_key(
            [self.decision_llmc], method="PolicyInduction", param="decision_llmc"
        )
        self._confirm_requests(self._estimate_predict_requests(samples))

        token_counter = TokenCounter()
        completed: Dict[str, Tuple[List[float], str]] = {}
        remaining_samples = samples

        fingerprint = self._predict_fingerprint(samples)
        ckpt = self._read_predict_ckpt(fingerprint)
        if ckpt is not None:
            completed = {
                sidx: (v["vector"], v["prediction"])
                for sidx, v in ckpt["completed"].items()
            }
            token_counter = TokenCounter.from_dict(ckpt["token_counter"])
            idx_type = type(samples.index[0]) if len(samples.index) else str
            for sidx_str, (vec, pred) in completed.items():
                yield (
                    idx_type(sidx_str),
                    np.array(vec, dtype=float),
                    cast(Literal["YES", "NO"], pred),
                    token_counter,
                )
            done = set(completed.keys())
            remaining_samples = samples[~samples.index.map(str).isin(done)]
            if completed:
                logger.info(
                    f"Resumed predict checkpoint: {len(completed)} samples pre-scored."
                )

        def save_ckpt() -> None:
            self._write_ckpt(
                self._PREDICT_CKPT_NAME,
                {
                    "completed": {
                        sidx: {"vector": vec, "prediction": pred}
                        for sidx, (vec, pred) in completed.items()
                    },
                    "token_counter": token_counter.to_dict(),
                    "run": fingerprint,
                },
            )

        used = [int(p) for p in np.flatnonzero(self._lr.coef_[0])]
        budget = JevBudget(self.decision_llmc.max_cost_usd)
        had_failures = False
        pbar = tqdm(
            total=len(samples),
            initial=len(completed),
            desc="[PREDICT]",
            unit="",
            bar_format=_BAR_FORMAT,
            ascii=_BAR_ASCII,
        )
        try:
            for start in range(0, len(remaining_samples), _PREDICT_CHUNK):
                chunk = remaining_samples.iloc[start : start + _PREDICT_CHUNK]
                try:
                    records = await self._predict_chunk(
                        chunk, used, budget, token_counter
                    )
                except (JevAuthError, JevCostCapError):
                    save_ckpt()
                    raise
                except LLMError as e:
                    # Every request failed: skip these samples and keep the
                    # checkpoint so a later run retries them.
                    had_failures = True
                    logger.warning(
                        "Predict failed for %d samples; skipping: %s", len(chunk), e
                    )
                    continue
                if len(records) < len(chunk):
                    had_failures = True
                for sidx, vec, pred in records:
                    completed[str(sidx)] = (vec.tolist(), pred)
                save_ckpt()
                pbar.update(len(records))
                for record in records:
                    yield record + (token_counter,)
        finally:
            pbar.close()
        if not had_failures:
            self._del_ckpt(self._PREDICT_CKPT_NAME)

    # ── Persistence ─────────────────────────────────────────────────────────────

    @staticmethod
    def _format_duration(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m}m {s}s"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"

    def _build_report(self) -> str:
        """Build a human-readable summary of the last fit: time + ranked weights."""
        lines = [f"# PolicyInduction Report: {self.name}", ""]

        if self._fit_completed_at is not None:
            lines.append(f"**Fit completed:** {self._fit_completed_at}")
        if self._fit_duration_seconds is not None:
            lines.append(
                f"**Fit duration:** {self._format_duration(self._fit_duration_seconds)}"
            )
        lines.append(f"**Token usage:** {self._token_counter.to_dict()}")
        lines.append("")

        if self._validation_result is not None:
            v = self._validation_result
            lines.append("## Validation")
            lines.append(f"- Best C: {v['best_C']}")
            lines.append(f"- CV F{v['beta']}: {v['avg_cv_fbeta']:.4f}")
            lines.append(f"- Decision threshold: {v['recommended_threshold']:.4f}")
            lines.append("")

        if self._lr is not None and self._policies:

            def clean(text: object) -> str:
                return str(text).replace("|", "\\|").replace("\n", " ")

            weights = self._lr.coef_[0].tolist()
            ranked = sorted(enumerate(weights), key=lambda kv: abs(kv[1]), reverse=True)
            used = [(p, w) for p, w in ranked if w != 0]
            dropped = [(p, w) for p, w in ranked if w == 0]

            lines.append(
                f"## Policies used by the model ({len(used)}/{len(ranked)}, "
                "ranked by |weight|)"
            )
            lines.append("")
            lines.append("| Rank | Weight | Policy |")
            lines.append("|---|---|---|")
            for rank, (p, w) in enumerate(used, 1):
                lines.append(f"| {rank} | {w:+.4f} | {clean(self._policies[p])} |")
            lines.append("")

            if dropped:
                lines.append(f"## Policies dropped (zero weight, {len(dropped)})")
                lines.append("")
                for p, _ in dropped:
                    lines.append(f"- {clean(self._policies[p])}")
                lines.append("")
        else:
            lines.append("## Policies")
            lines.append("Model has not been fitted yet — no weights available.")
            lines.append("")

        return "\n".join(lines)

    def save(
        self,
        dir_path: str | PathLike[str] | None = None,
        for_production: bool = False,
    ) -> None:
        """Persist model state to disk.

        Layout::

            policy_induction.json        manifest, config, state
            policies.parquet             policy texts
            policy_scores.parquet        Jev's P(YES) matrix    (dev only)
            data.parquet                 training data          (dev only)
            lr.joblib                    trained logistic regression
            report.md                    human-readable fit summary

        Args:
            dir_path: Target directory. Defaults to self.save_path.
            for_production: Strip training data; keep inference artifacts only.
        """
        base = Path(dir_path) if dir_path else self.save_path
        if base.is_file():
            raise ValueError("dir_path must be a directory, not a file.")
        base.mkdir(parents=True, exist_ok=True)
        if for_production:
            # Training files an earlier save left here must not survive.
            for name in ("data.parquet", "policy_scores.parquet"):
                (base / name).unlink(missing_ok=True)

        # Training data (dev mode only)
        if not for_production:
            if (self._X is None) != (self._y is None):
                raise ValueError(
                    "Corrupted state: X and y must both be set or both None."
                )
            if self._X is not None:
                df = deepcopy(self._X)
                df["y"] = self._y
                df.to_parquet(base / "data.parquet")

        # Policy texts
        pd.DataFrame(
            {
                "policy_id": [str(i) for i in range(len(self._policies))],
                "policy": self._policies,
            }
        ).to_parquet(base / "policies.parquet")

        # Scores on the training set (dev mode only)
        if not for_production and self._scores is not None and self._X is not None:
            scores_df = pd.DataFrame(
                self._scores,
                index=self._X.index,
                columns=pd.Index([str(i) for i in range(len(self._policies))]),
            )
            scores_df.to_parquet(base / "policy_scores.parquet")

        # LR model
        model_file = None
        if self._lr is not None:
            model_file = "lr.joblib"
            joblib_dump(self._lr, base / model_file)

        manifest = {
            "version": _MANIFEST_VERSION,
            "name": self.name,
            "gen_llmc": [
                lc if isinstance(lc, dict) else lc.model_dump() for lc in self.gen_llmc
            ],
            "decision_llmc": self.decision_llmc.model_dump(),
            "gen_batches": [list(self.gen_batches[0]), self.gen_batches[1]],
            "max_raw_policies": self.max_raw_policies,
            "random_state": self.random_state,
            "task_description": self._task_description,
            "policy_gen_instructions_template": (
                self._pgen_instructions_template if not for_production else None
            ),
            "token_counter": None if for_production else self._token_counter.to_dict(),
            "save_path": str(self.save_path) if not for_production else None,
            "threshold": self._threshold,
            "validation_result": self._validation_result,
            "n_policies": len(self._policies),
            "model_file": model_file,
            "config": asdict(self.config),
        }
        (base / "policy_induction.json").write_bytes(
            orjson.dumps(manifest, option=orjson.OPT_SERIALIZE_NUMPY)
        )

        # Human-readable report
        (base / "report.md").write_text(self._build_report(), encoding="utf-8")

        logger.info(f"Model saved to {base}")

    @classmethod
    def _load(cls, dir_path: str | PathLike[str]) -> "PolicyInduction":
        from joblib import load as joblib_load

        base = Path(dir_path)
        if not base.is_dir():
            raise ValueError("dir_path must be a directory.")
        manifest_path = base / "policy_induction.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"policy_induction.json not found in {base}")

        m = orjson.loads(manifest_path.read_bytes())
        if m.get("version", 0) < _MANIFEST_VERSION or "decision_llmc" not in m:
            raise ValueError(
                "This PolicyInduction was saved by an older version, which scored "
                "policies with chat models. Its policies and weights do not apply "
                "to Jev's probabilities; refit the model."
            )

        (pos, neg), n = m["gen_batches"]
        inst = cls(
            gen_llmc=m["gen_llmc"],
            decision_llmc=m["decision_llmc"],
            gen_batches=((pos, neg), n),
            max_raw_policies=m["max_raw_policies"],
            config=m.get("config"),
            random_state=m["random_state"],
            save_path=str(base),
            name=m["name"],
        )
        inst._task_description = m.get("task_description")
        inst._pgen_instructions_template = m.get("policy_gen_instructions_template")
        if tk := m.get("token_counter"):
            inst._token_counter = TokenCounter.from_dict(tk)
        inst._threshold = m.get("threshold", 0.0)
        inst._validation_result = m.get("validation_result")

        # Training data
        data_path = base / "data.parquet"
        if data_path.exists():
            data_df = pd.read_parquet(data_path)
            if "y" in data_df.columns:
                inst._y = data_df["y"].to_numpy(dtype=np.str_)
                inst._X = data_df.drop(columns=["y"])

        # Policies
        policies_path = base / "policies.parquet"
        if not policies_path.exists():
            raise FileNotFoundError(f"policies.parquet not found in {base}")
        p_df = pd.read_parquet(policies_path)
        if not {"policy_id", "policy"}.issubset(p_df.columns):
            raise ValueError(
                "policies.parquet must contain 'policy_id' and 'policy' columns."
            )
        inst._policies = (
            p_df.assign(_k=p_df["policy_id"].astype(int))
            .sort_values("_k")["policy"]
            .tolist()
        )

        # Scores on the training set
        scores_path = base / "policy_scores.parquet"
        if scores_path.exists() and inst._X is not None:
            inst._scores = pd.read_parquet(scores_path).to_numpy(dtype=float)

        # LR model
        if mf := m.get("model_file"):
            mp = base / mf
            if mp.exists():
                inst._lr = joblib_load(mp)

        return inst

    @classmethod
    def load(cls, dir_path: str | PathLike[str]) -> "PolicyInduction":
        """Load a previously saved PolicyInduction instance.

        Args:
            dir_path: Directory produced by save().

        Raises:
            ValueError: The model was saved by a version that scored policies
                with chat models, or the manifest is corrupted.
        """
        try:
            with loading_saved_model():
                return cls._load(dir_path)
        except KeyError as e:
            raise ValueError(f"Manifest corrupted or missing key: {e}") from e


__all__ = ["PolicyInduction", "Policies", "WeightTrainerConfig"]
