"""Offline tests for PolicyInduction: generation, merge, Jev scoring, fit, predict."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pandas as pd
import pytest

from think_reason_learn.core.exceptions import DataError, JevCostCapError, LLMError
from think_reason_learn.core.llms import JevBudget, JevChoice, NoulQuestion
from think_reason_learn.core.llms import OpenAIChoice
from think_reason_learn.policy_induction import PolicyInduction, WeightTrainerConfig
from think_reason_learn.policy_induction import _policy_induction as pi_module
from tests.fake_policy_llm import STRONG, TEMPLATE, FakePolicyLLM

pytestmark = pytest.mark.usefixtures("typesafe_key")

TASK = "Predict whether the founder succeeds."
ROWS = [f"strong founder {i}" for i in range(10)] + [
    f"weak founder {i}" for i in range(10)
]
LABELS = ["YES"] * 10 + ["NO"] * 10
X = pd.DataFrame({"text": ROWS})


def make_pi(tmp_path: Path, fake: FakePolicyLLM, **kw: Any) -> PolicyInduction:
    kw.setdefault("gen_batches", ((2, 2), 2))
    kw.setdefault("max_raw_policies", 6)
    return PolicyInduction(
        gen_llmc=[OpenAIChoice(model="gpt-4.1-nano")],
        decision_llmc=JevChoice(max_cost_usd=2.0),
        config={"cv_folds": 2},
        save_path=tmp_path,
        name="pi",
        confirm_requests=False,
        _llm=fake,
        **kw,
    )


async def fitted(
    tmp_path: Path, fake: FakePolicyLLM | None = None, **kw: Any
) -> PolicyInduction:
    pi = make_pi(tmp_path, fake or FakePolicyLLM(), **kw)
    await pi.set_task(TASK)
    return await pi.fit(X, LABELS)


# ── Construction ───────────────────────────────────────────────────────────────


def test_defaults() -> None:
    pi = PolicyInduction(gen_llmc=[OpenAIChoice(model="gpt-4.1-nano")])

    assert pi.decision_llmc == JevChoice()
    assert pi.gen_batches == ((5, 5), 7)
    assert pi.max_raw_policies == 30
    cfg = pi.config
    assert cfg.penalty == "l1" and cfg.one_se_rule
    assert max(cfg.Cs) <= 1.0 and min(cfg.Cs) <= 0.01


def test_decision_llmc_accepts_a_jev_dict() -> None:
    pi = PolicyInduction(
        gen_llmc=[OpenAIChoice(model="gpt-4.1-nano")],
        decision_llmc={"provider": "jev", "model": "jev-latest", "max_cost_usd": 3.0},
    )
    assert pi.decision_llmc == JevChoice(max_cost_usd=3.0)


@pytest.mark.parametrize(
    "bad",
    [
        OpenAIChoice(model="gpt-4.1-nano"),
        {"provider": "openai", "model": "gpt-4.1-nano"},
    ],
)
def test_only_jev_scores(bad: Any) -> None:
    with pytest.raises(ValueError, match="Jev"):
        PolicyInduction(
            gen_llmc=[OpenAIChoice(model="gpt-4.1-nano")], decision_llmc=bad
        )


@pytest.mark.parametrize(
    "bad", [(5, 7), ((5, 5), 0), ((0, 0), 3), ((-1, 5), 3), ((5.0, 5), 3), "x", None]
)
def test_bad_gen_batches_are_refused(bad: Any) -> None:
    with pytest.raises(ValueError, match="gen_batches"):
        PolicyInduction(gen_llmc=[OpenAIChoice(model="gpt-4.1-nano")], gen_batches=bad)


@pytest.mark.parametrize("bad", [0, -1, 500, 1.5, True, None])
def test_bad_max_raw_policies_are_refused(bad: Any) -> None:
    with pytest.raises(ValueError, match="max_raw_policies"):
        PolicyInduction(
            gen_llmc=[OpenAIChoice(model="gpt-4.1-nano")], max_raw_policies=bad
        )


# ── Generation batches ─────────────────────────────────────────────────────────


def test_batches_are_disjoint_balanced_and_reproducible(tmp_path: Path) -> None:
    pi = make_pi(tmp_path, FakePolicyLLM(), gen_batches=((3, 2), 3))
    pi._set_data(X, LABELS)

    batches = pi._batch_indices()

    assert len(batches) == 3
    assert all(len(b) == 5 for b in batches)
    flat = np.concatenate(batches)
    assert len(set(flat.tolist())) == len(flat)
    for b in batches:
        assert sum(pi._y[i] == "YES" for i in b) == 3  # type: ignore[index]
    again = make_pi(tmp_path, FakePolicyLLM(), gen_batches=((3, 2), 3))
    again._set_data(X, LABELS)
    assert [b.tolist() for b in again._batch_indices()] == [b.tolist() for b in batches]


def test_too_few_rows_for_the_batches_raises_with_a_way_out(tmp_path: Path) -> None:
    pi = make_pi(tmp_path, FakePolicyLLM(), gen_batches=((5, 5), 7))
    pi._set_data(X, LABELS)

    with pytest.raises(DataError, match=r"10 YES and 10 NO.*at most 2 fits"):
        pi._batch_indices()


@pytest.mark.asyncio
async def test_fit_refuses_before_any_spend_when_the_data_is_too_small(
    tmp_path: Path,
) -> None:
    fake = FakePolicyLLM()
    pi = make_pi(tmp_path, fake, gen_batches=((5, 5), 7))
    await pi.set_task(TASK)
    calls = len(fake.calls)

    with pytest.raises(DataError):
        await pi.fit(X, LABELS)

    assert len(fake.calls) == calls and fake.answer_requests == []


@pytest.mark.asyncio
async def test_each_batch_is_written_independently(tmp_path: Path) -> None:
    fake = FakePolicyLLM()
    pi = await fitted(tmp_path, fake)

    assert len(fake.gen_queries) == 2 and len(fake.merge_queries) == 1
    for query in fake.gen_queries:
        assert "EXISTING POLICIES" not in query
        assert "noise" not in query  # no other batch's policies
        assert query.count("[YES]") == 2 and query.count("[NO]") == 2
        assert TASK in query
    assert fake.template_calls == 1
    assert pi.policy_gen_instructions_template == TEMPLATE
    used = [c["instructions"] for c in fake.calls if c["query"] in fake.gen_queries]
    assert used == [TEMPLATE, TEMPLATE]


@pytest.mark.asyncio
async def test_set_task_is_always_written_by_the_llm(tmp_path: Path) -> None:
    pi = make_pi(tmp_path, FakePolicyLLM())

    assert await pi.set_task(TASK) == TEMPLATE
    assert pi.task_description == TASK
    with pytest.raises(TypeError):
        await pi.set_task(TASK, "my own template")  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_batches_run_concurrently(tmp_path: Path) -> None:
    import asyncio

    in_flight = 0
    peak = 0
    fake = FakePolicyLLM()
    original = fake.respond

    async def slow(*args: Any, **kwargs: Any) -> Any:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        try:
            return await original(*args, **kwargs)
        finally:
            in_flight -= 1

    fake.respond = slow  # type: ignore[method-assign]
    pi = make_pi(tmp_path, fake, gen_batches=((2, 2), 3))
    await pi.set_task(TASK)
    peak = 0
    await pi.fit(X, LABELS)

    assert peak == 3


# ── Merge ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_merge_compresses_to_the_cap_and_drops_repeats(tmp_path: Path) -> None:
    fake = FakePolicyLLM(noise_per_batch=5)  # 2 * 5 noise + 1 shared = 11 raw
    pi = await fitted(tmp_path, fake, max_raw_policies=4)

    assert len(pi.policies) == 4
    assert pi.policies[0] == STRONG
    assert fake.merge_queries[0].count("- " + STRONG) == 1
    assert (
        "AT MOST 4"
        in [c for c in fake.calls if "Merge" in str(c["instructions"])][0][
            "instructions"
        ]
    )


@pytest.mark.asyncio
async def test_merge_over_the_cap_is_asked_again_once(tmp_path: Path) -> None:
    too_many = [STRONG] + [f"extra {i}" for i in range(8)]
    fake = FakePolicyLLM(merge_replies=[too_many, [STRONG, "extra 0"]])
    pi = await fitted(tmp_path, fake, max_raw_policies=4)

    assert pi.policies == [STRONG, "extra 0"]
    assert len(fake.merge_queries) == 2
    assert "at most 4" in fake.merge_queries[1]


@pytest.mark.asyncio
async def test_merge_that_ignores_the_cap_twice_is_truncated(tmp_path: Path) -> None:
    too_many = [STRONG] + [f"extra {i}" for i in range(8)]
    fake = FakePolicyLLM(merge_replies=[too_many, too_many])
    pi = await fitted(tmp_path, fake, max_raw_policies=4)

    assert pi.policies == too_many[:4]


# ── Scoring ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_one_jev_request_per_sample_with_every_policy(tmp_path: Path) -> None:
    fake = FakePolicyLLM()
    pi = await fitted(tmp_path, fake)

    assert sorted(fake.answered_states.values()) == [1] * len(ROWS)
    for request in fake.answer_requests:
        assert len(request.questions) == len(pi.policies)
        for question, policy in zip(request.questions.values(), pi.policies):
            assert isinstance(question, NoulQuestion)
            assert policy in question.instructions


@pytest.mark.asyncio
async def test_policy_is_guidance_and_the_task_is_sent_once(tmp_path: Path) -> None:
    fake = FakePolicyLLM()
    await fitted(tmp_path, fake)

    request = fake.answer_requests[0]
    assert request.state.startswith(f"Task: {TASK}")
    assert request.state.count(TASK) == 1
    for question in request.questions.values():
        assert question.instructions.startswith("Use this policy as guidance")
        assert question.instructions.endswith("Is the answer to the task YES?")
        assert TASK not in question.instructions


@pytest.mark.asyncio
async def test_features_are_probabilities_not_yes_no(tmp_path: Path) -> None:
    pi = await fitted(tmp_path)

    assert pi._scores is not None
    strong = pi._scores[:, 0]
    assert set(np.round(strong, 6)) == {0.1, 0.9}
    assert set(np.round(pi._scores[:, 1], 6)) == {0.5}
    X_feat, _ = pi._feature_matrix()
    assert np.array_equal(X_feat, pi._scores)


@pytest.mark.asyncio
async def test_a_fit_shares_one_cost_budget_across_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pi_module, "_SCORE_CHUNK", 5)
    fake = FakePolicyLLM()
    await fitted(tmp_path, fake)

    budgets = {id(c["budget"]) for c in fake.answer_calls}
    assert len(fake.answer_calls) == 4 and len(budgets) == 1
    assert isinstance(fake.answer_calls[0]["budget"], JevBudget)
    assert fake.answer_calls[0]["budget"].remaining_usd == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_unanswered_cells_count_as_zero_and_are_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakePolicyLLM(
        prob=lambda s, i: None if "weak founder 3" in s and "noise" in i else 0.5
    )
    fake.prob = lambda s, i: (
        None if ("weak founder 3" in s) else (0.9 if "strong" in s else 0.1)
    )
    with caplog.at_level("WARNING"):
        pi = await fitted(tmp_path, fake)

    assert "could not be scored and count as 0" in caplog.text
    assert pi._scores is not None and np.isnan(pi._scores[13]).all()


@pytest.mark.asyncio
async def test_a_total_scoring_failure_raises(tmp_path: Path) -> None:
    fake = FakePolicyLLM(prob=lambda s, i: None)
    pi = make_pi(tmp_path, fake)
    await pi.set_task(TASK)

    with pytest.raises(LLMError, match="could be scored"):
        await pi.fit(X, LABELS)


# ── Fit: few policies, strong regularisation ───────────────────────────────────


@pytest.mark.asyncio
async def test_fit_keeps_only_the_informative_policy(tmp_path: Path) -> None:
    pi = await fitted(tmp_path)

    weights = pi.lr.coef_[0]
    assert weights[0] > 0
    assert np.count_nonzero(weights) == 1
    assert pi.validation_result["policies_used"] == 1
    assert 0.0 < pi.threshold < 1.0


@pytest.mark.asyncio
async def test_fit_removes_its_checkpoint_when_done(tmp_path: Path) -> None:
    await fitted(tmp_path)

    assert not (tmp_path / "fit_checkpoint.json").exists()


# ── Checkpointing ──────────────────────────────────────────────────────────────


def read_ckpt(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "fit_checkpoint.json").read_text())


@pytest.mark.asyncio
async def test_checkpoint_keeps_batches_merge_and_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pi_module, "_SCORE_CHUNK", 5)
    fake = FakePolicyLLM()
    calls = 0
    original = fake.answer_many

    async def capped(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise JevCostCapError("Spend passed the cap.")
        return await original(*args, **kwargs)

    fake.answer_many = capped  # type: ignore[method-assign]
    pi = make_pi(tmp_path, fake)
    await pi.set_task(TASK)
    with pytest.raises(JevCostCapError):
        await pi.fit(X, LABELS)

    ckpt = read_ckpt(tmp_path)
    assert set(ckpt["batches"]) == {"0", "1"}
    assert ckpt["merged"] == pi.policies
    # Two chunks of 5 samples were scored before the cap stopped the third.
    assert len(ckpt["scores"]["0"]) == 10
    assert set(ckpt["scores"]) == {str(p) for p in range(len(pi.policies))}


@pytest.mark.asyncio
async def test_resume_skips_generation_and_scored_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pi_module, "_SCORE_CHUNK", 5)
    first = FakePolicyLLM()
    calls = 0
    original = first.answer_many

    async def capped(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise JevCostCapError("Spend passed the cap.")
        return await original(*args, **kwargs)

    first.answer_many = capped  # type: ignore[method-assign]
    pi = make_pi(tmp_path, first)
    await pi.set_task(TASK)
    with pytest.raises(JevCostCapError):
        await pi.fit(X, LABELS)

    second = FakePolicyLLM()
    resumed = make_pi(tmp_path, second)
    resumed._task_description = TASK
    resumed._pgen_instructions_template = TEMPLATE
    await resumed.fit(X, LABELS)

    assert second.gen_queries == [] and second.merge_queries == []
    assert len(second.answer_requests) == 10  # only the 10 unscored samples
    assert resumed.policies == pi.policies
    assert resumed.lr.coef_[0][0] > 0


@pytest.mark.asyncio
async def test_a_checkpoint_for_other_data_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakePolicyLLM()
    pi = make_pi(tmp_path, fake)
    await pi.set_task(TASK)

    async def capped(*args: Any, **kwargs: Any) -> Any:
        raise JevCostCapError("Spend passed the cap.")

    fake.answer_many = capped  # type: ignore[method-assign]
    with pytest.raises(JevCostCapError):
        await pi.fit(X, LABELS)
    assert (tmp_path / "fit_checkpoint.json").exists()

    other = FakePolicyLLM()
    again = make_pi(tmp_path, other)
    await again.set_task(TASK)
    await again.fit(X.assign(text=[t + "!" for t in ROWS]), LABELS)

    assert len(other.gen_queries) == 2  # generated again, nothing reused


# ── Predict ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_predict_asks_only_weighted_policies(tmp_path: Path) -> None:
    fake = FakePolicyLLM()
    pi = await fitted(tmp_path, fake)
    fake.answer_requests.clear()
    fake.answer_calls.clear()

    X_new = pd.DataFrame({"text": ["strong founder x", "weak founder y"]})
    out = [(i, vec, pred) async for i, vec, pred, _ in pi.predict(X_new)]

    weighted = [p for p, w in enumerate(pi.lr.coef_[0]) if w != 0]
    assert len(fake.answer_calls) == 1 and len(fake.answer_requests) == 2
    assert all(len(r.questions) == len(weighted) for r in fake.answer_requests)
    by_index = {i: (vec, pred) for i, vec, pred in out}
    assert by_index[0][1] == "YES" and by_index[1][1] == "NO"
    assert by_index[0][0][0] == pytest.approx(0.9)  # P(YES), not 1.0
    assert by_index[1][0][0] == pytest.approx(0.1)
    assert by_index[0][0][1] == 0.0  # unused policy


@pytest.mark.asyncio
async def test_predict_skips_a_sample_jev_could_not_answer(tmp_path: Path) -> None:
    fake = FakePolicyLLM()
    pi = await fitted(tmp_path, fake)
    fake.prob = lambda state, instr: None if "unanswerable" in state else 0.9

    X_new = pd.DataFrame({"text": ["strong founder x", "unanswerable founder"]})
    out = [i async for i, _, _, _ in pi.predict(X_new)]

    assert out == [0]
    assert (tmp_path / pi._PREDICT_CKPT_NAME).exists()


@pytest.mark.asyncio
async def test_predict_skips_samples_when_every_request_fails(tmp_path: Path) -> None:
    fake = FakePolicyLLM()
    pi = await fitted(tmp_path, fake)

    async def all_failed(*args: Any, **kwargs: Any) -> Any:
        raise LLMError("Every Jev request failed.")

    fake.answer_many = all_failed  # type: ignore[method-assign]
    X_new = pd.DataFrame({"text": ["strong founder x", "weak founder y"]})

    for _ in range(2):  # the retry too
        assert [i async for i, _, _, _ in pi.predict(X_new)] == []


@pytest.mark.asyncio
async def test_predict_raises_when_jev_stops_on_its_cost_cap(tmp_path: Path) -> None:
    fake = FakePolicyLLM()
    pi = await fitted(tmp_path, fake)

    async def capped(*args: Any, **kwargs: Any) -> Any:
        raise JevCostCapError("Spend passed the cap.")

    fake.answer_many = capped  # type: ignore[method-assign]
    with pytest.raises(JevCostCapError):
        _ = [r async for r in pi.predict(pd.DataFrame({"text": ["strong x"]}))]


@pytest.mark.asyncio
async def test_predict_resumes_and_clears_its_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pi_module, "_PREDICT_CHUNK", 2)
    fake = FakePolicyLLM()
    pi = await fitted(tmp_path, fake)
    X_new = pd.DataFrame({"text": [f"strong founder {i}" for i in range(5)]})

    calls = 0
    original = fake.answer_many

    async def flaky(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise JevCostCapError("Spend passed the cap.")
        return await original(*args, **kwargs)

    fake.answer_many = flaky  # type: ignore[method-assign]
    with pytest.raises(JevCostCapError):
        _ = [r async for r in pi.predict(X_new)]
    assert (tmp_path / pi._PREDICT_CKPT_NAME).exists()

    fake.answer_many = original  # type: ignore[method-assign]
    fake.answer_requests.clear()
    out = [i async for i, _, _, _ in pi.predict(X_new)]

    assert sorted(out) == [0, 1, 2, 3, 4]
    assert len(fake.answer_requests) == 3  # the 3 samples not yet predicted
    assert not (tmp_path / pi._PREDICT_CKPT_NAME).exists()


@pytest.mark.asyncio
async def test_predict_before_fit_raises(tmp_path: Path) -> None:
    pi = make_pi(tmp_path, FakePolicyLLM())

    with pytest.raises(RuntimeError, match="not fitted"):
        _ = [r async for r in pi.predict(X)]


# ── Request estimates ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fit_estimate_counts_batches_merge_and_one_request_per_sample(
    tmp_path: Path,
) -> None:
    pi = make_pi(tmp_path, FakePolicyLLM())
    pi._set_data(X, LABELS)

    estimates = pi._estimate_fit_requests()

    assert estimates["gpt-4.1-nano (generation)"] == 2 + 1
    assert estimates["jev-latest (scoring)"] == len(ROWS)


@pytest.mark.asyncio
async def test_fit_estimate_is_zero_for_a_finished_checkpoint_run(
    tmp_path: Path,
) -> None:
    pi = await fitted(tmp_path)

    again = make_pi(tmp_path, FakePolicyLLM())
    again._policies = pi.policies
    again._set_data(X, LABELS)
    assert again._estimate_fit_requests()["gpt-4.1-nano (generation)"] == 0


# ── Persistence ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_load_round_trip(tmp_path: Path) -> None:
    pi = await fitted(tmp_path / "work")
    pi.save(tmp_path / "model")

    loaded = PolicyInduction.load(tmp_path / "model")

    assert loaded.policies == pi.policies
    assert loaded.threshold == pi.threshold
    assert loaded.gen_batches == pi.gen_batches
    assert loaded.max_raw_policies == pi.max_raw_policies
    assert loaded.decision_llmc == pi.decision_llmc
    assert np.allclose(loaded.lr.coef_, pi.lr.coef_)
    assert loaded.task_description == TASK
    assert loaded._scores is not None and np.allclose(loaded._scores, pi._scores)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_loaded_model_predicts(tmp_path: Path) -> None:
    pi = await fitted(tmp_path / "work")
    pi.save(tmp_path / "model", for_production=True)
    fake = FakePolicyLLM()
    loaded = PolicyInduction.load(tmp_path / "model")
    loaded._llm_instance = fake
    loaded.confirm_requests = False

    out = [
        (i, p)
        async for i, _, p, _ in loaded.predict(
            pd.DataFrame({"text": ["strong founder z", "weak founder z"]})
        )
    ]

    assert out == [(0, "YES"), (1, "NO")]


@pytest.mark.asyncio
async def test_production_save_drops_training_files(tmp_path: Path) -> None:
    pi = await fitted(tmp_path / "work")
    pi.save(tmp_path / "model")
    assert (tmp_path / "model" / "data.parquet").exists()
    assert (tmp_path / "model" / "policy_scores.parquet").exists()

    pi.save(tmp_path / "model", for_production=True)

    assert not (tmp_path / "model" / "data.parquet").exists()
    assert not (tmp_path / "model" / "policy_scores.parquet").exists()
    assert (tmp_path / "model" / "report.md").exists()


@pytest.mark.asyncio
async def test_report_lists_the_policies_used(tmp_path: Path) -> None:
    pi = await fitted(tmp_path)
    pi.save(tmp_path / "model")

    report = (tmp_path / "model" / "report.md").read_text()

    assert "Policies used by the model (1/" in report
    assert STRONG in report


def test_a_model_saved_by_the_chat_version_is_refused(tmp_path: Path) -> None:
    pi = make_pi(tmp_path, FakePolicyLLM())
    pi.save(tmp_path / "model")
    manifest_path = tmp_path / "model" / "policy_induction.json"
    manifest = orjson.loads(manifest_path.read_bytes())
    manifest["version"] = 3
    del manifest["decision_llmc"]
    manifest_path.write_bytes(orjson.dumps(manifest))

    with pytest.raises(ValueError, match="older version.*refit"):
        PolicyInduction.load(tmp_path / "model")


def test_config_cs_default_is_strong() -> None:
    cfg = WeightTrainerConfig()
    assert tuple(cfg.Cs) == (0.01, 0.03, 0.1, 0.3, 1.0)


# ── Through the real Jev client ────────────────────────────────────────────────


class _RealJev(FakePolicyLLM):
    """Fake generation, but ``answer_many`` goes through the real LLM and client."""

    def __init__(self, server: Any, cache_dir: Path) -> None:
        super().__init__()
        from think_reason_learn.core.llms import LLM
        from think_reason_learn.core.llms._jev.client import JevClient

        async def no_wait(seconds: float) -> None:
            return None

        self._real = LLM.__new__(LLM)
        self._real.jev_llm = JevClient(  # type: ignore[attr-defined]
            api_key="test-key",
            transport=server.transport(),
            cache_dir=cache_dir,
            sleep=no_wait,
        )

    async def answer_many(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return await self._real.answer_many(*args, **kwargs)


@pytest.mark.asyncio
async def test_fit_and_predict_through_the_real_jev_client(tmp_path: Path) -> None:
    import httpx

    from tests.fake_jev import FakeJevServer

    class ByFounder(FakeJevServer):
        """Strong founders get P(YES) 0.9 on every policy, weak ones 0.1."""

        def ok(self, body: dict[str, Any]) -> httpx.Response:
            p = 0.9 if "strong" in body["state"] else 0.1
            return httpx.Response(
                200,
                json={
                    "model": "jev-1.13.0",
                    "answers": {
                        q: {"type": "noul", "noul": p} for q in body["questions"]
                    },
                    "usage": {"input_tokens": 100, "output_tokens": 7},
                },
            )

    server = ByFounder()
    pi = make_pi(tmp_path / "pi", _RealJev(server, tmp_path / "cache"))
    await pi.set_task(TASK)

    await pi.fit(X, LABELS)

    assert len(server.bodies) == len(ROWS)  # one request per sample
    body = server.bodies[0]
    assert body["state"].startswith(f"Task: {TASK}")
    assert all(q["type"] == "noul" for q in body["questions"].values())
    assert len(body["questions"]) == len(pi.policies)
    assert pi._scores is not None
    assert set(np.round(pi._scores.ravel(), 6)) == {0.1, 0.9}  # as-is, not 0/1

    server.bodies.clear()
    X_new = pd.DataFrame({"text": ["strong founder x", "weak founder y"]})
    out = {i: p async for i, _, p, _ in pi.predict(X_new)}

    assert out == {0: "YES", 1: "NO"}
    assert len(server.bodies) == 2
    used = int(np.count_nonzero(pi.lr.coef_[0]))
    assert all(len(b["questions"]) == used for b in server.bodies)
