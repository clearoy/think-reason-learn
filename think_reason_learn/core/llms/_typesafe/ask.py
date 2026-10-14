from typing import Any, Dict, List
import asyncio
import logging
import time

import httpx

from .._schemas import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionQuestion,
    DecisionResponse,
    PredicateAnswer,
    PredicateQuestion,
    ScoreAnswer,
    TypeSafeDecisionChoice,
)
from .schemas import TypeSafeDecisionModel
from think_reason_learn.core._singleton import SingletonMeta


logger = logging.getLogger(__name__)

BASE_URL = "https://api.typesafe.ai"
RETRY_STATUSES = {429, 529}
MAX_RETRIES = 3


def _build_question(q: DecisionQuestion) -> Dict[str, Any]:
    if isinstance(q, PredicateQuestion):
        out: Dict[str, Any] = {"type": "noul", "instructions": q.instructions}
        if q.true_criteria or q.false_criteria:
            out["criteria"] = {
                "true": q.true_criteria or "",
                "false": q.false_criteria or "",
            }
        return out
    if isinstance(q, ChoiceQuestion):
        return {
            "type": "choice",
            "instructions": q.instructions,
            "criteria": q.choices,
        }
    return {"type": "score", "instructions": q.instructions, "criteria": q.levels}


def _parse_answer(name: str, raw: Dict[str, Any]) -> DecisionAnswer:
    kind = raw.get("type")
    if kind == "noul":
        return PredicateAnswer(name=name, probability=raw["noul"])
    if kind == "choice":
        return ChoiceAnswer(
            name=name,
            choice=raw["choice"],
            probabilities=raw["probabilities"],
            confidence=raw.get("confidence"),
        )
    if kind == "score":
        return ScoreAnswer(
            name=name,
            score=raw["score"],
            probabilities={str(k): v for k, v in raw["probabilities"].items()},
            confidence=raw.get("confidence"),
        )
    raise ValueError(f"Unknown TypeSafe answer type: {kind!r}")


class TypeSafeLLM(metaclass=SingletonMeta):
    """Client for TypeSafe's System One models (e.g. Jev)."""

    def __init__(self, api_key: str, timeout: float = 120.0) -> None:
        headers = {"Authorization": f"Bearer {api_key}"}
        self.client = httpx.Client(base_url=BASE_URL, headers=headers, timeout=timeout)
        self.aclient = httpx.AsyncClient(
            base_url=BASE_URL, headers=headers, timeout=timeout
        )

    def _body(
        self,
        model: TypeSafeDecisionModel,
        state: str,
        questions: List[DecisionQuestion],
    ) -> Dict[str, Any]:
        return {
            "model": model,
            "state": state,
            "questions": {q.name: _build_question(q) for q in questions},
        }

    def _parse(
        self, model: TypeSafeDecisionModel, data: Dict[str, Any]
    ) -> DecisionResponse:
        usage = data.get("usage") or {}
        total = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        return DecisionResponse(
            answers={k: _parse_answer(k, v) for k, v in data["answers"].items()},
            total_tokens=total if usage else None,
            provider_model=TypeSafeDecisionChoice(model=model),
        )

    def decide_sync(
        self,
        model: TypeSafeDecisionModel,
        state: str,
        questions: List[DecisionQuestion],
        raise_: bool = False,
    ) -> DecisionResponse | None:
        body = self._body(model, state, questions)
        try:
            for attempt in range(MAX_RETRIES + 1):
                response = self.client.post("/v1/systemone", json=body)
                if response.status_code in RETRY_STATUSES and attempt < MAX_RETRIES:
                    time.sleep(2**attempt)
                    continue
                response.raise_for_status()
                return self._parse(model, response.json())
        except Exception as e:
            logger.warning(f"Error deciding with TypeSafe: {e}", exc_info=True)
            if raise_:
                raise e
        return None

    async def decide(
        self,
        model: TypeSafeDecisionModel,
        state: str,
        questions: List[DecisionQuestion],
        raise_: bool = False,
    ) -> DecisionResponse | None:
        body = self._body(model, state, questions)
        try:
            for attempt in range(MAX_RETRIES + 1):
                response = await self.aclient.post("/v1/systemone", json=body)
                if response.status_code in RETRY_STATUSES and attempt < MAX_RETRIES:
                    await asyncio.sleep(2**attempt)
                    continue
                response.raise_for_status()
                return self._parse(model, response.json())
        except Exception as e:
            logger.warning(f"Error deciding with TypeSafe: {e}", exc_info=True)
            if raise_:
                raise e
        return None


def get_typesafe_llm(api_key: str) -> TypeSafeLLM | None:
    return TypeSafeLLM(api_key) if api_key else None
