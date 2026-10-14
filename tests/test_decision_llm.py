"""Tests for the decision-model providers (OpenAI Decisions, TypeSafe Jev)."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from think_reason_learn.core.llms import (
    ChoiceQuestion,
    PredicateAnswer,
    PredicateQuestion,
    ScoreQuestion,
)
from think_reason_learn.core.llms._ask import LLM
from think_reason_learn.core.llms._openai.ask import OpenAILLM
from think_reason_learn.core.llms._typesafe.ask import TypeSafeLLM

QUESTIONS = [
    PredicateQuestion(name="urgent", instructions="Is it urgent?"),
    ChoiceQuestion(name="team", instructions="Who?", choices={"a": "A", "b": "B"}),
    ScoreQuestion(name="anger", instructions="How angry?", levels=["calm", "angry"]),
]


def _typesafe(handler) -> TypeSafeLLM:
    llm = TypeSafeLLM.__new__(TypeSafeLLM)
    llm.client = httpx.Client(
        base_url="https://api.typesafe.ai", transport=httpx.MockTransport(handler)
    )
    return llm


def test_typesafe_request_and_response():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "urgent": {"type": "noul", "noul": 0.9},
                    "team": {
                        "type": "choice",
                        "choice": "a",
                        "probabilities": {"a": 0.8, "b": 0.2},
                        "confidence": 0.7,
                    },
                    "anger": {
                        "type": "score",
                        "score": 0.4,
                        "legend": {"0": "calm", "1": "angry"},
                        "probabilities": {"0": 0.6, "1": 0.4},
                        "confidence": 0.5,
                    },
                },
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        )

    res = _typesafe(handler).decide_sync("jev-latest", "state", QUESTIONS, raise_=True)
    assert res is not None
    assert res.probability("urgent") == 0.9
    assert res.answers["team"].choice == "a"  # type: ignore[union-attr]
    assert res.answers["anger"].score == 0.4  # type: ignore[union-attr]
    assert res.total_tokens == 12
    body = seen["body"].decode()
    assert '"type":"noul"' in body and '"criteria":["calm","angry"]' in body


def test_typesafe_retries_on_429():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429)
        return httpx.Response(
            200, json={"answers": {"urgent": {"type": "noul", "noul": 0.1}}}
        )

    with patch("think_reason_learn.core.llms._typesafe.ask.time.sleep"):
        res = _typesafe(handler).decide_sync("jev-latest", "s", QUESTIONS[:1])
    assert res is not None and res.probability("urgent") == 0.1
    assert len(calls) == 2


def test_openai_request_and_response():
    llm = OpenAILLM.__new__(OpenAILLM)
    llm.client = MagicMock()
    llm.client.post.return_value = {
        "answers": [
            {"type": "predicate", "name": "urgent", "probability": 0.3},
            {
                "type": "choice",
                "name": "team",
                "choice": "b",
                "probabilities": [
                    {"value": "a", "probability": 0.1},
                    {"value": "b", "probability": 0.9},
                ],
                "confidence": 0.8,
            },
            {
                "type": "score",
                "name": "anger",
                "score": 0.7,
                "probabilities": [
                    {"value": 0, "label": "0", "probability": 0.3},
                    {"value": 1, "label": "1", "probability": 0.7},
                ],
                "confidence": 0.6,
            },
        ],
        "usage": {"total_tokens": 42},
    }
    res = llm.decide_sync("gpt-6-luna", "state", QUESTIONS, raise_=True)
    assert res is not None
    assert res.probability("urgent") == 0.3
    assert res.answers["team"].probabilities == {"a": 0.1, "b": 0.9}  # type: ignore
    assert res.total_tokens == 42

    kwargs = llm.client.post.call_args.kwargs
    assert llm.client.post.call_args.args == ("/decisions",)
    assert kwargs["body"]["input"] == "state"
    assert [q["type"] for q in kwargs["body"]["questions"]] == [
        "predicate",
        "choice",
        "score",
    ]


def test_llm_decide_falls_back_to_next_provider():
    llm = LLM.__new__(LLM)
    llm.openai_llm = MagicMock()
    llm.openai_llm.decide_sync.return_value = None
    llm.typesafe_llm = MagicMock()
    expected = MagicMock()
    llm.typesafe_llm.decide_sync.return_value = expected

    res = llm.decide_sync(
        [
            {"provider": "openai", "model": "gpt-6-luna"},
            {"provider": "typesafe", "model": "jev-latest"},
        ],
        "state",
        QUESTIONS,
    )
    assert res is expected


def test_llm_decide_requires_api_key():
    llm = LLM.__new__(LLM)
    llm.typesafe_llm = None
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        llm.decide_sync(
            [{"provider": "typesafe", "model": "jev-latest"}], "s", QUESTIONS
        )


def test_predicate_probability_helper_ignores_other_types():
    from think_reason_learn.core.llms import DecisionResponse, TypeSafeDecisionChoice

    res = DecisionResponse(
        answers={"x": PredicateAnswer(name="x", probability=0.5)},
        provider_model=TypeSafeDecisionChoice(model="jev-latest"),
    )
    assert res.probability("x") == 0.5
    assert res.probability("missing") is None
