from typing import Type, Any, cast, Dict, List
import logging

from openai import AsyncOpenAI, OpenAI
from openai.types.responses import Response
from pydantic import BaseModel

from .._schemas import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionQuestion,
    DecisionResponse,
    LLMResponse,
    OpenAIChoice,
    OpenAIDecisionChoice,
    PredicateAnswer,
    PredicateQuestion,
    RefusalAnswer,
    ScoreAnswer,
    T,
)
from think_reason_learn.core._singleton import SingletonMeta
from .schemas import OpenAIChatModel, OpenAIDecisionModel, NOT_GIVEN, NotGiven


logger = logging.getLogger(__name__)

DECISIONS_PATH = "/decisions"


def _build_question(q: DecisionQuestion) -> Dict[str, Any]:
    if isinstance(q, PredicateQuestion):
        instructions = q.instructions
        if q.true_criteria:
            instructions += f"\nTrue when: {q.true_criteria}"
        if q.false_criteria:
            instructions += f"\nFalse when: {q.false_criteria}"
        return {"type": "predicate", "name": q.name, "instructions": instructions}
    if isinstance(q, ChoiceQuestion):
        return {
            "type": "choice",
            "name": q.name,
            "instructions": q.instructions,
            "choices": [{"value": v, "description": d} for v, d in q.choices.items()],
        }
    return {
        "type": "score",
        "name": q.name,
        "instructions": q.instructions,
        "levels": [{"label": str(i), "description": d} for i, d in enumerate(q.levels)],
    }


def _parse_answer(raw: Dict[str, Any]) -> DecisionAnswer:
    kind, name = raw.get("type"), raw["name"]
    if kind == "predicate":
        return PredicateAnswer(name=name, probability=raw["probability"])
    if kind == "choice":
        return ChoiceAnswer(
            name=name,
            choice=raw["choice"],
            probabilities={p["value"]: p["probability"] for p in raw["probabilities"]},
            confidence=raw.get("confidence"),
        )
    if kind == "score":
        return ScoreAnswer(
            name=name,
            score=raw["score"],
            probabilities={
                str(p.get("label", p["value"])): p["probability"]
                for p in raw["probabilities"]
            },
            confidence=raw.get("confidence"),
        )
    if kind == "refusal":
        return RefusalAnswer(name=name)
    raise ValueError(f"Unknown OpenAI decision answer type: {kind!r}")


class OpenAILLM(metaclass=SingletonMeta):
    def __init__(self, api_key: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.aclient = AsyncOpenAI(api_key=api_key)

    def _process_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v if v else (None if v is None else NOT_GIVEN)
            for k, v in kwargs.items()
            if k in self.client.responses.parse.__annotations__
        }

    def respond_sync(
        self,
        model: OpenAIChatModel,
        query: str = "",
        response_format: Type[T] = str,
        instructions: str | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        raise_: bool = False,
        **kwargs: Any,
    ) -> LLMResponse[T] | None:
        kwargs = self._process_kwargs(kwargs)

        try:
            if issubclass(response_format, BaseModel):
                response = self.client.responses.parse(
                    model=model,
                    input=kwargs.get("input", query),
                    instructions=instructions or None,
                    temperature=temperature or None,
                    text_format=response_format,
                    **kwargs,
                )
                return LLMResponse(
                    response=response.output_parsed,
                    logprobs=[],
                    total_tokens=(
                        response.usage.total_tokens if response.usage else None
                    ),
                    provider_model=OpenAIChoice(model=model),
                )

            response = self.client.responses.create(  # type: ignore
                model=model,
                input=kwargs.get("input", query),
                instructions=instructions or None,
                temperature=temperature or None,
                **kwargs,
            )
            response = cast(Response, response)

            return LLMResponse(
                response=cast(T, response.output_text),
                logprobs=[],
                total_tokens=response.usage.total_tokens if response.usage else None,
                provider_model=OpenAIChoice(model=model),
            )
        except Exception as e:
            logger.warning(f"Error responding with OpenAI: {e}", exc_info=True)
            if raise_:
                raise e
            return None

    async def respond(
        self,
        model: OpenAIChatModel,
        query: str = "",
        response_format: Type[T] = str,
        instructions: str | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        raise_: bool = False,
        **kwargs: Any,
    ) -> LLMResponse[T] | None:
        kwargs = self._process_kwargs(kwargs)

        try:
            if issubclass(response_format, BaseModel):
                response = await self.aclient.responses.parse(
                    model=model,
                    input=kwargs.get("input", query),
                    instructions=instructions or None,
                    temperature=temperature or None,
                    text_format=response_format,
                    **kwargs,
                )
                return LLMResponse(
                    response=response.output_parsed,
                    logprobs=[],
                    total_tokens=(
                        response.usage.total_tokens if response.usage else None
                    ),
                    provider_model=OpenAIChoice(model=model),
                )

            response = await self.aclient.responses.create(  # type: ignore
                model=model,
                input=kwargs.get("input", query),
                instructions=instructions or None,
                temperature=temperature or None,
                **kwargs,
            )
            response = cast(Response, response)
            return LLMResponse(
                response=cast(T, response.output_text),
                logprobs=[],
                total_tokens=response.usage.total_tokens if response.usage else None,
                provider_model=OpenAIChoice(model=model),
            )
        except Exception as e:
            logger.warning(f"Error responding with OpenAI: {e}", exc_info=True)
            if raise_:
                raise e
            return None

    def _decision_body(
        self,
        model: OpenAIDecisionModel,
        state: str,
        questions: List[DecisionQuestion],
    ) -> Dict[str, Any]:
        return {
            "model": model,
            "input": state,
            "questions": [_build_question(q) for q in questions],
        }

    def _parse_decision(
        self, model: OpenAIDecisionModel, data: Dict[str, Any]
    ) -> DecisionResponse:
        usage = data.get("usage") or {}
        total = usage.get("total_tokens")
        if total is None and usage:
            total = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        answers = [_parse_answer(a) for a in data["answers"]]
        return DecisionResponse(
            answers={a.name: a for a in answers},
            total_tokens=total,
            provider_model=OpenAIDecisionChoice(model=model),
        )

    def decide_sync(
        self,
        model: OpenAIDecisionModel,
        state: str,
        questions: List[DecisionQuestion],
        raise_: bool = False,
    ) -> DecisionResponse | None:
        # Posts directly so it works on openai SDK versions without
        # ``client.decisions``.
        try:
            data = self.client.post(
                DECISIONS_PATH,
                body=self._decision_body(model, state, questions),
                cast_to=object,
            )
            return self._parse_decision(model, cast(Dict[str, Any], data))
        except Exception as e:
            logger.warning(f"Error deciding with OpenAI: {e}", exc_info=True)
            if raise_:
                raise e
            return None

    async def decide(
        self,
        model: OpenAIDecisionModel,
        state: str,
        questions: List[DecisionQuestion],
        raise_: bool = False,
    ) -> DecisionResponse | None:
        try:
            data = await self.aclient.post(
                DECISIONS_PATH,
                body=self._decision_body(model, state, questions),
                cast_to=object,
            )
            return self._parse_decision(model, cast(Dict[str, Any], data))
        except Exception as e:
            logger.warning(f"Error deciding with OpenAI: {e}", exc_info=True)
            if raise_:
                raise e
            return None


def get_openai_llm(api_key: str) -> OpenAILLM | None:
    return OpenAILLM(api_key) if api_key else None
