from __future__ import annotations

import logging
from typing import Any, Dict, List, Type, cast

from pydantic import ValidationError

from think_reason_learn.core._config import settings
from think_reason_learn.core._singleton import SingletonMeta

from ._anthropic.ask import AnthropicLLM, get_anthropic_llm
from ._anthropic.schemas import AnthropicChoice
from ._google.ask import GeminiLLM, get_google_llm
from ._google.schemas import GoogleChoice
from ._openai.ask import OpenAILLM, get_openai_llm
from ._openai.schemas import OpenAIChoice
from ._schemas import (
    NOT_GIVEN,
    DecisionChoice,
    DecisionChoiceModel,
    DecisionQuestion,
    DecisionResponse,
    LLMChoice,
    LLMChoiceModel,
    LLMResponse,
    NotGiven,
    OpenAIDecisionChoice,
    T,
    TypeSafeDecisionChoice,
)
from ._typesafe.ask import TypeSafeLLM, get_typesafe_llm
from ._xai.ask import get_xai_llm, xAILLM
from ._xai.schemas import XAIChoice

logger = logging.getLogger(__name__)


class LLM(metaclass=SingletonMeta):
    """A singleton class that provides a unified interface for the LLMs."""

    def __init__(self) -> None:
        self.anthropic_llm = get_anthropic_llm(settings.ANTHROPIC_API_KEY)
        self.google_llm = get_google_llm(settings.GOOGLE_AI_API_KEY)
        self.openai_llm = get_openai_llm(settings.OPENAI_API_KEY)
        self.xai_llm = get_xai_llm(settings.XAI_API_KEY)
        self.typesafe_llm = get_typesafe_llm(settings.TYPESAFE_API_KEY)

    def _val_decision_priority_and_api_keys(
        self, decision_priority: List[DecisionChoice]
    ) -> List[DecisionChoiceModel]:
        """Validate the decision priority and ensure the API key is set."""
        models_map = {
            "openai": OpenAIDecisionChoice,
            "typesafe": TypeSafeDecisionChoice,
        }
        priority_models: List[DecisionChoiceModel] = []
        for dp in decision_priority:
            if isinstance(dp, dict):
                priority_class = models_map.get(dp["provider"])
                if priority_class is None:
                    raise ValueError(
                        f"Invalid decision priority: {dp['provider']}. "
                        f"Supported providers are {list(models_map.keys())}."
                    )
                dp = priority_class(**dp)  # type: ignore

            if getattr(self, f"{dp.provider}_llm") is None:
                raise ValueError(
                    f"Can't use {dp.model}. {dp.provider.upper()}_API_KEY not set! "
                    "Explicitly set it in your environment."
                )
            priority_models.append(dp)
        return priority_models

    def _val_llm_priority_and_api_keys(
        self, llm_priority: List[LLMChoice]
    ) -> List[LLMChoiceModel]:
        """Validate the LLMPriority and ensure the API key is set for such provider."""
        models_map = {
            "anthropic": AnthropicChoice,
            "google": GoogleChoice,
            "openai": OpenAIChoice,
            "xai": XAIChoice,
        }
        priority_models: List[LLMChoiceModel] = []
        for llmp in llm_priority:
            if isinstance(llmp, dict):
                priority_class = models_map.get(llmp["provider"])
                if priority_class is None:
                    raise ValueError(
                        f"Invalid LLMPriority: {llmp['provider']}. "
                        f"Supported providers are {list(models_map.keys())}."
                    )

                try:
                    llmp = priority_class(**llmp)  # type: ignore
                except ValidationError:
                    supported_models = priority_class.__annotations__["model"].__args__
                    raise ValueError(
                        f"Invalid LLMPriority: {llmp['model']} for {llmp['provider']}. "
                        f"Supported models are {supported_models}."
                    )

            else:
                llmp = llmp

            if getattr(self, f"{llmp.provider}_llm") is None:
                raise ValueError(
                    f"Can't use {llmp.model}. {llmp.provider.upper()}_API_KEY not set! "
                    "Explicitly set it in your environment."
                )
            priority_models.append(llmp)
        return priority_models

    def respond_sync(
        self,
        llm_priority: List[LLMChoice],
        query: str = "",
        response_format: Type[T] = str,
        instructions: str | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        **kwargs: Dict[str, Any],
    ) -> LLMResponse[T]:
        """Respond to a query using the LLM synchronously.

        Args:
            query: The query to respond to.
            llm_priority: LLMs to use in order of priority.
            response_format: The response format to use.
            instructions: Optional instructions to use.
            temperature: Optional temperature to use.
            **kwargs: Additional arguments to pass to the LLM. For:

                - OpenAI: ``openai.OpenAI.responses.parse`` or
                    ``openai.OpenAI.responses.create``.
                - Google: ``google.genai.types.GenerateContentConfig``.
                - XAI: ``xai_sdk.Client.chat.create``.
                - Anthropic: ``anthropic.Client.messages.create``.

        Note:
            Provided kwargs override the function arguments.

        Returns:
            The response from the LLM.

        Raises:
            AssertionError: If llm_priority is an empty list.
            ValueError: If none of the LLMs worked.
        """
        assert len(llm_priority) > 0, "llm_priority must be a non-empty list"
        raise_ = len(llm_priority) == 1

        llm_priority_models = self._val_llm_priority_and_api_keys(llm_priority)

        for idx, llmp in enumerate(llm_priority_models, 1):
            if idx > 1:
                logger.warning(f"Falling back to {llmp.model} by {llmp.provider}...")

            if llmp.provider == "google":
                google_llm = cast(GeminiLLM, self.google_llm)
                response = google_llm.respond_sync(
                    query=query,
                    model=llmp.model,
                    response_format=response_format,
                    instructions=instructions or None,
                    temperature=temperature or None,
                    raise_=raise_,
                    **kwargs,
                )
                if response is not None:
                    return response

            if llmp.provider == "openai":
                openai_llm = cast(OpenAILLM, self.openai_llm)
                response = openai_llm.respond_sync(
                    query=query,
                    model=llmp.model,
                    response_format=response_format,
                    instructions=(
                        instructions
                        if not isinstance(instructions, NotGiven)
                        else NOT_GIVEN.OPENAI_NOT_GIVEN
                    ),
                    temperature=(
                        temperature
                        if not isinstance(temperature, NotGiven)
                        else NOT_GIVEN.OPENAI_NOT_GIVEN
                    ),
                    raise_=raise_,
                    **kwargs,
                )
                if response is not None:
                    return response

            if llmp.provider == "anthropic":
                anthropic_llm = cast(AnthropicLLM, self.anthropic_llm)
                response = anthropic_llm.respond_sync(
                    query=query,
                    model=llmp.model,
                    response_format=response_format,
                    instructions=instructions or NOT_GIVEN.ANTHROPIC_NOT_GIVEN,
                    temperature=temperature or NOT_GIVEN.ANTHROPIC_NOT_GIVEN,
                    raise_=raise_,
                    **kwargs,
                )
                if response is not None:
                    return response

            if llmp.provider == "xai":
                xai_llm = cast(xAILLM, self.xai_llm)
                response = xai_llm.respond_sync(
                    query=query,
                    model=llmp.model,
                    response_format=response_format,
                    instructions=instructions or None,
                    temperature=temperature or None,
                    raise_=raise_,
                    **kwargs,
                )
                if response is not None:
                    return response

        llmps = [f"{llmp.provider}: {llmp.model}" for llmp in llm_priority_models]
        raise ValueError(
            f"Failed to respond with any of {llmps}\n"
            f"Query: {query}\n"
            f"Instructions: {instructions}\n"
            f"Temperature: {temperature}\n"
            f"**kwargs: {kwargs}"
        )

    async def respond(
        self,
        query: str,
        llm_priority: List[LLMChoice],
        response_format: Type[T],
        instructions: str | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        **kwargs: Dict[str, Any],
    ) -> LLMResponse[T]:
        """Respond to a query using the LLM asynchronously.

        Args:
            query: The query to respond to.
            llm_priority: LLMs to use in order of priority.
            response_format: The response format to use.
            instructions: Optional instructions to use.
            temperature: Optional temperature to use.
            **kwargs: Additional arguments to pass to the LLM. For:

                - OpenAI: ``openai.OpenAI.responses.parse`` or
                    ``openai.OpenAI.responses.create``.
                - Google: ``google.genai.types.GenerateContentConfig``.
                - XAI: ``xai_sdk.Client.chat.create``.
                - Anthropic: ``anthropic.Client.messages.create``.

        Note:
            Provided kwargs override the function arguments.

        Returns:
            The response from the LLM.

        Raises:
            AssertionError: If llm_priority is an empty list.
            ValueError: If none of the LLMs worked.
        """
        assert len(llm_priority) > 0, "llm_priority must be a non-empty list"
        raise_ = len(llm_priority) == 1

        llm_priority_models = self._val_llm_priority_and_api_keys(llm_priority)

        for idx, llmp in enumerate(llm_priority_models, 1):
            if idx > 1:
                logger.warning(f"Falling back to {llmp.model} from {llmp.provider}...")

            if llmp.provider == "google":
                google_llm = cast(GeminiLLM, self.google_llm)
                response = await google_llm.respond(
                    query=query,
                    model=llmp.model,
                    response_format=response_format,
                    instructions=instructions or None,
                    temperature=temperature or None,
                    raise_=raise_,
                    **kwargs,
                )
                if response is not None:
                    return response

            if llmp.provider == "openai":
                openai_llm = cast(OpenAILLM, self.openai_llm)
                response = await openai_llm.respond(
                    query=query,
                    model=llmp.model,
                    response_format=response_format,
                    instructions=(
                        instructions
                        if not isinstance(instructions, NotGiven)
                        else NOT_GIVEN.OPENAI_NOT_GIVEN
                    ),
                    temperature=(
                        temperature
                        if not isinstance(temperature, NotGiven)
                        else NOT_GIVEN.OPENAI_NOT_GIVEN
                    ),
                    raise_=raise_,
                    **kwargs,
                )
                if response is not None:
                    return response

            if llmp.provider == "anthropic":
                anthropic_llm = cast(AnthropicLLM, self.anthropic_llm)
                response = await anthropic_llm.respond(
                    query=query,
                    model=llmp.model,
                    response_format=response_format,
                    instructions=instructions or NOT_GIVEN.ANTHROPIC_NOT_GIVEN,
                    temperature=temperature or NOT_GIVEN.ANTHROPIC_NOT_GIVEN,
                    raise_=raise_,
                    **kwargs,
                )
                if response is not None:
                    return response

            if llmp.provider == "xai":
                xai_llm = cast(xAILLM, self.xai_llm)
                response = await xai_llm.respond(
                    query=query,
                    model=llmp.model,
                    response_format=response_format,
                    instructions=instructions or None,
                    temperature=temperature or None,
                    raise_=raise_,
                    **kwargs,
                )
                if response is not None:
                    return response

        llmps = [f"{llmp.provider}: {llmp.model}" for llmp in llm_priority_models]
        raise ValueError(f"Failed to respond with any of {llmps}")

    def decide_sync(
        self,
        decision_priority: List[DecisionChoice],
        state: str,
        questions: List[DecisionQuestion],
    ) -> DecisionResponse:
        """Ask a decision model typed questions about a state synchronously.

        Args:
            decision_priority: Decision models to use in order of priority.
            state: The shared evidence (text) the questions are asked about.
            questions: Predicate, choice and score questions with unique names.

        Returns:
            The answers keyed by question name.

        Raises:
            AssertionError: If decision_priority or questions is empty.
            ValueError: If none of the decision models worked.
        """
        assert len(decision_priority) > 0, "decision_priority must be non-empty"
        assert len(questions) > 0, "questions must be non-empty"
        raise_ = len(decision_priority) == 1

        models = self._val_decision_priority_and_api_keys(decision_priority)
        for idx, dp in enumerate(models, 1):
            if idx > 1:
                logger.warning(f"Falling back to {dp.model} by {dp.provider}...")

            provider_llm = cast(
                OpenAILLM | TypeSafeLLM, getattr(self, f"{dp.provider}_llm")
            )
            response = provider_llm.decide_sync(
                model=dp.model, state=state, questions=questions, raise_=raise_
            )
            if response is not None:
                return response

        dps = [f"{dp.provider}: {dp.model}" for dp in models]
        raise ValueError(f"Failed to decide with any of {dps}")

    async def decide(
        self,
        decision_priority: List[DecisionChoice],
        state: str,
        questions: List[DecisionQuestion],
    ) -> DecisionResponse:
        """Ask a decision model typed questions about a state asynchronously.

        Args:
            decision_priority: Decision models to use in order of priority.
            state: The shared evidence (text) the questions are asked about.
            questions: Predicate, choice and score questions with unique names.

        Returns:
            The answers keyed by question name.

        Raises:
            AssertionError: If decision_priority or questions is empty.
            ValueError: If none of the decision models worked.
        """
        assert len(decision_priority) > 0, "decision_priority must be non-empty"
        assert len(questions) > 0, "questions must be non-empty"
        raise_ = len(decision_priority) == 1

        models = self._val_decision_priority_and_api_keys(decision_priority)
        for idx, dp in enumerate(models, 1):
            if idx > 1:
                logger.warning(f"Falling back to {dp.model} by {dp.provider}...")

            provider_llm = cast(
                OpenAILLM | TypeSafeLLM, getattr(self, f"{dp.provider}_llm")
            )
            response = await provider_llm.decide(
                model=dp.model, state=state, questions=questions, raise_=raise_
            )
            if response is not None:
                return response

        dps = [f"{dp.provider}: {dp.model}" for dp in models]
        raise ValueError(f"Failed to decide with any of {dps}")
