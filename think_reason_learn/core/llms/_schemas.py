from __future__ import annotations

from typing import TypeVar, Tuple, Generic, List, TypeAlias, Literal
from typing import Dict, override, Union, Any
import math
from dataclasses import dataclass, field, asdict
import asyncio

from pydantic import BaseModel, Field

from ._anthropic.schemas import (
    AnthropicChatModel,
    AnthropicChoice,
    AnthropicChoiceDict,
    NOT_GIVEN as ANTHROPIC_NOT_GIVEN,
    NotGiven as AnthropicNotGiven,
)
from ._openai.schemas import (
    OpenAIChatModel,
    OpenAIChoice,
    OpenAIChoiceDict,
    OpenAIDecisionChoice,
    OpenAIDecisionChoiceDict,
    NotGiven as OpenAINotGiven,
    NOT_GIVEN as OPENAI_NOT_GIVEN,
)
from ._google.schemas import GoogleChatModel, GoogleChoice, GoogleChoiceDict
from ._xai.schemas import xAIChatModel, XAIChoice, XAIChoiceDict
from ._typesafe.schemas import (
    TypeSafeDecisionChoice,
    TypeSafeDecisionChoiceDict,
    TypeSafeDecisionModel,
)


T = TypeVar("T", bound=BaseModel | str, covariant=True)


LLMProvider: TypeAlias = Literal["anthropic", "google", "openai", "xai", "typesafe"]
LLMChatModel: TypeAlias = Union[
    AnthropicChatModel,
    GoogleChatModel,
    OpenAIChatModel,
    xAIChatModel,
    TypeSafeDecisionModel,
]
LLMChoiceModel: TypeAlias = Union[
    AnthropicChoice,
    GoogleChoice,
    OpenAIChoice,
    XAIChoice,
]
LLMChoiceDict: TypeAlias = Union[
    AnthropicChoiceDict,
    GoogleChoiceDict,
    OpenAIChoiceDict,
    XAIChoiceDict,
]
LLMChoice: TypeAlias = LLMChoiceModel | LLMChoiceDict


DecisionProvider: TypeAlias = Literal["openai", "typesafe"]
DecisionChoiceModel: TypeAlias = Union[OpenAIDecisionChoice, TypeSafeDecisionChoice]
DecisionChoiceDict: TypeAlias = Union[
    OpenAIDecisionChoiceDict, TypeSafeDecisionChoiceDict
]
DecisionChoice: TypeAlias = DecisionChoiceModel | DecisionChoiceDict


class PredicateQuestion(BaseModel):
    """A yes/no question. Answered with the probability that it is true.

    Maps to ``noul`` for TypeSafe and ``predicate`` for OpenAI.
    """

    type: Literal["predicate"] = "predicate"
    name: str = Field(description="Unique name; the answer is keyed by it.")
    instructions: str = Field(description="The yes/no question to evaluate.")
    true_criteria: str | None = Field(
        default=None, description="What makes the answer true."
    )
    false_criteria: str | None = Field(
        default=None, description="What makes the answer false."
    )


class ChoiceQuestion(BaseModel):
    """Pick exactly one option. Options are unordered."""

    type: Literal["choice"] = "choice"
    name: str = Field(description="Unique name; the answer is keyed by it.")
    instructions: str = Field(description="What to choose between.")
    choices: Dict[str, str] = Field(description="Option value -> its description.")


class ScoreQuestion(BaseModel):
    """Rate on an ordered rubric, from the lowest level to the highest."""

    type: Literal["score"] = "score"
    name: str = Field(description="Unique name; the answer is keyed by it.")
    instructions: str = Field(description="What to rate.")
    levels: List[str] = Field(
        min_length=2,
        max_length=10,
        description="Level descriptions ordered lowest to highest.",
    )


DecisionQuestion: TypeAlias = Union[PredicateQuestion, ChoiceQuestion, ScoreQuestion]


class PredicateAnswer(BaseModel):
    """Answer to a ``PredicateQuestion``."""

    type: Literal["predicate"] = "predicate"
    name: str
    probability: float = Field(description="Probability (0 to 1) that it is true.")


class ChoiceAnswer(BaseModel):
    """Answer to a ``ChoiceQuestion``."""

    type: Literal["choice"] = "choice"
    name: str
    choice: str = Field(description="The highest-probability option.")
    probabilities: Dict[str, float] = Field(description="Option -> probability.")
    confidence: float | None = None


class ScoreAnswer(BaseModel):
    """Answer to a ``ScoreQuestion``."""

    type: Literal["score"] = "score"
    name: str
    score: float = Field(
        description="Probability-weighted level index; can fall between levels."
    )
    probabilities: Dict[str, float] = Field(description="Level -> probability.")
    confidence: float | None = None


class RefusalAnswer(BaseModel):
    """The model declined to answer the question."""

    type: Literal["refusal"] = "refusal"
    name: str


DecisionAnswer: TypeAlias = Union[
    PredicateAnswer, ChoiceAnswer, ScoreAnswer, RefusalAnswer
]


class DecisionResponse(BaseModel):
    """A response from a decision model."""

    answers: Dict[str, DecisionAnswer] = Field(
        description="The answers keyed by question name."
    )
    total_tokens: int | None = Field(
        default=None, description="The total number of tokens used, if reported."
    )
    provider_model: DecisionChoiceModel = Field(
        description="The provider and model used to generate the response."
    )

    def probability(self, name: str) -> float | None:
        """Return the probability of a predicate answer, or None if unavailable."""
        answer = self.answers.get(name)
        return answer.probability if isinstance(answer, PredicateAnswer) else None


class LLMResponse(BaseModel, Generic[T]):
    """A response from an LLM."""

    response: T | None = Field(default=None, description="The response from the LLM.")
    logprobs: List[Tuple[str, float | None]] = Field(
        description="The log probabilities of the response."
    )
    total_tokens: int | None = Field(
        default=None,
        description="The total number of input and output tokens used "
        "to generate the response.",
    )
    provider_model: LLMChoiceModel = Field(
        description="The provider and model used to generate the response."
    )

    @property
    def average_confidence(self) -> float | None:
        if not self.logprobs:
            return

        lps = [lp[1] for lp in self.logprobs if lp[1] is not None]
        if not lps:
            return

        ave_lp = sum(lps) / len(lps)
        return math.exp(ave_lp)


class NotGiven:
    """A sentinel singleton class used to distinguish omitted keyword arguments.

    Examples:
        .. code-block:: python

            def get(timeout: int | NotGiven | None = NotGiven()) -> Response: ...

            get(timeout=1)      # 1s timeout
            get(timeout=None)   # No timeout
            get()               # Default timeout behavior; may not be statically
                                # known at the method definition.
    """

    ANTHROPIC_NOT_GIVEN: AnthropicNotGiven = ANTHROPIC_NOT_GIVEN
    OPENAI_NOT_GIVEN: OpenAINotGiven = OPENAI_NOT_GIVEN

    def __bool__(self) -> Literal[False]:
        return False

    @override
    def __repr__(self) -> str:
        return "NOT_GIVEN"


NOT_GIVEN = NotGiven()


@dataclass(slots=True)
class TokenCount:
    """A token count from an LLM."""

    provider: LLMProvider = field(metadata={"description": "The provider of the LLM."})
    model: LLMChatModel = field(metadata={"description": "The LLM model used."})
    number_of_calls: int = field(
        default=0, metadata={"description": "The number of calls to the LLM."}
    )
    value: int = field(default=0, metadata={"description": "The token count."})
    is_min_estimate: bool = field(
        default=False,
        metadata={
            "description": "Whether the token count is a minimum estimate. "
            "Will be True if the some calls to this LLM could not return "
            "the actual token count."
        },
    )
    callers: Dict[str, int] = field(
        default_factory=dict,
        metadata={
            "description": "The callers (functions/modules) and their number "
            "of calls to the LLM."
        },
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TokenCount:
        return cls(**d)


@dataclass(slots=True)
class TokenCounter:
    """Compactly represents token usage across multiple calls to LLMs."""

    token_counts: Dict[str, TokenCount] = field(
        default_factory=dict,
        metadata={"description": "The token counts for each provider/model pair."},
    )
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def append(
        self,
        model: LLMChatModel,
        provider: LLMProvider,
        value: int | None,
        caller: str,
    ) -> TokenCount:
        """Append a token count to the counter.

        Args:
            model: The model used.
            provider: The provider of the model.
            value: The token count.
            caller: The caller (function/module) of the LLM.
                Preferably the function name.

        Returns:
            The token count object.
        """
        key = f"{provider}/{model}"
        async with self._lock:
            token_count = self.token_counts.get(
                key, TokenCount(provider=provider, model=model)
            )
            token_count.number_of_calls += 1
            token_count.value += value if value is not None else 0
            token_count.callers[caller] = token_count.callers.get(caller, 0) + 1
            if not token_count.is_min_estimate and value is None:
                token_count.is_min_estimate = True
            self.token_counts[key] = token_count
            return token_count

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_lock", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TokenCounter:
        token_counts = {
            k: TokenCount.from_dict(v) for k, v in d["token_counts"].items()
        }
        self = cls(token_counts=token_counts)
        return self
