from typing import TypeAlias, Literal, TypedDict

from pydantic import BaseModel


TypeSafeDecisionModel: TypeAlias = str | Literal["jev-latest"]


class TypeSafeDecisionChoice(BaseModel):
    """A decision (System One) model from TypeSafe."""

    provider: Literal["typesafe"] = "typesafe"
    model: TypeSafeDecisionModel


class TypeSafeDecisionChoiceDict(TypedDict):
    """A decision (System One) model from TypeSafe."""

    provider: Literal["typesafe"]
    model: TypeSafeDecisionModel
