"""Deterministic fake LLM for offline PolicyInduction testing."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List

from think_reason_learn.core.llms import OpenAIChoice
from think_reason_learn.core.llms._schemas import LLMResponse
from think_reason_learn.policy_induction import Policies
from tests.fake_jev import FakeJevLLM

_FAKE_PROVIDER = OpenAIChoice(model="gpt-4.1-nano")

STRONG = "Give weight to the strong signal."
TEMPLATE = "Write general guidance policies from the labelled samples."


def default_prob(state: str, instructions: str) -> float:
    """Only the strong-signal policy carries information."""
    if STRONG in instructions:
        return 0.9 if "strong" in state else 0.1
    return 0.5


class FakePolicyLLM(FakeJevLLM):
    """Generation, merge and Jev answering for PolicyInduction.

    A generation batch returns the strong-signal policy plus ``noise_per_batch``
    policies of its own. The merge returns the first policies it was given, up
    to its cap, unless ``merge_replies`` scripts what each merge call returns.

    Args:
        noise_per_batch: Noise policies per generation batch.
        merge_replies: Policy lists returned by successive merge calls.
        **kwargs: Passed to ``FakeJevLLM`` (e.g. ``prob``).
    """

    def __init__(
        self,
        noise_per_batch: int = 2,
        merge_replies: List[List[str]] | None = None,
        prob: Callable[[str, str], float | None] = default_prob,
        **kwargs: Any,
    ) -> None:
        super().__init__(prob=prob, **kwargs)
        self.noise_per_batch = noise_per_batch
        self.merge_replies = list(merge_replies or [])
        self.gen_queries: List[str] = []
        self.merge_queries: List[str] = []
        self.template_calls = 0

    async def respond(
        self,
        query: str,
        llm_priority: List[Any],
        response_format: Any,
        instructions: Any = None,
        temperature: Any = None,
        **kwargs: Any,
    ) -> LLMResponse[Any]:
        self.calls.append(
            {
                "query": query,
                "response_format": response_format,
                "instructions": instructions,
            }
        )
        response: Any
        if response_format is str:
            self.template_calls += 1
            response = TEMPLATE
        elif response_format is Policies and "Merge the policies" in str(instructions):
            self.merge_queries.append(query)
            response = Policies(policies=self._merge(query, str(instructions)))
        elif response_format is Policies:
            batch = len(self.gen_queries)
            self.gen_queries.append(query)
            response = Policies(
                policies=[STRONG]
                + [
                    f"Weigh noise {k} of batch {batch}."
                    for k in range(self.noise_per_batch)
                ]
            )
        else:
            raise TypeError(f"unexpected response_format {response_format!r}")
        return LLMResponse(
            response=response,
            logprobs=[],
            total_tokens=10,
            provider_model=_FAKE_PROVIDER,
        )

    def _merge(self, query: str, instructions: str) -> List[str]:
        if self.merge_replies:
            return self.merge_replies.pop(0)
        cap = int(re.search(r"AT MOST (\d+)", instructions).group(1))  # type: ignore
        listed = re.findall(r"^- (.*)$", query, re.MULTILINE)
        return listed[:cap]

    @property
    def answered_states(self) -> Dict[str, int]:
        """How many requests each sample state received."""
        counts: Dict[str, int] = {}
        for request in self.answer_requests:
            counts[request.state] = counts.get(request.state, 0) + 1
        return counts
