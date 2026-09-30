from __future__ import annotations

from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetGuard:
    max_calls: int
    cost_ceiling_usd: float
    input_usd_per_million_tokens: float
    calls: int = 0
    input_tokens: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        return self.input_tokens * self.input_usd_per_million_tokens / 1_000_000

    def reserve_call(self) -> None:
        if self.calls >= self.max_calls:
            raise BudgetExceeded("Jev call budget exhausted")
        self.calls += 1

    def record_usage(self, input_tokens: int) -> None:
        projected = (
            (self.input_tokens + input_tokens)
            * self.input_usd_per_million_tokens
            / 1_000_000
        )
        if projected > self.cost_ceiling_usd:
            raise BudgetExceeded("Jev estimated cost ceiling exceeded")
        self.input_tokens += input_tokens
