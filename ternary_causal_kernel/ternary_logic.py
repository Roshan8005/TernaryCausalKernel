"""
Core algebraic engine implementing Kleene's Strong Three-Valued Logic (K3).
States:
  FALSE         = 0.0
  INDETERMINATE = 0.5  (⊥: Unobserved or causally unidentifiable)
  TRUE          = 1.0
"""
from enum import Enum
from typing import Union


class TernaryState(Enum):
    FALSE = 0.0
    INDETERMINATE = 0.5
    TRUE = 1.0

    @classmethod
    def from_value(cls, val: Union[float, int, bool]) -> "TernaryState":
        if isinstance(val, bool):
            return cls.TRUE if val else cls.FALSE
        if val in (1, 1.0):
            return cls.TRUE
        if val in (0, 0.0):
            return cls.FALSE
        return cls.INDETERMINATE

    def __repr__(self) -> str:
        if self == TernaryState.TRUE:
            return "1 (True)"
        elif self == TernaryState.FALSE:
            return "0 (False)"
        return "⊥ (Indeterminate)"


class KleeneEngine:
    """Implements truth-functional K3 operators."""

    @staticmethod
    def negation(a: TernaryState) -> TernaryState:
        # ¬x = 1 - x
        return TernaryState.from_value(round(1.0 - a.value, 1))

    @staticmethod
    def conjunction(a: TernaryState, b: TernaryState) -> TernaryState:
        # A ∧ B = min(A, B)
        # Note: 0 ∧ ⊥ = 0 (If one premise is false, the conclusion is false regardless of unknowns)
        return TernaryState.from_value(min(a.value, b.value))

    @staticmethod
    def disjunction(a: TernaryState, b: TernaryState) -> TernaryState:
        # A ∨ B = max(A, B)
        # Note: 1 ∨ ⊥ = 1 (If one premise is true, the disjunction holds)
        return TernaryState.from_value(max(a.value, b.value))

    @staticmethod
    def implies(a: TernaryState, b: TernaryState) -> TernaryState:
        # A → B is equivalent to (¬A ∨ B)
        not_a = KleeneEngine.negation(a)
        return KleeneEngine.disjunction(not_a, b)
