import pytest
from ternary_causal_kernel.ternary_logic import TernaryState, KleeneEngine

def test_ternary_state_parsing():
    assert TernaryState.from_value(1.0) == TernaryState.TRUE
    assert TernaryState.from_value(True) == TernaryState.TRUE
    assert TernaryState.from_value(0.0) == TernaryState.FALSE
    assert TernaryState.from_value(False) == TernaryState.FALSE
    assert TernaryState.from_value(0.5) == TernaryState.INDETERMINATE

def test_kleene_negation():
    assert KleeneEngine.negation(TernaryState.TRUE) == TernaryState.FALSE
    assert KleeneEngine.negation(TernaryState.FALSE) == TernaryState.TRUE
    assert KleeneEngine.negation(TernaryState.INDETERMINATE) == TernaryState.INDETERMINATE

def test_kleene_conjunction():
    T = TernaryState.TRUE
    F = TernaryState.FALSE
    I = TernaryState.INDETERMINATE

    assert KleeneEngine.conjunction(T, T) == T
    assert KleeneEngine.conjunction(T, F) == F
    assert KleeneEngine.conjunction(F, F) == F
    assert KleeneEngine.conjunction(T, I) == I
    assert KleeneEngine.conjunction(F, I) == F
    assert KleeneEngine.conjunction(I, I) == I

def test_kleene_disjunction():
    T = TernaryState.TRUE
    F = TernaryState.FALSE
    I = TernaryState.INDETERMINATE

    assert KleeneEngine.disjunction(T, T) == T
    assert KleeneEngine.disjunction(T, F) == T
    assert KleeneEngine.disjunction(F, F) == F
    assert KleeneEngine.disjunction(T, I) == T
    assert KleeneEngine.disjunction(F, I) == I
    assert KleeneEngine.disjunction(I, I) == I
