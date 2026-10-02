"""
The Ternary Causal Interceptor for LLM generation pipelines.
Catches hallucinated causal claims and projects them to ⊥.
"""
from typing import Dict, List, Set, Any
from ternary_causal_kernel.ternary_logic import TernaryState, KleeneEngine
from ternary_causal_kernel.causal_graph import CausalGraph


class TernaryCausalKernel:
    def __init__(self, causal_graph: CausalGraph):
        self.graph = causal_graph
        self.engine = KleeneEngine()

    def verify_premise_chain(self, premises: List[Dict[str, Any]], observed_context: Set[str]) -> TernaryState:
        """
        Validates a sequence of causal deductions against observed context.
        Each premise is a dict: {'treatment': str, 'outcome': str, 'explicit_truth': Optional[bool]}
        """
        current_state = TernaryState.TRUE

        for p in premises:
            treatment = p.get('treatment')
            outcome = p.get('outcome')
            explicit_val = p.get('explicit_truth')

            # 1. Direct explicit contradiction
            if explicit_val is False:
                premise_state = TernaryState.FALSE
            # 2. Direct explicit truth with verified causal identifiability
            elif explicit_val is True:
                identifiable = self.graph.is_identifiable(treatment, outcome, observed_context)
                premise_state = TernaryState.TRUE if identifiable else TernaryState.INDETERMINATE
            # 3. Missing/Unstated premise
            else:
                premise_state = TernaryState.INDETERMINATE

            # Conjoin with total deduction state: current_state = current_state ∧ premise_state
            current_state = self.engine.conjunction(current_state, premise_state)

            # Short-circuit: If anything is strictly FALSE, the entire conjunction is FALSE (0 ∧ ⊥ = 0)
            if current_state == TernaryState.FALSE:
                break

        return current_state

    def filter_response(self, candidate_answer: str, premises: List[Dict[str, Any]], observed_context: Set[str]) -> Dict[str, Any]:
        """
        Intercepts candidate model response and outputs either the confirmed answer
        or an explicit refusal with missing causal dependencies.
        """
        verdict = self.verify_premise_chain(premises, observed_context)

        if verdict == TernaryState.TRUE:
            return {
                "status": "CAUSALLY_GROUNDED",
                "state": repr(verdict),
                "output": candidate_answer
            }
        elif verdict == TernaryState.INDETERMINATE:
            return {
                "status": "HALLUCINATION_PREVENTED",
                "state": repr(verdict),
                "output": "[⊥ Indeterminate State]: Causal premises depend on unobserved confounding factors. Answer withheld to prevent hallucination.",
                "missing_dependencies": [
                    p for p in premises if not self.graph.is_identifiable(p['treatment'], p['outcome'], observed_context)
                ]
            }
        else:
            return {
                "status": "REFUTED",
                "state": repr(verdict),
                "output": "[0 Refuted]: Proposition contradicts verified causal axioms."
            }
