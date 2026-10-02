"""
Side-by-side demonstration of Standard Forced-Binary LLM vs. Ternary Causal Kernel.
"""
import sys
import os
import io

# Fix for Windows console printing Unicode (⊥)
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ternary_causal_kernel.causal_graph import CausalGraph
from ternary_causal_kernel.ternary_kernel import TernaryCausalKernel

def run_test():
    # 1. Define Causal Graph for a Clinical Scenario
    # Treatment: New Drug (X)
    # Outcome: Recovery (Y)
    # Latent Confounder: Patient Genetics (Z) -> affects both Drug sensitivity and Recovery
    graph = CausalGraph()
    graph.add_node("Drug_X", is_latent=False)
    graph.add_node("Recovery_Y", is_latent=False)
    graph.add_node("Genetics_Z", is_latent=True)  # Unobserved variable

    graph.add_edge("Genetics_Z", "Drug_X")
    graph.add_edge("Genetics_Z", "Recovery_Y")
    graph.add_edge("Drug_X", "Recovery_Y")

    kernel = TernaryCausalKernel(graph)

    # 2. Clinical Query Scenario:
    # Context provided only observes: Drug_X and Recovery_Y (Genetics_Z is unobserved)
    observed_context = {"Drug_X", "Recovery_Y"}

    # Model generates a causal claim: "Drug X directly causes Recovery Y in this cohort."
    candidate_llm_response = "Clinical evidence indicates that administering Drug X directly accelerates Recovery Y."
    deduced_premises = [
        {"treatment": "Drug_X", "outcome": "Recovery_Y", "explicit_truth": True}
    ]

    print("=== SCENARIO: Query with Latent Confounder (Unobserved Genetics) ===\n")
    
    # Standard Binary LLM behavior:
    print("[Standard Binary LLM]:")
    print("Forced to project onto [0, 1]. Prior correlation is high (~0.85).")
    print(f"Generated Output: \"{candidate_llm_response}\" (HALLUCINATION / SPURIOUS)\n")

    # Ternary Causal Kernel behavior:
    print("[RGAI Ternary Causal Kernel]:")
    result = kernel.filter_response(candidate_llm_response, deduced_premises, observed_context)
    print(f"Status: {result['status']}")
    print(f"Epistemic State: {result['state']}")
    print(f"Final Filtered Output: {result['output']}")
    if "missing_dependencies" in result:
        print(f"Flagged Unobserved Dependencies: {result['missing_dependencies']}")

if __name__ == "__main__":
    run_test()
