"""
Hugging Face Spaces Interactive Demo:
Ternary Causal Kernel: Indeterminate State Verification in Neural Representations
"""
import sys
import os
import gradio as gr

# Allow local imports if running locally
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ternary_causal_kernel.causal_graph import CausalGraph
from ternary_causal_kernel.ternary_kernel import TernaryCausalKernel
from ternary_causal_kernel.ternary_logic import TernaryState

# Pre-configured clinical scenarios
SCENARIOS = {
    "Sepsis Triage (Simpson's Paradox Trap)": {
        "treatment": "Steroid_Treatment",
        "outcome": "Mortality_Reduction",
        "confounder": "Baseline_Severity",
        "confounder_role": "Patient acuity affects both steroid prescription rate and baseline survival.",
        "candidate_output": "Based on cohort correlation (r=0.78), steroid protocol directly accelerates recovery. Recommend immediate steroid administration.",
        "default_latent": True,
        "explicit_truth": True
    },
    "Bacterial Clearance": {
        "treatment": "Antibiotic_X",
        "outcome": "Bacterial_Clearance",
        "confounder": "Dosage_Compliance",
        "confounder_role": "Patient compliance directly controls effective serum levels.",
        "candidate_output": "Antibiotic X demonstrates verifiable therapeutic clearance when administered at target concentration.",
        "default_latent": False,
        "explicit_truth": True
    },
    "Contraindicated Therapy": {
        "treatment": "Beta_Blocker",
        "outcome": "Heart_Rate_Stabilization",
        "confounder": "Acute_Bradycardia",
        "confounder_role": "Pre-existing severe low heart rate.",
        "candidate_output": "Administer beta-blocker protocol to regulate cardiovascular output.",
        "default_latent": False,
        "explicit_truth": False
    }
}

def run_causal_evaluation(scenario_name: str, is_confounder_observed: bool):
    scen = SCENARIOS[scenario_name]
    treatment = scen["treatment"]
    outcome = scen["outcome"]
    confounder = scen["confounder"]
    candidate = scen["candidate_output"]
    explicit_truth = scen["explicit_truth"]

    # Build Graph
    graph = CausalGraph()
    is_latent = not is_confounder_observed
    graph.add_node(treatment, is_latent=False)
    graph.add_node(outcome, is_latent=False)
    graph.add_node(confounder, is_latent=is_latent)
    
    graph.add_edge(confounder, treatment)
    graph.add_edge(confounder, outcome)
    graph.add_edge(treatment, outcome)

    observed_context = {treatment, outcome}
    if is_confounder_observed:
        observed_context.add(confounder)

    premises = [
        {"treatment": treatment, "outcome": outcome, "explicit_truth": explicit_truth}
    ]

    # Evaluate Kernel
    kernel = TernaryCausalKernel(graph)
    state = kernel.verify_premise_chain(premises, observed_context)
    result = kernel.filter_response(candidate, premises, observed_context)

    # Format Status Badges
    if state == TernaryState.TRUE:
        badge = "🟢 1 (TRUE / CAUSALLY GROUNDED)"
        explanation = f"Confounder [{confounder}] is fully observed. Back-door path is blocked. Output is causally identifiable."
    elif state == TernaryState.INDETERMINATE:
        badge = "🟡 ⊥ (INDETERMINATE / LATENT CONFOUNDER DETECTED)"
        explanation = f"Confounder [{confounder}] is UNMEASURED. The back-door path ({treatment} <- {confounder} -> {outcome}) is open. Statistical correlation cannot identify causal effect. Output intercepted."
    else:
        badge = "🔴 0 (FALSE / REFUTED BY AXIOMS)"
        explanation = "The proposition explicitly contradicts verified causal axioms."

    # Graph Topology Text
    topology = (
        f"Nodes: [{treatment}, {outcome}, {confounder} ({'Observed' if is_confounder_observed else 'Latent/Unmeasured'})]\n"
        f"Edges:\n"
        f" • {confounder} ➔ {treatment}\n"
        f" • {confounder} ➔ {outcome}\n"
        f" • {treatment} ➔ {outcome}"
    )

    return badge, explanation, topology, candidate, result["output"]

# Build Gradio UI
with gr.Blocks(title="Ternary Causal Kernel Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔬 RGAI Ternary Causal Kernel Playground")
    gr.Markdown(
        "Demonstrating **Kleene Strong Three-Valued Logic ($K_3$)** and **Pearl's do-calculus identifiability** "
        "to prevent neural network hallucinations under latent confounding."
    )
    with gr.Row():
        with gr.Column(scale=1):
            scenario_dropdown = gr.Dropdown(
                label="Select Clinical Scenario",
                choices=list(SCENARIOS.keys()),
                value="Sepsis Triage (Simpson's Paradox Trap)"
            )
            confounder_toggle = gr.Checkbox(
                label="Observe Confounder (Collect Laboratory Telemetry)",
                value=False,
                info="Toggle whether the critical confounding variable is measured or left latent."
            )
            eval_button = gr.Button("Evaluate Causal State", variant="primary")
            gr.Markdown("### Structural Causal Graph Topology")
            graph_display = gr.Textbox(label="Active DAG Structure", lines=5, interactive=False)
        with gr.Column(scale=1):
            status_box = gr.Textbox(label="Epistemic State Evaluation", interactive=False)
            analysis_box = gr.Textbox(label="Causal Analysis & Identifiability", lines=3, interactive=False)
            
            gr.Markdown("### Output Comparison")
            standard_llm_output = gr.Textbox(
                label="Standard Binary LLM (Forced Projection on [0, 1])",
                lines=3,
                interactive=False
            )
            kernel_output = gr.Textbox(
                label="RGAI Ternary Causal Interceptor (Withholding upon ⊥)",
                lines=3,
                interactive=False
            )
            
    eval_button.click(
        fn=run_causal_evaluation,
        inputs=[scenario_dropdown, confounder_toggle],
        outputs=[status_box, analysis_box, graph_display, standard_llm_output, kernel_output]
    )
    demo.load(
        fn=run_causal_evaluation,
        inputs=[scenario_dropdown, confounder_toggle],
        outputs=[status_box, analysis_box, graph_display, standard_llm_output, kernel_output]
    )

if __name__ == "__main__":
    demo.launch()
