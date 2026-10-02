# Ternary Causal Kernel

A standalone reference implementation of **Multi-Valued Paraconsistent Epistemic Logic** to eliminate LLM hallucinations by integrating **Kleene's Strong Three-Valued Logic ($K_3$)** and **Judea Pearl’s do-calculus**.

## Overview
Standard binary neural networks force all probabilistic generations into a $[0, 1]$ binary distribution. When unobserved confounding variables exist, models default to guessing based on correlation, producing hallucinations.

This kernel intercepts LLM causal deductions and applies structural identifiability checks. If a query relies on unobserved variables, the premise chain collapses into a formal **Indeterminate ($\bot$)** state rather than a false binary projection, halting the hallucination and explicitly flagging the missing dependencies.

## Structure
*   `ternary_logic.py`: The algebraic engine defining $\mathcal{S}_{\text{ter}} = \{0, 0.5, 1\}$ and Kleene continuous continuous min-max arithmetic.
*   `causal_graph.py`: The Structural Causal Model validator implementing Back-Door Identifiability.
*   `ternary_kernel.py`: The interceptor pipeline that filters LLM responses.

## Quickstart

```bash
# Run the demonstration benchmark
python examples/demo_benchmark.py
```

## Example Output
```text
=== SCENARIO: Query with Latent Confounder (Unobserved Genetics) ===

[Standard Binary LLM]:
Forced to project onto [0, 1]. Prior correlation is high (~0.85).
Generated Output: "Clinical evidence indicates that administering Drug X directly accelerates Recovery Y." (HALLUCINATION / SPURIOUS)

[RGAI Ternary Causal Kernel]:
Status: HALLUCINATION_PREVENTED
Epistemic State: ⊥ (Indeterminate)
Final Filtered Output: [⊥ Indeterminate State]: Causal premises depend on unobserved confounding factors. Answer withheld to prevent hallucination.
Flagged Unobserved Dependencies: [{'treatment': 'Drug_X', 'outcome': 'Recovery_Y', 'explicit_truth': True}]
```
