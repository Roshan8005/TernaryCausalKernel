"""
Structural Causal Model (SCM) validator implementing Pearl's 
Back-Door Identifiability Criterion.
"""
from typing import Set, Dict, List, Optional
from ternary_causal_kernel.ternary_logic import TernaryState, KleeneEngine


class CausalNode:
    def __init__(self, name: str, is_latent: bool = False):
        self.name = name
        self.is_latent = is_latent  # True if variable is unobserved/confounding


class CausalGraph:
    def __init__(self):
        self.nodes: Dict[str, CausalNode] = {}
        self.parents: Dict[str, Set[str]] = {}
        self.children: Dict[str, Set[str]] = {}

    def add_node(self, name: str, is_latent: bool = False):
        if name not in self.nodes:
            self.nodes[name] = CausalNode(name, is_latent)
            self.parents[name] = set()
            self.children[name] = set()

    def add_edge(self, u: str, v: str):
        """Directed edge u -> v (u causes v)"""
        self.parents[v].add(u)
        self.children[u].add(v)

    def is_identifiable(self, treatment: str, outcome: str, observed_vars: Set[str]) -> bool:
        """
        Evaluates whether P(outcome | do(treatment)) is identifiable.
        Checks for unblocked back-door paths containing unobserved (latent) confounders.
        """
        # Find common confounders (parents of both treatment and outcome)
        common_parents = self.parents.get(treatment, set()).intersection(
            self.parents.get(outcome, set())
        )
        
        for parent in common_parents:
            # If a common confounder is latent (unobserved), back-door is unblockable
            if self.nodes[parent].is_latent or parent not in observed_vars:
                return False
        return True
