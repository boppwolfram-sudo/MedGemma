# =====================================================================
# tree_core.py - The Engine (DAG Mechanics)
# =====================================================================
import math
from typing import Dict, Any, Tuple
import config
from llm_engine import MedGemmaEngine

def calculate_entropy(probabilities: Dict[str, float]) -> float:
    """
    Shannon Entropy H(S) = -sum(P(i) * log2(P(i)))
    Quantifies the exact diagnostic uncertainty at the current node.
    """
    entropy = 0.0
    for p in probabilities.values():
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

class DecisionNode:
    """Represents a rigorous, deterministic checkpoint in the diagnostic DAG."""
    def __init__(
        self, 
        node_id: str, 
        clinical_question: str, 
        token_map: Dict[str, Dict[str, Any]], 
        confidence_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
        temperature: float = config.DEFAULT_TEMPERATURE
    ):
        self.node_id = node_id
        self.clinical_question = clinical_question
        self.token_map = token_map
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        
        # Mandatory Contradiction Token for Intra-visit backtracking
        self.contradiction_token = config.CONTRADICTION_TOKEN
        if self.contradiction_token not in self.token_map:
            self.token_map[self.contradiction_token] = {
                "label": "Contradiction / None of the Above", 
                "next_node": "ROOT_REDIRECT",
                "cost": 0
            }

    def _generate_prompt(self, context_accumulator: str) -> str:
        mapping_str = "\n".join([f"{k}: {v['label']}" for k, v in self.token_map.items()])
        return config.get_brain_2_decision_prompt(context_accumulator, self.clinical_question, mapping_str)

    def calculate_eig_utility(self, current_entropy: float) -> Dict[str, Any]:
        """
        Calculates the Expected Information Gain (EIG) and Cost-Utility (ω) for candidate branches.
        Utility ω = (Current Entropy * Option Weight) / ln(Cost + e)
        This ensures $0 queries are prioritized, while expensive tests require high entropy to trigger.
        """
        utilities = {}
        # Euler's number for logarithmic smoothing
        e_val = math.e 
        
        for tok, data in self.token_map.items():
            if tok != self.contradiction_token:
                cost = data.get("cost", 0)
                
                # Heuristic: Higher cost tests usually yield higher absolute information gain if taken.
                # In a full deployment, this is E[H(S_next)], but here we approximate the gain potential.
                expected_gain = current_entropy * (0.5 if cost == 0 else 0.9) 
                
                # Logarithmic cost penalty prevents division by zero and smooths extreme values
                cost_penalty = math.log(cost + e_val)
                
                # Final Utility Metric (ω)
                utility = expected_gain / cost_penalty
                
                utilities[tok] = {
                    "label": data["label"],
                    "cost": cost,
                    "eig": expected_gain,
                    "utility": utility,
                    "next_node": data["next_node"]
                }
        return utilities

    def evaluate(self, context_accumulator: str, llm: MedGemmaEngine) -> Tuple[str, Dict[str, float], dict, float]:
        """
        Evaluates the node.
        Returns: status, probabilities, winning_data, entropy
        """
        prompt = self._generate_prompt(context_accumulator)
        target_tokens = list(self.token_map.keys())
        
        probs = llm.evaluate_node(prompt, target_tokens, self.temperature)
        entropy = calculate_entropy(probs)
        
        winning_token = max(probs, key=probs.get)
        winning_prob = probs[winning_token]
        winning_data = self.token_map[winning_token].copy() # Copy to avoid mutating original tree state
        
        # 1. Backtracking Trigger
        if winning_token == self.contradiction_token:
            return "ROOT_REDIRECT", probs, winning_data, entropy
            
        # 2. Safety Threshold Cleared -> Proceed
        if winning_prob >= self.confidence_threshold:
            return "PROCEED", probs, winning_data, entropy
            
        # 3. Safety Threshold Failed -> Trigger EIG Cost-Utility Routing
        else:
            winning_data["utilities"] = self.calculate_eig_utility(current_entropy=entropy)
            return "UNCERTAIN_EIG", probs, winning_data, entropy

class ClinicalTree:
    """Manages the active DAG node, traversal state, and metric tracking."""
    def __init__(self, nodes: Dict[str, DecisionNode], root_id: str, llm: MedGemmaEngine):
        self.nodes = nodes
        self.root_id = root_id
        self.llm = llm
        
        self.current_node_id = root_id
        self.context = ""
        self.history = []
        self.system_entropy = 0.0
        self.last_winning_prob = 0.0

    def start(self, initial_context: str):
        """Initializes the tree with the clean context from Brain 1."""
        self.context = initial_context
        self.current_node_id = self.root_id
        self.history = []
        self.system_entropy = 0.0
        self.last_winning_prob = 0.0
        
    def step(self) -> Dict[str, Any]:
        """Executes one pass through the currently active node."""
        if not self.current_node_id:
            return {"status": "END", "message": "Pathway complete."}
            
        node = self.nodes.get(self.current_node_id)
        if not node:
            return {"status": "ERROR", "message": f"Node '{self.current_node_id}' not found in pathway definitions."}
            
        status, probs, winning_data, entropy = node.evaluate(self.context, self.llm)
        
        prob_gain = max(probs.values()) - self.last_winning_prob if self.history else 0.0
        self.last_winning_prob = max(probs.values())
        self.system_entropy = entropy
        
        result = {
            "node_id": self.current_node_id,
            "question": node.clinical_question,
            "status": status,
            "probabilities": probs,
            "winning_data": winning_data,
            "entropy": entropy,
            "probability_gain": prob_gain
        }
        
        if status == "PROCEED":
            self.history.append(result)
            self.context += f"\nQ: {node.clinical_question}\nA: {winning_data['label']}"
            self.current_node_id = winning_data.get("next_node")
            
        elif status == "ROOT_REDIRECT":
            self.history.append(result)
            self.context += "\n[CONTRADICTION DETECTED. Re-evaluating.]"
            self.current_node_id = self.root_id
            
        elif status == "UNCERTAIN_EIG":
            # Select the path that provides the highest Utility (ω) per dollar spent
            utilities = winning_data.get("utilities", {})
            if utilities:
                best_action_token = max(utilities, key=lambda k: utilities[k]['utility'])
                fallback_data = utilities[best_action_token]
                
                result["fallback_trigger"] = fallback_data
                self.history.append(result)
                self.context += f"\n[UNCERTAIN. Cost-Utility Fallback triggered: {fallback_data['label']}]"
                self.current_node_id = fallback_data.get("next_node")
            else:
                 self.current_node_id = None # Dead end, no viable options
            
        return result
