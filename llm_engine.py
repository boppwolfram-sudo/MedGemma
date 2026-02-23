# =====================================================================
# llm_engine.py - The Math & Brain 2 (Single-pass inference)
# =====================================================================
import math
from typing import Dict, List
import config

class MedGemmaEngine:
    """
    Handles all interactions with MedGemma 4B for the diagnostic reasoning phase.
    Never uses .generate(); only runs forward passes to extract logits.
    """
    def __init__(self, model_name_or_path: str = config.MEDGEMMA_MODEL_PATH, use_mock: bool = config.USE_MOCK_INFERENCE):
        self.use_mock = use_mock
        self.model_name_or_path = model_name_or_path
        if not use_mock:
            # Placeholder for real initialization (transformers AutoModelForCausalLM)
            pass

    def get_raw_logits(self, prompt: str, target_tokens: List[str]) -> Dict[str, float]:
        """
        Passes the structured node prompt to the real model and extracts specific logit values.
        OPTIMIZED: Uses zero-cache forward passes and explicit VRAM garbage collection.
        """
        if self.use_mock:
            # Fallback mock logic for testing UI without VRAM
            # Simulate a realistic spread of logits for the UI charts
            return {t: float(10 - (i * 2)) for i, t in enumerate(target_tokens)}
            
        else:
            from brain_loader import MedGemmaLoader
            import torch
            
            loader = MedGemmaLoader()
            model, tokenizer = loader.get_model_and_tokenizer()
            
            # 1. Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # 2. Single Forward Pass (No Generation)
            # OPTIMIZATION: use_cache=False prevents VRAM bloat since we aren't generating text
            with torch.no_grad():
                outputs = model(**inputs, use_cache=False)
                
            # 3. Extract logits for the very last token position
            next_token_logits = outputs.logits[0, -1, :]
            
            # 4. Map the string tokens ("A", "B", "Z") to their vocabulary IDs and get the values
            result_logits = {}
            for token_str in target_tokens:
                # Note: Gemma tokenization often prepends a space to single letters. 
                # We extract the exact ID for predictability.
                token_id = tokenizer.encode(token_str, add_special_tokens=False)[-1]
                logit_val = next_token_logits[token_id].item()
                result_logits[token_str] = logit_val
            
            # OPTIMIZATION: Aggressive VRAM cleanup to prevent Edge OOM crashes
            del inputs
            del outputs
            del next_token_logits
            torch.cuda.empty_cache()
                
            return result_logits

    def apply_temperature_scaling(self, raw_logits: Dict[str, float], temperature: float) -> Dict[str, float]:
        """
        Implements the logit calibration math: z_i / T
        """
        return {token: logit / temperature for token, logit in raw_logits.items()}

    def calculate_softmax(self, scaled_logits: Dict[str, float]) -> Dict[str, float]:
        """
        Normalizes the scores into exact percentages.
        P(y_i|x) = exp(z_i/T) / sum(exp(z_j/T))
        """
        max_logit = max(scaled_logits.values()) # For numerical stability
        exp_vals = {token: math.exp(val - max_logit) for token, val in scaled_logits.items()}
        sum_exp = sum(exp_vals.values())
        return {token: exp_val / sum_exp for token, exp_val in exp_vals.items()}
        
    def evaluate_node(self, prompt: str, target_tokens: List[str], temperature: float) -> Dict[str, float]:
        """Convenience method for the full single-pass inference chain."""
        raw_logits = self.get_raw_logits(prompt, target_tokens)
        scaled_logits = self.apply_temperature_scaling(raw_logits, temperature)
        probabilities = self.calculate_softmax(scaled_logits)
        return probabilities
