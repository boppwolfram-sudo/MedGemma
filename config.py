# =====================================================================
# config.py - Environment & Constants
# =====================================================================
import os

# Model Configuration
MEDGEMMA_MODEL_PATH = "google/medgemma-1.5-4b-it" 
USE_MOCK_INFERENCE = False # Set to False to use the real model!
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Inference Mathematics & Safety Constants
DEFAULT_TEMPERATURE = 1.5           # T > 1 to flatten overconfidence
DEFAULT_CONFIDENCE_THRESHOLD = 0.70 # Minimum Softmax probability required to proceed
CONTRADICTION_TOKEN = "Z"           # Token indicating 'None of the Above' / Contradiction

# Prompts
BRAIN_1_SYSTEM_PROMPT = """You are P-ATHENA, an empathetic Clinical Intake Assistant. Your ONLY goal is to gather the patient's chief complaint, symptom duration, and any accompanying symptoms. 
Do not provide medical advice. Do not diagnose the patient under any circumstances. 
Ask exactly ONE short, clarifying question at a time. 
If the patient states they have nothing else to add, you must strictly reply exactly with: '[READY_FOR_SUMMARY]' and nothing else."""

BRAIN_1_SUMMARY_PROMPT = """You are a clinical summarizer. Read the following patient conversation history. 
Extract exactly their symptoms, duration, and context into a clean, objective bulleted list. 
Do not include any pleasantries or conversational text. Output ONLY the bulleted list.

At the very end of your summary, you MUST include a routing tag that classifies the primary chief complaint.
Available pathways: [cough, headache, abdominal_pain, chest_pain, fever, back_pain, shortness_of_breath]
Format the tag exactly like this on its own line:
[ROUTING_TAG: cough]"""

def get_brain_2_decision_prompt(context_accumulator: str, clinical_question: str, mapping_str: str) -> str:
    """Strictly formatted prompt for the deterministic DecisionNode."""
    return f"""Clinical Triage Task.

[PATIENT HISTORY]
{context_accumulator}

[CLINICAL QUESTION]
{clinical_question}

[OPTIONS]
{mapping_str}

Respond strictly with a single letter from the options above. Do not output any other text.
Answer: """
