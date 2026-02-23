# =====================================================================
# pathways.py - The Clinical DAG Scenarios (Generalized Builder)
# =====================================================================
from typing import Dict, Any
from tree_core import DecisionNode

def build_pathway(schema: Dict[str, Any]) -> Dict[str, DecisionNode]:
    """
    Dynamically builds a DAG of DecisionNodes from a generic dictionary schema.
    This allows pathways to be loaded easily from JSON files, databases, or API responses.
    """
    nodes = {}
    for node_id, data in schema.items():
        nodes[node_id] = DecisionNode(
            node_id=node_id,
            clinical_question=data["clinical_question"],
            token_map=data["token_map"],
            confidence_threshold=data.get("confidence_threshold", 0.70)
        )
    return nodes

# =====================================================================
# DATA SCHEMAS (These could easily be moved to external .json files)
# =====================================================================

RESPIRATORY_TRIAGE_SCHEMA = {
    "cough_L1": {
        "clinical_question": "Are there signs of hemoptysis or severe dyspnea at rest?",
        "token_map": {
            "A": {"label": "Emergency ER", "next_node": None, "cost": 0},
            "B": {"label": "Routine Care - No Red Flags", "next_node": "cough_L2", "cost": 0},
            "C": {"label": "Need more info: Ask patient 'Are you coughing up any blood, or struggling to breathe while resting?'", "next_node": "PHASE_1_CLARIFY", "cost": 0}
        },
        "confidence_threshold": 0.85
    },
    "cough_L2": {
        "clinical_question": "What is the duration of the cough symptom?",
        "token_map": {
            "A": {"label": "Acute < 3 weeks", "next_node": "cough_L3", "cost": 0},
            "B": {"label": "Sub-acute 3-8 weeks", "next_node": "cough_L3", "cost": 0},
            "C": {"label": "Chronic > 8 weeks", "next_node": "dx_chronic", "cost": 0},
            "D": {"label": "Need more info: Ask patient 'Exactly how many weeks have you had this cough?'", "next_node": "PHASE_1_CLARIFY", "cost": 0}
        },
        "confidence_threshold": 0.80
    },
    "cough_L3": {
        "clinical_question": "What is the most likely etiology based on accompanying symptoms?",
        "token_map": {
            "A": {"label": "Post-infectious (Prior cold symptoms)", "next_node": "dx_post_infectious", "cost": 0},
            "B": {"label": "Pertussis (Whooping sound, vomiting after cough)", "next_node": "dx_pertussis", "cost": 50}, 
            "C": {"label": "Pneumonia / Unclear (Fever, chills)", "next_node": "order_cxr", "cost": 150}, 
            "D": {"label": "Need more info: Ask patient 'Did this start with a normal cold, or do you have a high fever?'", "next_node": "PHASE_1_CLARIFY", "cost": 0}
        },
        "confidence_threshold": 0.70
    },
    "dx_post_infectious": {
        "clinical_question": "Final Diagnosis: Post-infectious cough. Resolves with time.",
        "token_map": {"A": {"label": "Confirm Diagnosis", "next_node": None, "cost": 0}}
    },
    "dx_pertussis": {
        "clinical_question": "Final Diagnosis: Suspected Pertussis. Recommend Macrolide antibiotics.",
        "token_map": {"A": {"label": "Confirm Diagnosis", "next_node": None, "cost": 0}}
    },
    "dx_chronic": {
        "clinical_question": "Final Diagnosis: Chronic Cough. Evaluate for Asthma, GERD, or ACE-inhibitor use.",
        "token_map": {"A": {"label": "Confirm Diagnosis", "next_node": None, "cost": 0}}
    },
    "order_cxr": {
        "clinical_question": "Action: Ordering Chest X-Ray to rule out Pneumonia.",
        "token_map": {"A": {"label": "Confirm Action", "next_node": None, "cost": 0}}
    }
}

HEADACHE_TRIAGE_SCHEMA = {
    "headache_L1": {
        "clinical_question": "Are there red flags: sudden 'thunderclap' onset, worst headache of life, or recent head trauma?",
        "token_map": {
            "A": {"label": "Emergency ER (Rule out Subarachnoid Hemorrhage/Bleed)", "next_node": None, "cost": 0},
            "B": {"label": "No red flags present. Gradual or typical onset.", "next_node": "headache_L2", "cost": 0},
            "C": {"label": "Need more info: Ask patient 'Did this headache start completely suddenly within seconds, or build up gradually?'", "next_node": "PHASE_1_CLARIFY", "cost": 0},
        },
        "confidence_threshold": 0.85
    },
    "headache_L2": {
        "clinical_question": "What are the specific characteristics and associated symptoms of the headache?",
        "token_map": {
            "A": {"label": "Throbbing, unilateral, photophobia, nausea (Migraine)", "next_node": "dx_migraine", "cost": 0},
            "B": {"label": "Bilateral, band-like, non-throbbing, mild-moderate (Tension)", "next_node": "dx_tension", "cost": 0},
            "C": {"label": "Severe, unilateral orbital pain with eye tearing/redness (Cluster)", "next_node": "dx_cluster", "cost": 0},
            "D": {"label": "Need more info: Ask patient 'Is the pain throbbing on one side, or does it feel like a tight band around your head?'", "next_node": "PHASE_1_CLARIFY", "cost": 0}
        },
        "confidence_threshold": 0.75
    },
    "dx_migraine": {
        "clinical_question": "Final Diagnosis: Migraine Pathway. Proceed with migraine-specific abortive therapy (Triptans).",
        "token_map": {"A": {"label": "Confirm Diagnosis", "next_node": None, "cost": 0}}
    },
    "dx_tension": {
        "clinical_question": "Final Diagnosis: Tension-Type Headache. Proceed with NSAIDs or Acetaminophen.",
        "token_map": {"A": {"label": "Confirm Diagnosis", "next_node": None, "cost": 0}}
    },
    "dx_cluster": {
        "clinical_question": "Final Diagnosis: Cluster Headache. Proceed with High-Flow Oxygen.",
        "token_map": {"A": {"label": "Confirm Diagnosis", "next_node": None, "cost": 0}}
    }
}

CHEST_PAIN_TRIAGE_SCHEMA = {
    "cp_L1_red_flags": {
        "clinical_question": "Does the patient describe a sudden onset of severe ripping/tearing pain to the back, or sudden unilateral pleuritic pain?",
        "token_map": {
            "A": {"label": "Activate Resuscitation Protocol (Dissection/PTX)", "next_node": None, "cost": 0},
            "B": {"label": "No red flag descriptors present", "next_node": "cp_L2_ischemic_nature", "cost": 0},
            "C": {"label": "Need more info: Ask patient 'Does the pain feel like it is tearing into your back, or did it start after violent vomiting?'", "next_node": "PHASE_1_CLARIFY", "cost": 0}
        },
        "confidence_threshold": 0.85
    },
    
    "cp_L2_ischemic_nature": {
        "clinical_question": "What is the primary character of the chest pain and associated symptoms?",
        "token_map": {
            "A": {"label": "Squeezing, crushing, heavy pressure, radiation to jaw/arm, diaphoresis (Cardiac)", "next_node": "cp_L3_risk_factors", "cost": 0},
            "B": {"label": "Sharp, pleuritic, reproducible by palpation, or fleeting (Noncardiac/Possibly Cardiac)", "next_node": "cp_L3_risk_factors", "cost": 0},
            "C": {"label": "Burning sensation after eating, worse when lying down (Gastrointestinal)", "next_node": "dx_gerd", "cost": 0},
            "D": {"label": "Need more info: Ask patient 'Does the pain feel like a heavy weight, a sharp pain when breathing, or a burning in your throat?'", "next_node": "PHASE_1_CLARIFY", "cost": 0}
        },
        "confidence_threshold": 0.80
    },

    "cp_L3_risk_factors": {
        "clinical_question": "How many atherosclerotic risk factors (HTN, DM, Obesity, Smoking, Age >65) does the patient have?",
        "token_map": {
            "A": {"label": "Zero to Two risk factors, Age < 65", "next_node": "cp_L4_biomarkers", "cost": 0},
            "B": {"label": "Three or more risk factors, Age >= 65, or known CAD/prior MI", "next_node": "cp_L4_biomarkers", "cost": 0},
            "C": {"label": "Need more info: Ask patient 'Do you have high blood pressure, diabetes, or a history of heart issues?'", "next_node": "PHASE_1_CLARIFY", "cost": 0}
        },
        "confidence_threshold": 0.75
    },

    "cp_L4_biomarkers": {
        "clinical_question": "Action required: Perform baseline ECG and High-Sensitivity Cardiac Troponin.",
        "token_map": {
            "A": {"label": "Order ECG + hs-Troponin to rule out Acute Coronary Syndrome", "next_node": None, "cost": 54}
        },
        "confidence_threshold": 0.50
    },

    "dx_gerd": {
        "clinical_question": "Final Diagnosis: Suspected GERD / Acid Reflux. Recommend antacids/PPI.",
        "token_map": {"A": {"label": "Confirm Diagnosis", "next_node": None, "cost": 0}}
    }
}

# =====================================================================
# PATHWAY REGISTRY & LOADER
# =====================================================================

PATHWAY_REGISTRY = {
    "cough": RESPIRATORY_TRIAGE_SCHEMA,
    "headache": HEADACHE_TRIAGE_SCHEMA,
    "chest_pain": CHEST_PAIN_TRIAGE_SCHEMA
}

def get_pathway(pathway_name: str) -> Dict[str, DecisionNode]:
    """
    Fetches the generic schema by name and builds the live DAG for the engine.
    Example: get_pathway("headache")
    """
    schema = PATHWAY_REGISTRY.get(pathway_name.lower())
    if not schema:
        raise ValueError(f"Pathway '{pathway_name}' not found in registry. Available: {list(PATHWAY_REGISTRY.keys())}")
    
    return build_pathway(schema)
