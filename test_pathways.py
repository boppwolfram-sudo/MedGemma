# =====================================================================
# test_pathways.py - End-to-End DAG Traversal Tests for P-ATHENA
# =====================================================================
# Tests 3 complete clinical scenarios without requiring the real LLM.
# Uses a MockLLM that returns scripted probability distributions.
#
# KEY DESIGN NOTE:
# PHASE_1_CLARIFY is detected in app.py AFTER tree.step() returns.
# If the winning token's next_node == "PHASE_1_CLARIFY", app.py intercepts.
# This means the mock must make the PHASE_1_CLARIFY token WIN (highest prob).
# The tree itself treats it as a normal PROCEED — app.py does the interception.
# =====================================================================
import sys
import re

# --------------- Mock LLM ---------------
class MockLLM:
    """Queue-based mock. Each evaluate_node() call pops the next response."""
    def __init__(self):
        self._queue = []

    def queue_response(self, probs: dict):
        self._queue.append(probs)

    def evaluate_node(self, prompt: str, target_tokens: list, temperature: float) -> dict:
        if self._queue:
            scripted = self._queue.pop(0)
            for tok in target_tokens:
                if tok not in scripted:
                    scripted[tok] = 0.0
            return scripted
        return {tok: (1.0 if i == 0 else 0.0) for i, tok in enumerate(target_tokens)}


# --------------- Utilities ---------------
def header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def log_step(step, result, extra=""):
    status = result["status"]
    node_id = result.get("node_id", "?")
    question = result.get("question", "")[:55]
    
    tag = {"PROCEED": "OK  ", "UNCERTAIN_EIG": "EIG ", "ROOT_REDIRECT": "BACK",
           "END": "DONE", "ERROR": "ERR "}.get(status, "????")
    
    print(f"  [{tag}] Step {step}: {node_id}")
    print(f"         Q: {question}...")
    
    if status == "PROCEED":
        w = result["winning_data"]
        conf = max(result["probabilities"].values())
        print(f"         -> {w.get('label', '?')[:60]} (conf={conf:.0%})")
        nxt = w.get("next_node")
        print(f"         -> next: {nxt if nxt else 'TERMINAL'}")
    elif status == "UNCERTAIN_EIG":
        fb = result.get("fallback_trigger", {})
        print(f"         -> EIG Fallback: {fb.get('label', '?')[:60]}")
        print(f"         -> next: {fb.get('next_node', '?')}")
    elif status in ("END", "ERROR"):
        print(f"         -> {result.get('message', 'Done')}")
    if extra:
        print(f"         ** {extra}")


def detect_phase1_clarify(result):
    """
    Mirrors app.py logic: checks if the step result leads to PHASE_1_CLARIFY.
    Works for both PROCEED and UNCERTAIN_EIG statuses.
    """
    wd = result.get("winning_data", {})
    next_node = wd.get("next_node", "")
    label = wd.get("label", "")
    
    if result["status"] == "UNCERTAIN_EIG":
        fb = result.get("fallback_trigger", {})
        if fb:
            next_node = fb.get("next_node", next_node)
            label = fb.get("label", label)
    
    if next_node == "PHASE_1_CLARIFY":
        q_match = re.search(r"Ask patient[:\s]*['\"](.+?)['\"]", label, re.IGNORECASE)
        return True, q_match.group(1) if q_match else label
    return False, None


def run_scenario(tree, mock, clarify_answers, max_steps=15):
    """Run tree to completion, intercepting PHASE_1_CLARIFY like app.py does."""
    step = 0
    results = []
    clarify_idx = 0

    while step < max_steps:
        step += 1
        result = tree.step()
        
        is_clarify, question = detect_phase1_clarify(result)
        
        if is_clarify:
            answer = clarify_answers[clarify_idx] if clarify_idx < len(clarify_answers) else "Not sure"
            log_step(step, result, f"PHASE_1_CLARIFY: \"{question}\"")
            print(f"         ** Patient: \"{answer}\"")
            
            # Rewind: update context, keep same node
            tree.context += f"\nFollow-up Q: {question} | Patient A: {answer}"
            tree.current_node_id = result["node_id"]
            result["status"] = "PHASE_1_CLARIFY"
            results.append(result)
            clarify_idx += 1
            continue
        
        log_step(step, result)
        results.append(result)
        
        if result["status"] in ("END", "ERROR"):
            break
        if result["status"] == "PROCEED" and result["winning_data"].get("next_node") is None:
            step += 1
            final = tree.step()
            log_step(step, final)
            results.append(final)
            break

    return results


# =====================================================================
# SCENARIO 1: Cough -> PHASE_1_CLARIFY -> Post-Infectious
#
# The PHASE_1_CLARIFY token (C) must WIN with conf >= threshold (0.85)
# so tree.step() returns PROCEED with next_node=PHASE_1_CLARIFY.
# Then app.py (simulated here) intercepts it.
# =====================================================================
def test_scenario_1():
    from pathways import get_pathway
    from tree_core import ClinicalTree

    header("SCENARIO 1: Cough + PHASE_1_CLARIFY -> Post-Infectious")
    print("  Patient: Cough 5 weeks, started after cold")
    print("  Path: L1(clarify) -> L1(B) -> L2(B) -> L3(A) -> dx_post_infectious")
    print()

    nodes = get_pathway("cough")
    mock = MockLLM()

    # 1. cough_L1: C wins (PHASE_1_CLARIFY) with 0.88 > 0.85 threshold
    mock.queue_response({"A": 0.02, "B": 0.05, "C": 0.88, "Z": 0.05})
    # 2. cough_L1 re-eval: B wins (No Red Flags) after patient clarified
    mock.queue_response({"A": 0.02, "B": 0.92, "C": 0.04, "Z": 0.02})
    # 3. cough_L2: B wins (Sub-acute 3-8 weeks)
    mock.queue_response({"A": 0.05, "B": 0.88, "C": 0.04, "D": 0.03})
    # 4. cough_L3: A wins (Post-infectious)
    mock.queue_response({"A": 0.82, "B": 0.08, "C": 0.06, "D": 0.04})
    # 5. dx_post_infectious: terminal
    mock.queue_response({"A": 1.0})

    tree = ClinicalTree(nodes, "cough_L1", mock)
    tree.start("Symptoms: Cough. Duration: 5 weeks. Started after a cold.")

    results = run_scenario(tree, mock,
        clarify_answers=["No blood, I can breathe fine at rest"])

    statuses = [r["status"] for r in results]
    assert "PHASE_1_CLARIFY" in statuses, f"Expected PHASE_1_CLARIFY, got: {statuses}"
    assert any("Post-infectious" in r.get("question", "") for r in results), \
        "Expected Post-infectious diagnosis"
    
    print(f"\n  PASSED: {len(results)} steps | 1x clarify | Dx: Post-Infectious Cough")
    return True


# =====================================================================
# SCENARIO 2: Headache -> Clean Path -> Migraine
# =====================================================================
def test_scenario_2():
    from pathways import get_pathway
    from tree_core import ClinicalTree

    header("SCENARIO 2: Headache -> Clean Path -> Migraine")
    print("  Patient: Throbbing headache left side, nausea, photophobia, 3 days")
    print("  Path: L1(B) -> L2(A) -> dx_migraine")
    print()

    nodes = get_pathway("headache")
    mock = MockLLM()

    mock.queue_response({"A": 0.03, "B": 0.90, "C": 0.05, "Z": 0.02})
    mock.queue_response({"A": 0.85, "B": 0.07, "C": 0.04, "D": 0.04})
    mock.queue_response({"A": 1.0})

    tree = ClinicalTree(nodes, "headache_L1", mock)
    tree.start("Symptoms: Headache, throbbing, left side, nausea, photophobia. Duration: 3 days.")

    results = run_scenario(tree, mock, clarify_answers=[])

    statuses = [r["status"] for r in results]
    assert "PHASE_1_CLARIFY" not in statuses, f"Should not clarify: {statuses}"
    assert any("Migraine" in r.get("question", "") for r in results), \
        "Expected Migraine diagnosis"
    
    print(f"\n  PASSED: {len(results)} steps | No clarification | Dx: Migraine")
    return True


# =====================================================================
# SCENARIO 3: Chest Pain -> 2x PHASE_1_CLARIFY -> GERD
# =====================================================================
def test_scenario_3():
    from pathways import get_pathway
    from tree_core import ClinicalTree

    header("SCENARIO 3: Chest Pain -> 2x PHASE_1_CLARIFY -> GERD")
    print("  Patient: Burning chest pain after eating, worse lying down, 2 weeks")
    print("  Path: L1(clarify->B) -> L2(clarify->C) -> dx_gerd")
    print()

    nodes = get_pathway("chest_pain")
    mock = MockLLM()

    # 1. cp_L1: C wins (PHASE_1_CLARIFY) conf=0.88 > 0.85
    mock.queue_response({"A": 0.02, "B": 0.05, "C": 0.88, "Z": 0.05})
    # 2. cp_L1 re-eval: B wins (no red flags) after clarify
    mock.queue_response({"A": 0.02, "B": 0.93, "C": 0.03, "Z": 0.02})
    # 3. cp_L2: D wins (PHASE_1_CLARIFY) conf=0.85 > 0.80
    mock.queue_response({"A": 0.03, "B": 0.04, "C": 0.05, "D": 0.85, "Z": 0.03})
    # 4. cp_L2 re-eval: C wins (GI/GERD) after clarify
    mock.queue_response({"A": 0.03, "B": 0.04, "C": 0.88, "D": 0.03, "Z": 0.02})
    # 5. dx_gerd: terminal
    mock.queue_response({"A": 1.0})

    tree = ClinicalTree(nodes, "cp_L1_red_flags", mock)
    tree.start("Symptoms: Chest pain, burning. Duration: 2 weeks. Worse after eating, lying down.")

    results = run_scenario(tree, mock, clarify_answers=[
        "No, no tearing feeling and no vomiting.",
        "It is a burning sensation in my throat after meals."
    ])

    statuses = [r["status"] for r in results]
    clarify_count = statuses.count("PHASE_1_CLARIFY")
    assert clarify_count == 2, f"Expected 2 clarify loops, got {clarify_count}: {statuses}"
    assert any("GERD" in r.get("question", "") for r in results), \
        "Expected GERD diagnosis"
    
    print(f"\n  PASSED: {len(results)} steps | 2x clarify | Dx: GERD/Acid Reflux")
    return True


# =====================================================================
if __name__ == "__main__":
    header("P-ATHENA End-to-End DAG Test Suite")
    print("  3 clinical scenarios | Mock LLM | No GPU required")
    
    passed = 0
    failed = 0
    
    for test_fn in [test_scenario_1, test_scenario_2, test_scenario_3]:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"\n  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    header("FINAL RESULTS")
    total = passed + failed
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {failed}/{total}")
    print(f"\n  {'ALL SCENARIOS PASSED' if failed == 0 else f'{failed} SCENARIO(S) FAILED'}")
    if failed > 0:
        sys.exit(1)
