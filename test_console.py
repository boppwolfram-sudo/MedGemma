# =====================================================================
# test_console.py - Console-based test for Brain 1 (Phase 1 Chat)
# =====================================================================
# Run with: python test_console.py
# This tests the real MedGemma inference loop WITHOUT Streamlit overhead.
# Output is also logged to test_output.txt for inspection.

import traceback
import sys
import config
from brain_loader import MedGemmaLoader
from gatherer import PatientGatherer

class Tee:
    """Write to both stdout and a file."""
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()
    def isatty(self):
        return self.stdout.isatty()
    def flush(self):
        self.stdout.flush()
        self.file.flush()

def main():
    sys.stdout = Tee("test_output.txt")
    
    print("=" * 60)
    print("DIRECT Console Test - Brain 1 (Patient Intake)")
    print(f"Mode: {'MOCK' if config.USE_MOCK_INFERENCE else 'REAL INFERENCE'}")
    print("=" * 60)
    
    # 1. Load Model (only if real inference)
    if not config.USE_MOCK_INFERENCE:
        print(f"\nLoading {config.MEDGEMMA_MODEL_PATH}...")
        loader = MedGemmaLoader()
        loader.load_model(config.HF_TOKEN)
        print("Model loaded successfully!\n")
    else:
        print("Running in MOCK mode (no model loaded)\n")
    
    # 2. Create Gatherer
    gatherer = PatientGatherer()
    
    # 3. Automated test: send predefined messages
    test_messages = [
        "I have a cough since 1 year",
        "No other symptoms, nothing else to add"
    ]
    
    print("--- Chat Started ---")
    print(f"Assistant: {gatherer.visible_history[0]['content']}")
    print()
    
    turn = 0
    for msg in test_messages:
        if gatherer.ready_for_doctor:
            break
        turn += 1
        print(f"You: {msg}")
        print(f"\n[Turn {turn} - Generating response...]")
        try:
            reply = gatherer.ingest_patient_input(msg)
            print(f"\nAssistant: {reply}")
            print(f"[ready_for_doctor={gatherer.ready_for_doctor}]")
            print()
        except Exception as e:
            print(f"\n[ERROR during ingest_patient_input]")
            import traceback as tb
            print(tb.format_exc())
            break
    
    # 4. Show Summary
    print("\n" + "=" * 60)
    print("ANTI-HALLUCINATION GATE - Structured Summary:")
    print("=" * 60)
    summary = gatherer.get_structured_summary()
    if summary:
        print(summary)
    else:
        print("[No summary generated - model did not trigger [READY_FOR_SUMMARY]]")
    
    print("\nFull visible chat history:")
    for msg in gatherer.visible_history:
        print(f"  {msg['role'].upper()}: {msg['content']}")
    
    print("\n[Phase 1 Complete.]")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        # Also print to raw stdout in case Tee is failing
        sys.__stdout__.write("\n[FATAL ERROR TRACEBACK]\n")
        sys.__stdout__.write(traceback.format_exc())
        print(f"\n[FATAL ERROR]")
        traceback.print_exc()
