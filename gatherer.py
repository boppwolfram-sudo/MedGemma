# =====================================================================
# gatherer.py - Brain 1: Patient Data
# =====================================================================
import config

class PatientGatherer:
    """
    The generative brain for patient intake. 
    Handles conversational data gathering, enforces a strict turn limit, 
    and triggers the Anti-Hallucination check.
    """
    def __init__(self, max_turns: int = 5):
        self.system_prompt = config.BRAIN_1_SYSTEM_PROMPT
        self.max_turns = max_turns
        self.current_turn = 0
        
        # We start the chat history with the system prompt as the first "user" turn
        # and the greeting as the first "assistant" turn.
        # Gemma REQUIRES strict user/assistant/user/assistant alternation.
        self.chat_history = [
            {"role": "user", "content": config.BRAIN_1_SYSTEM_PROMPT},
            {"role": "assistant", "content": "Understood. I will gather the patient's history empathetically and ask one clarifying question at a time. What is the main reason you are seeking care today?"}
        ]
        
        # This is for the UI display (hiding system prompts)
        self.visible_history = [{"role": "assistant", "content": (
            "Hello, it is so nice to meet you. I am P-ATHENA.\n\n"
            "You might wonder why an AI like me exists in a clinic. Well, let me explain: "
            "modern healthcare moves incredibly fast, and there simply isn't always enough time "
            "for doctors to sit down and just listen. And yet, listening is exactly where healing begins.\n\n"
            "Because I am an AI, I am not bound by a 10-minute appointment window. I am here to give you "
            "all the unhurried time you need. My job is to listen carefully and dig deeper into your history, "
            "making sure no subtle detail is missed. When we are finished, I will organize everything for "
            "your doctor so they can focus entirely on your care.\n\n"
            "Take a deep breath, take as much time as you need, and tell me: "
            "**What is the main reason you are seeking care today?**"
        )}]
        
        self.structured_summary = ""
        self.is_confirmed = False
        self.ready_for_doctor = False
        self.in_eig_loop = False # Tracks if we are in the Brain 2 callback loop

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Strips MedGemma's internal chain-of-thought markers from output."""
        if '<unused95>' in text:
            text = text.split('<unused95>')[-1].strip()
        if '<unused94>' in text:
            text = text.split('<unused94>')[0].strip()
        return text

    def _ensure_alternating(self):
        """Repairs chat_history to ensure strict user/assistant alternation.
        Merges consecutive same-role messages to prevent Jinja template errors."""
        if len(self.chat_history) < 2:
            return
        cleaned = [self.chat_history[0]]
        for msg in self.chat_history[1:]:
            if cleaned[-1]["role"] == msg["role"]:
                # Merge into previous message
                cleaned[-1]["content"] += "\n" + msg["content"]
            else:
                cleaned.append(msg)
        self.chat_history = cleaned

    def ingest_patient_input(self, text_input: str) -> str:
        """Stores the patient input and streams a response from the LLM."""
        
        # If Brain 2 has injected a specific question, handle it via the specialized EIG loop
        if self.in_eig_loop:
            return self.ingest_eig_response(text_input)
            
        self.chat_history.append({"role": "user", "content": text_input})
        self.visible_history.append({"role": "user", "content": text_input})
        self.is_confirmed = False
        self.current_turn += 1
        
        if config.USE_MOCK_INFERENCE:
            # Fallback mock logic for testing UI without VRAM
            if self.current_turn >= 2: # Mock cuts to the chase quickly
                reply = "Thank you. I have compiled your history. Please review the structured summary to your right and click 'Confirm & Send' if it is completely accurate."
                self.visible_history.append({"role": "assistant", "content": reply})
                self.structured_summary = f"Patient confirms the following history:\n- Symptom: {text_input}\n- Duration: 2 weeks"
                self.ready_for_doctor = True
                return reply
            else:
                reply = "Can you tell me how long this has been going on?"
                self.visible_history.append({"role": "assistant", "content": reply})
                self.chat_history.append({"role": "assistant", "content": reply})
                return reply

        # ---- REAL INFERENCE MODE ----
        import torch
        from brain_loader import MedGemmaLoader
        
        loader = MedGemmaLoader()
        model, tokenizer = loader.get_model_and_tokenizer()
        
        # Optimization 1: Force summary if maximum turns reached to prevent infinite loops
        force_summary = self.current_turn >= self.max_turns

        if force_summary:
             # Manually inject the trigger word into the system's thought process
             reply = "[READY_FOR_SUMMARY]"
        else:
            # 1. Format history with Chat Template
            self._ensure_alternating()  # Safety: repair any broken alternation
            prompt = tokenizer.apply_chat_template(
                self.chat_history, 
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # 2. Generate Chat Response - Optimized parameters for strict, short Q&A
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=200,     # Room for thinking tokens (~120) + actual reply (~80)
                    do_sample=True,
                    temperature=0.4,        # Optimization 3: Lower temp for more focused, clinical questions
                    repetition_penalty=1.15 # Optimization 4: Prevent model from asking the same thing twice
                )
                
            # 3. Decode only the new generated tokens
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0, input_length:]
            reply = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Strip MedGemma's internal chain-of-thought before showing to user
            reply = self._strip_thinking(reply)
            
            # Fallback: if thinking consumed all tokens and reply is empty, ask a generic follow-up
            if not reply:
                reply = "Can you tell me more about when this started and how it has changed over time?"
            
            # Optimization 5: VRAM Cleanup to prevent OOM on Edge Devices
            del inputs
            del outputs
            torch.cuda.empty_cache()
        
        # 4. Intercept Logic: Is it ready for summary?
        if "[READY_FOR_SUMMARY]" in reply or force_summary:
            self.ready_for_doctor = True
            ui_reply = "Thank you. I have gathered your clinical history. Please review the structured summary to your right and click 'Confirm' if it is completely accurate."
            self.chat_history.append({"role": "assistant", "content": ui_reply})
            self.visible_history.append({"role": "assistant", "content": ui_reply})
            self._generate_real_summary(model, tokenizer)
            return ui_reply
        else:
            # Normal conversational turn
            self.chat_history.append({"role": "assistant", "content": reply})
            self.visible_history.append({"role": "assistant", "content": reply})
            return reply

    def _generate_real_summary(self, model, tokenizer):
        """Forces the LLM to write the objective bulleted list using chat template."""
        import torch
        
        # Build a clean chat history string from the visible conversation
        chat_text = ""
        for msg in self.visible_history:
            role = msg['role'].capitalize()
            content = msg['content']
            # Strip any thinking tokens that leaked into history
            if '<unused95>' in content:
                content = content.split('<unused95>')[-1].strip()
            if '<unused94>' in content:
                content = content.split('<unused94>')[0].strip()
            chat_text += f"{role}: {content}\n"
        
        # Use chat template for proper formatting (prevents hallucination)
        summary_messages = [
            {"role": "user", "content": config.BRAIN_1_SUMMARY_PROMPT + "\n\nChat History:\n" + chat_text},
            {"role": "assistant", "content": "Here is the clinical summary:\n-"}
        ]
        
        prompt = tokenizer.apply_chat_template(
            summary_messages,
            tokenize=False,
            add_generation_prompt=False
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Strict Deterministic Decoding with repetition penalty for clinical extraction
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=250, 
                do_sample=False,
                repetition_penalty=1.3  # Prevent repetitive "I will provide" loops
            )
            
        generated_tokens = outputs[0, inputs.input_ids.shape[1]:]
        summary = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Strip any thinking tokens from summary output
        summary = self._strip_thinking(summary)
        
        # Prepend the seeded bullet point
        self.structured_summary = "- " + summary if not summary.startswith("-") else summary
        
        # VRAM Cleanup
        del inputs
        del outputs
        torch.cuda.empty_cache()

    # =====================================================================
    # STAGE 2 LOOP (EIG & BACKTRACKING) METHODS
    # =====================================================================

    def inject_eig_question(self, eig_question: str) -> str:
        """
        STAGE 2 HOOK: Brain 2 has paused due to low confidence and high Expected Information Gain.
        It injects the optimal, lowest-cost question back into Brain 1 to ask the patient.
        """
        self.in_eig_loop = True
        self.ready_for_doctor = False # Re-open the chat interface
        
        # The conversational UI wrapper — presented as an assistant message to maintain alternation
        ui_reply = f"I need a bit more information to narrow this down. {eig_question}"
        self.chat_history.append({"role": "assistant", "content": ui_reply})
        self.visible_history.append({"role": "assistant", "content": ui_reply})
        
        # Ensure alternation is intact
        self._ensure_alternating()
        
        return ui_reply

    def ingest_eig_response(self, text_input: str) -> str:
        """
        STAGE 2 HOOK: Captures the patient's answer to the injected EIG question.
        Appends it directly to the structured summary and closes the loop back to Brain 2.
        """
        self.chat_history.append({"role": "user", "content": text_input})
        self.visible_history.append({"role": "user", "content": text_input})
        
        # For maximum edge efficiency, we map the Q&A directly rather than running another generative pass
        last_question = self.visible_history[-2]["content"].replace("I need a bit more information to narrow this down. ", "")
        
        # Format the new fact and append to the locked Context Accumulator
        new_clinical_fact = f"Follow-up Q: {last_question} | Patient A: {text_input}"
        self.structured_summary += f"\n- {new_clinical_fact}"
        
        # Close the loop and send back to Brain 2
        self.in_eig_loop = False
        self.ready_for_doctor = True
        self.is_confirmed = True # Auto-confirm targeted EIG follow-ups to save user friction
        
        ui_reply = "Got it. Updating your clinical profile..."
        self.chat_history.append({"role": "assistant", "content": ui_reply})
        self.visible_history.append({"role": "assistant", "content": ui_reply})
        
        return ui_reply

    def get_structured_summary(self) -> str:
        """Returns the drafted summary for UI display (with routing tag stripped)."""
        import re
        return re.sub(r'\[ROUTING_TAG:\s*\w+\]', '', self.structured_summary).strip()

    def get_routing_tag(self) -> str:
        """Extracts the routing tag from the summary to determine which clinical pathway to load.
        Uses a two-layer approach: explicit tag first, keyword matching as fallback."""
        import re
        
        summary_lower = self.structured_summary.lower()
        
        # Layer 1: Try explicit routing tag
        match = re.search(r'\[ROUTING_TAG:\s*(\w+)\]', summary_lower)
        if match:
            return match.group(1)
        
        # Layer 2: Keyword-based fallback by scanning the summary content
        keyword_map = {
            "chest_pain": ["chest pain", "chest tightness", "chest pressure", "angina"],
            "headache": ["headache", "head pain", "migraine", "head ache"],
            "abdominal_pain": ["abdominal", "stomach pain", "belly pain", "stomach ache"],
            "fever": ["fever", "high temperature", "chills", "feeling hot"],
            "back_pain": ["back pain", "lower back", "spine pain", "backache"],
            "shortness_of_breath": ["shortness of breath", "difficulty breathing", "breathless", "dyspnea", "can't breathe"],
            "cough": ["cough", "coughing"],
        }
        
        for pathway, keywords in keyword_map.items():
            if any(kw in summary_lower for kw in keywords):
                return pathway
        
        return "cough"  # Ultimate default fallback

    def confirm_summary(self) -> bool:
        """
        The Anti-Hallucination check. 
        Pauses execution until the user verifies the summary is accurate.
        """
        if not self.structured_summary:
            return False
        self.is_confirmed = True
        return self.is_confirmed
