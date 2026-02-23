import streamlit as st
import pandas as pd
from llm_engine import MedGemmaEngine
from gatherer import PatientGatherer
from tree_core import ClinicalTree
from pathways import get_pathway

# --- Page Config ---
st.set_page_config(page_title="P-ATHENA Clinical Triage", layout="wide")
st.title("P-ATHENA Clinical Triage Dashboard")

# --- Session State ---
if "phase" not in st.session_state:
    st.session_state.phase = 1 # 1: Patient Intake, 2: Doctor Dashboard
    st.session_state.llm = MedGemmaEngine()
    st.session_state.gatherer = PatientGatherer()
    st.session_state.nodes = None   # Will be loaded dynamically after Brain 1
    st.session_state.tree = None    # Will be created after pathway selection
    st.session_state.raw_symptoms = ""
    st.session_state.verified_summary = ""
    st.session_state.tree_results = []
    st.session_state.tree_active = False
    st.session_state.selected_pathway = None

# --- Helper: Render Graphviz DAG ---
def render_dag(nodes, results):
    dot = "digraph DAG {\n"
    dot += "  node [shape=box, style=filled, color=lightgray];\n"
    
    # Identify taken path and current active node
    taken_nodes = [r["node_id"] for r in results]
    active_node = st.session_state.tree.current_node_id
    
    # Draw nodes
    for nid, node in nodes.items():
        label = node.clinical_question[:30] + "..." if len(node.clinical_question) > 30 else node.clinical_question
        color = "lightgray"
        if nid in taken_nodes:
            color = "lightgreen"
        if nid == active_node and st.session_state.tree_active:
             color = "lightblue"
             
        dot += f'  "{nid}" [label="{nid}\\n{label}", fillcolor="{color}"];\n'
        
        # Draw edges
        for tok, dat in node.token_map.items():
            nn = dat.get("next_node")
            if nn and nn in nodes:
                # Highlight edge if taken
                edge_color = "black"
                for r in results:
                    if r["node_id"] == nid and r.get("winning_data", {}).get("next_node") == nn:
                        edge_color = "green"
                        break
                dot += f'  "{nid}" -> "{nn}" [label="{tok}", color="{edge_color}"];\n'
                
    dot += "}"
    return dot

# =====================================================================
# PHASE 1: Patient Intake (Brain 1)
# =====================================================================
if st.session_state.phase == 1:
    st.header("Phase 1: Patient Data Gatherer")
    
    # --- Real Inference Model Loader ---
    import config
    if not config.USE_MOCK_INFERENCE:
        from brain_loader import MedGemmaLoader
        loader = MedGemmaLoader()
        
        if not loader._is_loaded:
            st.warning("Running in **Real Inference Mode**. MedGemma is loading...")
            with st.spinner("Loading model weights..."):
                loader.load_model(config.HF_TOKEN)
            st.success("Model loaded successfully. Ready for triage.")
            st.rerun()
    
    # Supported Protocols Info
    st.caption("Supported Triage Protocols: **Respiratory (Cough)** | **Headache** | **Chest Pain** — Additional protocols can be added to the pathway registry.")
    
    col_chat, col_summary = st.columns([1.5, 1])
    
    with col_chat:
        st.subheader("Conversational Intake")
        
        # Display chat history dynamically
        for msg in st.session_state.gatherer.visible_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
        # Chat input element loops until ready for doctor
        if not st.session_state.gatherer.ready_for_doctor:
            if prompt := st.chat_input("Type your response here..."):
                st.session_state.gatherer.ingest_patient_input(prompt)
                st.rerun()
            
    with col_summary:
        summary = st.session_state.gatherer.get_structured_summary()
        if st.session_state.gatherer.ready_for_doctor and summary:
            st.subheader("Anti-Hallucination Gate")
            st.info(summary)
            
            # Show the detected routing tag
            routing_tag = st.session_state.gatherer.get_routing_tag()
            pathway_display = routing_tag.replace("_", " ").title()
            st.caption(f"Auto-detected pathway: **{pathway_display}**")
            
            st.warning("Please verify the structured summary above matches your exact symptoms. Do not proceed if inaccurate.")
            if st.button("Confirm and Send to Doctor", use_container_width=True, type="primary"):
                st.session_state.gatherer.confirm_summary()
                
                # Dynamic Pathway Loading based on routing tag
                st.session_state.selected_pathway = routing_tag
                st.session_state.nodes = get_pathway(routing_tag)
                root_node_id = list(st.session_state.nodes.keys())[0]
                
                st.session_state.tree = ClinicalTree(st.session_state.nodes, root_node_id, st.session_state.llm)
                st.session_state.tree.start(summary)
                st.session_state.tree_results = []
                st.session_state.tree_active = True
                st.session_state.phase = 2
                st.rerun()

# ==========================================
# PHASE 2: Doctor Dashboard (Brain 2)
# ==========================================
elif st.session_state.phase == 2:
    pathway_label = (st.session_state.selected_pathway or "unknown").replace("_", " ").title()
    st.header(f"Phase 2: Diagnostic Dashboard - {pathway_label} Pathway")
    
    # --- Check if we are in a PHASE_1_CLARIFY loop ---
    if getattr(st.session_state, "clarify_active", False):
        st.warning(f"**Additional information needed from patient.**")
        st.subheader("Patient Clarification")
        
        # Show the chat with the injected question
        for msg in st.session_state.gatherer.visible_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        if prompt := st.chat_input("Patient response..."):
            st.session_state.gatherer.ingest_eig_response(prompt)
            
            # Update ONLY the tree's context (do NOT call tree.start() — that resets position)
            st.session_state.tree.context = st.session_state.gatherer.get_structured_summary()
            
            # current_node_id is already set to clarify_return_node, so next step() re-evaluates
            st.session_state.clarify_active = False
            st.rerun()
        st.stop()  # Don't render rest of Phase 2 while clarifying
    
    # --- Patient Information ---
    st.subheader("Patient Information")
    summary = st.session_state.gatherer.get_structured_summary()
    st.success("**Verified Patient Summary:**\n" + summary)
    
    st.divider()
    
    # --- Decision History Log ---
    st.markdown("##### Decision History")
    for idx, r in enumerate(st.session_state.tree_results):
        if r["status"] == "PROCEED":
             st.write(f"**{r['node_id']}**: {r['winning_data'].get('label')} (Conf: {max(r['probabilities'].values()):.1%})")
        elif r["status"] == "ROOT_REDIRECT":
             st.error(f"**{r['node_id']}**: Contradiction detected. Triggered redirect.")
        elif r["status"] == "UNCERTAIN_EIG":
             st.warning(f"**{r['node_id']}**: Threshold not met. Triggered EIG fallback.")
        elif r["status"] == "PHASE_1_CLARIFY":
             st.info(f"**{r['node_id']}**: Clarification requested from patient.")
             
    if not st.session_state.tree_active:
         st.info("Pathway terminal reached.")
    
    st.divider()
    
    # --- Clinical Decision Tree ---
    st.subheader("Clinical Decision Tree")
    if st.button("Step Forward (Evaluate Active Node)", use_container_width=True):
         result = st.session_state.tree.step()
         
         # Determine the actual next_node — could be from direct PROCEED or from EIG fallback
         winning_data = result.get("winning_data", {})
         next_node = winning_data.get("next_node", "")
         
         # If UNCERTAIN_EIG, the tree auto-picked the best utility option — check that path too
         if result["status"] == "UNCERTAIN_EIG":
             fallback = result.get("fallback_trigger", {})
             if fallback:
                 next_node = fallback.get("next_node", next_node)
         
         if next_node == "PHASE_1_CLARIFY":
             # Extract the question from the label (after "Ask patient")
             import re
             # Get the label from whatever source triggered clarify
             if result["status"] == "UNCERTAIN_EIG" and result.get("fallback_trigger"):
                 label = result["fallback_trigger"].get("label", "")
             else:
                 label = winning_data.get("label", "")
             
             q_match = re.search(r"Ask patient[:\s]*['\"](.+?)['\"]", label, re.IGNORECASE)
             question = q_match.group(1) if q_match else label
             
             # Rewind: set current_node back to the node that needs re-evaluation
             st.session_state.tree.current_node_id = result["node_id"]
             
             # Inject question back to Phase 1
             st.session_state.gatherer.inject_eig_question(question)
             st.session_state.clarify_active = True
             st.session_state.clarify_return_node = result["node_id"]
             
             result["status"] = "PHASE_1_CLARIFY"
             st.session_state.tree_results.append(result)
             st.rerun()
         else:
             st.session_state.tree_results.append(result)
             if result["status"] in ("END", "ERROR"):
                  st.session_state.tree_active = False
             st.rerun()
         
    dot_code = ""
    try:
        dot_code = render_dag(st.session_state.nodes, st.session_state.tree_results)
        st.graphviz_chart(dot_code, use_container_width=True)
    except Exception as e:
        st.error(f"Graphviz rendering failed: {e}")
        if dot_code:
            st.code(dot_code)

    st.divider()

    # --- CoAI Action Center ---
    st.subheader("CoAI Action Center")
    
    # Display the result of the LAST action taken
    if len(st.session_state.tree_results) > 0:
         last_res = st.session_state.tree_results[-1]
         if last_res["status"] in ("END", "ERROR"):
             st.success("Pathway Complete.")
         else:
             st.markdown(f"**Evaluating Node**: `{last_res['node_id']}`")
             
             # Metrics Row
             met1, met2 = st.columns(2)
             met1.metric("System Entropy", f"{last_res.get('entropy', 0.0):.3f} bits")
             pg = last_res.get('probability_gain', 0.0)
             met2.metric("Probability Gain", f"{pg*100:+.1f}%")
             
             # Edge Case UI Banners
             if last_res["status"] == "UNCERTAIN_EIG":
                 st.warning("**Confidence Threshold Not Met.** Evaluating Feature Utilities.")
                 st.markdown("##### Cost-Utility Recommendations:")
                 utilities = last_res['winning_data'].get('utilities', {})
                 for tok, udata in utilities.items():
                     st.markdown(f"- **Option [{tok}]**: {udata['label']}\n  - Cost: ${udata['cost']:.2f}\n  - EIG: {udata['eig']:.3f} bits\n  - **Utility: {udata['utility']:.3f} bits/$**")
                 
                 best_util_tok = max(utilities, key=lambda k: utilities[k]['utility'])
                 st.success(f"[System Recommendation] Optimal Action: **{utilities[best_util_tok]['label']}**")
                 
             elif last_res["status"] == "ROOT_REDIRECT":
                 st.error("**Contradiction Detected in Path.** High Surprisal Value.")
                 st.write("Tree is automatically breaking current branch and re-routing to Root.")
             else:
                 st.success("Decision Cleared Safety Threshold.")

             # Probability Bars
             st.markdown("##### Token Probabilities (Scaled Softmax)")
             node_obj = st.session_state.nodes.get(last_res['node_id'])
             if node_obj:
                 st.caption(f"Safety Threshold: {node_obj.confidence_threshold*100:.0f}%")
                 
                 for tok, p in last_res['probabilities'].items():
                     label = node_obj.token_map.get(tok, {}).get("label", tok)
                     st.progress(p, text=f"[{tok}] {label} - {p*100:.1f}%")

    st.sidebar.button("Reset Simulation", on_click=lambda: st.session_state.clear())
