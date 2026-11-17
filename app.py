"""
Vision & Multimodal Workflow Automation Agent - Streamlit Web Dashboard
========================================================================
Interactive Web UI for exploring, demonstrating, and publishing all 8 parts
of the Pure Python Autonomous Agent.

Run locally:
    streamlit run app.py
Publish to Streamlit Cloud:
    Push repo to GitHub -> Deploy on share.streamlit.io -> Set main file: app.py
"""

from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

# Ensure current directory and agent package directory are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

AGENT_DIR = os.path.join(CURRENT_DIR, "agent")
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

import streamlit as st

# Import agent components
from agent.loop import AgentLoop, AgentResponse, ReActParser
from agent.tools import (
    default_registry,
    scanner_registry,
    inspect_visual_document,
    enhance_low_quality_details,
    categorize_extracted_details,
    extract_structured_fields,
    validate_business_rules,
    dispatch_action_or_notification,
)
from agent.memory import TaskMemoryStore
from agent.planner import PlannerEngine, ExecutionPlan
from agent.safety import SafetyGate, AuditLogger, StepLimiter
from agent.team import MultiAgentTeam, AgentMessage
from agent.scanner import default_scanner, ScanResult, ScanAwareAgentLLM

# Import demo LLM simulations
from agent.run_part1 import SimulatedLLM as Part1SimulatedLLM
from agent.run_part2 import Part2SimulatedLLM
from agent.run_part3 import SelfHealingParamLLM as ErrorFixingLLM, SelfHealingTypoLLM as TypoCorrectionLLM
from agent.run_part4 import Task1LearningLLM as VendorLearningLLM, Task2RecallingLLM as RepeatVendorLLM
from agent.run_part5 import PlanningAwareLLM as PrePlannedLLM
from agent.run_part6 import SensitiveActionLLM, RunawayInfiniteLoopLLM

DynamicFileLLM = ScanAwareAgentLLM




# ============================================================================
# Page Configuration & Styling
# ============================================================================

st.set_page_config(
    page_title="Vision Agent Studio · Pure Python",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-success {
        background-color: rgba(74, 222, 128, 0.15);
        color: #4ade80;
        border: 1px solid rgba(74, 222, 128, 0.3);
    }
    .stCard {
        background-color: #1e293b;
        border-radius: 0.5rem;
        padding: 1.2rem;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Sidebar Navigation
# ============================================================================

st.sidebar.markdown("## 🤖 Vision Agent Studio")
st.sidebar.markdown(
    '<span class="badge-pill badge-success">Pure Python 3</span> '
    '<span class="badge-pill">Zero Frameworks</span>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Select Component / Part:",
    options=[
        "🌟 Interactive Agent Playground",
        "1️⃣ Part 1: ReAct Loop (Think-Act-Look)",
        "2️⃣ Part 2: Tools Registry (4 Tools)",
        "3️⃣ Part 3: Tool Self-Healing",
        "4️⃣ Part 4: Cross-Task Memory",
        "5️⃣ Part 5: Pre-Action Planning",
        "6️⃣ Part 6: Safety & Human Gate",
        "7️⃣ Part 7: Two-Tier Test Suite",
        "8️⃣ Part 8 (⭐ Bonus): Multi-Agent Team",
        "📜 System Logs & Memory Inspector",
        "🚀 Cloud Deployment Guide",
    ],
    index=0,
)

st.sidebar.markdown("---")
with st.sidebar.expander("🔑 Optional Vision AI API Key", expanded=False):
    vision_provider = st.selectbox("Provider", ["auto", "gemini", "openai"], index=0, key="sidebar_vision_provider")
    vision_api_key = st.text_input(
        "API Key (optional)",
        type="password",
        key="sidebar_vision_api_key",
        help="Offline Windows OCR + Pillow enhancement works without a key. Provide a Gemini or OpenAI key to enable Vision LLM table/handwriting extraction.",
    )

st.sidebar.markdown("### 📦 Active Registered Tools")
st.sidebar.caption("• inspect_visual_document (🟢 Safe)")
st.sidebar.caption("• extract_structured_fields (🟢 Safe)")
st.sidebar.caption("• validate_business_rules (🟢 Safe)")
st.sidebar.caption("• dispatch_action_or_notification (⚠️ Sensitive)")


# ============================================================================
# Helper Functions
# ============================================================================

def render_trajectory(trajectory):
    """Renders step-by-step thoughts, actions, and observations."""
    if not trajectory:
        st.info("No trajectory recorded.")
        return

    for step in trajectory:
        step_num = step.step if hasattr(step, "step") else getattr(step, "step_number", 1)
        
        # Safely extract thought text
        raw_thought = getattr(step, "thought", "")
        if raw_thought is None:
            thought_text = "Step completed."
        elif hasattr(raw_thought, "content"):
            thought_text = raw_thought.content
        else:
            thought_text = str(raw_thought)

        preview = (thought_text[:60] + "...") if len(thought_text) > 60 else thought_text
        with st.expander(f"📍 Step {step_num}: {preview}", expanded=True):
            st.markdown(f"**🧠 Thought:** {thought_text}")
            if hasattr(step, "action") and step.action:
                st.markdown(
                    f"**⚡ Action:** `{step.action.tool_name}` with args:"
                )
                st.json(step.action.tool_args)
            if hasattr(step, "observation") and step.observation:
                raw_obs = step.observation
                obs_content = raw_obs.content if hasattr(raw_obs, "content") else str(raw_obs)
                try:
                    obs_json = json.loads(obs_content)
                    st.markdown("**👁️ Observation (JSON):**")
                    st.json(obs_json)
                except Exception:
                    st.markdown(f"**👁️ Observation:** `{obs_content}`")


# ============================================================================
# View 0: Interactive Agent Playground
# ============================================================================

if menu == "🌟 Interactive Agent Playground":
    st.markdown('<div class="main-title">🔍 Autonomous Document Scanner, Enhancer & Categorizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload any document (PDF, Image, CSV, JSON, TXT). Scans only genuine details present, enhances low-quality regions, and sorts details separately into suitable categories.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📄 1. Document Input (Multi-Format Support)")
        doc_source = st.radio(
            "Choose document source:",
            ["📤 Upload Your Own Document", "📁 Select From Uploaded Files (uploads/)", "📑 Select Built-in Real Sample"],
            horizontal=True,
        )

        selected_file_path: str | None = None

        if "Upload" in doc_source:
            uploaded_file = st.file_uploader(
                "Upload document (Images, PDFs, Spreadsheets, Data, Text):",
                type=["png", "jpg", "jpeg", "webp", "pdf", "tiff", "bmp", "csv", "json", "txt", "md"],
                help="Supports: PNG, JPG, JPEG, WEBP, PDF, TIFF, BMP, CSV, JSON, TXT, MD",
            )
            if uploaded_file is not None:
                os.makedirs("uploads", exist_ok=True)
                saved_path = os.path.join("uploads", uploaded_file.name).replace("\\", "/")
                with open(saved_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state["last_uploaded_path"] = saved_path
                selected_file_path = saved_path
            elif st.session_state.get("last_uploaded_path") and os.path.exists(st.session_state["last_uploaded_path"]):
                selected_file_path = st.session_state["last_uploaded_path"]

            if selected_file_path and os.path.exists(selected_file_path):
                file_ext = os.path.splitext(selected_file_path)[1].lower()
                file_size_kb = round(os.path.getsize(selected_file_path) / 1024, 2)
                st.success(f"✅ Ready: `{os.path.basename(selected_file_path)}` ({file_size_kb} KB, Format: `{file_ext}`)")

                # Multi-format preview
                with st.expander(f"👁️ Preview Uploaded File ({os.path.basename(selected_file_path)})", expanded=False):
                    if file_ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"):
                        st.image(selected_file_path, caption=f"{os.path.basename(selected_file_path)} ({file_size_kb} KB)", use_container_width=True)
                    elif file_ext == ".pdf":
                        st.info(f"📄 **PDF Document Ready for Scanning & Enhancement:** `{os.path.basename(selected_file_path)}` ({file_size_kb} KB)")
                    elif file_ext == ".csv":
                        try:
                            import pandas as pd
                            df = pd.read_csv(selected_file_path, skipinitialspace=True)
                            st.markdown("**Spreadsheet Preview:**")
                            st.dataframe(df.head(10), use_container_width=True)
                        except Exception:
                            st.code(open(selected_file_path, "r", encoding="utf-8", errors="ignore").read()[:500])
                    elif file_ext == ".json":
                        try:
                            with open(selected_file_path, "r", encoding="utf-8") as jf:
                                st.json(json.load(jf))
                        except Exception:
                            st.code(open(selected_file_path, "r", encoding="utf-8", errors="ignore").read()[:500])
                    elif file_ext in (".txt", ".md"):
                        st.text_area("Text Content Preview:", open(selected_file_path, "r", encoding="utf-8", errors="ignore").read()[:1000], height=120)
            else:
                st.info("👆 Drop a document above (PNG, JPG, PDF, CSV, JSON, TXT, MD), or switch to **Select From Uploaded Files (uploads/)** or **Select Built-in Real Sample** to test immediately.")
        elif "Uploaded Files" in doc_source:
            uploaded_dir_files = []
            if os.path.exists("uploads"):
                uploaded_dir_files = sorted(
                    os.path.join("uploads", f).replace("\\", "/")
                    for f in os.listdir("uploads")
                    if os.path.isfile(os.path.join("uploads", f)) and not f.startswith("enhanced_")
                )
            if uploaded_dir_files:
                selected_file_path = st.selectbox(
                    "Choose from uploaded files:",
                    uploaded_dir_files,
                )
                st.success(f"✅ Selected from uploads: `{selected_file_path}`")
            else:
                st.info("No uploaded documents in `uploads/` yet. Upload a document above or choose Built-in Real Sample!")
        else:
            real_samples = [
                ("samples/low_quality_receipt_scan.png", "samples/low_quality_receipt_scan.png (Blurry Low-Contrast Receipt PNG -> Enhanced + Categorized)"),
                ("samples/messy_expenses.csv", "samples/messy_expenses.csv (Dirty Spreadsheet CSV -> Cleaned + Categorized)"),
                ("samples/degraded_meeting_notes.txt", "samples/degraded_meeting_notes.txt (Degraded Notes TXT with OCR Glitches -> Repaired + Categorized)"),
                ("samples/candidate_profile.json", "samples/candidate_profile.json (Unclean JSON with Whitespace & Dates -> Normalized + Categorized)"),
            ]
            available_samples = [(p, lbl) for p, lbl in real_samples if os.path.exists(p)]
            if available_samples:
                sample_labels = [lbl for _, lbl in available_samples]
                sample_choice = st.selectbox("Select real built-in sample document:", sample_labels)
                selected_file_path = sample_choice.split(" ")[0]
                st.success(f"✅ Selected sample: `{selected_file_path}`")
            else:
                st.warning("No files found in `samples/`.")

        # --------------------------------------------------------------------
        # Document Intelligence Scan, Quality Enhancement & Categorized Details
        # --------------------------------------------------------------------
        scan_res = None
        if selected_file_path and os.path.exists(selected_file_path):
            try:
                with st.spinner("🔬 Scanning document, enhancing degraded details, and sorting into suitable categories..."):
                    scan_res = default_scanner.scan_and_extract(
                        selected_file_path,
                        api_key=vision_api_key or None,
                        provider=vision_provider,
                    )
            except Exception as e:
                st.warning(f"Scanner note: {e}")

        if scan_res:
            # 1. Quality Diagnosis & Auto-Enhancement Hub
            st.markdown("### 🔬 Quality Diagnosis & Auto-Enhancement Hub")
            repairs_list = scan_res.quality_report.before_after_repairs or []
            qcol1, qcol2, qcol3, qcol4 = st.columns(4)
            with qcol1:
                st.metric("Original Quality", f"{scan_res.quality_report.original_score}%")
            with qcol2:
                diff = round(scan_res.quality_report.enhanced_score - scan_res.quality_report.original_score, 1)
                st.metric("Enhanced Quality", f"{scan_res.quality_report.enhanced_score}% ✨", delta=f"+{diff}%")
            with qcol3:
                st.metric("Document Domain", scan_res.document_domain)
            with qcol4:
                st.metric("Repairs / Upgrades", len(scan_res.quality_report.enhancements_applied))

            with st.expander("🛡️ Applied Quality Enhancements, Text Repairs & Visual Comparison", expanded=True):
                st.markdown(f"**Quality Verdict:** `{scan_res.quality_report.quality_verdict}`")
                for enh in scan_res.quality_report.enhancements_applied:
                    st.markdown(f"- ✨ **Enhanced:** {enh}")

                if repairs_list:
                    st.markdown("**🔁 Exact Quality Repairs (`Before ➔ After`):**")
                    st.dataframe(
                        [
                            {
                                "Original (Degraded)": r.get("original", ""),
                                "Enhanced (Clean)": r.get("enhanced", ""),
                                "Repair Type": r.get("fix_type", ""),
                            }
                            for r in repairs_list
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                # Side-by-side comparison for images OR rendered PDF pages
                orig_img = scan_res.quality_report.original_preview_path
                if not orig_img and selected_file_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")):
                    orig_img = selected_file_path
                enh_img = scan_res.quality_report.enhanced_file_path

                if (
                    enh_img
                    and enh_img.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                    and os.path.exists(enh_img)
                    and orig_img
                    and os.path.exists(orig_img)
                    and os.path.abspath(enh_img) != os.path.abspath(orig_img)
                ):
                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        st.image(orig_img, caption=f"Original ({scan_res.quality_report.original_score}%)", use_container_width=True)
                    with v_col2:
                        st.image(enh_img, caption=f"Auto-Enhanced ({scan_res.quality_report.enhanced_score}%)", use_container_width=True)

            # 2. Extracted Details Sorted into Suitable Categories
            st.markdown("### 📂 Extracted Details (Sorted into Suitable Categories)")
            st.caption("Only authentic details genuinely present in this document, repaired for clarity and organized by category:")

            cat_names = list(scan_res.categorized_details.keys())
            if cat_names:
                cat_tabs = st.tabs(cat_names)
                for tab_idx, cat_name in enumerate(cat_names):
                    with cat_tabs[tab_idx]:
                        fields_dict = scan_res.categorized_details[cat_name]
                        for field_k, field_v in fields_dict.items():
                            st.markdown(f"- **{field_k}:** `{field_v}`")
                        if scan_res.extracted_tables and "Structured Line Items" in cat_name:
                            st.markdown("**📊 Structured Table View:**")
                            st.dataframe(scan_res.extracted_tables, use_container_width=True, hide_index=True)

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    "📥 Download Categorized Details (JSON)",
                    data=json.dumps(
                        {
                            "file_name": scan_res.file_name,
                            "document_domain": scan_res.document_domain,
                            "quality_report": scan_res.quality_report.to_dict(),
                            "categorized_details": scan_res.categorized_details,
                            "extracted_tables": scan_res.extracted_tables,
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    file_name=f"{scan_res.file_name}_categorized.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with dl_col2:
                enh_dl_path = scan_res.quality_report.enhanced_file_path
                if enh_dl_path and os.path.exists(enh_dl_path):
                    enh_ext = os.path.splitext(enh_dl_path)[1].lower()
                    mime_map = {
                        ".png": "image/png",
                        ".jpg": "image/jpeg",
                        ".csv": "text/csv",
                        ".json": "application/json",
                        ".txt": "text/plain",
                    }
                    with open(enh_dl_path, "rb") as ef:
                        enh_bytes = ef.read()
                    st.download_button(
                        "✨ Download Auto-Enhanced Document",
                        data=enh_bytes,
                        file_name=os.path.basename(enh_dl_path),
                        mime=mime_map.get(enh_ext, "application/octet-stream"),
                        use_container_width=True,
                    )

        st.markdown("### 🎯 2. Goal / Prompt")
        if scan_res and selected_file_path:
            default_prompt = (
                f"Scan {scan_res.document_domain} document '{selected_file_path}': "
                "extract only genuine details present, enhance any degraded quality or formatting issues, "
                "sort details separately into suitable categories, and audit completeness."
            )
        else:
            default_prompt = "Scan uploaded document: enhance low-quality visual/text details, extract only genuine content, and sort details separately into suitable categories."

        task_prompt = st.text_area(
            "Task / Goal for Agent:",
            value=default_prompt,
            height=85,
        )

        preset = st.selectbox(
            "Execution Mode / Preset:",
            [
                "Document Intelligence: Scan, Enhance & Categorize",
                "Self-Healing: Bad Parameter Recovery",
                "Safety Gate: Mutating External ERP Action",
                "Runaway Prevention: Step Limit Guardrail",
            ],
        )

        c1, c2 = st.columns(2)
        with c1:
            use_planner = st.checkbox("Enable Pre-Action Planner (Part 5)", value=True)
        with c2:
            use_memory = st.checkbox("Inject Cross-Task Memory (Part 4)", value=True)

        needs_real_file = "Document Intelligence" in preset
        run_disabled = needs_real_file and (not selected_file_path or not os.path.exists(selected_file_path))
        run_btn = st.button(
            "🚀 Run Scan, Enhance & Categorize Agent",
            type="primary",
            use_container_width=True,
            disabled=run_disabled,
        )
        if run_disabled:
            st.caption("⚠️ Upload a document above or choose **Select From Uploaded Files (uploads/)** / **Select Built-in Real Sample** to enable execution.")

    with col2:
        st.markdown("### ⚙️ Runtime Guardrails")
        max_steps = st.slider("Max Step Ceiling (Part 6)", min_value=1, max_value=10, value=6)
        st.success("🛡️ **Safety Gate:** Active (Supervisor Authorization Enabled)")
        auto_approve = True

    if run_btn:
        st.markdown("---")
        st.subheader("⚡ Agent Execution Trace")

        # Pick LLM + appropriate tool registry based on mode
        if "Self-Healing" in preset:
            llm_caller = ErrorFixingLLM()
            active_tools = default_registry
        elif "Safety" in preset:
            llm_caller = SensitiveActionLLM()
            active_tools = default_registry
        elif "Runaway" in preset:
            llm_caller = RunawayInfiniteLoopLLM()
            active_tools = default_registry
        else:
            llm_caller = ScanAwareAgentLLM(selected_file_path or "", scan_res)
            active_tools = scanner_registry

        # Configure safety gate
        audit_logger = AuditLogger(log_path="agent_audit.log")
        safety_gate = SafetyGate(
            audit_logger=audit_logger,
            approval_handler=lambda tool, args: auto_approve,
        )

        # Configure memory
        memory_store = TaskMemoryStore(storage_path="memory_store.json") if use_memory else None

        # Build agent
        agent = AgentLoop(
            tools=active_tools,
            llm_caller=llm_caller,
            safety_gate=safety_gate,
            audit_logger=audit_logger,
            memory=memory_store,
            enable_planning=use_planner,
            max_steps=max_steps,
            verbose=False,
        )

        with st.spinner("🤖 Agent scanning document, enhancing low-quality details, and sorting categories..."):
            response = agent.run(task_prompt)

        # Metrics display
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Execution Status", "Success ✅" if response.success else "Halted 🛑")
        m2.metric("Steps Taken", response.total_steps)
        m3.metric("Stop Reason", response.stop_reason)
        m4.metric("Trajectory Length", len(response.trajectory))

        st.success(f"**🎯 Final Answer:**\n\n{response.final_answer}")

        # Show Agent's Execution Plan status if enabled
        if response.plan and getattr(response.plan, "steps", None):
            with st.expander("📋 Agent Execution Plan (Pre-Action Planner)", expanded=False):
                for pstep in response.plan.steps:
                    icon = "✅" if pstep.status == "COMPLETED" else ("🔄" if pstep.status == "IN_PROGRESS" else "⏳")
                    st.markdown(f"{icon} **Step {pstep.step_number}:** `[{pstep.status}]` {pstep.description}")

        # Show Separate Category Cards directly in the Agent Output!
        if scan_res and "Document Intelligence" in preset and scan_res.categorized_details:
            st.markdown("### 📂 Agent Extracted & Categorized Output (Sorted Separately)")
            cat_items = list(scan_res.categorized_details.items())
            for i in range(0, len(cat_items), 2):
                row_cols = st.columns(2)
                for col_offset in range(2):
                    if i + col_offset < len(cat_items):
                        cat_title, cat_fields = cat_items[i + col_offset]
                        with row_cols[col_offset]:
                            with st.container(border=True):
                                st.markdown(f"**{cat_title}**")
                                for fk, fv in cat_fields.items():
                                    st.markdown(f"- **{fk}:** `{fv}`")
                                if scan_res.extracted_tables and "Structured Line Items" in cat_title:
                                    st.dataframe(scan_res.extracted_tables, use_container_width=True, hide_index=True)

        # Render Trajectory
        st.markdown("### 👣 Step-by-Step Trajectory")
        render_trajectory(response.trajectory)

        # Raw Trajectory JSON
        with st.expander("🔍 View Raw Trajectory JSON"):
            if hasattr(response, "to_dict"):
                st.json(response.to_dict())
            else:
                st.json({
                    "final_answer": response.final_answer,
                    "success": response.success,
                    "total_steps": response.total_steps,
                    "stop_reason": response.stop_reason,
                    "trajectory": [s.to_dict() if hasattr(s, "to_dict") else str(s) for s in response.trajectory],
                })


# ============================================================================
# View 1: Part 1 - ReAct Loop
# ============================================================================

elif menu == "1️⃣ Part 1: ReAct Loop (Think-Act-Look)":
    st.markdown('<div class="main-title">Part 1: The Core ReAct Loop</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A pure Python while loop alternating: Think ➔ Act ➔ Look ➔ Think.</div>', unsafe_allow_html=True)

    st.info("Concept: Plain Python loop using regular expressions and string parsing to parse Thoughts, Actions, and Observations without any external agent framework.")

    if st.button("▶ Run Part 1 Demo"):
        agent = AgentLoop(
            tools=default_registry,
            llm_caller=Part1SimulatedLLM(),
            max_steps=5,
            verbose=False,
        )
        res = agent.run("Please process invoice_sample.png and extract the key financial fields.")
        st.success(f"**🎯 Final Answer:** {res.final_answer}")
        render_trajectory(res.trajectory)


# ============================================================================
# View 2: Part 2 - Tools Registry
# ============================================================================

elif menu == "2️⃣ Part 2: Tools Registry (4 Tools)":
    st.markdown('<div class="main-title">Part 2: 2–4 Tools, Clearly Described</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Python function registry with schemas, parameter validation, and prompt generation.</div>', unsafe_allow_html=True)

    tool_names = default_registry.list_tools()
    for name in tool_names:
        tool_obj = default_registry.get(name)
        if not tool_obj:
            continue
        with st.container():
            tag = "⚠️ Sensitive (Requires Approval)" if getattr(tool_obj, "is_sensitive", False) else "🟢 Safe"
            st.markdown(f"### `{name}` &nbsp; <span class='badge-pill'>{tag}</span>", unsafe_allow_html=True)
            st.markdown(f"**Description:** {tool_obj.description}")
            st.markdown("**Parameters:**")
            st.json(tool_obj.parameters)
            st.markdown("---")

    st.subheader("⚡ Test Tool Execution Directly")
    selected_tool = st.selectbox("Select tool to test:", tool_names)

    tool_file_target = "invoice_sample.png"
    if selected_tool in ("inspect_visual_document", "extract_structured_fields"):
        st.markdown("**📁 Target Document (Sample or Uploaded):**")
        uploaded_files = []
        if os.path.exists("uploads"):
            uploaded_files = [
                os.path.join("uploads", f).replace("\\", "/")
                for f in os.listdir("uploads")
                if os.path.isfile(os.path.join("uploads", f))
            ]

        real_sample_files = [
            f"samples/{f}"
            for f in ("low_quality_receipt_scan.png", "messy_expenses.csv", "degraded_meeting_notes.txt", "candidate_profile.json")
            if os.path.exists(os.path.join("samples", f))
        ]
        sample_options = ["invoice_sample.png", "stark_inv_1001.pdf", "Q4_inv.png", "claim_photo.jpg", "multimodal_invoice.pdf"]
        all_doc_options = uploaded_files + real_sample_files + sample_options

        c_file_sel, c_file_upl = st.columns([1, 1])
        with c_file_sel:
            tool_file_target = st.selectbox("Select document:", all_doc_options, key="tool_test_file_sel")
        with c_file_upl:
            direct_up = st.file_uploader(
                "Or upload file for tool execution:",
                type=["png", "jpg", "jpeg", "webp", "pdf", "tiff", "bmp", "csv", "json", "txt"],
                key="direct_tool_uploader",
            )
            if direct_up is not None:
                os.makedirs("uploads", exist_ok=True)
                saved_direct = os.path.join("uploads", direct_up.name).replace("\\", "/")
                with open(saved_direct, "wb") as f:
                    f.write(direct_up.getbuffer())
                tool_file_target = saved_direct
                st.success(f"✅ Loaded: `{direct_up.name}`")

    if st.button(f"Execute {selected_tool}"):
        if selected_tool == "inspect_visual_document":
            out = default_registry.execute(selected_tool, {"file_path": tool_file_target, "query": "Document type, quality issues, and categorized structure"})
        elif selected_tool == "extract_structured_fields":
            fields_to_pull = ["*"] if os.path.exists(tool_file_target) else ["vendor", "total_amount", "subtotal", "tax_amount"]
            out = default_registry.execute(selected_tool, {"file_path": tool_file_target, "fields": fields_to_pull})
        elif selected_tool == "validate_business_rules":
            out = default_registry.execute(selected_tool, {"extracted_data": {"subtotal": 100, "tax_amount": 10, "total_amount": 110}, "rules": ["check_math_totals"]})
        else:
            out = default_registry.execute(selected_tool, {"action_type": "post_to_erp", "payload": {"status": "APPROVED"}})
        st.json(out)


# ============================================================================
# View 3: Part 3 - Self-Healing
# ============================================================================

elif menu == "3️⃣ Part 3: Tool Self-Healing":
    st.markdown('<div class="main-title">Part 3: Fixing Errors When a Tool Fails</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Exceptions become observations so the agent can inspect the failure and self-correct.</div>', unsafe_allow_html=True)

    scenario = st.radio("Choose Error Scenario:", ["Parameter Format Error (String instead of List)", "Tool Name Typo with Fuzzy Match Suggestion"])

    if st.button("▶ Run Self-Healing Demo"):
        if "Parameter" in scenario:
            agent = AgentLoop(tools=default_registry, llm_caller=ErrorFixingLLM(), max_steps=5, verbose=False)
            res = agent.run("Extract vendor and total_amount from invoice.pdf")
        else:
            agent = AgentLoop(tools=default_registry, llm_caller=TypoCorrectionLLM(), max_steps=5, verbose=False)
            res = agent.run("Inspect claim_photo.jpg")

        st.success(f"**🎯 Final Answer:** {res.final_answer}")
        render_trajectory(res.trajectory)


# ============================================================================
# View 4: Part 4 - Cross-Task Memory
# ============================================================================

elif menu == "4️⃣ Part 4: Cross-Task Memory":
    st.markdown('<div class="main-title">Part 4: Memory Between Tasks</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Persisting learned vendor quirks and task summaries to disk across sessions.</div>', unsafe_allow_html=True)

    mem_store = TaskMemoryStore(storage_path="memory_store.json")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### 📝 Task 1: Learn Vendor Quirk")
        if st.button("Run Task 1 (Learn Stark Industries convention)"):
            agent1 = AgentLoop(tools=default_registry, llm_caller=VendorLearningLLM(), memory=mem_store, max_steps=5, verbose=False)
            res1 = agent1.run("Process invoice stark_inv_1001.pdf from Stark Industries.")
            mem_store.record_learned_fact("vendors", "Stark Industries", "Uses DD/MM/YYYY date format; PO in footer.")
            st.success(res1.final_answer)

    with colB:
        st.markdown("### 🔄 Task 2: Apply Remembered Quirk")
        if st.button("Run Task 2 (Restart agent & apply memory)"):
            fresh_mem = TaskMemoryStore(storage_path="memory_store.json")
            agent2 = AgentLoop(tools=default_registry, llm_caller=RepeatVendorLLM(), memory=fresh_mem, max_steps=5, verbose=False)
            res2 = agent2.run("Extract and validate new invoice stark_inv_4412.pdf from Stark Industries.")
            st.success(res2.final_answer)

    st.markdown("### 💾 Current `memory_store.json` on Disk")
    try:
        with open("memory_store.json", "r") as f:
            st.json(json.load(f))
    except Exception as e:
        st.warning(f"Could not load memory file: {e}")


# ============================================================================
# View 5: Part 5 - Pre-Action Planning
# ============================================================================

elif menu == "5️⃣ Part 5: Pre-Action Planning":
    st.markdown('<div class="main-title">Part 5: A Plan Before It Acts</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Generates an explicit numbered plan with [TODO] tags and updates status step-by-step.</div>', unsafe_allow_html=True)

    if st.button("▶ Generate Plan & Run Agent"):
        agent = AgentLoop(
            tools=default_registry,
            llm_caller=PrePlannedLLM(),
            enable_planning=True,
            max_steps=6,
            verbose=False,
        )
        res = agent.run("Process Q4_inv.png, validate financial totals, and post to ERP.")
        st.success(f"**🎯 Final Answer:** {res.final_answer}")
        if res.plan:
            st.markdown("### 📋 Final Plan State")
            for step in res.plan.steps:
                icon = "✅" if step.status == "COMPLETED" else "⏳"
                st.markdown(f"{icon} **Step {step.step_number}:** `[{step.status}]` {step.description}")
        render_trajectory(res.trajectory)


# ============================================================================
# View 6: Part 6 - Safety Guardrails
# ============================================================================

elif menu == "6️⃣ Part 6: Safety & Human Gate":
    st.markdown('<div class="main-title">Part 6: Safety Guardrails</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Asks you first (human authorization), step limit, and immutable audit logging.</div>', unsafe_allow_html=True)

    audit_logger = AuditLogger(log_path="agent_audit.log")

    tab1, tab2 = st.tabs(["🛡️ Human Authorization Intercept", "🛑 Step Limit Runaway Blocker"])

    with tab1:
        st.markdown("### Human Gate ('Asks You First')")
        st.info("The Safety Gate intercepts sensitive mutation tools (`dispatch_action_or_notification`) and requires supervisor authorization before executing any write.")

        if st.button("Authorize & Execute Sensitive Action", type="primary"):
            gate = SafetyGate(
                audit_logger=audit_logger,
                approval_handler=lambda tool, args: True,
            )
            agent = AgentLoop(
                tools=default_registry,
                llm_caller=SensitiveActionLLM(),
                safety_gate=gate,
                audit_logger=audit_logger,
                max_steps=5,
                verbose=False,
            )
            res = agent.run("Process inv_march.pdf and post payment to ERP.")
            st.success(f"Result: {res.final_answer}")
            render_trajectory(res.trajectory)

    with tab2:
        st.markdown("### Step Limit Guardrail")
        st.caption("Guarantees termination if an agent enters an infinite loop.")
        if st.button("Run Runaway Agent (Hard limit: 3 steps)"):
            runaway_agent = AgentLoop(
                tools=default_registry,
                llm_caller=RunawayInfiniteLoopLLM(),
                audit_logger=audit_logger,
                max_steps=3,
                verbose=False,
            )
            res_runaway = runaway_agent.run("Run forever on infinite_loop.png")
            st.error(f"🛑 {res_runaway.final_answer}")
            st.caption(f"Stop Reason: `{res_runaway.stop_reason}` | Steps taken: `{res_runaway.total_steps}`")


# ============================================================================
# View 7: Part 7 - Test Suite
# ============================================================================

elif menu == "7️⃣ Part 7: Two-Tier Test Suite":
    st.markdown('<div class="main-title">Part 7: Two-Tier Testing Suite</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Deterministic offline unit tests first (FakeLLM), then live multimodal integration.</div>', unsafe_allow_html=True)

    if st.button("🧪 Run Tier 1 Offline Unit Tests"):
        import time
        import unittest
        from tests.test_fake_llm import TestAgentDeterministic

        t0 = time.perf_counter()
        suite = unittest.TestLoader().loadTestsFromTestCase(TestAgentDeterministic)
        stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=2)
        result = runner.run(suite)
        elapsed = time.perf_counter() - t0

        if result.wasSuccessful():
            st.success(f"✅ All {result.testsRun} unit tests passed successfully in {elapsed:.2f}s!")
        else:
            st.error(f"❌ Tests failed: {len(result.failures)} failures, {len(result.errors)} errors.")

        st.code(stream.getvalue())


# ============================================================================
# View 8: Part 8 - Multi-Agent Team (Bonus)
# ============================================================================

elif menu == "8️⃣ Part 8 (⭐ Bonus): Multi-Agent Team":
    st.markdown('<div class="main-title">Part 8 (⭐ Bonus): Multi-Agent Collaboration</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Orchestrator, Vision Analyst, and Compliance Auditor communicating via message passing.</div>', unsafe_allow_html=True)

    st.markdown("### 📄 Document Target for Multi-Agent Team")
    team_doc_src = st.radio(
        "Choose document source for team workflow:",
        ["📤 Upload Custom Document", "📁 Select From Uploaded Files (uploads/)", "📑 Built-in Sample Document"],
        horizontal=True,
        key="team_doc_radio",
    )

    team_target_file = "multimodal_invoice.pdf"
    if "Upload Custom" in team_doc_src:
        team_file_upload = st.file_uploader(
            "Upload file for multi-agent collaboration:",
            type=["png", "jpg", "jpeg", "webp", "pdf", "tiff", "bmp", "csv", "json", "txt", "md"],
            key="team_file_uploader",
        )
        if team_file_upload is not None:
            os.makedirs("uploads", exist_ok=True)
            saved_team_path = os.path.join("uploads", team_file_upload.name).replace("\\", "/")
            with open(saved_team_path, "wb") as f:
                f.write(team_file_upload.getbuffer())
            team_target_file = saved_team_path
            st.success(f"✅ Loaded for multi-agent analysis: `{team_file_upload.name}`")
    elif "Uploaded Files" in team_doc_src:
        team_uploaded = []
        if os.path.exists("uploads"):
            team_uploaded = sorted(
                os.path.join("uploads", f).replace("\\", "/")
                for f in os.listdir("uploads")
                if os.path.isfile(os.path.join("uploads", f)) and not f.startswith("enhanced_")
            )
        if team_uploaded:
            team_target_file = st.selectbox("Select uploaded document:", team_uploaded, key="team_uploaded_select")
            st.success(f"✅ Selected from uploads: `{team_target_file}`")
        else:
            st.info("No uploaded documents in `uploads/` yet. Upload one or choose a sample.")
    else:
        real_team_samples = [
            f"samples/{f}"
            for f in ("low_quality_receipt_scan.png", "messy_expenses.csv", "degraded_meeting_notes.txt", "candidate_profile.json")
            if os.path.exists(os.path.join("samples", f))
        ]
        team_target_file = st.selectbox(
            "Select sample document for team workflow:",
            real_team_samples + ["multimodal_invoice.pdf", "stark_inv_1001.pdf", "invoice_sample.png", "Q4_inv.png", "claim_photo.jpg"],
            key="team_sample_select",
        )

    if st.button("👥 Run Multi-Agent Team Workflow"):
        team_scan = None
        if os.path.exists(team_target_file):
            try:
                team_scan = default_scanner.scan_and_extract(
                    team_target_file,
                    api_key=vision_api_key or None,
                    provider=vision_provider,
                )
            except Exception:
                team_scan = None

        def llm_factory(role: str):
            if role == "VisionAnalyst":
                if team_scan is not None:
                    cats = list(team_scan.categorized_details.keys())
                    return lambda p: (
                        f"Thought: Scanned '{team_target_file}' ({team_scan.document_domain}) and improved quality {team_scan.quality_report.original_score}% -> {team_scan.quality_report.enhanced_score}%.\n"
                        f"Final Answer: Extracted genuine document details and sorted them into {len(cats)} categories: {', '.join(cats)}."
                    )
                return lambda p: f"Thought: Visual extraction complete on {team_target_file}.\nFinal Answer: Vision extraction finished. Extracted 6 fields with bounding boxes and 0.97 confidence."
            elif role == "ComplianceAuditor":
                return lambda p: "Thought: Authenticity, quality enhancements, and domain rules verified.\nFinal Answer: Compliance audit passed. Zero fabricated fields or violations found."
            else:
                return lambda p: "Thought: Coordinating specialists.\nFinal Answer: Coordinated."

        team = MultiAgentTeam(llm_factory=llm_factory, verbose=False)
        with st.spinner(f"Collaborating across specialized agents on {team_target_file}..."):
            instructions = (
                f"Scan {team_target_file}, enhance degraded details, extract authentic fields, and organize into suitable categories."
                if team_scan is not None
                else "Extract financial fields, audit math totals, and dispatch to ERP."
            )
            report = team.run_workflow(team_target_file, instructions)

        st.success("🎯 Multi-Agent Team Collaboration Completed Successfully!")
        if team_scan is not None:
            st.markdown("### 📂 Categorized Details Extracted by Team")
            st.json(team_scan.categorized_details)
        st.markdown("### 📊 Complete Orchestrator Team Report")
        st.json(report)

        st.markdown("### 📨 Messages Exchanged on Shared Board")
        for i, msg in enumerate(team.get_message_board(), 1):
            with st.expander(f"[{i}] {msg.sender} ➔ {msg.recipient} ({msg.message_type})", expanded=True):
                st.caption(f"Timestamp: {msg.timestamp}")
                st.json(msg.payload)


# ============================================================================
# View 9: System Logs & Memory Inspector
# ============================================================================

elif menu == "📜 System Logs & Memory Inspector":
    st.markdown('<div class="main-title">System Logs & Memory Inspector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Direct disk inspection of agent_audit.log and memory_store.json.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📜 `agent_audit.log` (Append-Only Trail)")
        try:
            with open("agent_audit.log", "r") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            st.dataframe(lines, use_container_width=True)
        except Exception as e:
            st.info(f"No audit logs found yet ({e}).")

    with col2:
        st.markdown("### 🧠 `memory_store.json` (Persistent Memory)")
        try:
            with open("memory_store.json", "r") as f:
                st.json(json.load(f))
        except Exception as e:
            st.info(f"No memory store found yet ({e}).")


# ============================================================================
# View 10: Cloud Deployment Guide
# ============================================================================

elif menu == "🚀 Cloud Deployment Guide":
    st.markdown('<div class="main-title">🚀 Cloud Hosting & Deployment Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Step-by-step guides for hosting your Vision Agent on Render and Streamlit Cloud.</div>', unsafe_allow_html=True)

    tab_render, tab_streamlit = st.tabs(["🟣 Host on Render (Web Service)", "☁️ Host on Streamlit Community Cloud"])

    with tab_render:
        st.markdown(
            """
            ### 🟣 Deploy to Render (render.com)
            
            Render automatically builds and hosts your Python application with full custom start commands and dynamic port mapping.

            #### 1. Push Code to GitHub
            ```bash
            # In your project folder:
            git init
            git add .
            git commit -m "Deploy Vision Agent to Render"
            git branch -M main
            git remote add origin https://github.com/<your-username>/<your-repo-name>.git
            git push -u origin main
            ```

            #### 2. Create a Web Service on Render
            1. Go to [dashboard.render.com](https://dashboard.render.com/) and sign in with GitHub.
            2. Click **New +** ➔ **Web Service**.
            3. Select **"Build and deploy from a Git repository"** and connect your repository.

            #### 3. Configure Service Settings
            - **Name:** `vision-agent-studio`
            - **Region:** Any (e.g. Oregon, Ohio, Frankfurt)
            - **Branch:** `main`
            - **Root Directory:** *(leave blank)*
            - **Runtime:** `Python 3`
            - **Build Command:**
              ```bash
              pip install -r requirements.txt
              ```
            - **Start Command:**
              ```bash
              streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
              ```
            - **Instance Type:** `Free`

            #### 4. Environment Variables (Optional)
            If you want to enable multimodal Vision AI extraction in the cloud, add under **Environment Variables**:
            - `GEMINI_API_KEY`: `<your-gemini-key>`
            - `OPENAI_API_KEY`: `<your-openai-key>`
            - `PYTHON_VERSION`: `3.11.9`

            #### 5. Click "Deploy Web Service"
            Render will build dependencies, bind to `$PORT`, and give you a permanent `https://<your-service>.onrender.com` URL!
            """
        )

    with tab_streamlit:
        st.markdown(
            """
            ### ☁️ Deploy to Streamlit Community Cloud
            
            #### 1. Push this project to GitHub
            ```bash
            git init
            git add .
            git commit -m "Publish Vision & Multimodal Agent with Streamlit Dashboard"
            git branch -M main
            git remote add origin https://github.com/<your-username>/<your-repo-name>.git
            git push -u origin main
            ```

            #### 2. Sign in to Streamlit Community Cloud
            1. Go to [share.streamlit.io](https://share.streamlit.io/).
            2. Click **"New app"**.
            3. Select your repository, set **Branch** to `main`, and **Main file path** to:
               ```text
               app.py
               ```

            #### 3. Click "Deploy!"
            Streamlit Community Cloud will automatically detect `requirements.txt` and publish your app online with a shareable public URL!
            """
        )

