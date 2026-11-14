import os
import sys
import uuid
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from flows.flow_auto import AutoClaimFlow
from flows.flow_home import HomeClaimFlow
from infrastructure.bq_logger import BQLogger

# TODO(security): Integrate OAuth 2.0 / OpenID Connect and MFA via Cloud IAP in production deployment.
st.set_page_config(page_title="Agentic Insurance Fleet", layout="wide")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB limit


def validate_and_save_upload(uploaded_file, upload_dir: str = "temp_uploads") -> str:
    """
    Securely validates and stores user-uploaded image files:
    - Enforces maximum byte size (5 MB)
    - Validates extension against strict allow-list
    - Inspects magic-byte file headers (JPEG / PNG)
    - Uses UUID filename and verifies sandbox directory boundary against path traversal
    TODO(security): Integrate Cloud DLP / Antivirus malware scanning and CDR image re-encoding in production.
    """
    raw_bytes = bytes(uploaded_file.getbuffer())
    if len(raw_bytes) == 0:
        raise ValueError("Uploaded file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Uploaded file exceeds the 5 MB security size limit.")

    safe_original_name = os.path.basename(str(uploaded_file.name))
    _, ext = os.path.splitext(safe_original_name.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension '{ext}'. Allowed: .jpg, .jpeg, .png")

    is_png = raw_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = raw_bytes.startswith(b"\xff\xd8\xff")
    if not (is_png or is_jpeg):
        raise ValueError("File header magic bytes do not match a valid PNG or JPEG image.")

    base_dir = os.path.realpath(os.path.abspath(upload_dir))
    os.makedirs(base_dir, mode=0o750, exist_ok=True)

    unique_filename = f"upload_{uuid.uuid4().hex}{ext}"
    resolved_target = os.path.realpath(os.path.join(base_dir, unique_filename))
    if not resolved_target.startswith(base_dir + os.sep):
        raise ValueError("Path traversal detected.")

    with open(resolved_target, "wb") as f:
        f.write(raw_bytes)

    return resolved_target


# --- RESOURCE CACHING ---
@st.cache_resource
def get_auto_flow():
    return AutoClaimFlow()


@st.cache_resource
def get_home_flow():
    return HomeClaimFlow()


@st.cache_data
def load_data(filename):
    safe_name = os.path.basename(filename)
    with open(os.path.join("data", safe_name), "r", encoding="utf-8") as f:
        return json.load(f)


# --- VIEWS ---

def render_solution_architecture():
    st.title("🏗️ Solution Architecture")
    st.markdown("""
    ### Agentic 3-Loop Architecture
    This system demonstrates a **Multi-Agent 3-Loop Architecture** designed for autonomous high-stakes insurance claim adjudication across five core engineering pillars:
    1. **Tool & Interface Design**: Strongly-typed ADK tools (`lookup_official_policy`, `retrieve_institutional_knowledge`, `query_override_telemetry`, `propose_policy_amendment`) with schema validators.
    2. **Context & Memory**: Hybrid similarity RAG retriever + Two-tier `MemoryStore` (short-term session state & persistent episodic claim/override memory).
    3. **Orchestration & Logic**: `ClaimOrchestrator` with domain routing, 3-loop adjudication, and safety/confidence escalation guardrails.
    4. **Observability & Tracing**: OpenTelemetry-compatible `ObservabilityTracer` spans, tool call instrumentation, and `BQLogger` audit sink.
    5. **Infrastructure & CI/CD**: Hardened non-root Docker container, automated test suite, Cloud Build & GitHub Actions pipelines.
    """)

    st.graphviz_chart("""
    digraph G {
        rankdir=LR;
        node [shape=box, style=filled, fontname="Helvetica"];
        
        subgraph cluster_0 {
            label = "Perception";
            style=filled;
            color=lightgrey;
            VisualAgent [label="Visual Agent\\n(Vertex AI Vision)", fillcolor="#bbdefb"];
        }

        subgraph cluster_1 {
            label = "Cognition (Adjudicator)";
            style=filled;
            color="#fff3e0";
            
            Base [label="Layer 1\\nBase Policy", fillcolor=white];
            Inner [label="Layer 2\\nFast Loop\\n(RAG Context)", fillcolor="#fff9c4"];
            Outer [label="Layer 3\\nSlow Loop\\n(Evolution)", fillcolor="#c8e6c9"];
            
            Base -> Inner [label="Findings"];
            Inner -> Outer [label="Decision"];
        }
        
        subgraph cluster_2 {
            label = "Memory & Observability";
            style=dashed;
            
            PolicyDB [shape=cylinder, label="Policy DB"];
            KnowledgeBase [shape=cylinder, label="Institutional\\nKnowledge\\n(Ranked RAG)"];
            FeedbackStore [shape=cylinder, label="Episodic Memory\\n& BQ Audit Sink"];
        }

        Input [shape=circle, label="Evidence"];
        Output [shape=star, label="Final\\nDecision"];

        Input -> VisualAgent;
        VisualAgent -> Base [label="Validated\\nFindings"];
        
        PolicyDB -> Base;
        KnowledgeBase -> Inner [label="Top-K Retrieval"];
        FeedbackStore -> Outer [label="Override Patterns"];
        
        Outer -> Output;
    }
    """)

    st.info("Key Atomic Agents & Guardrails:")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Visual Agent (`AdkVisualAgent`)**")
        st.caption("Turns unstructured multimodal evidence into schema-validated structured findings.")
    with c2:
        st.markdown("**2. Adjudicator Agent (`AdkAdjudicatorAgent`)**")
        st.caption("Executes 3-loop reasoning across Base Policy, Fast-Loop RAG, and Slow-Loop Policy Diffs.")
    with c3:
        st.markdown("**3. Orchestrator & Escalation Gate**")
        st.caption("Enforces HV/Hazmat safety escalations, confidence thresholds, and distributed span tracing.")


def render_agentic_fleet():
    st.title("🛡️ Agentic Insurance Fleet")
    st.markdown("### Self-Evolving & Multi-Modal Claim Management")

    scenarios_db = load_data("scenarios.json")
    auto_cases = scenarios_db.get("auto", [])
    home_cases = scenarios_db.get("home", [])

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.header("Configuration")

    input_mode = st.sidebar.radio("Input Source:", ["📂 Sample Case", "📤 Upload Image"], horizontal=True)
    model_selection = st.sidebar.radio("Model Selection:", ["Model 1", "Model 2"], horizontal=True)

    if "input_mode_prev" not in st.session_state:
        st.session_state["input_mode_prev"] = input_mode
    if st.session_state["input_mode_prev"] != input_mode:
        st.session_state.pop("adjudication_result", None)
        st.session_state["input_mode_prev"] = input_mode

    selected_case = None
    claim_type = "auto"
    img_path = None

    if input_mode == "📤 Upload Image":
        st.sidebar.markdown("---")
        uploaded_file = st.sidebar.file_uploader("Upload evidence...", type=["jpg", "jpeg", "png"])
        claim_type_input = st.sidebar.radio("Claim Type", ["Auto", "Home"], horizontal=True)
        claim_type = claim_type_input.lower()

        if uploaded_file is not None:
            try:
                safe_path = validate_and_save_upload(uploaded_file)
                img_path = safe_path
                safe_display_name = os.path.basename(str(uploaded_file.name))
                selected_case = {
                    "id": "USER_UPLOAD",
                    "title": f"Custom Upload ({safe_display_name})",
                    "image_file": safe_display_name,
                }
            except ValueError as err:
                st.sidebar.error(f"Upload rejected: {err}")
        else:
            st.sidebar.warning("Please upload an image.")
    else:
        case_options = [f"🚘 [Auto] {c['id']}: {c['title']}" for c in auto_cases]
        case_options += [f"🏠 [Home] {c['id']}: {c['title']}" for c in home_cases]

        selected_option = st.sidebar.selectbox(
            "Select Case",
            case_options,
            on_change=lambda: st.session_state.pop("adjudication_result", None),
        )

        is_auto = "🚘" in selected_option
        claim_type = "auto" if is_auto else "home"

        if is_auto:
            selected_case = next((c for c in auto_cases if f"🚘 [Auto] {c['id']}: {c['title']}" == selected_option), None)
        else:
            selected_case = next((c for c in home_cases if f"🏠 [Home] {c['id']}: {c['title']}" == selected_option), None)

        if selected_case:
            img_path = os.path.join("assets", os.path.basename(selected_case["image_file"]))

    # Knowledge Base Sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("🧠 Knowledge Base")

    with st.sidebar.expander("📜 Official Policies", expanded=False):
        if claim_type == "auto":
            st.subheader("Auto Policy")
            st.json(load_data("policy_auto.json"))
        else:
            st.subheader("Home Policy")
            st.json(load_data("policy_home.json"))

    with st.sidebar.expander("💬 Institutional Knowledge", expanded=False):
        st.json(load_data("institutional_knowledge.json"))

    st.sidebar.markdown("---")
    st.sidebar.caption(f"System Status: ● Online\nENV: {os.getenv('ENV', 'LOCAL')}")

    # --- MAIN CONTENT AREA ---
    if not selected_case or not img_path:
        st.warning("Select a case or upload an image.")
        return

    st.info(f"**Context:** {selected_case['title']}  |  **Type:** {claim_type.title()}")

    col1, col2 = st.columns([0.45, 0.55], gap="large")

    with col1:
        st.markdown("#### 📷 Visual Evidence")
        if os.path.exists(img_path):
            st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
        else:
            st.error(f"Image not found: {img_path}")

        st.markdown("---")
        if st.button("🚀 Process Claim", type="primary", use_container_width=True):
            with st.spinner("🤖 Orchestrating Multi-Agent Claim Flow..."):
                flow = get_auto_flow() if claim_type == "auto" else get_home_flow()
                if model_selection == "Model 1":
                    try:
                        cached_db = load_data("cached_results.json")
                        case_id = selected_case.get("id") if selected_case else "unknown"
                        cached_result = cached_db.get(case_id)

                        if cached_result:
                            # Run orchestrator to capture full observability & memory trace, using cached decision trace
                            live_run = flow.run(img_path)
                            live_run["decision_trace"] = cached_result
                            result = live_run
                        else:
                            result = flow.run(img_path)
                    except Exception:
                        result = flow.run(img_path)
                else:
                    result = flow.run(img_path)

                st.session_state["adjudication_result"] = result

    with col2:
        if "adjudication_result" in st.session_state:
            result = st.session_state["adjudication_result"]
            findings = result["findings"]
            trace = result["decision_trace"]
            escalation = result.get("escalation", {})
            observability = result.get("observability", {})

            st.success(f"✅ Adjudication Complete (Claim ID: `{result.get('claim_id', 'N/A')[:8]}`)")

            if escalation.get("requires_human_signoff"):
                st.warning(
                    f"⚠️ **Escalation Gate [{escalation.get('status')}]:** {escalation.get('reason')}"
                )

            with st.container():
                st.markdown("#### 👁️ Computer Vision Analysis")
                c1, c2, c3 = st.columns(3)
                c1.metric("Detected Object", findings.get("object_detected", "Unknown"))
                c2.metric("Damage Type", str(findings.get("damage_type", "Unknown")).title())
                if "severity" in findings and findings["severity"]:
                    c3.metric("Severity", str(findings["severity"]).title())
                elif "affected_area_percentage" in findings:
                    c3.metric("Coverage", f"{findings['affected_area_percentage']}%")

            st.divider()

            # --- 3 LAYER + OBSERVABILITY VISUALIZATION ---
            st.markdown("### 🧬 Decision Evolution & Observability")

            tab1, tab2, tab3, tab4 = st.tabs([
                "1. Base Policy",
                "2. Fast Loop (Context)",
                "3. Slow Loop (System Update)",
                "4. 🔍 Observability & Trace",
            ])

            with tab1:
                st.caption("Standard rules engine. Agnostic to recent events.")
                d = trace.get("layer_1_base", {})
                if not d:
                    st.info("No data for Layer 1")
                else:
                    outcome = d.get("outcome", "UNKNOWN")
                    if "TOTAL_LOSS" in outcome or "REFUSE" in outcome or "DENY" in outcome:
                        st.error(f"## {outcome}")
                    else:
                        st.success(f"## {outcome}")
                    st.write(f"**Reason:** {d.get('reason', 'N/A')}")
                    st.metric("Confidence", f"{float(d.get('confidence', 0.0)):.2f}")

            with tab2:
                st.caption("Incorporates Ranked Institutional Knowledge (Chat Logs, Safety Bulletins).")
                layer_data = trace.get("layer_2_inner", {})
                d = layer_data.get("decision", {})
                meta = layer_data.get("context_metadata", {})

                if d:
                    outcome = d.get("outcome", "UNKNOWN")
                    if "TOTAL_LOSS" in outcome or "REFUSE" in outcome or "DENY" in outcome:
                        st.error(f"## {outcome}")
                    else:
                        st.success(f"## {outcome}")
                    st.write(f"**Reason:** {d.get('reason', 'N/A')}")
                else:
                    st.info("No decision data for Layer 2")

                if meta.get("retrieved_rules"):
                    st.divider()
                    st.markdown("##### 💬 Relevant Context Retrieved (Hybrid RAG):")
                    for rule in meta["retrieved_rules"]:
                        if isinstance(rule, dict):
                            author = rule.get("author", "System")
                            source = rule.get("source", "Knowledge Base")
                            content = rule.get("content", "")
                            context = rule.get("context", "General")
                            score = rule.get("similarity_score")
                            score_badge = f" | Relevance: `{score:.2f}`" if score is not None else ""
                            with st.chat_message("user", avatar="👤"):
                                st.write(f"**{author}** in *{source}*{score_badge}")
                                st.caption(f"Context: {context}")
                                st.markdown(f"> {content}")
                        else:
                            st.info(f"Rule: {rule}")
                else:
                    st.caption("No overrides applied from Institutional Knowledge.")

            with tab3:
                st.caption("System learns from episodic feedback signals to propose policy updates.")
                layer_data = trace.get("layer_3_outer", {})
                d = layer_data.get("decision", {})
                signals = layer_data.get("signals", {})

                if d:
                    outcome = d.get("outcome", "UNKNOWN")
                    if "TOTAL_LOSS" in outcome or "REFUSE" in outcome or "DENY" in outcome:
                        st.error(f"## {outcome}")
                    else:
                        st.success(f"## {outcome}")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric(
                            "Confidence",
                            f"{float(d.get('confidence', 0.0)):.2f}",
                            delta="Reinforced by Feedback" if signals.get("frequency_stats") else None,
                        )
                    with col_b:
                        if signals.get("frequency_stats"):
                            stats = signals["frequency_stats"]
                            st.metric(stats.get("metric", "Trend"), stats.get("value", 0), delta=stats.get("trend", "Flat"))

                if signals.get("draft_policy_diff"):
                    st.markdown("##### 📝 Proposed Policy Amendment")
                    st.code(signals["draft_policy_diff"], language="diff")
                else:
                    st.info("No sufficient patterns detected for policy evolution yet.")

            with tab4:
                st.caption("Distributed execution trace, tool call telemetry, and BigQuery audit logs.")
                if observability:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Trace ID", observability.get("trace_id", "N/A")[:14])
                    m2.metric("Total Latency", f"{observability.get('total_duration_ms', 0.0):.1f} ms")
                    m3.metric("Tool Invocations", observability.get("tool_call_count", 0))

                    st.markdown("##### ⏱️ Multi-Agent Span Waterfall")
                    for span_item in observability.get("spans", []):
                        st.markdown(
                            f"- **`{span_item['agent']}.{span_item['name']}`** — "
                            f"`{span_item['duration_ms']} ms` | Status: `{span_item['status']}`"
                        )
                        for tc in span_item.get("tool_calls", []):
                            st.caption(
                                f"  ↳ 🛠️ Tool `{tc['tool_name']}` ({tc['duration_ms']} ms) — Args: `{json.dumps(tc['arguments'])}`"
                            )
                with st.expander("📦 Raw BigQuery Audit Logs (Recent)", expanded=False):
                    st.json(BQLogger.get_recent_logs(limit=5))

            # --- HUMAN-IN-THE-LOOP (HITL) FEEDBACK INTERFACE ---
            st.divider()
            with st.expander("🧑‍⚖️ Human-in-the-Loop (HITL) Adjuster Override & Memory Feedback", expanded=False):
                st.caption("Submit an adjuster override to update BigQuery, Episodic Memory, and Institutional Knowledge.")
                current_l2_outcome = trace.get("layer_2_inner", {}).get("decision", {}).get("outcome", "APPROVE_REPAIR")
                corrected_outcome = st.selectbox(
                    "Override Outcome",
                    [
                        "APPROVE_REPAIR",
                        "TOTAL_LOSS",
                        "TOTAL_LOSS + HAZMAT FEE",
                        "APPROVE_PDR_SERVICE",
                        "APPROVE_REPAIR (PLASTIC WELD)",
                        "REFUSE_REPAIR_FULL_REPLACEMENT",
                        "MANUAL_REVIEW",
                    ],
                )
                adjuster_notes = st.text_input(
                    "Adjuster Rationale / Institutional Rule Note",
                    placeholder="e.g., Field inspection confirmed hairline battery casing impact; enforce total loss.",
                )
                if st.button("💾 Submit Human Override to Memory & BigQuery"):
                    flow = get_auto_flow() if claim_type == "auto" else get_home_flow()
                    feedback_res = flow.orchestrator.submit_human_feedback(
                        claim_id=result.get("claim_id", "manual"),
                        claim_type=claim_type,
                        damage_type=findings.get("damage_type", "unknown"),
                        original_outcome=current_l2_outcome,
                        corrected_outcome=corrected_outcome,
                        adjuster_notes=adjuster_notes or "Manual adjuster override.",
                    )
                    st.success("✅ Override persisted to Episodic Memory (`claims_db.json`), BigQuery Audit Log, and RAG Knowledge Base!")
                    st.json(feedback_res)


# --- MAIN APP ROUTER ---
def main():
    st.sidebar.image("assets/google_cloud_uploaded.png", use_container_width=True)
    st.sidebar.title("Navigation")
    view = st.sidebar.radio("Go to:", ["Agentic Insurance Fleet", "Solution Architecture"])

    st.sidebar.divider()

    if view == "Agentic Insurance Fleet":
        render_agentic_fleet()
    elif view == "Solution Architecture":
        render_solution_architecture()


if __name__ == "__main__":
    main()

