# 🛡️ Agentic Insurance Fleet — Self-Evolving Multi-Agent Adjudication System

An enterprise-grade **Multi-Agent 3-Loop Insurance Adjudication System** built on **Google Agent Development Kit (ADK)**, **Vertex AI (Gemini 2.0 Flash)**, **BigQuery**, and **Cloud Run**.

The system processes multimodal claim evidence (Auto & Home damage imagery) across three cognitive loops:
1. **Layer 1 — Base Policy Loop**: Deterministic evaluation against codified policy terms (`data/policy_auto.json`, `data/policy_home.json`).
2. **Layer 2 — Fast Loop (Contextual RAG)**: Real-time retrieval of institutional knowledge (safety alerts, Slack memos, vendor routing guidelines) to override outdated base policies when necessary.
3. **Layer 3 — Slow Loop (Meta-Learning & Policy Evolution)**: Aggregates recurring human adjuster overrides from Episodic Memory to automatically draft git-style policy amendments (`draft_policy_diff`).

---

## 🏆 Certification Rubric Alignment (5 Pillars)

| Evaluation Pillar | Key Implementation Files | Architectural Highlights |
| :--- | :--- | :--- |
| **1. Tool & Interface Design** | [`agents/tools.py`](./agents/tools.py)<br>[`agents/adk_adjudicator.py`](./agents/adk_adjudicator.py)<br>[`ui/dashboard.py`](./ui/dashboard.py) | • **4 Typed ADK Agent Tools**: `lookup_official_policy`, `retrieve_institutional_knowledge`, `query_override_telemetry`, `propose_policy_amendment`.<br>• **Strict Schema Contracts**: `validate_visual_findings` and `validate_adjudication_trace` guarantee type safety across agent boundaries.<br>• **Secure Multimodal & HITL UI**: Hardened image upload validation (magic-byte checks, 5 MB limit, UUID naming, path traversal guard) and interactive Human-in-the-Loop (HITL) adjuster override panel. |
| **2. Context & Memory** | [`infrastructure/rag_retriever.py`](./infrastructure/rag_retriever.py)<br>[`infrastructure/memory_store.py`](./infrastructure/memory_store.py)<br>[`data/institutional_knowledge.json`](./data/institutional_knowledge.json) | • **Hybrid Similarity RAG**: `RAGRetriever` scores institutional rules using lexical overlap, domain synonym expansion, and authority weight boosting.<br>• **Two-Tier `MemoryStore`**: Combines short-term session event tracking (`append_session_event`) with persistent long-term episodic memory (`data/claims_db.json`) for historical claims, human overrides, and pattern frequencies. |
| **3. Orchestration & Logic** | [`flows/orchestrator.py`](./flows/orchestrator.py)<br>[`flows/flow_auto.py`](./flows/flow_auto.py)<br>[`flows/flow_home.py`](./flows/flow_home.py)<br>[`agents/adjudicator.py`](./agents/adjudicator.py) | • **Multi-Stage `ClaimOrchestrator`**: Coordinates Intake Triage → `AdkVisualAgent` Perception → `AdkAdjudicatorAgent` 3-Loop Reasoning → Safety/Confidence Escalation Gate → Memory & Audit Commit.<br>• **Safety & Confidence Guardrails**: Automatically flags high-voltage EV battery/hazmat risks (`CRITICAL_SAFETY_ESCALATION`) and low-confidence decisions (`< 0.75`) for human sign-off.<br>• **Resilient Hybrid Fallback**: Seamlessly falls back to the deterministic 3-loop `AdjudicatorAgent` when executed in offline/CI environments. |
| **4. Observability & Tracing** | [`infrastructure/observability.py`](./infrastructure/observability.py)<br>[`infrastructure/bq_logger.py`](./infrastructure/bq_logger.py)<br>[`ui/dashboard.py`](./ui/dashboard.py) | • **Distributed Span Tracing**: `ObservabilityTracer` and `Span` record `trace_id`, `span_id`, parent-child hierarchies, per-stage latency (`duration_ms`), tool call arguments/outputs, and confidence progression across layers.<br>• **Cloud Logging & BigQuery Sink**: Structured JSON formatter + `BQLogger` streaming decisions and human corrections to `antigravity_logs.claims_log`, inspectable live in the UI's **🔍 Observability & Trace** tab. |
| **5. Infrastructure & CI/CD** | [`Dockerfile`](./Dockerfile)<br>[`cloudbuild.yaml`](./cloudbuild.yaml)<br>[`.github/workflows/ci.yml`](./.github/workflows/ci.yml)<br>[`tests/test_fleet.py`](./tests/test_fleet.py) | • **Hardened Container**: Non-root `appuser`, `HEALTHCHECK`, XSRF protection enabled.<br>• **Automated CI/CD**: Both GitHub Actions (`.github/workflows/ci.yml`) and Cloud Build (`cloudbuild.yaml`) run the 5-pillar automated test suite (`tests/test_fleet.py`) prior to container build and Cloud Run deployment. |

---

## 🚀 Quickstart

### 1. Local Setup & Execution
```bash
./setup_env.sh
source venv/bin/activate
streamlit run ui/dashboard.py
```

### 2. Run the 5-Pillar Automated Test Suite
```bash
ADK_OFFLINE_MODE=1 python3 -m unittest discover -s tests -v
```

### 3. Cloud Run Deployment
See [`DEPLOYMENT.md`](./DEPLOYMENT.md) for full Google Cloud Build and BigQuery setup instructions:
```bash
gcloud builds submit --config cloudbuild.yaml .
```
