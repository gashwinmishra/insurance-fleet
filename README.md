# 🛡️ Agentic Insurance Fleet — Self-Evolving Multi-Agent Adjudication System

An enterprise-grade **Multi-Agent 3-Loop Insurance Adjudication System** built on **Google Agent Development Kit (ADK)**, **Vertex AI (Gemini 2.0 Flash & Gemini 1.5 Pro)**, **BigQuery**, **SQLite**, **Terraform**, and **Cloud Run**.

The system processes multimodal claim evidence (Auto & Home damage imagery) across three cognitive loops:
1. **Layer 1 — Base Policy Loop**: Deterministic evaluation against codified policy terms (`data/policy_auto.json`, `data/policy_home.json`).
2. **Layer 2 — Fast Loop (Contextual RAG)**: Real-time retrieval of institutional knowledge (safety alerts, Slack memos, vendor routing guidelines) to override outdated base policies when necessary.
3. **Layer 3 — Slow Loop (Meta-Learning & Policy Evolution)**: Aggregates recurring human adjuster overrides from Episodic Memory to automatically draft git-style policy amendments (`draft_policy_diff`).

---

## 🏆 Certification Rubric Alignment (5 Pillars)

| Evaluation Pillar | Key Implementation Files | Architectural Highlights |
| :--- | :--- | :--- |
| **1. Tool & Interface Design** | [`agents/tools.py`](./agents/tools.py)<br>[`agents/adk_adjudicator.py`](./agents/adk_adjudicator.py)<br>[`ui/dashboard.py`](./ui/dashboard.py) | • **Formal JSON Schema Constraints**: Pydantic `BaseModel` schemas (`PolicyLookupArgs`, `InstitutionalKnowledgeQueryArgs`, `OverrideTelemetryQueryArgs`, `PolicyAmendmentProposalArgs`, `VisualFindingsSchema`, `AdjudicationTraceSchema`) and Google GenAI `types.FunctionDeclaration` / `response_schema` configurations (`TOOL_JSON_SCHEMAS`, `ADK_GENAI_TOOLS`).<br>• **Actionable Tool Error Recovery Guidance**: `format_tool_error_with_recovery` returns structured JSON error payloads containing `"status": "ERROR"`, `"error_code"`, `"valid_options"`, and step-by-step `"recovery_guidance"` so the LLM can autonomously self-correct.<br>• **Secure Multimodal & HITL UI**: Hardened image upload validation (magic-byte checks, 5 MB limit, UUID naming, path traversal guard) and interactive Human-in-the-Loop (HITL) adjuster override panel. |
| **2. Context & Memory** | [`infrastructure/memory_store.py`](./infrastructure/memory_store.py)<br>[`infrastructure/rag_retriever.py`](./infrastructure/rag_retriever.py) | • **Token-Budget Semantic History Compaction**: `HistoryCompactor` estimates token usage and distills older conversational turns and episodic claims into structured `COMPACTED_SESSION_DIGEST` summaries (preserving entities, decisions, and overrides) while retaining recent turns verbatim.<br>• **Durable SQLite + JSON Conversational Session Persistence**: Conversational session turns and compacted summaries are persisted in SQLite (`data/sessions_memory.db`) alongside `data/claims_db.json` and synced to BigQuery, surviving process restarts.<br>• **Non-Blocking Async Memory & RAG Operations**: Full `async`/`await` API (`aappend_session_event`, `aget_session_history`, `acompact_session_history`, `arecord_claim`, `arecord_human_override`, `aretrieve`, `alog_feedback`) backed by `asyncio.Lock` and `asyncio.to_thread` to prevent UI blocking. |
| **3. Orchestration & Logic** | [`flows/orchestrator.py`](./flows/orchestrator.py)<br>[`agents/adk_adjudicator.py`](./agents/adk_adjudicator.py)<br>[`flows/flow_auto.py`](./flows/flow_auto.py)<br>[`flows/flow_home.py`](./flows/flow_home.py) | • **Strategic Model Routing (`ModelRouter`)**: Dynamically routes low-latency multimodal perception and routine claims to **`gemini-2.0-flash-001`**, while routing complex multi-layer policy conflicts, high-voltage/hazmat safety hazards, and Layer 3 Slow-Loop policy synthesis to **`gemini-1.5-pro-002`**.<br>• **Safety & Confidence Guardrails**: Automatically flags high-voltage EV battery/hazmat risks (`CRITICAL_SAFETY_ESCALATION`) and low-confidence decisions (`< 0.75`) for human sign-off. |
| **4. Observability & Tracing** | [`infrastructure/observability.py`](./infrastructure/observability.py)<br>[`infrastructure/bq_logger.py`](./infrastructure/bq_logger.py)<br>[`ui/dashboard.py`](./ui/dashboard.py) | • **Automated PII Redaction Engine (`PIIRedactor`)**: Recursively scrubs SSNs (`[REDACTED_SSN]`), emails (`[REDACTED_EMAIL]`), phone numbers (`[REDACTED_PHONE]`), credit cards (`[REDACTED_CC]`), 17-char VINs (`[REDACTED_VIN]`), sensitive keys, and bearer tokens/API keys (`[REDACTED_SECRET]`) across all log records, span attributes, tool inputs/outputs, and BigQuery payloads.<br>• **Distributed Span Tracing**: `ObservabilityTracer` and `Span` record `trace_id`, `span_id`, tool call intent vs. outcome, model routing decisions, and PII redaction counts. |
| **5. Infrastructure & CI/CD** | [`terraform/main.tf`](./terraform/main.tf)<br>[`evals/run_eval.py`](./evals/run_eval.py)<br>[`evals/golden_dataset.json`](./evals/golden_dataset.json)<br>[`.github/workflows/ci.yml`](./.github/workflows/ci.yml)<br>[`cloudbuild.yaml`](./cloudbuild.yaml) | • **Declarative Infrastructure as Code (Terraform)**: `terraform/` provisions Artifact Registry, BigQuery dataset & `claims_log` schema, least-privilege IAM Service Account, Secret Manager, and Cloud Run v2 Service.<br>• **Generative Evaluation Suite & Golden Dataset**: `evals/run_eval.py` evaluates the agent system against `evals/golden_dataset.json` across Tool Trajectory Recall, Model Routing Accuracy, Layer 1 Policy Accuracy, Layer 2 RAG Groundedness, and Layer 3 Judge Rubric Score, enforced in GitHub Actions and Cloud Build. |

---

## 🚀 Quickstart

### 1. Local Setup & Execution
```bash
./setup_env.sh
source venv/bin/activate
streamlit run ui/dashboard.py
```

### 2. Run Unit Tests & Generative Golden Evaluation Suite
```bash
ADK_OFFLINE_MODE=1 python3 -m unittest discover -s tests -v
ADK_OFFLINE_MODE=1 python3 evals/run_eval.py
```

### 3. Provision Infrastructure with Terraform & Deploy
```bash
cd terraform
terraform init
terraform apply -var="project_id=YOUR_PROJECT_ID"
```
See [`DEPLOYMENT.md`](./DEPLOYMENT.md) for full Terraform and Cloud Build deployment details.
