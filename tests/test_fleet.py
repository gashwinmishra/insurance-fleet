import asyncio
import json
import os
import tempfile
import unittest

os.environ["ADK_OFFLINE_MODE"] = "1"
os.environ["ENV"] = "LOCAL"

from agents.tools import (
    ADK_TOOL_FUNCTION_DECLARATIONS,
    TOOL_JSON_SCHEMAS,
    lookup_official_policy,
    retrieve_institutional_knowledge,
    query_override_telemetry,
    propose_policy_amendment,
    validate_visual_findings,
    validate_adjudication_trace,
)
from agents.adk_adjudicator import ModelRouter
from infrastructure.rag_retriever import RAGRetriever
from infrastructure.memory_store import HistoryCompactor, MemoryStore
from infrastructure.observability import ObservabilityTracer, PIIRedactor
from infrastructure.bq_logger import BQLogger
from flows.orchestrator import ClaimOrchestrator
from flows.flow_auto import AutoClaimFlow
from flows.flow_home import HomeClaimFlow
from evals.run_eval import GenerativeEvalRunner
from ui.dashboard import validate_and_save_upload


class DummyUploadFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getbuffer(self):
        return memoryview(self._content)


class TestInsuranceFleetPillars(unittest.TestCase):
    """Comprehensive Automated Test Suite covering all 5 Certification Rubric Pillars."""

    # --- Pillar 1: Tool & Interface Design (JSON Schemas & Actionable Recovery Guidance) ---
    def test_pillar_1_formal_json_schemas_and_error_recovery(self):
        # Verify formal JSON Schema declarations exist for all tools and responses
        self.assertEqual(len(ADK_TOOL_FUNCTION_DECLARATIONS), 4)
        self.assertIn("lookup_official_policy", TOOL_JSON_SCHEMAS)
        self.assertIn("adjudication_trace_response", TOOL_JSON_SCHEMAS)

        # Verify happy-path tool execution
        policy_text = lookup_official_policy("auto", "scratch")
        self.assertIn("EV_COMMERCIAL_FLEET_v2.0", policy_text)

        # Verify actionable recovery guidance on invalid tool inputs
        err_policy = json.loads(lookup_official_policy("invalid_domain", "scratch"))
        self.assertEqual(err_policy["status"], "ERROR")
        self.assertIn("recovery_guidance", err_policy)
        self.assertIn("valid_options", err_policy)

        err_rag = json.loads(retrieve_institutional_knowledge("", claim_type="auto"))
        self.assertEqual(err_rag["status"], "ERROR")
        self.assertIn("recovery_guidance", err_rag)

        err_amend = json.loads(propose_policy_amendment("auto", "k", "a", "b", "short"))
        self.assertEqual(err_amend["status"], "ERROR")
        self.assertIn("recovery_guidance", err_amend)

        norm_findings = validate_visual_findings({"damage_type": "SCRATCH", "depth_mm": "1.5"}, claim_type="auto")
        self.assertEqual(norm_findings["damage_type"], "scratch")
        self.assertEqual(norm_findings["depth_mm"], 1.5)

    def test_pillar_1_secure_upload_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            valid_png = DummyUploadFile("car.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
            saved_path = validate_and_save_upload(valid_png, upload_dir=tmpdir)
            self.assertTrue(os.path.exists(saved_path))
            self.assertNotIn("car.png", os.path.basename(saved_path))

            spoofed_file = DummyUploadFile("exploit.png", b"<script>alert(1)</script>")
            with self.assertRaises(ValueError):
                validate_and_save_upload(spoofed_file, upload_dir=tmpdir)

    # --- Pillar 2: Context & Memory (Semantic Compaction, SQLite Persistence, Async API) ---
    def test_pillar_2_semantic_compaction_sqlite_persistence_and_async(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_db = os.path.join(tmpdir, "claims_db.json")
            sqlite_db = os.path.join(tmpdir, "sessions_memory.db")

            async def run_async_memory_workflow():
                mem = MemoryStore(db_path=json_db, sqlite_path=sqlite_db)
                rag = RAGRetriever()

                # Async RAG retrieval
                hits = await rag.aretrieve("EV battery plate scratch fire", claim_type="auto", top_k=3)
                self.assertEqual(hits[0]["id"], "MSG-8921")

                # Append 6 turns via async API and trigger semantic compaction
                for idx in range(6):
                    await mem.aappend_session_event(
                        "sess_persist_1",
                        "visual_agent" if idx % 2 == 0 else "adjudicator_agent",
                        {
                            "stage": f"step_{idx}",
                            "object_detected": "EV Battery",
                            "damage_type": "scratch",
                            "location": "battery_casing",
                            "outcome": "TOTAL_LOSS",
                            "confidence": 0.95,
                        },
                        auto_compact_threshold=10,
                    )

                compaction = await mem.acompact_session_history("sess_persist_1", max_tokens=50, keep_recent=2)
                self.assertTrue(compaction["compacted"])
                self.assertEqual(compaction["summary_digest"]["type"], "COMPACTED_SESSION_DIGEST")
                self.assertLess(compaction["tokens_after"], compaction["tokens_before"])

                # Re-instantiate MemoryStore from disk/SQLite to verify durable session persistence
                mem_reloaded = MemoryStore(db_path=json_db, sqlite_path=sqlite_db)
                history = await mem_reloaded.aget_session_history("sess_persist_1")
                self.assertEqual(len(history), 3)  # 1 compacted digest + 2 recent verbatim turns
                self.assertEqual(history[0]["role"], "system_memory_compactor")

            asyncio.run(run_async_memory_workflow())

    # --- Pillar 3: Orchestration & Logic (Strategic Model Routing: Flash vs Pro) ---
    def test_pillar_3_strategic_model_routing_and_orchestrator(self):
        # High-stakes battery scratch -> Routed to Gemini Pro (gemini-1.5-pro-002)
        pro_route = ModelRouter.route(
            {"damage_type": "scratch", "location": "battery_casing", "severity": "negligible"},
            claim_type="auto",
        )
        self.assertEqual(pro_route["selected_model"], ModelRouter.PRO_MODEL)
        self.assertEqual(pro_route["tier"], "PRO_COMPLEX_PLANNING")

        # Routine minor bumper crack -> Routed to Gemini Flash (gemini-2.0-flash-001)
        flash_route = ModelRouter.route(
            {"damage_type": "crack", "location": "rear_bumper_plastic", "severity": "minor"},
            claim_type="auto",
        )
        self.assertEqual(flash_route["selected_model"], ModelRouter.FLASH_MODEL)
        self.assertEqual(flash_route["tier"], "FLASH_ROUTINE_ADJUDICATION")

        # End-to-end Auto & Home flows
        auto_flow = AutoClaimFlow()
        res_auto = auto_flow.run("assets/car_scratch_shallow.png")
        self.assertEqual(res_auto["model_routing"]["selected_model"], ModelRouter.PRO_MODEL)
        self.assertEqual(res_auto["decision_trace"]["layer_2_inner"]["decision"]["outcome"], "TOTAL_LOSS")

        home_flow = HomeClaimFlow()
        res_home = home_flow.run("assets/roof_damage.png")
        self.assertEqual(
            res_home["decision_trace"]["layer_2_inner"]["decision"]["outcome"],
            "REFUSE_REPAIR_FULL_REPLACEMENT",
        )

    # --- Pillar 4: Observability & Tracing (PII Redaction & Distributed Spans) ---
    def test_pillar_4_pii_redaction_and_distributed_tracing(self):
        raw_payload = {
            "policyholder_name": "John Doe",
            "ssn": "123-45-6789",
            "notes": "Contact claimant at john.doe@example.com or 415-555-0199 regarding VIN 5YJ3E1EA7JF000001 and SSN 123-45-6789.",
        }
        sanitized, count = PIIRedactor.redact(raw_payload)
        self.assertGreaterEqual(count, 5)
        self.assertEqual(sanitized["policyholder_name"], "[REDACTED_POLICYHOLDER_NAME]")
        self.assertEqual(sanitized["ssn"], "[REDACTED_SSN]")
        self.assertNotIn("john.doe@example.com", sanitized["notes"])
        self.assertNotIn("415-555-0199", sanitized["notes"])
        self.assertNotIn("5YJ3E1EA7JF000001", sanitized["notes"])
        self.assertIn("[REDACTED_EMAIL]", sanitized["notes"])
        self.assertIn("[REDACTED_PHONE]", sanitized["notes"])
        self.assertIn("[REDACTED_VIN]", sanitized["notes"])

        # Verify BQLogger scrubs PII before storing
        logger = BQLogger()
        entry = logger.log_feedback(
            claim_id="test_pii_claim",
            agent_decision={"outcome": "APPROVE_REPAIR", "notes": "Email jane@example.com"},
            human_correction={"ssn": "999-00-1111"},
        )
        self.assertNotIn("jane@example.com", entry["agent_decision"])
        self.assertNotIn("999-00-1111", entry["human_correction"])

    # --- Pillar 5: Infrastructure & CI/CD (Golden Dataset Generative Evaluation Suite) ---
    def test_pillar_5_golden_dataset_evaluation_suite(self):
        runner = GenerativeEvalRunner()
        report = runner.run_suite()
        self.assertEqual(report["total_cases"], 8)
        self.assertGreaterEqual(report["mean_composite_score"], 0.95)
        self.assertEqual(report["pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
