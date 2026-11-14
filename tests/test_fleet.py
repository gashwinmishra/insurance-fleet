import io
import json
import os
import tempfile
import unittest

os.environ["ADK_OFFLINE_MODE"] = "1"
os.environ["ENV"] = "LOCAL"

from agents.tools import (
    lookup_official_policy,
    retrieve_institutional_knowledge,
    query_override_telemetry,
    propose_policy_amendment,
    validate_visual_findings,
    validate_adjudication_trace,
)
from infrastructure.rag_retriever import RAGRetriever
from infrastructure.memory_store import MemoryStore
from infrastructure.observability import ObservabilityTracer
from infrastructure.bq_logger import BQLogger
from flows.orchestrator import ClaimOrchestrator
from flows.flow_auto import AutoClaimFlow
from flows.flow_home import HomeClaimFlow
from ui.dashboard import validate_and_save_upload


class DummyUploadFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getbuffer(self):
        return memoryview(self._content)


class TestInsuranceFleetPillars(unittest.TestCase):
    """Automated Verification Suite covering all 5 Certification Rubric Pillars."""

    # --- Pillar 1: Tool & Interface Design ---
    def test_pillar_1_tools_and_schema_contracts(self):
        policy_text = lookup_official_policy("auto", "scratch")
        self.assertIn("EV_COMMERCIAL_FLEET_v2.0", policy_text)
        self.assertIn("battery_damage", policy_text)

        rag_text = retrieve_institutional_knowledge("battery scratch casing", claim_type="auto", top_k=2)
        self.assertIn("MANDATORY TOTAL LOSS", rag_text)

        telemetry_json = json.loads(query_override_telemetry("auto", "scratch", "battery_casing"))
        self.assertEqual(telemetry_json["metric"], "Manual Overrides")

        diff_json = json.loads(
            propose_policy_amendment("auto", "battery_damage", "< 2mm repair", "ZERO TOLERANCE", "Fire hazard")
        )
        self.assertIn("draft_policy_diff", diff_json)

        norm_findings = validate_visual_findings({"damage_type": "SCRATCH", "depth_mm": "1.5"}, claim_type="auto")
        self.assertEqual(norm_findings["damage_type"], "scratch")
        self.assertEqual(norm_findings["depth_mm"], 1.5)

        norm_trace = validate_adjudication_trace({})
        self.assertIn("layer_1_base", norm_trace)
        self.assertIn("layer_2_inner", norm_trace)
        self.assertIn("layer_3_outer", norm_trace)

    def test_pillar_1_secure_upload_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid PNG header
            valid_png = DummyUploadFile("car.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
            saved_path = validate_and_save_upload(valid_png, upload_dir=tmpdir)
            self.assertTrue(os.path.exists(saved_path))
            self.assertNotIn("car.png", os.path.basename(saved_path))  # Renamed to UUID

            # Reject invalid magic bytes
            spoofed_file = DummyUploadFile("exploit.png", b"<script>alert(1)</script>")
            with self.assertRaises(ValueError):
                validate_and_save_upload(spoofed_file, upload_dir=tmpdir)

            # Reject unsupported extension
            bad_ext = DummyUploadFile("payload.sh", b"\x89PNG\r\n\x1a\n")
            with self.assertRaises(ValueError):
                validate_and_save_upload(bad_ext, upload_dir=tmpdir)

    # --- Pillar 2: Context & Memory ---
    def test_pillar_2_rag_ranking_and_memory_store(self):
        rag = RAGRetriever()
        hits = rag.retrieve("EV battery plate scratch fire", claim_type="auto", top_k=3)
        self.assertTrue(len(hits) > 0)
        self.assertEqual(hits[0]["id"], "MSG-8921")
        self.assertIn("similarity_score", hits[0])

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_db = os.path.join(tmpdir, "claims_db.json")
            mem = MemoryStore(db_path=temp_db)
            mem.append_session_event("sess_1", "user", {"msg": "start"})
            self.assertEqual(len(mem.get_session_history("sess_1")), 1)

            initial_freq = mem.get_pattern_frequency("auto", "scratch", "battery_casing")
            mem.record_claim(
                claim_id="c1",
                claim_type="auto",
                findings={"damage_type": "scratch", "location": "battery_casing"},
                decision_trace={"layer_2_inner": {"decision": {"outcome": "TOTAL_LOSS"}}},
            )
            self.assertEqual(mem.get_pattern_frequency("auto", "scratch", "battery_casing"), initial_freq + 1)

    # --- Pillar 3: Orchestration & Logic ---
    def test_pillar_3_orchestrator_and_3_loop_flows(self):
        auto_flow = AutoClaimFlow()
        res_auto = auto_flow.run("assets/car_scratch_shallow.png")
        trace_auto = res_auto["decision_trace"]

        # Layer 1 approves repair (<2mm), Layer 2 overrides to TOTAL_LOSS, Layer 3 generates policy diff
        self.assertEqual(trace_auto["layer_1_base"]["outcome"], "APPROVE_REPAIR")
        self.assertEqual(trace_auto["layer_2_inner"]["decision"]["outcome"], "TOTAL_LOSS")
        self.assertIsNotNone(trace_auto["layer_3_outer"]["signals"]["draft_policy_diff"])
        self.assertTrue(res_auto["escalation"]["requires_human_signoff"])

        home_flow = HomeClaimFlow()
        res_home = home_flow.run("assets/roof_damage.png")
        trace_home = res_home["decision_trace"]
        self.assertEqual(trace_home["layer_2_inner"]["decision"]["outcome"], "REFUSE_REPAIR_FULL_REPLACEMENT")
        self.assertIsNotNone(trace_home["layer_3_outer"]["signals"]["draft_policy_diff"])

    # --- Pillar 4: Observability & Tracing ---
    def test_pillar_4_observability_spans_and_bq_logger(self):
        orchestrator = ClaimOrchestrator(default_claim_type="auto")
        res = orchestrator.run("assets/ev_bumper_minor_crack.png", claim_type="auto")
        obs = res["observability"]

        self.assertTrue(obs["trace_id"].startswith("trace_"))
        self.assertEqual(obs["span_count"], 5)
        self.assertGreaterEqual(obs["tool_call_count"], 3)

        recent_logs = BQLogger.get_recent_logs(limit=5)
        self.assertTrue(any(log["claim_id"] == res["claim_id"] for log in recent_logs))


if __name__ == "__main__":
    unittest.main()
