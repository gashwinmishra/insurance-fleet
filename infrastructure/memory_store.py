import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class MemoryStore:
    """
    Two-Tier Context & Memory Manager:
    1. Short-Term Session Memory: Active conversation/session context per claim run.
    2. Long-Term Episodic Memory: Persistent store (`data/claims_db.json`) tracking adjudicated
       claims, human-in-the-loop (HITL) overrides, and pattern frequencies that drive Layer 3
       (Slow Loop) policy evolution.
    """

    def __init__(self, db_path: str = "data/claims_db.json"):
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(base_path, os.path.normpath(db_path))
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}
        self._episodic_db: Dict[str, Any] = self._load_db()

    def _load_db(self) -> Dict[str, Any]:
        default_state = {
            "claims": [],
            "human_overrides": [],
            "pattern_counts": {
                "auto:scratch:battery_casing": 15,
                "auto:crack:rear_bumper_plastic": 24,
                "auto:dent:side_door_panel": 42,
                "home:granule_loss:roof": 19,
                "home:missing_material:roof": 31,
            },
        }
        if not os.path.exists(self.db_path):
            return default_state
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "claims" in data:
                    # Merge defaults for pattern_counts if missing
                    if "pattern_counts" not in data:
                        data["pattern_counts"] = default_state["pattern_counts"]
                    if "human_overrides" not in data:
                        data["human_overrides"] = []
                    return data
                return default_state
        except (OSError, json.JSONDecodeError):
            return default_state

    def _save_db(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self._episodic_db, f, indent=2)
        except OSError:
            pass

    # --- Short-Term Session Context ---
    def append_session_event(self, session_id: str, role: str, content: Any) -> None:
        """Records an interaction step in short-term session memory."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "content": content,
        })

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Returns the short-term conversation/event history for a given session."""
        return list(self._sessions.get(session_id, []))

    # --- Long-Term Episodic Memory ---
    def record_claim(
        self,
        claim_id: str,
        claim_type: str,
        findings: Dict[str, Any],
        decision_trace: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Saves an adjudicated claim into episodic memory and updates pattern counters."""
        damage_type = str(findings.get("damage_type", "unknown")).lower()
        location = str(findings.get("location", "roof" if claim_type == "home" else "general")).lower()
        pattern_key = f"{claim_type}:{damage_type}:{location}"

        counts = self._episodic_db.setdefault("pattern_counts", {})
        counts[pattern_key] = int(counts.get(pattern_key, 0)) + 1

        layer_2_outcome = (
            decision_trace.get("layer_2_inner", {}).get("decision", {}).get("outcome", "UNKNOWN")
        )
        record = {
            "claim_id": claim_id,
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "claim_type": claim_type,
            "damage_type": damage_type,
            "location": location,
            "outcome": layer_2_outcome,
            "pattern_key": pattern_key,
        }
        claims_list = self._episodic_db.setdefault("claims", [])
        claims_list.append(record)
        # Keep bounded history in JSON file
        if len(claims_list) > 200:
            self._episodic_db["claims"] = claims_list[-200:]

        self._save_db()
        return record

    def record_human_override(
        self,
        claim_id: str,
        claim_type: str,
        damage_type: str,
        original_outcome: str,
        corrected_outcome: str,
        adjuster_notes: str,
        adjuster_name: str = "Senior Adjuster",
    ) -> Dict[str, Any]:
        """Persists a Human-in-the-Loop (HITL) override for continuous system learning."""
        override_entry = {
            "claim_id": claim_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "claim_type": claim_type,
            "damage_type": damage_type,
            "original_outcome": original_outcome,
            "corrected_outcome": corrected_outcome,
            "adjuster_notes": adjuster_notes,
            "adjuster_name": adjuster_name,
        }
        overrides = self._episodic_db.setdefault("human_overrides", [])
        overrides.append(override_entry)

        # Increment pattern count for slow-loop reinforcement
        pattern_key = f"{claim_type}:{damage_type.lower()}:override"
        counts = self._episodic_db.setdefault("pattern_counts", {})
        counts[pattern_key] = int(counts.get(pattern_key, 0)) + 1

        self._save_db()
        return override_entry

    def get_pattern_frequency(self, claim_type: str, damage_type: str, location: str = "") -> int:
        """Returns historical occurrence/override count for a damage pattern."""
        counts = self._episodic_db.get("pattern_counts", {})
        exact_key = f"{claim_type}:{damage_type.lower()}:{location.lower()}"
        if exact_key in counts:
            return int(counts[exact_key])

        # Prefix match fallback
        prefix = f"{claim_type}:{damage_type.lower()}:"
        total = sum(int(v) for k, v in counts.items() if k.startswith(prefix))
        return total if total > 0 else 1

    def get_recent_claims(self, limit: int = 10) -> List[Dict[str, Any]]:
        claims = self._episodic_db.get("claims", [])
        return list(reversed(claims[-limit:]))

    def get_recent_overrides(self, limit: int = 10) -> List[Dict[str, Any]]:
        overrides = self._episodic_db.get("human_overrides", [])
        return list(reversed(overrides[-limit:]))
