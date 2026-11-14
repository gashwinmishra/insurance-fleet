import asyncio
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from infrastructure.observability import PIIRedactor


class HistoryCompactor:
    """
    Semantic & Token-Budget History Compaction Engine (Pillar 2: Context & Memory).
    Replaces rudimentary array truncation with token-aware rolling semantic summarization.
    Distills older conversational turns and episodic claims into structured memory digests
    preserving key entities, overrides, risk flags, and decision rationales while keeping
    recent turns verbatim within the context window budget.
    """

    DEFAULT_TOKEN_BUDGET = 1200
    RECENT_VERBATIM_WINDOW = 4

    @staticmethod
    def estimate_tokens(payload: Any) -> int:
        """Estimates token count (~4 chars per token heuristic)."""
        text = json.dumps(payload) if not isinstance(payload, str) else payload
        return max(1, len(text) // 4)

    @classmethod
    def compact_session_events(
        cls,
        events: List[Dict[str, Any]],
        max_tokens: int = DEFAULT_TOKEN_BUDGET,
        keep_recent: int = RECENT_VERBATIM_WINDOW,
    ) -> Dict[str, Any]:
        """
        Compacts conversational session events when token count or turn count exceeds budget.
        Extracts semantic facts (entities, damage types, outcomes, overrides) from older turns
        into a structured summary digest and retains the most recent `keep_recent` turns verbatim.
        """
        total_tokens_before = sum(cls.estimate_tokens(e) for e in events)
        if len(events) <= keep_recent and total_tokens_before <= max_tokens:
            return {
                "compacted": False,
                "summary_digest": None,
                "retained_events": list(events),
                "tokens_before": total_tokens_before,
                "tokens_after": total_tokens_before,
            }

        older_events = events[:-keep_recent] if keep_recent > 0 else events
        recent_events = events[-keep_recent:] if keep_recent > 0 else []

        entities: List[str] = []
        decisions: List[str] = []
        stages_completed: List[str] = []

        for ev in older_events:
            content = ev.get("content", {})
            if isinstance(content, dict):
                if "stage" in content:
                    stages_completed.append(str(content["stage"]))
                if "object_detected" in content or "damage_type" in content:
                    entities.append(
                        f"{content.get('object_detected', 'Asset')}:{content.get('damage_type', 'unknown')}@{content.get('location', 'general')}"
                    )
                if "outcome" in content:
                    decisions.append(
                        f"{content.get('outcome')} (conf={content.get('confidence', 'N/A')})"
                    )
            elif isinstance(content, str):
                decisions.append(content[:80])

        digest_text = (
            f"Compacted {len(older_events)} prior conversational turns. "
            f"Stages: {', '.join(sorted(set(stages_completed))) or 'triage'}. "
            f"Entities inspected: {', '.join(sorted(set(entities))) or 'none'}. "
            f"Prior outcomes: {', '.join(decisions) or 'in-progress'}."
        )

        summary_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": "system_memory_compactor",
            "content": {
                "type": "COMPACTED_SESSION_DIGEST",
                "compacted_turn_count": len(older_events),
                "entities": sorted(set(entities)),
                "decisions": decisions,
                "narrative_summary": digest_text,
            },
        }

        compacted_list = [summary_event] + recent_events
        tokens_after = sum(cls.estimate_tokens(e) for e in compacted_list)

        return {
            "compacted": True,
            "summary_digest": summary_event["content"],
            "retained_events": compacted_list,
            "tokens_before": total_tokens_before,
            "tokens_after": tokens_after,
        }

    @classmethod
    def compact_episodic_claims(
        cls,
        claims: List[Dict[str, Any]],
        max_active_claims: int = 50,
        keep_recent: int = 25,
    ) -> Dict[str, Any]:
        """
        Compacts historical episodic claims into aggregated domain/outcome digests
        rather than discarding history via naive array slicing.
        """
        if len(claims) <= max_active_claims:
            return {"compacted": False, "digest": None, "retained_claims": claims}

        to_summarize = claims[:-keep_recent]
        retained = claims[-keep_recent:]

        outcome_distribution: Dict[str, int] = {}
        domain_distribution: Dict[str, int] = {}
        for item in to_summarize:
            out = str(item.get("outcome", "UNKNOWN"))
            dom = str(item.get("claim_type", "auto"))
            outcome_distribution[out] = outcome_distribution.get(out, 0) + 1
            domain_distribution[dom] = domain_distribution.get(dom, 0) + 1

        digest = {
            "compacted_at": datetime.now(timezone.utc).isoformat(),
            "archived_claim_count": len(to_summarize),
            "outcome_distribution": outcome_distribution,
            "domain_distribution": domain_distribution,
        }
        return {
            "compacted": True,
            "digest": digest,
            "retained_claims": retained,
        }


class MemoryStore:
    """
    Two-Tier Persistent & Asynchronous Context & Memory Manager (Pillar 2: Context & Memory):
    1. Durable Conversational Session Memory: Persisted in SQLite (`data/sessions_memory.db`)
       and mirrored in `data/claims_db.json` so conversational session state survives process
       restarts. Includes automatic token-budget semantic compaction via `HistoryCompactor`.
    2. Long-Term Episodic Memory: Persists adjudicated claims, human-in-the-loop (HITL)
       overrides, compacted historical digests, and pattern frequencies.
    3. Full Asynchronous (`async`/`await`) API: Non-blocking `aappend_session_event`,
       `aget_session_history`, `acompact_session_history`, `arecord_claim`, and
       `arecord_human_override` methods using `asyncio.Lock` and `asyncio.to_thread`.
    """

    def __init__(
        self,
        db_path: str = "data/claims_db.json",
        sqlite_path: Optional[str] = None,
    ):
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if os.path.isabs(db_path):
            self.db_path = db_path
        else:
            self.db_path = os.path.join(base_path, os.path.normpath(db_path))

        if sqlite_path:
            self.sqlite_path = (
                sqlite_path if os.path.isabs(sqlite_path) else os.path.join(base_path, sqlite_path)
            )
        else:
            self.sqlite_path = os.path.join(os.path.dirname(self.db_path), "sessions_memory.db")

        self._async_lock: Optional[asyncio.Lock] = None
        self._init_sqlite()
        self._episodic_db: Dict[str, Any] = self._load_db()

    def _get_lock(self) -> asyncio.Lock:
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    def _init_sqlite(self) -> None:
        """Initializes durable SQLite storage for conversational session state and summaries."""
        try:
            os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
            with closing(sqlite3.connect(self.sqlite_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversational_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content_json TEXT NOT NULL,
                        token_estimate INTEGER NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_session_id
                    ON conversational_sessions(session_id)
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_summaries (
                        session_id TEXT PRIMARY KEY,
                        compacted_at TEXT NOT NULL,
                        summary_json TEXT NOT NULL,
                        tokens_before INTEGER NOT NULL,
                        tokens_after INTEGER NOT NULL
                    )
                    """
                )
                conn.commit()
        except sqlite3.Error:
            pass

    def _load_db(self) -> Dict[str, Any]:
        default_state: Dict[str, Any] = {
            "claims": [],
            "compacted_claim_digests": [],
            "human_overrides": [],
            "persistent_sessions": {},
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
                    for key, default_val in default_state.items():
                        if key not in data:
                            data[key] = default_val
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

    # --- Persistent Conversational Session State (Synchronous & Async) ---

    def append_session_event(
        self,
        session_id: str,
        role: str,
        content: Any,
        auto_compact_threshold: int = 8,
    ) -> Dict[str, Any]:
        """
        Persists a conversational turn to durable SQLite + JSON storage (with PII redaction)
        and triggers semantic compaction if the session exceeds `auto_compact_threshold` turns.
        """
        sanitized_content, _ = PIIRedactor.redact(content)
        now_iso = datetime.now(timezone.utc).isoformat()
        token_est = HistoryCompactor.estimate_tokens(sanitized_content)
        event = {
            "timestamp": now_iso,
            "role": role,
            "content": sanitized_content,
            "token_estimate": token_est,
        }

        # 1. Persist to SQLite durable session table
        try:
            with closing(sqlite3.connect(self.sqlite_path)) as conn:
                conn.execute(
                    """
                    INSERT INTO conversational_sessions (session_id, timestamp, role, content_json, token_estimate)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, now_iso, role, json.dumps(sanitized_content), token_est),
                )
                conn.commit()
        except sqlite3.Error:
            pass

        # 2. Mirror in persistent JSON session store & run semantic compaction if needed
        sessions_map = self._episodic_db.setdefault("persistent_sessions", {})
        session_events = sessions_map.setdefault(session_id, [])
        session_events.append(event)

        if len(session_events) > auto_compact_threshold:
            self.compact_session_history(session_id)
        else:
            self._save_db()

        return event

    async def aappend_session_event(
        self,
        session_id: str,
        role: str,
        content: Any,
        auto_compact_threshold: int = 8,
    ) -> Dict[str, Any]:
        """Asynchronous non-blocking conversational session event persistence."""
        async with self._get_lock():
            return await asyncio.to_thread(
                self.append_session_event,
                session_id,
                role,
                content,
                auto_compact_threshold,
            )

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Loads conversational session history from durable SQLite storage, falling back
        to persistent JSON storage so session state survives across process restarts.
        """
        persisted_sessions = self._episodic_db.get("persistent_sessions", {})
        if session_id in persisted_sessions and persisted_sessions[session_id]:
            return list(persisted_sessions[session_id])

        try:
            with closing(sqlite3.connect(self.sqlite_path)) as conn:
                cursor = conn.execute(
                    """
                    SELECT timestamp, role, content_json, token_estimate
                    FROM conversational_sessions
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (session_id,),
                )
                rows = cursor.fetchall()
                if rows:
                    return [
                        {
                            "timestamp": r[0],
                            "role": r[1],
                            "content": json.loads(r[2]),
                            "token_estimate": r[3],
                        }
                        for r in rows
                    ]
        except sqlite3.Error:
            pass
        return []

    async def aget_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Asynchronous non-blocking session history retrieval."""
        async with self._get_lock():
            return await asyncio.to_thread(self.get_session_history, session_id)

    def compact_session_history(
        self,
        session_id: str,
        max_tokens: int = 800,
        keep_recent: int = 3,
    ) -> Dict[str, Any]:
        """
        Executes semantic compaction on a session's history, persisting the compacted
        digest to both SQLite (`session_summaries`) and `claims_db.json`.
        """
        events = self.get_session_history(session_id)
        compaction_result = HistoryCompactor.compact_session_events(
            events, max_tokens=max_tokens, keep_recent=keep_recent
        )
        if compaction_result["compacted"]:
            sessions_map = self._episodic_db.setdefault("persistent_sessions", {})
            sessions_map[session_id] = compaction_result["retained_events"]
            self._save_db()

            try:
                with closing(sqlite3.connect(self.sqlite_path)) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO session_summaries
                        (session_id, compacted_at, summary_json, tokens_before, tokens_after)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            datetime.now(timezone.utc).isoformat(),
                            json.dumps(compaction_result["summary_digest"]),
                            compaction_result["tokens_before"],
                            compaction_result["tokens_after"],
                        ),
                    )
                    conn.commit()
            except sqlite3.Error:
                pass

        return compaction_result

    async def acompact_session_history(
        self,
        session_id: str,
        max_tokens: int = 800,
        keep_recent: int = 3,
    ) -> Dict[str, Any]:
        """Asynchronous non-blocking semantic session history compaction."""
        async with self._get_lock():
            return await asyncio.to_thread(
                self.compact_session_history, session_id, max_tokens, keep_recent
            )

    # --- Long-Term Episodic Memory (Synchronous & Async) ---

    def record_claim(
        self,
        claim_id: str,
        claim_type: str,
        findings: Dict[str, Any],
        decision_trace: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Saves an adjudicated claim into episodic memory and runs semantic compaction when needed."""
        clean_findings, _ = PIIRedactor.redact(findings)
        damage_type = str(clean_findings.get("damage_type", "unknown")).lower()
        location = str(clean_findings.get("location", "roof" if claim_type == "home" else "general")).lower()
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

        # Semantic compaction of episodic claim history (instead of naive truncation)
        compaction = HistoryCompactor.compact_episodic_claims(claims_list, max_active_claims=60, keep_recent=30)
        if compaction["compacted"]:
            self._episodic_db["claims"] = compaction["retained_claims"]
            digests = self._episodic_db.setdefault("compacted_claim_digests", [])
            digests.append(compaction["digest"])

        self._save_db()
        return record

    async def arecord_claim(
        self,
        claim_id: str,
        claim_type: str,
        findings: Dict[str, Any],
        decision_trace: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Asynchronous non-blocking episodic claim recording."""
        async with self._get_lock():
            return await asyncio.to_thread(
                self.record_claim,
                claim_id,
                claim_type,
                findings,
                decision_trace,
                trace_id,
            )

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
        """Persists a PII-scrubbed Human-in-the-Loop (HITL) override for continuous system learning."""
        clean_notes, _ = PIIRedactor.redact_text(adjuster_notes)
        override_entry = {
            "claim_id": claim_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "claim_type": claim_type,
            "damage_type": damage_type,
            "original_outcome": original_outcome,
            "corrected_outcome": corrected_outcome,
            "adjuster_notes": clean_notes,
            "adjuster_name": adjuster_name,
        }
        overrides = self._episodic_db.setdefault("human_overrides", [])
        overrides.append(override_entry)

        pattern_key = f"{claim_type}:{damage_type.lower()}:override"
        counts = self._episodic_db.setdefault("pattern_counts", {})
        counts[pattern_key] = int(counts.get(pattern_key, 0)) + 1

        self._save_db()
        return override_entry

    async def arecord_human_override(
        self,
        claim_id: str,
        claim_type: str,
        damage_type: str,
        original_outcome: str,
        corrected_outcome: str,
        adjuster_notes: str,
        adjuster_name: str = "Senior Adjuster",
    ) -> Dict[str, Any]:
        """Asynchronous non-blocking HITL override persistence."""
        async with self._get_lock():
            return await asyncio.to_thread(
                self.record_human_override,
                claim_id,
                claim_type,
                damage_type,
                original_outcome,
                corrected_outcome,
                adjuster_notes,
                adjuster_name,
            )

    def get_pattern_frequency(self, claim_type: str, damage_type: str, location: str = "") -> int:
        """Returns historical occurrence/override count for a damage pattern."""
        counts = self._episodic_db.get("pattern_counts", {})
        exact_key = f"{claim_type}:{damage_type.lower()}:{location.lower()}"
        if exact_key in counts:
            return int(counts[exact_key])

        prefix = f"{claim_type}:{damage_type.lower()}:"
        total = sum(int(v) for k, v in counts.items() if k.startswith(prefix))
        return total if total > 0 else 1

    async def aget_pattern_frequency(self, claim_type: str, damage_type: str, location: str = "") -> int:
        """Asynchronous pattern frequency lookup."""
        return await asyncio.to_thread(self.get_pattern_frequency, claim_type, damage_type, location)

    def get_recent_claims(self, limit: int = 10) -> List[Dict[str, Any]]:
        claims = self._episodic_db.get("claims", [])
        return list(reversed(claims[-limit:]))

    def get_recent_overrides(self, limit: int = 10) -> List[Dict[str, Any]]:
        overrides = self._episodic_db.get("human_overrides", [])
        return list(reversed(overrides[-limit:]))
