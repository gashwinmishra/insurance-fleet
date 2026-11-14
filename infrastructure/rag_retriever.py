import asyncio
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class RAGRetriever:
    """
    Context & Memory Engine for Institutional Knowledge (Fast Loop RAG).
    Scores institutional knowledge rules using hybrid lexical token overlap,
    domain synonym expansion, and authority weight boosting.
    Supports both synchronous (`retrieve`, `add_knowledge_rule`) and non-blocking
    asynchronous (`aretrieve`, `aadd_knowledge_rule`) operations.
    """

    SYNONYM_MAP = {
        "scratch": {"scratch", "scratches", "scrape", "scuff", "abrasion", "casing", "battery"},
        "puncture": {"puncture", "puncturing", "hole", "leak", "hazmat", "battery", "hv"},
        "crack": {"crack", "cracks", "bumper", "plastic", "welding", "fascia"},
        "dent": {"dent", "dents", "ding", "dings", "hail", "pdr", "door", "hood", "frunk"},
        "corrosion": {"corrosion", "rust", "oxidation", "hv", "connector", "flashing", "chimney"},
        "granule_loss": {"granule", "shingles", "asphalt", "roof", "patch", "replacement"},
        "missing_material": {"missing", "shingles", "wind", "windstorm", "slope", "tab", "sealing"},
        "chip": {"chip", "windshield", "glass", "adas", "calibration", "sensor"},
        "scuff": {"scuff", "paint", "transfer", "buff", "polish", "cosmetic", "fender"},
        "abrasion": {"abrasion", "curb", "rash", "wheel", "turbine", "refinished"},
        "rust": {"rust", "flashing", "chimney", "leak", "reseal", "water"},
    }

    def __init__(self, knowledge_path: str = "data/institutional_knowledge.json"):
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.knowledge_path = os.path.join(base_path, os.path.normpath(knowledge_path))
        self._knowledge_base: List[Dict[str, Any]] = self._load_knowledge()
        self._async_lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    def _load_knowledge(self) -> List[Dict[str, Any]]:
        try:
            with open(self.knowledge_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except FileNotFoundError:
            return []
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _tokenize(text: str) -> set:
        if not text:
            return set()
        tokens = set(re.findall(r"[a-z0-9_]+", text.lower()))
        expanded = set(tokens)
        for token in tokens:
            if token in RAGRetriever.SYNONYM_MAP:
                expanded.update(RAGRetriever.SYNONYM_MAP[token])
        return expanded

    def score_rule(self, query_tokens: set, rule: Dict[str, Any], claim_type: Optional[str] = None) -> float:
        """Computes a normalized relevance score in [0.0, 1.0] for a knowledge rule."""
        if not query_tokens:
            return float(rule.get("weight", 0.5))

        searchable_text = " ".join([
            str(rule.get("content", "")),
            str(rule.get("context", "")),
            str(rule.get("source", "")),
        ])
        rule_tokens = self._tokenize(searchable_text)
        if not rule_tokens:
            return 0.0

        overlap = query_tokens.intersection(rule_tokens)
        if not overlap:
            return 0.0

        lexical_score = len(overlap) / max(len(query_tokens), 1)
        authority_weight = float(rule.get("weight", 0.8))

        domain_bonus = 0.0
        source_lower = str(rule.get("source", "")).lower()
        if claim_type == "home" and ("home" in source_lower or "catastrophe" in source_lower):
            domain_bonus = 0.15
        elif claim_type == "auto" and any(k in source_lower for k in ("auto", "ev", "hv", "vendor", "tech")):
            domain_bonus = 0.15

        combined = (lexical_score * 0.65) + (authority_weight * 0.20) + domain_bonus
        return round(min(combined, 1.0), 4)

    def retrieve(
        self,
        query: Optional[str] = None,
        claim_type: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.15,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant institutional knowledge rules ranked by hybrid similarity score.
        Falls back to returning all rules if query is empty so legacy callers remain supported.
        """
        if not query or not str(query).strip():
            return list(self._knowledge_base)

        query_tokens = self._tokenize(str(query))
        scored_hits: List[Dict[str, Any]] = []

        for rule in self._knowledge_base:
            score = self.score_rule(query_tokens, rule, claim_type=claim_type)
            if score >= min_score:
                enriched = dict(rule)
                enriched["similarity_score"] = score
                scored_hits.append(enriched)

        scored_hits.sort(key=lambda r: (r.get("similarity_score", 0.0), r.get("weight", 0.0)), reverse=True)

        if scored_hits:
            return scored_hits[:top_k]

        fallback = sorted(self._knowledge_base, key=lambda r: r.get("weight", 0.0), reverse=True)
        return fallback[:top_k]

    async def aretrieve(
        self,
        query: Optional[str] = None,
        claim_type: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.15,
    ) -> List[Dict[str, Any]]:
        """Asynchronous non-blocking RAG retrieval to prevent UI/event-loop blocking."""
        return await asyncio.to_thread(self.retrieve, query, claim_type, top_k, min_score)

    def add_knowledge_rule(
        self,
        content: str,
        author: str = "Human Adjuster (HITL)",
        source: str = "Adjuster Override Feedback",
        context: str = "Live Claim Override",
        weight: float = 0.90,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Adds a new rule from human feedback into the institutional knowledge base."""
        new_rule = {
            "id": f"MSG-{len(self._knowledge_base) + 9000}",
            "source": source,
            "author": author,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context,
            "content": content,
            "weight": round(float(weight), 2),
        }
        self._knowledge_base.append(new_rule)
        if persist:
            try:
                with open(self.knowledge_path, "w", encoding="utf-8") as f:
                    json.dump(self._knowledge_base, f, indent=2)
            except OSError:
                pass
        return new_rule

    async def aadd_knowledge_rule(
        self,
        content: str,
        author: str = "Human Adjuster (HITL)",
        source: str = "Adjuster Override Feedback",
        context: str = "Live Claim Override",
        weight: float = 0.90,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Asynchronous non-blocking rule insertion."""
        async with self._get_lock():
            return await asyncio.to_thread(
                self.add_knowledge_rule, content, author, source, context, weight, persist
            )
