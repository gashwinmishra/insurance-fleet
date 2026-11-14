import json
import os
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ValidationError
from google.genai import types

from infrastructure.rag_retriever import RAGRetriever
from infrastructure.memory_store import MemoryStore


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAG_INSTANCE = RAGRetriever()
_MEMORY_INSTANCE = MemoryStore()

VALID_CLAIM_TYPES = ("auto", "home")
CANONICAL_DAMAGE_TYPES = (
    "scratch",
    "puncture",
    "crack",
    "dent",
    "scrape",
    "corrosion",
    "scuff",
    "abrasion",
    "chip",
    "granule_loss",
    "missing_material",
    "rust",
)


# --- Formal Pydantic & JSON Schema Declarations (Pillar 1: Tool & Interface Design) ---

class PolicyLookupArgs(BaseModel):
    claim_type: Literal["auto", "home"] = Field(
        ...,
        description="Insurance domain to query ('auto' for commercial EV fleet, 'home' for residential property).",
    )
    damage_type: str = Field(
        ...,
        min_length=2,
        description="Canonical damage category (e.g., 'scratch', 'puncture', 'crack', 'dent', 'granule_loss').",
    )


class InstitutionalKnowledgeQueryArgs(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        description="Search query combining detected object, damage type, and component location.",
    )
    claim_type: Literal["auto", "home"] = Field(
        default="auto",
        description="Domain filter ('auto' or 'home') used for authority & domain bonus scoring.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of ranked institutional knowledge rules to return (1-10).",
    )


class OverrideTelemetryQueryArgs(BaseModel):
    claim_type: Literal["auto", "home"] = Field(
        ...,
        description="Insurance domain ('auto' or 'home').",
    )
    damage_type: str = Field(
        ...,
        min_length=2,
        description="Canonical damage type to query in Episodic Memory.",
    )
    location: str = Field(
        default="",
        description="Specific component or structural location (e.g., 'battery_casing', 'rear_bumper_plastic').",
    )


class PolicyAmendmentProposalArgs(BaseModel):
    claim_type: Literal["auto", "home"] = Field(..., description="Insurance domain ('auto' or 'home').")
    clause_key: str = Field(..., min_length=2, description="Policy JSON key to amend (e.g., 'battery_damage').")
    current_rule: str = Field(..., min_length=2, description="Existing codified policy rule text.")
    proposed_rule: str = Field(..., min_length=2, description="Proposed evolved policy rule text.")
    justification: str = Field(..., min_length=5, description="Telemetry and safety/cost justification for the diff.")


class VisualFindingsSchema(BaseModel):
    object_detected: str = Field(..., description="Detected vehicle model, component, or home structure.")
    damage_type: str = Field(..., description="Canonical damage classification.")
    location: str = Field(..., description="Component or structural zone where damage is located.")
    severity: str = Field(..., description="Damage severity ('negligible', 'minor', 'moderate', 'critical').")
    depth_mm: float = Field(default=0.0, ge=0.0, description="Estimated damage depth in millimeters.")
    affected_area_percentage: int = Field(default=0, ge=0, le=100, description="Percentage of surface affected.")
    visual_evidence: str = Field(..., description="Concise forensic summary of visual evidence.")


class LayerDecisionSchema(BaseModel):
    outcome: str = Field(..., description="Adjudication decision code (e.g. 'APPROVE_REPAIR', 'TOTAL_LOSS').")
    reason: str = Field(..., description="Detailed justification citing policy clause or override rule.")
    source: str = Field(default="Official Policy", description="Rule authority source.")
    confidence: float = Field(default=0.95, ge=0.0, le=1.0, description="Confidence score in [0.0, 1.0].")


class RetrievedRuleSchema(BaseModel):
    author: str = Field(default="System", description="Author of the institutional rule.")
    source: str = Field(default="Knowledge Base", description="Channel or bulletin source.")
    content: str = Field(..., description="Full rule text.")


class Layer2ContextMetadataSchema(BaseModel):
    rag_override_reason: Optional[str] = Field(default=None, description="High-level override category.")
    retrieved_rules: List[RetrievedRuleSchema] = Field(default_factory=list, description="Matched RAG rules.")


class Layer2InnerSchema(BaseModel):
    decision: LayerDecisionSchema
    context_metadata: Layer2ContextMetadataSchema


class FrequencyStatsSchema(BaseModel):
    metric: str = Field(..., description="Name of the tracked slow-loop metric.")
    value: Any = Field(..., description="Observed metric value or count.")
    trend: str = Field(..., description="Directional trend indicator.")


class Layer3SignalsSchema(BaseModel):
    draft_policy_diff: Optional[str] = Field(default=None, description="Git-style unified diff for policy update.")
    frequency_stats: Optional[FrequencyStatsSchema] = Field(default=None, description="Historical override stats.")


class Layer3OuterSchema(BaseModel):
    decision: LayerDecisionSchema
    signals: Layer3SignalsSchema


class AdjudicationTraceSchema(BaseModel):
    layer_1_base: LayerDecisionSchema
    layer_2_inner: Layer2InnerSchema
    layer_3_outer: Layer3OuterSchema


# Formal JSON Schema Registry & Google GenAI FunctionDeclarations for LLM Tool Constraints
TOOL_JSON_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "lookup_official_policy": PolicyLookupArgs.model_json_schema(),
    "retrieve_institutional_knowledge": InstitutionalKnowledgeQueryArgs.model_json_schema(),
    "query_override_telemetry": OverrideTelemetryQueryArgs.model_json_schema(),
    "propose_policy_amendment": PolicyAmendmentProposalArgs.model_json_schema(),
    "visual_findings_response": VisualFindingsSchema.model_json_schema(),
    "adjudication_trace_response": AdjudicationTraceSchema.model_json_schema(),
}

ADK_TOOL_FUNCTION_DECLARATIONS: List[types.FunctionDeclaration] = [
    types.FunctionDeclaration(
        name="lookup_official_policy",
        description="Looks up codified official insurance policy terms, deductibles, and coverage limits for Layer 1 adjudication.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "claim_type": types.Schema(
                    type="STRING",
                    enum=["auto", "home"],
                    description="Insurance domain ('auto' or 'home').",
                ),
                "damage_type": types.Schema(
                    type="STRING",
                    description="Canonical damage category (e.g., 'scratch', 'puncture', 'crack', 'dent', 'granule_loss').",
                ),
            },
            required=["claim_type", "damage_type"],
        ),
    ),
    types.FunctionDeclaration(
        name="retrieve_institutional_knowledge",
        description="Retrieves ranked institutional knowledge rules (safety alerts, memos, vendor rules) for Layer 2 Fast Loop overrides.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "query": types.Schema(
                    type="STRING",
                    description="Search query combining object, damage type, and component location.",
                ),
                "claim_type": types.Schema(
                    type="STRING",
                    enum=["auto", "home"],
                    description="Insurance domain ('auto' or 'home').",
                ),
                "top_k": types.Schema(
                    type="INTEGER",
                    description="Maximum number of ranked rules to return (1-10).",
                ),
            },
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="query_override_telemetry",
        description="Queries historical human adjuster override frequency and cost/safety trends from Episodic Memory for Layer 3 Slow Loop.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "claim_type": types.Schema(
                    type="STRING",
                    enum=["auto", "home"],
                    description="Insurance domain ('auto' or 'home').",
                ),
                "damage_type": types.Schema(
                    type="STRING",
                    description="Canonical damage category.",
                ),
                "location": types.Schema(
                    type="STRING",
                    description="Component or surface location (e.g., 'battery_casing').",
                ),
            },
            required=["claim_type", "damage_type"],
        ),
    ),
    types.FunctionDeclaration(
        name="propose_policy_amendment",
        description="Generates a formal git-style policy amendment diff for Layer 3 Slow Loop policy evolution.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "claim_type": types.Schema(type="STRING", enum=["auto", "home"]),
                "clause_key": types.Schema(type="STRING", description="Policy clause key to amend."),
                "current_rule": types.Schema(type="STRING", description="Existing policy text."),
                "proposed_rule": types.Schema(type="STRING", description="Proposed policy text."),
                "justification": types.Schema(type="STRING", description="Telemetry justification."),
            },
            required=["claim_type", "clause_key", "current_rule", "proposed_rule", "justification"],
        ),
    ),
]

ADK_GENAI_TOOLS: List[types.Tool] = [
    types.Tool(function_declarations=ADK_TOOL_FUNCTION_DECLARATIONS)
]


def format_tool_error_with_recovery(
    tool_name: str,
    error_code: str,
    message: str,
    recovery_guidance: str,
    valid_options: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Formats a structured JSON tool error with actionable self-correction recovery guidance
    for the calling LLM agent.
    """
    payload: Dict[str, Any] = {
        "status": "ERROR",
        "tool_name": tool_name,
        "error_code": error_code,
        "message": message,
        "recovery_guidance": recovery_guidance,
    }
    if valid_options:
        payload["valid_options"] = valid_options
    return json.dumps(payload)


def _load_policy_file(claim_type: str) -> Dict[str, Any]:
    normalized = "home" if str(claim_type).lower().strip() == "home" else "auto"
    filename = "policy_home.json" if normalized == "home" else "policy_auto.json"
    filepath = os.path.join(_BASE_DIR, "data", filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


# --- ADK Agent Tools with Schema Validation & Actionable Recovery Guidance ---

def lookup_official_policy(claim_type: str, damage_type: str) -> str:
    """
    Looks up codified official insurance policy terms, deductibles, and coverage limits
    for a given claim_type ('auto' or 'home') and damage_type (e.g., 'scratch', 'granule_loss').
    Used in Layer 1 (Base Loop) adjudication.
    """
    norm_claim = str(claim_type or "").lower().strip()
    norm_damage = str(damage_type or "").lower().strip()

    if norm_claim not in VALID_CLAIM_TYPES:
        return format_tool_error_with_recovery(
            tool_name="lookup_official_policy",
            error_code="INVALID_CLAIM_TYPE",
            message=f"Unsupported claim_type '{claim_type}'.",
            recovery_guidance=(
                "ACTION REQUIRED: Re-invoke `lookup_official_policy` with `claim_type` set to "
                "either 'auto' (for vehicle/EV claims) or 'home' (for roof/property claims)."
            ),
            valid_options={"claim_type": list(VALID_CLAIM_TYPES), "damage_type": list(CANONICAL_DAMAGE_TYPES)},
        )

    if not norm_damage or len(norm_damage) < 2:
        return format_tool_error_with_recovery(
            tool_name="lookup_official_policy",
            error_code="MISSING_DAMAGE_TYPE",
            message="Parameter `damage_type` cannot be empty.",
            recovery_guidance=(
                "ACTION REQUIRED: Extract the primary damage category from Visual Findings "
                f"(one of {list(CANONICAL_DAMAGE_TYPES)}) and retry `lookup_official_policy`, "
                "or call `retrieve_institutional_knowledge` with a descriptive text query."
            ),
            valid_options={"damage_type": list(CANONICAL_DAMAGE_TYPES)},
        )

    policy = _load_policy_file(norm_claim)
    if not policy:
        return format_tool_error_with_recovery(
            tool_name="lookup_official_policy",
            error_code="POLICY_DATA_UNAVAILABLE",
            message=f"Official policy store for '{norm_claim}' could not be loaded.",
            recovery_guidance=(
                "FALLBACK ACTION: Proceed immediately to Layer 2 by calling "
                f"`retrieve_institutional_knowledge(query='{norm_damage}', claim_type='{norm_claim}')` "
                "and mark Layer 1 outcome as 'MANUAL_REVIEW' with confidence 0.50."
            ),
        )

    terms = policy.get("terms", {})
    deductibles = policy.get("deductibles", {})
    policy_version = policy.get("policy_type", "STANDARD_POLICY")

    matched_clauses: List[str] = []
    for clause_name, clause_text in terms.items():
        if norm_damage in clause_name.lower() or norm_damage in clause_text.lower():
            matched_clauses.append(f"- {clause_name}: {clause_text}")

    if not matched_clauses:
        matched_clauses = [f"- {k}: {v}" for k, v in terms.items()]

    return (
        f"Policy Version: {policy_version}\n"
        f"Deductibles: {json.dumps(deductibles)}\n"
        f"Applicable Terms:\n" + "\n".join(matched_clauses)
    )


def retrieve_institutional_knowledge(query: str, claim_type: str = "auto", top_k: int = 3) -> str:
    """
    Retrieves ranked institutional knowledge rules (safety alerts, Slack memos, vendor routing
    guidelines) matching the query using hybrid similarity scoring.
    Used in Layer 2 (Fast Loop) adjudication to determine if official policy should be overridden.
    """
    clean_query = str(query or "").strip()
    norm_claim = str(claim_type or "auto").lower().strip()

    if not clean_query or len(clean_query) < 2:
        return format_tool_error_with_recovery(
            tool_name="retrieve_institutional_knowledge",
            error_code="EMPTY_SEARCH_QUERY",
            message="Search query is empty or too short to compute similarity scores.",
            recovery_guidance=(
                "ACTION REQUIRED: Re-invoke `retrieve_institutional_knowledge` with a non-empty `query` "
                "combining `<object_detected> <damage_type> <location>` (e.g., 'EV battery scratch battery_casing')."
            ),
        )

    if norm_claim not in VALID_CLAIM_TYPES:
        norm_claim = "auto"

    try:
        safe_top_k = max(1, min(int(top_k), 10))
    except (ValueError, TypeError):
        safe_top_k = 3

    hits = _RAG_INSTANCE.retrieve(query=clean_query, claim_type=norm_claim, top_k=safe_top_k)
    if not hits:
        return format_tool_error_with_recovery(
            tool_name="retrieve_institutional_knowledge",
            error_code="NO_KNOWLEDGE_HITS",
            message=f"No institutional rules matched query '{clean_query}'.",
            recovery_guidance=(
                "RECOVERY GUIDANCE: Broaden your query to use canonical damage keywords "
                f"({list(CANONICAL_DAMAGE_TYPES)}) or retain the Layer 1 Base Policy decision without override."
            ),
        )

    results = []
    for hit in hits:
        score_str = f", Relevance: {hit.get('similarity_score', 0.0):.2f}" if "similarity_score" in hit else ""
        results.append(
            f"Rule [{hit.get('id', 'N/A')}]: {hit.get('content')} "
            f"(Author: {hit.get('author', 'System')}, Source: {hit.get('source')}, "
            f"Weight: {hit.get('weight', 0.8)}{score_str})"
        )
    return "\n".join(results)


def retrieve_knowledge(query: str) -> str:
    """
    Backwards-compatible alias for retrieve_institutional_knowledge.
    Retrieves institutional knowledge rules relevant to the query.
    """
    return retrieve_institutional_knowledge(query=query, claim_type="auto", top_k=3)


def query_override_telemetry(claim_type: str, damage_type: str, location: str = "") -> str:
    """
    Queries historical human override frequency and operational telemetry from Episodic Memory
    to determine whether a Fast Loop override should be codified into an official Policy Diff
    in Layer 3 (Slow Loop).
    """
    norm_claim = str(claim_type or "").lower().strip()
    dmg = str(damage_type or "").lower().strip()
    loc = str(location or "").lower().strip()

    if norm_claim not in VALID_CLAIM_TYPES or not dmg:
        return format_tool_error_with_recovery(
            tool_name="query_override_telemetry",
            error_code="INVALID_TELEMETRY_PARAMETERS",
            message=f"Invalid parameters claim_type='{claim_type}', damage_type='{damage_type}'.",
            recovery_guidance=(
                "ACTION REQUIRED: Retry `query_override_telemetry` with `claim_type` in ['auto', 'home'] "
                f"and a non-empty `damage_type` from {list(CANONICAL_DAMAGE_TYPES)}."
            ),
            valid_options={"claim_type": list(VALID_CLAIM_TYPES), "damage_type": list(CANONICAL_DAMAGE_TYPES)},
        )

    count = _MEMORY_INSTANCE.get_pattern_frequency(claim_type=norm_claim, damage_type=dmg, location=loc)

    if "battery" in loc and dmg == "scratch":
        return json.dumps({
            "status": "OK",
            "pattern_occurrences": count,
            "metric": "Manual Overrides",
            "value": str(count),
            "trend": "UP (+40%)",
            "recommendation": "Codify ZERO TOLERANCE for battery casing scratches into v2.1 policy.",
        })
    if dmg == "crack" and "bumper" in loc:
        return json.dumps({
            "status": "OK",
            "pattern_occurrences": count,
            "metric": "Avg Repair Savings",
            "value": "$1,200",
            "trend": "SAVED (98% Weld Success)",
            "recommendation": "Codify PLASTIC_WELD repair for bumper cracks < 3 inches.",
        })
    if dmg == "dent":
        return json.dumps({
            "status": "OK",
            "pattern_occurrences": count,
            "metric": "Cycle Time Reduction",
            "value": "4 Days",
            "trend": "FASTER",
            "recommendation": "Route dents < 50mm to Mobile PDR vendor in official policy.",
        })
    if dmg == "granule_loss":
        return json.dumps({
            "status": "OK",
            "pattern_occurrences": count,
            "metric": "Re-work Cost Avoidance",
            "value": "$74,500",
            "trend": "HIGH_RISK_PATCH",
            "recommendation": "Lower roof patch threshold from 20% to 15% granule loss.",
        })
    if dmg == "missing_material":
        return json.dumps({
            "status": "OK",
            "pattern_occurrences": count,
            "metric": "Storm Claims Vol",
            "value": "2,400",
            "trend": "PEAK",
            "recommendation": "Add catastrophe Tab Sealing clause for minor wind shingle loss.",
        })

    return json.dumps({
        "status": "OK",
        "pattern_occurrences": count,
        "metric": "Historical Overrides",
        "value": str(count),
        "trend": "STABLE",
        "recommendation": "Continue monitoring override frequency before amending base policy.",
    })


def propose_policy_amendment(
    claim_type: str,
    clause_key: str,
    current_rule: str,
    proposed_rule: str,
    justification: str,
) -> str:
    """
    Generates a standardized git-style policy amendment diff for Layer 3 (Slow Loop)
    when recurring institutional overrides indicate the base policy is outdated.
    """
    try:
        validated = PolicyAmendmentProposalArgs(
            claim_type="home" if str(claim_type).lower().strip() == "home" else "auto",
            clause_key=str(clause_key).strip(),
            current_rule=str(current_rule).strip(),
            proposed_rule=str(proposed_rule).strip(),
            justification=str(justification).strip(),
        )
    except ValidationError as exc:
        return format_tool_error_with_recovery(
            tool_name="propose_policy_amendment",
            error_code="SCHEMA_VALIDATION_FAILED",
            message=f"Amendment arguments failed schema validation: {exc.errors()}",
            recovery_guidance=(
                "ACTION REQUIRED: Ensure `clause_key`, `current_rule`, `proposed_rule` (min 2 chars), "
                "and `justification` (min 5 chars) are all non-empty strings, then re-invoke `propose_policy_amendment`."
            ),
        )

    clean_key = validated.clause_key.replace('"', "")
    clean_curr = validated.current_rule.replace('"', "'")
    clean_prop = validated.proposed_rule.replace('"', "'")
    diff_str = f'- "{clean_key}": "{clean_curr}"\n+ "{clean_key}": "{clean_prop}"'
    return json.dumps({
        "status": "OK",
        "claim_type": validated.claim_type,
        "clause_key": clean_key,
        "draft_policy_diff": diff_str,
        "justification": validated.justification,
    })


# --- Interface & Contract Schema Validators ---

def validate_visual_findings(raw_findings: Any, claim_type: str = "auto") -> Dict[str, Any]:
    """
    Validates and normalizes the Visual Agent output contract using `VisualFindingsSchema`.
    """
    if not isinstance(raw_findings, dict):
        raw_findings = {"visual_evidence": str(raw_findings)}

    candidate = {
        "object_detected": str(raw_findings.get("object_detected") or ("Vehicle" if claim_type == "auto" else "Dwelling")),
        "damage_type": str(raw_findings.get("damage_type") or "unknown").lower(),
        "location": str(raw_findings.get("location") or ("roof_surface" if claim_type == "home" else "exterior")),
        "severity": str(raw_findings.get("severity") or "moderate").lower(),
        "depth_mm": max(0.0, float(raw_findings.get("depth_mm") or 0.0)),
        "affected_area_percentage": max(0, min(100, int(raw_findings.get("affected_area_percentage") or 0))),
        "visual_evidence": str(raw_findings.get("visual_evidence") or "Visual inspection completed."),
    }
    model = VisualFindingsSchema(**candidate)
    return model.model_dump()


def validate_adjudication_trace(raw_trace: Any) -> Dict[str, Any]:
    """
    Validates and normalizes the 3-Layer Adjudication Trace schema so UI rendering and
    BigQuery logging never encounter missing keys or invalid types.
    """
    if not isinstance(raw_trace, dict):
        raw_trace = {}

    l1 = raw_trace.get("layer_1_base") if isinstance(raw_trace.get("layer_1_base"), dict) else {}
    l2 = raw_trace.get("layer_2_inner") if isinstance(raw_trace.get("layer_2_inner"), dict) else {}
    l3 = raw_trace.get("layer_3_outer") if isinstance(raw_trace.get("layer_3_outer"), dict) else {}

    l1_norm = {
        "outcome": str(l1.get("outcome", "MANUAL_REVIEW")),
        "reason": str(l1.get("reason", "Evaluated against base policy.")),
        "source": str(l1.get("source", "Official Policy")),
        "confidence": float(l1.get("confidence", 0.90)),
    }

    l2_dec = l2.get("decision") if isinstance(l2.get("decision"), dict) else {}
    l2_meta = l2.get("context_metadata") if isinstance(l2.get("context_metadata"), dict) else {}
    l2_norm = {
        "decision": {
            "outcome": str(l2_dec.get("outcome", l1_norm["outcome"])),
            "reason": str(l2_dec.get("reason", l1_norm["reason"])),
            "source": str(l2_dec.get("source", l1_norm["source"])),
            "confidence": float(l2_dec.get("confidence", l1_norm["confidence"])),
        },
        "context_metadata": {
            "rag_override_reason": l2_meta.get("rag_override_reason"),
            "retrieved_rules": (
                l2_meta.get("retrieved_rules")
                if isinstance(l2_meta.get("retrieved_rules"), list)
                else []
            ),
        },
    }

    l3_dec = l3.get("decision") if isinstance(l3.get("decision"), dict) else {}
    l3_sig = l3.get("signals") if isinstance(l3.get("signals"), dict) else {}
    l3_norm = {
        "decision": {
            "outcome": str(l3_dec.get("outcome", l2_norm["decision"]["outcome"])),
            "reason": str(l3_dec.get("reason", l2_norm["decision"]["reason"])),
            "source": str(l3_dec.get("source", "Evolved Policy Engine")),
            "confidence": float(l3_dec.get("confidence", l2_norm["decision"]["confidence"])),
        },
        "signals": {
            "draft_policy_diff": l3_sig.get("draft_policy_diff"),
            "frequency_stats": (
                l3_sig.get("frequency_stats")
                if isinstance(l3_sig.get("frequency_stats"), dict)
                else None
            ),
        },
    }

    return {
        "layer_1_base": l1_norm,
        "layer_2_inner": l2_norm,
        "layer_3_outer": l3_norm,
    }
