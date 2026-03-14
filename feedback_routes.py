"""
Survey feedback routes for AVIRA response evaluation.
Two forms: per-response feedback and periodic pattern checks.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth import get_current_user, get_supabase

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


# --- Request models ---

class ResponseFeedback(BaseModel):
    session_id: str
    response_number: int
    tester_role: str | None = None
    question_topic: str | None = None
    difficulty_tier: str = Field(
        ..., description="easy | medium | hard | very_complex"
    )
    q1_relevance: str = Field(
        ...,
        description=(
            "different_question | partial | exact | anticipated_followup"
        ),
    )
    q2_scientific_accuracy: str = Field(
        ...,
        description=(
            "factually_wrong | correct_direction_wrong_details | "
            "correct_and_complete | exceptional"
        ),
    )
    q3_depth: str = Field(
        ...,
        description=(
            "needed_more_equations | right_depth | phd_level"
        ),
    )
    q4_length_format: str = Field(
        ...,
        description=(
            "way_too_long | slightly_long | just_right | "
            "too_short | ignored_format"
        ),
    )
    q5_critical_content: str = Field(
        ...,
        description=(
            "all_present | mostly_present | core_missing"
        ),
    )
    q7_comment: str = Field(
        ..., min_length=1, description="Mandatory comment — cannot be blank"
    )
    quick_score: int = Field(..., ge=1, le=10)
    conversation_id: str | None = None
    message_id: str | None = None


class PatternCheck(BaseModel):
    session_id: str
    checkpoint: int = Field(..., description="5, 10, or 15")
    strengths: list[str] = Field(
        default_factory=list,
        description=(
            "conceptual_explanations | math_derivations | "
            "industrial_examples | regulatory_context | "
            "calculations | format_compliance | no_strength_yet"
        ),
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description=(
            "quantitative_depth | specific_factual_recall | "
            "answering_exactly_asked | brief_vs_detailed | "
            "confidence_accuracy_mismatch | no_weakness_yet"
        ),
    )
    confidence_vs_accuracy: str = Field(
        ..., description="concern | minor | no_issue"
    )
    comparison_to_usual_tool: str = Field(
        ..., description="better | same | worse | mixed"
    )
    comparison_detail: str | None = None


# --- Endpoints ---

@router.post("")
def submit_response_feedback(
    req: ResponseFeedback,
    user: dict = Depends(get_current_user),
):
    sb = get_supabase()
    row = {
        "tester_id": user["user_id"],
        "session_id": req.session_id,
        "response_number": req.response_number,
        "tester_role": req.tester_role,
        "question_topic": req.question_topic,
        "difficulty_tier": req.difficulty_tier,
        "q1_relevance": req.q1_relevance,
        "q2_scientific_accuracy": req.q2_scientific_accuracy,
        "q3_depth": req.q3_depth,
        "q4_length_format": req.q4_length_format,
        "q5_critical_content": req.q5_critical_content,
        "q7_comment": req.q7_comment,
        "quick_score": req.quick_score,
        "conversation_id": req.conversation_id,
        "message_id": req.message_id,
    }
    try:
        result = sb.table("response_feedback").insert(row).execute()
        return result.data[0]
    except Exception as e:
        logger.exception("Failed to save response feedback")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pattern-check")
def submit_pattern_check(
    req: PatternCheck,
    user: dict = Depends(get_current_user),
):
    sb = get_supabase()
    row = {
        "tester_id": user["user_id"],
        "session_id": req.session_id,
        "checkpoint": req.checkpoint,
        "strengths": req.strengths,
        "weaknesses": req.weaknesses,
        "confidence_vs_accuracy": req.confidence_vs_accuracy,
        "comparison_to_usual_tool": req.comparison_to_usual_tool,
        "comparison_detail": req.comparison_detail,
    }
    try:
        result = sb.table("pattern_checks").insert(row).execute()
        return result.data[0]
    except Exception as e:
        logger.exception("Failed to save pattern check")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}")
def get_session_feedback(
    session_id: str,
    user: dict = Depends(get_current_user),
):
    sb = get_supabase()
    feedback = (
        sb.table("response_feedback")
        .select("*")
        .eq("session_id", session_id)
        .eq("tester_id", user["user_id"])
        .order("response_number", desc=False)
        .execute()
    )
    patterns = (
        sb.table("pattern_checks")
        .select("*")
        .eq("session_id", session_id)
        .eq("tester_id", user["user_id"])
        .order("checkpoint", desc=False)
        .execute()
    )
    return {
        "response_feedback": feedback.data,
        "pattern_checks": patterns.data,
    }


@router.get("/stats")
def get_feedback_stats(user: dict = Depends(get_current_user)):
    sb = get_supabase()
    feedback = (
        sb.table("response_feedback")
        .select("quick_score, q1_relevance, q2_scientific_accuracy, q3_depth, q4_length_format, q5_critical_content, difficulty_tier")
        .execute()
    )
    rows = feedback.data or []
    if not rows:
        return {"count": 0}

    scores = [r["quick_score"] for r in rows]
    avg_score = sum(scores) / len(scores)

    def count_field(field):
        counts = {}
        for r in rows:
            val = r.get(field)
            if val:
                counts[val] = counts.get(val, 0) + 1
        return counts

    return {
        "count": len(rows),
        "avg_quick_score": round(avg_score, 2),
        "score_distribution": {s: scores.count(s) for s in range(1, 11) if scores.count(s)},
        "q1_relevance": count_field("q1_relevance"),
        "q2_scientific_accuracy": count_field("q2_scientific_accuracy"),
        "q3_depth": count_field("q3_depth"),
        "q4_length_format": count_field("q4_length_format"),
        "q5_critical_content": count_field("q5_critical_content"),
        "difficulty_tier": count_field("difficulty_tier"),
    }
