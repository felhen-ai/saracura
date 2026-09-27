"""Optional-ML-free contracts and pure functions for generic shadow evaluation."""

from saracura.shadow.evaluation import (
    evaluate_shadow_feedback,
    project_shadow_decision,
    validate_shadow_item_state,
)
from saracura.shadow.models import (
    ShadowDecisionRecord,
    ShadowEvaluationSummary,
    ShadowFeedbackRecord,
    ShadowItem,
    ShadowPolicy,
    ShadowRankingEntry,
    parse_shadow_decision_json,
    parse_shadow_feedback_json,
    parse_shadow_item_json,
    parse_shadow_policy_json,
    shadow_policy_digest,
)

__all__ = [
    "ShadowDecisionRecord",
    "ShadowEvaluationSummary",
    "ShadowFeedbackRecord",
    "ShadowItem",
    "ShadowPolicy",
    "ShadowRankingEntry",
    "evaluate_shadow_feedback",
    "parse_shadow_decision_json",
    "parse_shadow_feedback_json",
    "parse_shadow_item_json",
    "parse_shadow_policy_json",
    "project_shadow_decision",
    "shadow_policy_digest",
    "validate_shadow_item_state",
]
