"""Deterministic content-free projection and feedback evaluation."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import ValidationError

from saracura.shadow.models import (
    SHADOW_SUMMARY_SCHEMA,
    ShadowDecisionRecord,
    ShadowEvaluationSummary,
    ShadowFeedbackRecord,
    ShadowPolicy,
    _invalid,
    _revalidate,
    project_shadow_decision,
    shadow_policy_digest,
    validate_shadow_item_state,
    validate_shadow_state_capacity,
)


def evaluate_shadow_feedback(
    policy: ShadowPolicy,
    decisions: Sequence[ShadowDecisionRecord],
    feedback: Sequence[ShadowFeedbackRecord],
) -> ShadowEvaluationSummary:
    """Join exact opaque references and return descriptive agreement counts."""

    policy = _revalidate(policy, ShadowPolicy, "Shadow policy is invalid.")
    digest = shadow_policy_digest(policy)
    policy_labels = tuple(criterion.id for criterion in policy.criteria)
    labels = set(policy_labels)
    decision_by_ref: dict[str, ShadowDecisionRecord] = {}
    model_reference = None
    for raw_decision in decisions:
        decision = _revalidate(raw_decision, ShadowDecisionRecord, "Shadow decision is invalid.")
        if decision.item_ref in decision_by_ref:
            raise _invalid("Duplicate shadow decision reference.")
        if decision.policy_sha256 != digest or decision.model.revision != policy.model:
            raise _invalid("Shadow decision does not match the policy.")
        if model_reference is None:
            model_reference = decision.model
        elif decision.model != model_reference:
            raise _invalid("Shadow decisions use mixed model references.")
        ranking_labels = tuple(item.label for item in decision.ranking)
        if (
            set(ranking_labels) != labels
            or len(ranking_labels) != len(labels)
            or decision.suggested_label not in labels
            or decision.suggested_label != decision.ranking[0].label
        ):
            raise _invalid("Shadow decision labels do not match the policy.")
        policy_order = {label: index for index, label in enumerate(policy_labels)}
        expected_ranking = tuple(
            sorted(
                decision.ranking,
                key=lambda item: (-item.ranking_weight, policy_order.get(item.label, len(labels))),
            )
        )
        if decision.ranking != expected_ranking:
            raise _invalid("Shadow decision ranking is not in deterministic order.")
        decision_by_ref[decision.item_ref] = decision

    feedback_by_ref: dict[str, ShadowFeedbackRecord] = {}
    for raw_record in feedback:
        record = _revalidate(raw_record, ShadowFeedbackRecord, "Shadow feedback is invalid.")
        if record.item_ref in feedback_by_ref:
            raise _invalid("Duplicate shadow feedback reference.")
        if record.policy_sha256 != digest:
            raise _invalid("Shadow feedback does not match the policy.")
        if record.item_ref not in decision_by_ref:
            raise _invalid("Shadow feedback reference has no prediction.")
        if record.disposition == "labeled" and record.operator_label not in labels:
            raise _invalid("Shadow feedback label is outside the policy.")
        if record.disposition == "skipped" and record.operator_label is not None:
            raise _invalid("Skipped shadow feedback cannot have a label.")
        feedback_by_ref[record.item_ref] = record

    matrix = {predicted: {actual: 0 for actual in policy_labels} for predicted in policy_labels}
    labeled_count = 0
    skipped_count = 0
    agreement_count = 0
    for item_ref, record in feedback_by_ref.items():
        if record.disposition == "skipped":
            skipped_count += 1
            continue
        operator_label = record.operator_label
        if operator_label is None:
            raise _invalid("Labeled shadow feedback requires a label.")
        predicted_label = decision_by_ref[item_ref].suggested_label
        matrix[predicted_label][operator_label] += 1
        labeled_count += 1
        if predicted_label == operator_label:
            agreement_count += 1

    prediction_count = len(decision_by_ref)
    feedback_count = len(feedback_by_ref)
    agreement_rate = agreement_count / labeled_count if labeled_count else None
    coverage = feedback_count / prediction_count if prediction_count else 0.0
    try:
        summary = ShadowEvaluationSummary(
            schema_version=SHADOW_SUMMARY_SCHEMA,
            policy_sha256=digest,
            prediction_count=prediction_count,
            feedback_count=feedback_count,
            labeled_count=labeled_count,
            skipped_count=skipped_count,
            unreviewed_prediction_count=prediction_count - feedback_count,
            feedback_coverage=coverage,
            agreement_count=agreement_count,
            agreement_rate=agreement_rate,
            confusion_matrix=matrix,
        )
    except (ValidationError, ValueError, TypeError):
        invalid = True
    else:
        invalid = False
    if invalid:
        raise _invalid("Shadow evaluation summary is invalid.")
    return summary


__all__ = [
    "evaluate_shadow_feedback",
    "project_shadow_decision",
    "validate_shadow_item_state",
    "validate_shadow_state_capacity",
]
