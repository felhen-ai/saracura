from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.synthetic_research import (
    AUTHOR_MAX_TOKENS,
    AUTHOR_MODEL,
    LABELS,
    REVIEW_MAX_TOKENS,
    BudgetLedger,
    OpenRouterClient,
    SyntheticError,
    _author_messages,
    _review_messages,
    build_plan,
    family_id,
    packet_manifest,
    request_worst_case,
    review_records,
    validate_author_records,
    validate_packet_manifest,
    validate_plan,
)


def test_plan_is_exactly_250_and_stable() -> None:
    assert LABELS == (
        "billing",
        "technical_support",
        "account_access",
        "subscription_cancellation",
        "order_delivery",
    )
    plan = build_plan()
    validate_plan(plan)
    assert len(plan["families"]) == 250
    assert family_id("billing", 0) == plan["families"][0]["family_id"]
    for label in LABELS:
        rows = [row for row in plan["families"] if row["candidate_label"] == label]
        assert len(rows) == 50
        assert {row["split"] for row in rows} == {
            "synthetic_train",
            "synthetic_dev",
            "synthetic_holdout",
        }


def test_budget_has_independent_hard_stage_and_total_gates() -> None:
    ledger = BudgetLedger()
    ledger.record("author", 0.149, "a")
    with pytest.raises(SyntheticError):
        ledger.reserve("author", 0.002)
    ledger.record("review", 0.06, "r")
    with pytest.raises(SyntheticError):
        ledger.reserve("jev", 0.042)


def test_fake_transport_uses_literal_host_and_redacts_nothing_into_result() -> None:
    captured: dict[str, object] = {}

    def fake(
        method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> tuple[int, dict[str, str], bytes]:
        captured.update(method=method, url=url, headers=headers, body=body)
        return 200, {}, b'{"data":{"limit":1}}'

    client = OpenRouterClient(api_key="test-secret", allow_network=True, transport=fake)
    result = client.preflight()
    assert result == {"data": {"limit": 1}}
    assert captured["url"] == "https://openrouter.ai/api/v1/key"
    assert "test-secret" not in json.dumps(result)


def test_network_is_rejected_without_explicit_allowance() -> None:
    with pytest.raises(SyntheticError):
        OpenRouterClient(api_key="test-secret")


def test_chat_is_pinned_and_requires_server_cost() -> None:
    def fake(*_: object) -> tuple[int, dict[str, str], bytes]:
        return 200, {}, b'{"id":"x","choices":[],"usage":{}}'

    client = OpenRouterClient(api_key="test-secret", allow_network=True, transport=fake)
    with pytest.raises(SyntheticError, match=r"usage\.cost"):
        client.chat(
            stage="author",
            model=AUTHOR_MODEL,
            messages=[{"role": "user", "content": "fictional"}],
            max_tokens=1800,
            temperature=0.8,
            worst_case=0.01,
        )
    assert client.ledger.uncertain is True


def test_strict_payload_and_reviewer_blindness() -> None:
    captured: dict[str, object] = {}

    def fake(
        method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> tuple[int, dict[str, str], bytes]:
        captured.update(method=method, url=url, headers=headers, body=body)
        return 200, {}, b'{"id":"x","choices":[],"usage":{"cost":0.0001}}'

    plan = build_plan()["families"][:10]
    author_messages = _author_messages(plan)
    client = OpenRouterClient(api_key="test-secret", allow_network=True, transport=fake)
    client.chat(
        stage="author",
        model=AUTHOR_MODEL,
        messages=author_messages,
        max_tokens=AUTHOR_MAX_TOKENS,
        temperature=0.8,
    )
    raw_body = captured["body"]
    assert isinstance(raw_body, bytes)
    body = json.loads(raw_body)
    assert body["provider"] == {
        "data_collection": "deny",
        "zdr": True,
        "enforce_distillable_text": True,
    }
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["reasoning"] == {"enabled": False}

    records = [
        {"family_id": row["family_id"], "text": "Preciso de ajuda com isso."} for row in plan
    ]
    reviewer_payload = json.dumps(_review_messages(records), ensure_ascii=False)
    assert "candidate_label" not in reviewer_payload
    assert "synthetic_train" not in reviewer_payload
    assert "axes" not in reviewer_payload
    assert "defina fictional=true" in reviewer_payload
    assert "review_label deve conter exatamente um rótulo" in reviewer_payload


def test_all_planned_chat_requests_fit_stage_reservations() -> None:
    plan = build_plan()["families"]
    author_total = 0.0
    reviewer_total = 0.0
    for start in range(0, 250, 10):
        batch = plan[start : start + 10]
        author_total += request_worst_case("author", _author_messages(batch), AUTHOR_MAX_TOKENS)
        maximal_records = [{"family_id": row["family_id"], "text": "x" * 320} for row in batch]
        reviewer_total += request_worst_case(
            "review", _review_messages(maximal_records), REVIEW_MAX_TOKENS
        )
    assert author_total < 0.15
    assert reviewer_total < 0.06


def test_provider_record_order_is_not_treated_as_provenance() -> None:
    planned = build_plan()["families"][:2]
    authored = [
        {"family_id": planned[1]["family_id"], "text": "Meu cartão foi debitado duas vezes."},
        {
            "family_id": planned[0]["family_id"],
            "text": "O boleto venceu, mas o pagamento ainda não compensou.",
        },
    ]
    records = validate_author_records(authored, planned)
    assert [record["family_id"] for record in records] == [row["family_id"] for row in planned]
    reviews = [
        {
            "family_id": record["family_id"],
            "status": "accepted",
            "review_label": "billing",
            "reason_codes": [],
            "natural_ptbr": True,
            "fictional": True,
            "single_owner": True,
            "contains_sensitive_pattern": False,
        }
        for record in reversed(records)
    ]
    accepted, excluded = review_records(
        records, reviews, {row["family_id"]: row for row in planned}
    )
    assert len(accepted) == 2
    assert excluded == []


def test_author_privacy_and_cross_batch_near_duplicate_are_rejected() -> None:
    planned = build_plan()["families"][:1]
    unsafe = [
        {
            "family_id": planned[0]["family_id"],
            "text": "Escreva para pessoa@example.com sobre minha cobrança.",
        }
    ]
    with pytest.raises(SyntheticError, match="privacy"):
        validate_author_records(unsafe, planned)

    clean = validate_author_records(
        [{**unsafe[0], "text": "Quero entender esta cobrança que apareceu ontem."}], planned
    )
    reviews = [
        {
            "family_id": planned[0]["family_id"],
            "status": "accepted",
            "review_label": planned[0]["candidate_label"],
            "reason_codes": [],
            "natural_ptbr": True,
            "fictional": True,
            "single_owner": True,
            "contains_sensitive_pattern": False,
        }
    ]
    accepted, excluded = review_records(
        clean,
        reviews,
        {planned[0]["family_id"]: planned[0]},
        ["Quero entender esta cobrança que apareceu ontem!"],
    )
    assert accepted == []
    assert excluded[0]["reason_codes"][0] == "near_duplicate"


def test_packet_manifest_detects_mutation_and_extras(tmp_path: Path) -> None:
    files = {
        "plan.json": b"{}\n",
        "accepted.jsonl": b'{"synthetic_only":true}\n',
        "excluded.jsonl": b'{"reason_codes":[]}\n',
        "author-responses.jsonl": b'{"batch":0}\n',
        "review-responses.jsonl": b'{"batch":0}\n',
        "cost-ledger.json": b"{}\n",
        "usage-snapshots.json": b"{}\n",
        "duplicate-report.json": b"{}\n",
        "completed-batches.jsonl": b'{"batch":0}\n',
    }
    for name, payload in files.items():
        (tmp_path / name).write_bytes(payload)
    manifest = packet_manifest(tmp_path)
    (tmp_path / "packet-manifest.json").write_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    validate_packet_manifest(tmp_path / "packet-manifest.json")
    (tmp_path / "extra.txt").write_text("no", encoding="utf-8")
    with pytest.raises(SyntheticError):
        validate_packet_manifest(tmp_path / "packet-manifest.json")
