"""Offline-first Phase 3B-S synthetic research lane.

This module deliberately keeps the provider boundary small: the default HTTP
transport is only constructed after ``--allow-network`` and the test surface
can inject a callable transport without importing an HTTP client package.
Generated records are never written to the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import ssl
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol, cast

from benchmarks.data_policy_registry import load_bundled_registry
from benchmarks.first_party_packet import privacy_matches
from benchmarks.io import atomic_create
from saracura.serialization import canonical_json_bytes

ROOT = Path(__file__).parents[1]
POLICY_PATH = ROOT / "benchmarks/manifests/synthetic-research-policy.v1.json"
SOURCE_POLICY_PATH = ROOT / "benchmarks/manifests/training-data-source-policies.v1.json"
PROTOCOL_PATH = ROOT / "benchmarks/manifests/support-routing-protocol.v1.json"
WORKFLOW_REVISION = "phase3b-synthetic-research-training.v2"
SEED = "saracura-phase3bs-pilot-20260921-v2"
LABELS = (
    "billing",
    "technical_support",
    "account_access",
    "subscription_cancellation",
    "order_delivery",
)
SPLITS = ("synthetic_train", "synthetic_dev", "synthetic_holdout")
AUTHOR_MODEL = "qwen/qwen3.5-9b"
REVIEWER_MODEL = "mistralai/ministral-8b-2512"
JEV_MODEL = "typesafe/jev-1.13"
HOST = "openrouter.ai"
TOTAL_BUDGET = 0.25
STAGE_BUDGETS = {"author": 0.15, "review": 0.06, "jev": 0.04}
PRICE_CEILINGS = {
    "author_input": 0.12,
    "author_output": 0.20,
    "review_input": 0.20,
    "review_output": 0.60,
}
JEV_INPUT_PRICE_CEILING = 0.05
PROTOCOL_TOKEN_ALLOWANCE = 256
MAX_BATCH_SIZE = 10
AUTHOR_MAX_TOKENS = 1800
REVIEW_MAX_TOKENS = 1600
AXES = {
    "message_style": ("conversacional", "objetivo", "fragmentado", "educado"),
    "length_bucket": ("curto", "medio", "longo"),
    "spelling_noise_level": ("none", "light", "moderate"),
    "explicitness": ("explicit", "implicit"),
    "negation": ("none", "present"),
    "urgency": ("normal", "urgent"),
    "regional_neutral_wording": ("neutral",),
}
TAXONOMY = {
    "account_access": (
        "autenticação, identidade, credencial ou acesso à conta; tem prioridade quando bloqueia "
        "outra ação"
    ),
    "billing": (
        "cobrança, pagamento, estorno, fatura ou método de pagamento, salvo cancelamento puro"
    ),
    "subscription_cancellation": (
        "parar assinatura, renovação ou plano recorrente sem cobrança ou estorno separado"
    ),
    "order_delivery": "envio, rastreio, entrega, pacote ausente/danificado ou pedido físico",
    "technical_support": (
        "falha, configuração, compatibilidade ou ajuda de uso não bloqueada por acesso"
    ),
}
PRIORITY = (
    "account_access",
    "billing",
    "subscription_cancellation",
    "order_delivery",
    "technical_support",
)


class SyntheticError(ValueError):
    """Fail-closed synthetic lane error."""


class Transport(Protocol):
    def __call__(
        self, method: str, url: str, headers: Mapping[str, str], body: bytes | None
    ) -> Any: ...


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> bytes:
    return canonical_json_bytes(value)


def _nfc(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    if normalized != value:
        raise SyntheticError("text is not NFC")
    return normalized


def _token_bound(messages: Sequence[Mapping[str, Any]]) -> int:
    """Conservative, deterministic token bound for the provider preflight."""
    payload = _canonical(list(messages))
    # One UTF-8 byte per token is deliberately pessimistic for this small pilot.
    return len(payload) + PROTOCOL_TOKEN_ALLOWANCE


def request_worst_case(stage: str, messages: Sequence[Mapping[str, Any]], max_tokens: int) -> float:
    if stage == "author":
        input_price, output_price = PRICE_CEILINGS["author_input"], PRICE_CEILINGS["author_output"]
    elif stage == "review":
        input_price, output_price = PRICE_CEILINGS["review_input"], PRICE_CEILINGS["review_output"]
    else:
        raise SyntheticError("unknown budget stage")
    return (_token_bound(messages) * input_price + max_tokens * output_price) / 1_000_000


def _provider_controls() -> dict[str, Any]:
    return {"data_collection": "deny", "zdr": True, "enforce_distillable_text": True}


def _author_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["records"],
        "properties": {
            "records": {
                "type": "array",
                "minItems": 10,
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["family_id", "text"],
                    "properties": {
                        "family_id": {"type": "string"},
                        "text": {"type": "string", "minLength": 12, "maxLength": 320},
                    },
                },
            }
        },
    }


def _review_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["reviews"],
        "properties": {
            "reviews": {
                "type": "array",
                "minItems": 10,
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "family_id",
                        "status",
                        "review_label",
                        "reason_codes",
                        "natural_ptbr",
                        "fictional",
                        "single_owner",
                        "contains_sensitive_pattern",
                    ],
                    "properties": {
                        "family_id": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["accepted", "rejected", "ambiguous", "out_of_scope"],
                        },
                        "review_label": {"type": ["string", "null"], "enum": [*LABELS, None]},
                        "reason_codes": {"type": "array", "items": {"type": "string"}},
                        "natural_ptbr": {"type": "boolean"},
                        "fictional": {"type": "boolean"},
                        "single_owner": {"type": "boolean"},
                        "contains_sensitive_pattern": {"type": "boolean"},
                    },
                },
            }
        },
    }


def validate_synthetic_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    """Validate the exact closed policy manifest and its source-policy binding."""
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=_no_duplicate_keys)
    except (OSError, ValueError, TypeError) as exc:
        raise SyntheticError("synthetic policy is invalid") from exc
    expected = {
        "schema_version",
        "workflow_revision",
        "id",
        "author_model",
        "reviewer_model",
        "jev_model",
        "labels",
        "splits",
        "prices",
        "provider",
        "accessed_date",
        "privacy_controls",
        "authorizations",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise SyntheticError("synthetic policy shape")
    if (
        value["schema_version"] != "synthetic-research-policy.v1"
        or value["workflow_revision"] != WORKFLOW_REVISION
    ):
        raise SyntheticError("synthetic policy revision")
    if (
        value["id"] != "synthetic-research"
        or value["author_model"] != AUTHOR_MODEL
        or value["reviewer_model"] != REVIEWER_MODEL
        or value["jev_model"] != JEV_MODEL
    ):
        raise SyntheticError("synthetic policy model binding")
    if value["labels"] != list(LABELS) or value["splits"] != list(SPLITS):
        raise SyntheticError("synthetic policy taxonomy")
    if value["prices"] != PRICE_CEILINGS or value["accessed_date"] != "2026-09-21":
        raise SyntheticError("synthetic policy prices or date")
    if value["provider"] != {
        "host": HOST,
        "zdr": True,
        "data_collection": "deny",
        "enforce_distillable_text": True,
    }:
        raise SyntheticError("provider controls")
    if value["authorizations"] != {
        "synthetic_generation_authorized": True,
        "synthetic_research_training_authorized": True,
        "human_original": False,
        "canonical_training_authorized": False,
        "calibration_authorized": False,
        "blind_test_authorized": False,
        "publication_authorized": False,
        "quality_claims_allowed": False,
        "automation_authorized": False,
    }:
        raise SyntheticError("authorization boundary")
    if not isinstance(value["privacy_controls"], list) or set(value["privacy_controls"]) != {
        "fictional_only",
        "mechanical_scan",
        "no_credentials",
        "no_private_data",
        "no_provider_storage",
    }:
        raise SyntheticError("privacy controls")
    load_bundled_registry()
    return value


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, val in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = val
    return result


def family_id(label: str, index: int) -> str:
    if label not in LABELS or not 0 <= index < 50:
        raise SyntheticError("invalid family identity")
    return _digest(f"{WORKFLOW_REVISION}\0{label}\0{index}".encode())


def build_plan() -> dict[str, Any]:
    """Build the text-free immutable 250-family plan."""
    families: list[dict[str, Any]] = []
    for label in LABELS:
        ranked = sorted(range(50), key=lambda i: _digest(f"{SEED}\0{label}\0{i}".encode()))
        split_for = {
            index: SPLITS[position // 10] if position < 30 else "synthetic_holdout"
            for position, index in enumerate(ranked)
        }
        # The first 30 ranked items are train, the next 10 dev, the final 10 holdout.
        split_for = {
            index: (
                "synthetic_train"
                if pos < 30
                else "synthetic_dev"
                if pos < 40
                else "synthetic_holdout"
            )
            for pos, index in enumerate(ranked)
        }
        for index in range(50):
            digest = _digest(f"{SEED}\0{label}\0{index}".encode())
            families.append(
                {
                    "family_id": family_id(label, index),
                    "candidate_label": label,
                    "label_index": index,
                    "split": split_for[index],
                    "axes": {
                        axis: values[int(digest[offset : offset + 2], 16) % len(values)]
                        for offset, (axis, values) in enumerate(AXES.items())
                    },
                }
            )
    return {
        "schema_version": "synthetic-plan.v1",
        "workflow_revision": WORKFLOW_REVISION,
        "seed": SEED,
        "labels": list(LABELS),
        "families": families,
    }


def validate_plan(plan: Mapping[str, Any]) -> None:
    if _canonical(plan) != _canonical(build_plan()):
        raise SyntheticError("immutable plan mismatch")
    if len(plan["families"]) != 250:
        raise SyntheticError("plan size")
    for label in LABELS:
        counts = {
            split: sum(
                row["candidate_label"] == label and row["split"] == split
                for row in plan["families"]
            )
            for split in SPLITS
        }
        if counts != {"synthetic_train": 30, "synthetic_dev": 10, "synthetic_holdout": 10}:
            raise SyntheticError("plan split counts")


def _privacy_or_duplicate(text: str, prior: Iterable[str]) -> tuple[str, ...]:
    hits = list(privacy_matches(text))
    normalized = " ".join(text.casefold().split())
    if any(normalized == item for item in prior):
        hits.append("duplicate")
    return tuple(dict.fromkeys(hits))


def validate_author_records(records: Any, planned: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(records, list) or len(records) != len(planned):
        raise SyntheticError("author record count")
    expected_ids = {item["family_id"] for item in planned}
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"family_id", "text"}:
            raise SyntheticError("author schema")
        family = record["family_id"]
        text = record["text"]
        if not isinstance(text, str) or not 12 <= len(text) <= 320:
            raise SyntheticError("author text length")
        _nfc(text)
        if not isinstance(family, str) or family in by_id or family not in expected_ids:
            raise SyntheticError("author provenance mismatch")
        if privacy_matches(text):
            raise SyntheticError("author privacy or duplicate id")
        by_id[family] = record
    if set(by_id) != expected_ids:
        raise SyntheticError("author provenance mismatch")
    accepted: list[dict[str, Any]] = []
    for item in planned:
        record = by_id[item["family_id"]]
        accepted.append(
            {
                "family_id": record["family_id"],
                "candidate_label": item["candidate_label"],
                "text": record["text"],
                "axes": item["axes"],
            }
        )
    return accepted


def validate_review_records(
    reviews: Any, records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if not isinstance(reviews, list) or len(reviews) != len(records):
        raise SyntheticError("review record count")
    expected_ids = {record["family_id"] for record in records}
    by_id: dict[str, dict[str, Any]] = {}
    for review in reviews:
        if not isinstance(review, dict) or set(review) != {
            "family_id",
            "status",
            "review_label",
            "reason_codes",
            "natural_ptbr",
            "fictional",
            "single_owner",
            "contains_sensitive_pattern",
        }:
            raise SyntheticError("review schema")
        family = review["family_id"]
        if not isinstance(family, str) or family in by_id or family not in expected_ids:
            raise SyntheticError("review provenance mismatch")
        by_id[family] = review
    if set(by_id) != expected_ids:
        raise SyntheticError("review provenance mismatch")
    return [by_id[record["family_id"]] for record in records]


def review_records(
    records: Sequence[Mapping[str, Any]],
    reviews: Any,
    plan_by_id: Mapping[str, Mapping[str, Any]],
    prior_texts: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reviews = validate_review_records(reviews, records)
    seen_text: set[str] = {" ".join(text.casefold().split()) for text in prior_texts}
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for record, review in zip(records, reviews, strict=True):
        fid = record.get("family_id")
        if not isinstance(fid, str):
            raise SyntheticError("review input provenance")
        code = "review_rejected"
        if fid not in plan_by_id:
            excluded.append({"family_id": fid, "reason_codes": ["unknown_family"]})
            continue
        expected = plan_by_id[fid]["candidate_label"]
        normalized = " ".join(record["text"].casefold().split())
        if (
            review["status"] not in {"accepted", "rejected", "ambiguous", "out_of_scope"}
            or review["review_label"] not in (*LABELS, None)
            or not isinstance(review["reason_codes"], list)
            or any(not isinstance(code, str) or not code for code in review["reason_codes"])
            or any(
                not isinstance(review[key], bool)
                for key in (
                    "natural_ptbr",
                    "fictional",
                    "single_owner",
                    "contains_sensitive_pattern",
                )
            )
        ):
            excluded.append({"family_id": fid, "reason_codes": ["review_value_schema"]})
            continue
        reasons = list(dict.fromkeys(review["reason_codes"]))
        if review["status"] != "accepted":
            code = review["status"]
        elif review["review_label"] != expected:
            code = "wrong_label"
        elif not (review["natural_ptbr"] and review["fictional"] and review["single_owner"]):
            code = "quality_boolean"
        elif review["contains_sensitive_pattern"] or privacy_matches(record["text"]):
            code = "sensitive_pattern"
        elif normalized in seen_text:
            code = "duplicate"
        elif any(
            SequenceMatcher(None, normalized, previous).ratio() >= 0.92 for previous in seen_text
        ):
            code = "near_duplicate"
        if code != "review_rejected":
            excluded.append({"family_id": fid, "reason_codes": [code, *reasons]})
            continue
        seen_text.add(normalized)
        accepted.append(
            {
                **record,
                "review": {"status": "accepted", "review_label": expected, "reason_codes": reasons},
                "synthetic_only": True,
            }
        )
    return accepted, excluded


@dataclass
class BudgetLedger:
    """Hard local budget gate. Server-reported costs are authoritative."""

    total_limit: float = TOTAL_BUDGET
    stage_limits: dict[str, float] = field(default_factory=lambda: dict(STAGE_BUDGETS))
    entries: list[dict[str, Any]] = field(default_factory=list)
    uncertain: bool = False

    @property
    def total(self) -> float:
        return sum(float(item["cost"]) for item in self.entries)

    def spent(self, stage: str) -> float:
        return sum(float(item["cost"]) for item in self.entries if item["stage"] == stage)

    def reserve(self, stage: str, worst_case: float) -> None:
        if self.uncertain or stage not in self.stage_limits or worst_case < 0:
            raise SyntheticError("budget is uncertain")
        if (
            self.spent(stage) + worst_case > self.stage_limits[stage] + 1e-12
            or self.total + worst_case > self.total_limit + 1e-12
        ):
            raise SyntheticError("budget ceiling")

    def record(
        self, stage: str, cost: float, request_id: str | None, status: str = "complete"
    ) -> None:
        if not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost < 0:
            self.uncertain = True
            raise SyntheticError("invalid reported cost")
        self.entries.append(
            {"stage": stage, "cost": float(cost), "request_id": request_id, "status": status}
        )
        if (
            self.spent(stage) > self.stage_limits[stage] + 1e-12
            or self.total > self.total_limit + 1e-12
        ):
            raise SyntheticError("reported cost crossed budget")

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> BudgetLedger:
        if value.get("schema_version") != "cost-ledger.v1" or value.get("final") not in {
            True,
            False,
        }:
            raise SyntheticError("cost ledger is not final")
        ledger = cls()
        entries = value.get("entries")
        if not isinstance(entries, list):
            raise SyntheticError("cost ledger entries")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {
                "stage",
                "cost",
                "request_id",
                "status",
            }:
                raise SyntheticError("cost ledger entry shape")
            if entry["status"] not in {"complete", "billed_failure"}:
                raise SyntheticError("cost ledger entry status")
            ledger.record(entry["stage"], entry["cost"], entry["request_id"], entry["status"])
        if abs(float(value.get("total", -1)) - ledger.total) > 1e-9:
            raise SyntheticError("cost ledger total mismatch")
        return ledger


class OpenRouterClient:
    """Pinned OpenRouter client with an injectable HTTP transport."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        allow_network: bool = False,
        transport: Any = None,
        ledger: BudgetLedger | None = None,
    ) -> None:
        if not allow_network:
            raise SyntheticError("--allow-network is required")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise SyntheticError("OPENROUTER_API_KEY is required")
        self.transport = transport or self._urllib_transport
        self.ledger = ledger or BudgetLedger()
        self.pre_usage: float | None = None
        self.usage_snapshots: list[dict[str, Any]] = []

    def _urllib_transport(
        self, method: str, url: str, headers: Mapping[str, str], body: bytes | None
    ) -> tuple[int, Mapping[str, str], bytes]:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != HOST:
            raise SyntheticError("literal OpenRouter host required")
        request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
        context = ssl.create_default_context()
        try:
            with urllib.request.urlopen(request, timeout=30, context=context) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.URLError as exc:
            raise SyntheticError("OpenRouter request failed") from exc

    def request(self, path: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if not path.startswith("/api/") or "?" in path:
            raise SyntheticError("invalid OpenRouter path")
        body = None if payload is None else _canonical(payload)
        result = self.transport(
            "POST" if body is not None else "GET",
            f"https://{HOST}{path}",
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            body,
        )
        if isinstance(result, tuple):
            status, _, raw = result
            if status >= 400:
                raise SyntheticError("provider error")
            value = json.loads(raw)
        else:
            value = result
        if not isinstance(value, dict):
            raise SyntheticError("provider response shape")
        return value

    def preflight(self) -> dict[str, Any]:
        """Perform only the documented read-only key check."""
        result = self.request("/api/v1/key")
        if not isinstance(result.get("data", result), dict):
            raise SyntheticError("key preflight response shape")
        snapshot = self._usage_snapshot(result)
        self.pre_usage = snapshot
        self.usage_snapshots.append({"kind": "pre", "usage": snapshot})
        return result

    @staticmethod
    def _usage_snapshot(result: Mapping[str, Any]) -> float | None:
        data = result.get("data", result)
        if not isinstance(data, Mapping):
            return None
        for key in ("usage", "usage_usd", "total_usage"):
            value = data.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                return float(value)
        return None

    def _reconcile(self, cost: float) -> None:
        if self.pre_usage is None:
            return
        tolerance = max(0.000002, abs(cost) * 0.10)
        last_usage: float | None = None
        for attempt in range(3):
            post = self.request("/api/v1/key")
            post_usage = self._usage_snapshot(post)
            last_usage = post_usage
            self.usage_snapshots.append(
                {"kind": "post", "attempt": attempt + 1, "usage": post_usage}
            )
            if post_usage is not None:
                delta = post_usage - self.pre_usage
                if delta >= -1e-9 and abs(delta - cost) <= tolerance:
                    self.pre_usage = post_usage
                    return
            if attempt < 2:
                time.sleep(0.2)
        # The read-only key counter is eventually consistent. The response's
        # usage.cost remains authoritative for the local hard gate; retain the
        # mismatch as audit evidence without issuing an unbounded retry.
        if last_usage is not None and last_usage < self.pre_usage - 1e-9:
            self.usage_snapshots.append(
                {
                    "kind": "reconciliation_non_monotonic",
                    "previous_usage": self.pre_usage,
                    "usage": last_usage,
                }
            )
        self.usage_snapshots.append(
            {"kind": "reconciliation_delayed", "reported_cost": cost, "usage": last_usage}
        )
        if last_usage is not None:
            self.pre_usage = max(self.pre_usage, last_usage)

    def chat(
        self,
        *,
        stage: str,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int,
        temperature: float,
        worst_case: float | None = None,
    ) -> dict[str, Any]:
        if stage not in {"author", "review"} or model not in {AUTHOR_MODEL, REVIEWER_MODEL}:
            raise SyntheticError("model or stage is not pinned")
        # Never trust a caller-supplied estimate: derive the reservation from
        # the exact payload limits and the reviewed price ceilings.
        calculated_worst_case = request_worst_case(stage, messages, max_tokens)
        self.ledger.reserve(stage, calculated_worst_case)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning": {"enabled": False},
            "provider": _provider_controls(),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "synthetic_author_batch"
                    if stage == "author"
                    else "synthetic_review_batch",
                    "strict": True,
                    "schema": _author_schema() if stage == "author" else _review_schema(),
                },
            },
        }
        result = self.request("/api/v1/chat/completions", payload)
        usage = result.get("usage")
        if (
            not isinstance(usage, dict)
            or not isinstance(usage.get("cost"), (int, float))
            or isinstance(usage.get("cost"), bool)
        ):
            self.ledger.uncertain = True
            raise SyntheticError("chat usage.cost is required")
        self.ledger.record(stage, float(usage["cost"]), result.get("id"))
        self._reconcile(float(usage["cost"]))
        return result

    def jev(
        self, *, state: Mapping[str, Any], question: Mapping[str, Any], worst_case: float
    ) -> dict[str, Any]:
        payload = {
            "model": JEV_MODEL,
            "state": state,
            "questions": {"routing": question},
            "provider": _provider_controls(),
        }
        calculated = (
            (len(_canonical(payload)) + PROTOCOL_TOKEN_ALLOWANCE)
            * JEV_INPUT_PRICE_CEILING
            / 1_000_000
        )
        self.ledger.reserve("jev", calculated)
        result = self.request("/api/alpha/decisions", payload)
        usage = result.get("usage")
        if (
            not isinstance(usage, dict)
            or not isinstance(usage.get("cost"), (int, float))
            or isinstance(usage.get("cost"), bool)
        ):
            self.ledger.uncertain = True
            raise SyntheticError("Jev usage.cost is required")
        self.ledger.record("jev", float(usage["cost"]), result.get("id"))
        self._reconcile(float(usage["cost"]))
        return result


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Append a batch atomically; existing bytes are never rewritten."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = b"".join(_canonical(row) + b"\n" for row in rows)
    if not payload:
        raise SyntheticError("empty append")
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise SyntheticError("artifact is not a regular file")
    rows: list[dict[str, Any]] = []
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise SyntheticError("truncated append-only artifact")
        value = json.loads(line, object_pairs_hook=_no_duplicate_keys)
        if not isinstance(value, dict):
            raise SyntheticError("JSONL row must be an object")
        rows.append(value)
    return rows


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == value
        and abs(value) != float("inf")
    )


def _read_accepted(packet: Path) -> list[dict[str, Any]]:
    validate_packet_manifest(packet / "packet-manifest.json")
    plan = json.loads((packet / "plan.json").read_bytes(), object_pairs_hook=_no_duplicate_keys)
    validate_plan(plan)
    ledger_value = json.loads(
        (packet / "cost-ledger.json").read_bytes(), object_pairs_hook=_no_duplicate_keys
    )
    BudgetLedger.from_json(ledger_value)
    if ledger_value.get("final") is not True:
        raise SyntheticError("packet cost ledger is not final")
    rows = load_jsonl(packet / "accepted.jsonl")
    if any(row.get("synthetic_only") is not True or "review" not in row for row in rows):
        raise SyntheticError("accepted packet lacks synthetic provenance")
    return rows


def train_head_from_embeddings(
    embeddings: Mapping[str, Any],
    *,
    seed: int = 20260921,
    max_epochs: int = 80,
    patience: int = 10,
) -> dict[str, Any]:
    """Train two CPU-only deterministic linear heads from sealed embeddings.

    The encoder is intentionally not imported here. Callers must supply sealed
    float32 vectors produced by the existing MPS encoder gate.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SyntheticError("encoder-eval extra is required for training") from exc
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    required = {"synthetic_train", "synthetic_dev", "synthetic_holdout"}
    if set(embeddings) != required:
        raise SyntheticError("sealed embedding splits")
    vectors = {
        split: torch.as_tensor(value[0], dtype=torch.float32) for split, value in embeddings.items()
    }
    targets = {
        split: torch.as_tensor(value[1], dtype=torch.long) for split, value in embeddings.items()
    }
    if any(vector.ndim != 2 or vector.shape[1] != 384 for vector in vectors.values()):
        raise SyntheticError("embedding width")
    if any(not torch.isfinite(vector).all() for vector in vectors.values()):
        raise SyntheticError("non-finite embedding")
    try:
        from safetensors.torch import load, save  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SyntheticError("safetensors is required for training") from exc
    checkpoints: list[bytes] = []
    run_ledgers: list[list[dict[str, Any]]] = []
    metrics: dict[str, Any] = {}
    for _run in range(2):
        torch.manual_seed(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        head = torch.nn.Linear(384, 5, device="cpu")
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()
        best = float("-inf")
        stale = 0
        best_state: dict[str, Any] | None = None
        selected_epoch = 0
        epoch_ledger: list[dict[str, Any]] = []
        train_vectors = vectors["synthetic_train"]
        train_targets = targets["synthetic_train"]
        for epoch in range(1, max_epochs + 1):
            head.train()
            permutation = torch.randperm(len(train_targets), generator=generator)
            total_loss = 0.0
            for offset in range(0, len(permutation), 32):
                indices = permutation[offset : offset + 32]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(head(train_vectors[indices]), train_targets[indices])
                loss_value = float(loss.detach())
                if not _finite(loss_value):
                    raise SyntheticError("non-finite training loss")
                loss.backward()
                optimizer.step()
                total_loss += loss_value * len(indices)
            with torch.no_grad():
                pred = head(vectors["synthetic_dev"]).argmax(dim=1)
                score = _macro_f1(pred.tolist(), targets["synthetic_dev"].tolist(), 5)
            epoch_ledger.append(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / len(train_targets),
                    "dev_macro_f1": score,
                }
            )
            if score > best + 0.001:
                best, stale, selected_epoch, best_state = (
                    score,
                    0,
                    epoch,
                    {key: value.detach().clone() for key, value in head.state_dict().items()},
                )
            else:
                stale += 1
            if stale >= patience:
                break
        if best_state is None:
            raise SyntheticError("no finite checkpoint")
        head.load_state_dict(best_state)
        checkpoint = save(
            {key: value.detach().cpu().contiguous() for key, value in head.state_dict().items()}
        )
        reloaded = torch.nn.Linear(384, 5, device="cpu")
        reloaded.load_state_dict(load(checkpoint))
        with torch.no_grad():
            for split in SPLITS:
                torch.testing.assert_close(
                    head(vectors[split]), reloaded(vectors[split]), rtol=1e-6, atol=1e-6
                )
            train_predicted = head(train_vectors).argmax(dim=1).tolist()
            dev_predicted = head(vectors["synthetic_dev"]).argmax(dim=1).tolist()
        checkpoints.append(checkpoint)
        run_ledgers.append(epoch_ledger)
        metrics = {
            "synthetic_only": True,
            "epochs": len(epoch_ledger),
            "selected_epoch": selected_epoch,
            "epoch_ledger": epoch_ledger,
            "train": _classification_metrics(train_predicted, train_targets.tolist()),
            "dev": _classification_metrics(dev_predicted, targets["synthetic_dev"].tolist()),
        }
    if checkpoints[0] != checkpoints[1]:
        raise SyntheticError("CPU head determinism failure")
    if run_ledgers[0] != run_ledgers[1]:
        raise SyntheticError("CPU training ledger determinism failure")
    return {
        "checkpoint": checkpoints[0],
        "checkpoint_sha256": _digest(checkpoints[0]),
        "metrics": metrics,
    }


def _macro_f1(predicted: Sequence[int], actual: Sequence[int], classes: int) -> float:
    values: list[float] = []
    for label in range(classes):
        tp = sum(p == label and a == label for p, a in zip(predicted, actual, strict=True))
        fp = sum(p == label and a != label for p, a in zip(predicted, actual, strict=True))
        fn = sum(p != label and a == label for p, a in zip(predicted, actual, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(values) / classes


def _classification_metrics(predicted: Sequence[int], actual: Sequence[int]) -> dict[str, Any]:
    if len(predicted) != len(actual) or not predicted:
        raise SyntheticError("classification metric inputs")
    confusion = [[0 for _ in LABELS] for _ in LABELS]
    per_label: dict[str, dict[str, float | int]] = {}
    for expected, observed in zip(actual, predicted, strict=True):
        if expected not in range(len(LABELS)) or observed not in range(len(LABELS)):
            raise SyntheticError("classification label range")
        confusion[expected][observed] += 1
    for index, label in enumerate(LABELS):
        tp = confusion[index][index]
        fp = sum(confusion[row][index] for row in range(len(LABELS)) if row != index)
        fn = sum(confusion[index][column] for column in range(len(LABELS)) if column != index)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "support": sum(confusion[index]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "count": len(actual),
        "accuracy": sum(left == right for left, right in zip(predicted, actual, strict=True))
        / len(actual),
        "macro_f1": _macro_f1(predicted, actual, len(LABELS)),
        "confusion_matrix": confusion,
        "per_label": per_label,
    }


def _latency_summary(values_ms: Sequence[float]) -> dict[str, float | int]:
    if not values_ms or any(not _finite(value) or value < 0 for value in values_ms):
        raise SyntheticError("latency inputs")
    ordered = sorted(values_ms)

    def percentile(fraction: float) -> float:
        index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5)))
        return ordered[index]

    return {
        "count": len(ordered),
        "min_ms": ordered[0],
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "max_ms": ordered[-1],
    }


def _packet_embedding_arrays(packet_manifest_path: Path) -> dict[str, Any]:
    """Extract sealed MiniLM embeddings locally; this function never downloads."""
    try:
        import torch

        from benchmarks.encoder_acquisition import snapshot_path
        from benchmarks.encoder_loader import VerifiedSnapshot, load_encoder
        from benchmarks.encoder_registry import get_candidate
    except ImportError as exc:
        raise SyntheticError("encoder-eval extra is required for training") from exc
    if not torch.backends.mps.is_available():
        raise SyntheticError("MPS is required; CPU encoder fallback is disabled")
    packet = packet_manifest_path.parent
    rows = _read_accepted(packet)
    candidate = get_candidate("multilingual-minilm-l12")
    snapshot = snapshot_path(candidate)
    verified = VerifiedSnapshot.create(candidate, snapshot)
    loaded = load_encoder(candidate, snapshot, verified=verified)
    model = loaded.model.to("mps")
    model.eval()
    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    plan = json.loads((packet / "plan.json").read_bytes(), object_pairs_hook=_no_duplicate_keys)
    by_id = {item["family_id"]: item["split"] for item in plan["families"]}
    for row in rows:
        by_split[by_id[row["family_id"]]].append(row)
    result: dict[str, Any] = {}
    with torch.inference_mode():
        for split in SPLITS:
            batch = by_split[split]
            texts = [str(row["text"]) for row in batch]
            encoded = loaded.tokenizer(
                texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
            )
            inputs = {key: value.to("mps") for key, value in encoded.items()}
            output = model(**inputs).last_hidden_state.float()
            pooled = (output * inputs["attention_mask"].unsqueeze(-1)).sum(1) / inputs[
                "attention_mask"
            ].sum(1, keepdim=True)
            torch.mps.synchronize()
            result[split] = (
                pooled.cpu(),
                torch.tensor(
                    [LABELS.index(row["candidate_label"]) for row in batch], dtype=torch.long
                ),
            )
    return result


def train_from_packet(packet_manifest_path: Path, output_dir: Path) -> Path:
    rows = _read_accepted(packet_manifest_path.parent)
    counts = {label: {split: 0 for split in SPLITS} for label in LABELS}
    plan = json.loads((packet_manifest_path.parent / "plan.json").read_bytes())
    split_by_id = {item["family_id"]: item["split"] for item in plan["families"]}
    for row in rows:
        counts[row["candidate_label"]][split_by_id[row["family_id"]]] += 1
    if any(
        counts[label][split] < minimum
        for label in LABELS
        for split, minimum in (
            ("synthetic_train", 20),
            ("synthetic_dev", 5),
            ("synthetic_holdout", 5),
        )
    ):
        raise SyntheticError("accepted minimum per label was not met")
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    arrays = _packet_embedding_arrays(packet_manifest_path)
    try:
        import safetensors  # type: ignore[import-not-found]
        import torch
        from safetensors.torch import load, save

        from benchmarks.encoder_registry import get_candidate, registry_digest
    except ImportError as exc:
        raise SyntheticError("safetensors is required for training") from exc
    embedding_tensors = {}
    for split, (vectors, targets) in arrays.items():
        embedding_tensors[f"{split}.vectors"] = vectors
        embedding_tensors[f"{split}.targets"] = targets
    embedding_bytes = save(embedding_tensors)
    atomic_create(output_dir / "embeddings.safetensors", embedding_bytes)
    os.chmod(output_dir / "embeddings.safetensors", 0o600)
    sealed = load((output_dir / "embeddings.safetensors").read_bytes())
    sealed_arrays = {
        split: (sealed[f"{split}.vectors"], sealed[f"{split}.targets"]) for split in SPLITS
    }
    result = train_head_from_embeddings(sealed_arrays)
    atomic_create(output_dir / "checkpoint.safetensors", result["checkpoint"])
    os.chmod(output_dir / "checkpoint.safetensors", 0o600)
    candidate = get_candidate("multilingual-minilm-l12")
    manifest = {
        "schema_version": "synthetic-training-manifest.v1",
        "workflow_revision": WORKFLOW_REVISION,
        "synthetic_only": True,
        "packet_manifest_sha256": _digest(packet_manifest_path.read_bytes()),
        "training_code_sha256": _digest(Path(__file__).read_bytes()),
        "protocol_sha256": _digest(PROTOCOL_PATH.read_bytes()),
        "encoder": {
            "candidate": "multilingual-minilm-l12",
            "revision": candidate.revision,
            "registry_sha256": registry_digest(),
            "device": "mps",
            "frozen": True,
        },
        "labels": list(LABELS),
        "seed": 20260921,
        "head": {
            "width": 384,
            "classes": 5,
            "optimizer": "AdamW",
            "lr": 0.01,
            "weight_decay": 0.01,
            "batch_size": 32,
            "maximum_epochs": 80,
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 0.001,
        },
        "files": {
            "embeddings.safetensors": _digest(embedding_bytes),
            "checkpoint.safetensors": _digest(result["checkpoint"]),
        },
        "metrics": result["metrics"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "safetensors": safetensors.__version__,
            "platform": platform.platform(),
        },
        "authorizations": {"calibration": False, "automation": False, "quality_claims": False},
    }
    manifest["manifest_sha256"] = _digest(_canonical(manifest))
    _write_json(output_dir / "training-manifest.json", manifest)
    return output_dir / "training-manifest.json"


def validate_training_manifest(path: Path, packet_manifest_path: Path) -> dict[str, Any]:
    if path.name != "training-manifest.json" or path.is_symlink() or not path.is_file():
        raise SyntheticError("training manifest path")
    training = json.loads(path.read_bytes(), object_pairs_hook=_no_duplicate_keys)
    expected_keys = {
        "schema_version",
        "workflow_revision",
        "synthetic_only",
        "packet_manifest_sha256",
        "training_code_sha256",
        "protocol_sha256",
        "encoder",
        "labels",
        "seed",
        "head",
        "files",
        "metrics",
        "environment",
        "authorizations",
        "manifest_sha256",
    }
    if not isinstance(training, dict) or set(training) != expected_keys:
        raise SyntheticError("training manifest shape")
    if (
        training["schema_version"] != "synthetic-training-manifest.v1"
        or training["workflow_revision"] != WORKFLOW_REVISION
        or training["synthetic_only"] is not True
        or training["labels"] != list(LABELS)
        or training["authorizations"]
        != {"calibration": False, "automation": False, "quality_claims": False}
    ):
        raise SyntheticError("training manifest boundary")
    unhashed = dict(training)
    recorded_manifest_digest = unhashed.pop("manifest_sha256")
    if recorded_manifest_digest != _digest(_canonical(unhashed)):
        raise SyntheticError("training manifest self digest")
    if (
        training["packet_manifest_sha256"] != _digest(packet_manifest_path.read_bytes())
        or training["training_code_sha256"] != _digest(Path(__file__).read_bytes())
        or training["protocol_sha256"] != _digest(PROTOCOL_PATH.read_bytes())
    ):
        raise SyntheticError("training provenance digest")
    files = training["files"]
    if not isinstance(files, dict) or set(files) != {
        "embeddings.safetensors",
        "checkpoint.safetensors",
    }:
        raise SyntheticError("training file ledger")
    expected_names = {"training-manifest.json", *files}
    actual_names = {item.name for item in path.parent.iterdir()}
    if actual_names != expected_names:
        raise SyntheticError("training artifact file set")
    for name, digest in files.items():
        artifact = path.parent / name
        if (
            artifact.is_symlink()
            or not artifact.is_file()
            or _digest(artifact.read_bytes()) != digest
        ):
            raise SyntheticError("training artifact digest")
    return training


def evaluate_from_training(
    training_manifest_path: Path,
    packet_manifest_path: Path,
    output_dir: Path,
    *,
    allow_network: bool,
    budget_usd: str,
    prior_cost_ledger_path: Path | None = None,
) -> Path:
    training = validate_training_manifest(training_manifest_path, packet_manifest_path)
    checkpoint_path = training_manifest_path.parent / "checkpoint.safetensors"
    if _digest(checkpoint_path.read_bytes()) != training["files"]["checkpoint.safetensors"]:
        raise SyntheticError("checkpoint digest mismatch")
    if budget_usd != "0.25":
        raise SyntheticError("approved budget must be exactly 0.25")
    try:
        import torch
        from safetensors.torch import load
    except ImportError as exc:
        raise SyntheticError("encoder-eval extra is required for evaluation") from exc
    state = load(checkpoint_path.read_bytes())
    head = torch.nn.Linear(384, 5)
    head.load_state_dict(state)
    embeddings_path = training_manifest_path.parent / "embeddings.safetensors"
    if _digest(embeddings_path.read_bytes()) != training["files"]["embeddings.safetensors"]:
        raise SyntheticError("embedding digest mismatch")
    embeddings = load(embeddings_path.read_bytes())
    vectors = embeddings["synthetic_holdout.vectors"]
    targets = embeddings["synthetic_holdout.targets"]
    predicted: list[int] = []
    latencies: list[float] = []
    with torch.no_grad():
        for vector in vectors:
            started = time.perf_counter_ns()
            predicted.append(int(head(vector.unsqueeze(0)).argmax(dim=1).item()))
            latencies.append((time.perf_counter_ns() - started) / 1_000_000)
    metrics = {
        "synthetic_only": True,
        **_classification_metrics(predicted, targets.tolist()),
        "head_latency": _latency_summary(latencies),
    }
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": "synthetic-evaluation.v1",
        "sealed": True,
        "local": metrics,
        "jev": {"status": "external_control_unavailable"},
    }
    if allow_network:
        packet = packet_manifest_path.parent
        ledger_path = prior_cost_ledger_path or (packet / "cost-ledger.json")
        ledger = BudgetLedger.from_json(
            json.loads(ledger_path.read_bytes(), object_pairs_hook=_no_duplicate_keys)
        )
        client = OpenRouterClient(allow_network=True, ledger=ledger)
        client.preflight()
        question = {
            "type": "choice",
            "instructions": (
                "Classifique a mensagem de suporte em exatamente uma categoria. "
                "Aplique as definições e a prioridade descritas nos critérios."
            ),
            "criteria": {label: TAXONOMY[label] for label in LABELS},
        }
        rows = _read_accepted(packet)
        holdout_ids = {
            row["family_id"]
            for row in json.loads((packet / "plan.json").read_bytes())["families"]
            if row["split"] == "synthetic_holdout"
        }
        holdout = [row for row in rows if row["family_id"] in holdout_ids]
        jev_predicted: list[int] = []
        jev_actual: list[int] = []
        digests: list[str] = []
        resolved_models: set[str] = set()
        jev_latencies: list[float] = []
        jev_cost = 0.0
        try:
            for row in holdout:
                started = time.perf_counter_ns()
                response = client.jev(
                    state={
                        "description": "Uma mensagem fictícia de cliente pedindo suporte.",
                        "text": row["text"],
                    },
                    question=question,
                    worst_case=0.0,
                )
                jev_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
                answers = response.get("answers")
                answer = answers.get("routing") if isinstance(answers, dict) else None
                if (
                    not isinstance(answer, dict)
                    or answer.get("type") != "choice"
                    or answer.get("choice") not in LABELS
                ):
                    raise SyntheticError("Jev choice response shape")
                model = response.get("model")
                if not isinstance(model, str) or not model.startswith(f"{JEV_MODEL}-"):
                    raise SyntheticError("Jev resolved revision")
                usage = response.get("usage")
                if not isinstance(usage, dict) or not _finite(usage.get("cost")):
                    raise SyntheticError("Jev usage response shape")
                jev_predicted.append(LABELS.index(answer["choice"]))
                jev_actual.append(LABELS.index(row["candidate_label"]))
                resolved_models.add(model)
                jev_cost += float(usage["cost"])
                digests.append(_digest(_canonical(response)))
                _write_json_mutable(
                    output_dir / "jev-cost-ledger.json",
                    {
                        "schema_version": "cost-ledger.v1",
                        "final": False,
                        "entries": client.ledger.entries,
                        "total": client.ledger.total,
                    },
                )
        except SyntheticError as exc:
            report["jev"] = {
                "status": "external_control_unavailable",
                "completed": len(jev_predicted),
                "reason": str(exc),
                "digests": digests,
            }
        else:
            report["jev"] = {
                "status": "complete",
                "resolved_models": sorted(resolved_models),
                "metrics": _classification_metrics(jev_predicted, jev_actual),
                "latency": _latency_summary(jev_latencies),
                "usage": {"cost": jev_cost},
                "digests": digests,
            }
        _write_json_mutable(
            output_dir / "jev-cost-ledger.json",
            {
                "schema_version": "cost-ledger.v1",
                "final": True,
                "entries": client.ledger.entries,
                "total": client.ledger.total,
            },
        )
        report["jev_usage_snapshots"] = client.usage_snapshots
    _write_json(
        output_dir / "evaluation-manifest.json",
        {
            **report,
            "training_manifest_sha256": _digest(training_manifest_path.read_bytes()),
            "packet_manifest_sha256": _digest(packet_manifest_path.read_bytes()),
        },
    )
    return output_dir / "evaluation-manifest.json"


def _content(response: Mapping[str, Any]) -> Any:
    try:
        content = response["choices"][0]["message"]["content"]
        return json.loads(content) if isinstance(content, str) else content
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise SyntheticError("provider content is not strict JSON") from exc


def _envelope(response: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    value = _content(response)
    if not isinstance(value, dict) or set(value) != {key} or not isinstance(value[key], list):
        raise SyntheticError("provider envelope schema mismatch")
    return cast(list[dict[str, Any]], value[key])


def _author_messages(
    batch_plan: Sequence[Mapping[str, Any]], *, repair: bool = False
) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": (
                "Você é o autor sintético de pesquisa da Saracura. Responda somente no envelope "
                "JSON schema exigido. Crie casos naturais em PT-BR, exclusivamente fictícios, sem "
                "nomes, empresas, domínios, URLs, handles, contas, segredos ou alegações de "
                "autoria humana. Respeite exatamente a taxonomia, definição, prioridade e eixos "
                "fornecidos. Devolva somente family_id e text; não repita nem crie rótulos, eixos, "
                "splits ou metadados. Cada texto deve ser uma mensagem recebida de um cliente "
                "pedindo suporte; nunca escreva a resposta de um atendente. Para "
                "technical_support, use falhas de uso, configuração, dispositivo ou aplicativo "
                "sem login, senha, "
                "verificação de identidade ou bloqueio de conta. Para account_access, o problema "
                "central deve ser autenticação, identidade, credencial ou acesso à conta. Para "
                "billing, não peça cancelamento de assinatura; para subscription_cancellation, "
                "não inclua cobrança, pagamento ou estorno."
            ),
        },
        {
            "role": "user",
            "content": _canonical(
                {
                    "families": list(batch_plan),
                    "taxonomy": TAXONOMY,
                    "priority": list(PRIORITY),
                }
            ).decode(),
        },
    ]
    if repair:
        messages.insert(
            1,
            {
                "role": "system",
                "content": (
                    "Reparo único de schema: a resposta anterior foi inválida ou truncada. "
                    "Mantenha cada text entre 12 e 160 caracteres e devolva os dez registros "
                    "completos no JSON exigido."
                ),
            },
        )
    return messages


def _review_messages(
    records: Sequence[Mapping[str, Any]], *, repair: bool = False
) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": (
                "Você é um revisor independente. Responda somente no envelope JSON schema exigido. "
                "Avalie naturalidade PT-BR, ficcionalidade, um único responsável e padrões "
                "sensíveis. Você está cego ao rótulo candidato: não receba nem infira qualquer "
                "label planejado, split ou eixo; use apenas o texto e o family_id opaco. Escolha "
                "somente a taxonomia "
                "fechada quando o texto permitir e use códigos de motivo curtos. Todos os textos "
                "foram gerados sinteticamente: defina fictional=true, exceto se o próprio texto "
                "nomear uma pessoa, empresa, conta ou identificador real. Se status=accepted, "
                "review_label deve conter exatamente um rótulo da taxonomia e reason_codes deve "
                "ser vazio."
            ),
        },
        {
            "role": "user",
            "content": _canonical(
                {
                    "records": [{"family_id": r["family_id"], "text": r["text"]} for r in records],
                    "taxonomy": TAXONOMY,
                    "priority": list(PRIORITY),
                }
            ).decode(),
        },
    ]
    if repair:
        messages.insert(
            1,
            {
                "role": "system",
                "content": (
                    "Reparo único de schema: a resposta anterior foi inválida ou truncada. "
                    "Devolva os dez reviews completos, sem prosa e no JSON exigido."
                ),
            },
        )
    return messages


def _write_json_mutable(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_canonical(value) + b"\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _duplicate_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    exact: list[list[str]] = []
    normalized: list[list[str]] = []
    near: list[list[str]] = []
    seen_exact: dict[str, str] = {}
    seen_normalized: dict[str, str] = {}
    texts: list[tuple[str, str]] = []
    for row in rows:
        fid, text = str(row["family_id"]), str(row["text"])
        norm = " ".join(text.casefold().split())
        if text in seen_exact:
            exact.append([seen_exact[text], fid])
        else:
            seen_exact[text] = fid
        if norm in seen_normalized:
            normalized.append([seen_normalized[norm], fid])
        else:
            seen_normalized[norm] = fid
        for previous_id, previous_text in texts:
            if SequenceMatcher(None, norm, previous_text).ratio() >= 0.92 and norm != previous_text:
                near.append([previous_id, fid])
        texts.append((fid, norm))
    return {
        "schema_version": "duplicate-report.v1",
        "exact": exact,
        "normalized": normalized,
        "near": near,
    }


def run_generation(client: OpenRouterClient, output_dir: Path) -> Path:
    """Run the pinned author/reviewer flow with deterministic append-only files."""
    validate_synthetic_policy()
    sealed = output_dir / "packet-manifest.json"
    if sealed.exists():
        validate_packet_manifest(sealed)
        return sealed
    plan = build_plan()
    validate_plan(plan)
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(output_dir, 0o700)
    plan_path = output_dir / "plan.json"
    if plan_path.exists():
        if plan_path.read_bytes() != _canonical(plan) + b"\n":
            raise SyntheticError("resume plan mismatch")
    else:
        _write_json(plan_path, plan)
    plan_by_id = {item["family_id"]: item for item in plan["families"]}
    accepted_path = output_dir / "accepted.jsonl"
    excluded_path = output_dir / "excluded.jsonl"
    ledger_path = output_dir / "cost-ledger.json"
    if ledger_path.exists():
        client.ledger = BudgetLedger.from_json(
            json.loads(ledger_path.read_bytes(), object_pairs_hook=_no_duplicate_keys)
        )
    completed_path = output_dir / "completed-batches.jsonl"
    completed = load_jsonl(completed_path) if completed_path.exists() else []
    done_batches = {int(row["batch"]) for row in completed if row.get("status") == "complete"}
    existing = load_jsonl(accepted_path) if accepted_path.exists() else []
    prior_texts = [str(row["text"]) for row in existing]
    author_by_batch: dict[int, list[dict[str, Any]]] = {}
    author_attempts: dict[int, int] = {}
    author_log = output_dir / "author-responses.jsonl"
    if author_log.exists():
        for row in load_jsonl(author_log):
            try:
                batch_number = int(row["batch"])
                author_attempts[batch_number] = author_attempts.get(batch_number, 0) + 1
                records = _envelope(row["response"], "records")
            except (KeyError, TypeError, ValueError, SyntheticError):
                continue
            author_by_batch[batch_number] = records
    review_by_batch: dict[int, list[dict[str, Any]]] = {}
    review_attempts: dict[int, int] = {}
    review_log = output_dir / "review-responses.jsonl"
    if review_log.exists():
        for row in load_jsonl(review_log):
            try:
                batch_number = int(row["batch"])
                review_attempts[batch_number] = review_attempts.get(batch_number, 0) + 1
                reviews = _envelope(row["response"], "reviews")
            except (KeyError, TypeError, ValueError, SyntheticError):
                continue
            review_by_batch[batch_number] = reviews
    for start in range(0, 250, 10):
        batch_number = start // 10
        batch_plan = plan["families"][start : start + 10]
        if batch_number in done_batches:
            continue
        cached_records = author_by_batch.get(batch_number)
        if cached_records is not None:
            records = validate_author_records(cached_records, batch_plan)
        else:
            attempts = author_attempts.get(batch_number, 0)
            if attempts >= 2:
                raise SyntheticError("author repair limit exceeded")
            messages = _author_messages(batch_plan, repair=attempts == 1)
            response = client.chat(
                stage="author",
                model=AUTHOR_MODEL,
                messages=messages,
                max_tokens=AUTHOR_MAX_TOKENS,
                temperature=0.8,
                worst_case=request_worst_case("author", messages, AUTHOR_MAX_TOKENS),
            )
            _write_json_mutable(
                ledger_path,
                {
                    "schema_version": "cost-ledger.v1",
                    "final": False,
                    "entries": client.ledger.entries,
                    "total": client.ledger.total,
                },
            )
            append_jsonl(
                author_log,
                [
                    {
                        "batch": batch_number,
                        "response": response,
                    }
                ],
            )
            records = validate_author_records(_envelope(response, "records"), batch_plan)
            author_by_batch[batch_number] = records
        cached_reviews = review_by_batch.get(batch_number)
        if cached_reviews is not None:
            try:
                reviews = validate_review_records(cached_reviews, records)
            except SyntheticError:
                cached_reviews = None
        if cached_reviews is None:
            attempts = review_attempts.get(batch_number, 0)
            if attempts >= 2:
                raise SyntheticError("review repair limit exceeded")
            review_messages = _review_messages(records, repair=attempts == 1)
            review = client.chat(
                stage="review",
                model=REVIEWER_MODEL,
                messages=review_messages,
                max_tokens=REVIEW_MAX_TOKENS,
                temperature=0,
                worst_case=request_worst_case("review", review_messages, REVIEW_MAX_TOKENS),
            )
            _write_json_mutable(
                ledger_path,
                {
                    "schema_version": "cost-ledger.v1",
                    "final": False,
                    "entries": client.ledger.entries,
                    "total": client.ledger.total,
                },
            )
            append_jsonl(
                review_log,
                [
                    {
                        "batch": batch_number,
                        "response": review,
                    }
                ],
            )
            reviews = validate_review_records(_envelope(review, "reviews"), records)
            review_by_batch[batch_number] = reviews
        accepted, excluded = review_records(records, reviews, plan_by_id, prior_texts)
        if accepted:
            append_jsonl(accepted_path, accepted)
            prior_texts.extend(str(row["text"]) for row in accepted)
        if excluded:
            append_jsonl(excluded_path, excluded)
        append_jsonl(completed_path, [{"batch": batch_number, "status": "complete"}])
        done_batches.add(batch_number)
        _write_json_mutable(
            ledger_path,
            {
                "schema_version": "cost-ledger.v1",
                "final": False,
                "entries": client.ledger.entries,
                "total": client.ledger.total,
            },
        )
    all_accepted = load_jsonl(accepted_path) if accepted_path.exists() else []
    counts = {label: {split: 0 for split in SPLITS} for label in LABELS}
    for row in all_accepted:
        counts[row["candidate_label"]][plan_by_id[row["family_id"]]["split"]] += 1
    for label in LABELS:
        if any(
            counts[label][split] < minimum
            for split, minimum in (
                ("synthetic_train", 20),
                ("synthetic_dev", 5),
                ("synthetic_holdout", 5),
            )
        ):
            raise SyntheticError("accepted minimum per label was not met")
    _write_json_mutable(
        ledger_path,
        {
            "schema_version": "cost-ledger.v1",
            "final": True,
            "entries": client.ledger.entries,
            "total": client.ledger.total,
        },
    )
    author_rows: list[dict[str, Any]] = []
    if author_log.exists():
        for envelope in load_jsonl(author_log):
            try:
                parsed = _envelope(envelope["response"], "records")
            except SyntheticError:
                continue
            author_rows.extend(parsed)
    _write_json_mutable(output_dir / "duplicate-report.json", _duplicate_report(author_rows))
    _write_json_mutable(output_dir / "usage-snapshots.json", {"snapshots": client.usage_snapshots})
    manifest = packet_manifest(output_dir)
    _write_json_mutable(output_dir / "packet-manifest.json", manifest)
    return output_dir / "packet-manifest.json"


def packet_manifest(packet_dir: Path) -> dict[str, Any]:
    allowed = {
        "plan.json",
        "accepted.jsonl",
        "excluded.jsonl",
        "author-responses.jsonl",
        "review-responses.jsonl",
        "cost-ledger.json",
        "usage-snapshots.json",
        "duplicate-report.json",
        "completed-batches.jsonl",
    }
    required = allowed
    files = []
    for path in sorted(packet_dir.iterdir()):
        if path.name == "packet-manifest.json":
            continue
        if path.name not in allowed or not path.is_file() or path.is_symlink():
            raise SyntheticError("packet file allowlist")
        files.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": _digest(path.read_bytes())}
        )
    if {item["path"] for item in files} != required:
        raise SyntheticError("packet required file set")
    value = {
        "schema_version": "synthetic-packet-manifest.v1",
        "workflow_revision": WORKFLOW_REVISION,
        "files": files,
        "root_sha256": "",
    }
    value["root_sha256"] = _digest(_canonical(value))
    return value


def validate_packet_manifest(path: Path) -> dict[str, Any]:
    if path.name != "packet-manifest.json" or path.is_symlink():
        raise SyntheticError("packet manifest path")
    packet_dir = path.parent
    value = json.loads(path.read_bytes(), object_pairs_hook=_no_duplicate_keys)
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "workflow_revision",
        "files",
        "root_sha256",
    }:
        raise SyntheticError("packet manifest shape")
    expected = packet_manifest(packet_dir)
    if value != expected:
        raise SyntheticError("packet manifest digest or contents mismatch")
    if len(value["files"]) == 0:
        raise SyntheticError("empty packet")
    return value


def _write_json(path: Path, value: Any) -> None:
    atomic_create(path, _canonical(value) + b"\n")


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    atomic_create(path, b"".join(_canonical(row) + b"\n" for row in rows))


def _make_plan(output: Path) -> dict[str, Any]:
    plan = build_plan()
    validate_plan(plan)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chmod(output, 0o700)
    _write_json(output / "plan.json", plan)
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.synthetic_research")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--allow-network", action="store_true")
    generate.add_argument("--budget-usd", required=True)
    validate = sub.add_parser("validate-packet")
    validate.add_argument("--packet", type=Path, required=True)
    train = sub.add_parser("train")
    train.add_argument("--packet", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--training", type=Path, required=True)
    evaluate.add_argument("--packet", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--allow-network", action="store_true")
    evaluate.add_argument("--budget-usd", default="0.25")
    evaluate.add_argument("--prior-cost-ledger", type=Path)
    args = parser.parse_args(argv)
    try:
        validate_synthetic_policy()
        if args.command == "validate-packet":
            validate_packet_manifest(args.packet)
            print("synthetic packet: valid")
            return 0
        if args.command in {"train", "evaluate"}:
            packet_path = (
                args.packet
                if args.packet.name == "packet-manifest.json"
                else args.packet / "packet-manifest.json"
            )
            validate_packet_manifest(packet_path)
            if args.command == "train":
                train_from_packet(packet_path, args.output_dir)
            else:
                evaluate_from_training(
                    args.training,
                    packet_path,
                    args.output_dir,
                    allow_network=args.allow_network,
                    budget_usd=args.budget_usd,
                    prior_cost_ledger_path=args.prior_cost_ledger,
                )
            return 0
        if args.budget_usd != "0.25":
            raise SyntheticError("approved budget must be exactly 0.25")
        if not args.output_dir.exists():
            _make_plan(args.output_dir)
        elif not (args.output_dir / "plan.json").is_file():
            raise SyntheticError("existing generation directory lacks plan")
        if not args.allow_network:
            print("plan created; network generation not authorized", file=sys.stderr)
            return 2
        client = OpenRouterClient(allow_network=True)
        client.preflight()
        run_generation(client, args.output_dir)
        print("synthetic packet: written")
        return 0
    except (SyntheticError, OSError, json.JSONDecodeError) as exc:
        print(f"synthetic research: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
