"""Offline command line gate for the Phase 3A first-party packet."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from benchmarks.first_party_packet import (
    GUIDE_PATH,
    HEX64,
    MANIFEST_PATH,
    POLICY_PATH,
    DigestMismatch,
    DuplicateBlocked,
    PacketError,
    PrivacyBlocked,
    ResourceLimited,
    WriteConflict,
    build_plan,
    parse_plan,
    read_states,
    strict_json,
    validate_plan,
    write_plan,
)


def validate_protocol_bytes(raw: bytes) -> tuple[dict[str, object], bytes, str]:
    value = strict_json(raw)
    if not isinstance(value, dict):
        raise PacketError("manifest object required")
    expected = {
        "schema_version",
        "workflow_id",
        "workflow_revision",
        "locale",
        "labels",
        "review_outcomes",
        "guide_path",
        "guide_sha256",
        "policy_registry_path",
        "policy_registry_sha256",
        "base_state_schema",
        "split_algorithm",
        "split_proportions",
        "normalization",
        "unicode_behavior",
        "near_duplicate",
        "limits",
        "training_authorized",
        "quality_claims_allowed",
        "publication_authorized",
    }
    if set(value) != expected or value["schema_version"] != "support-routing-protocol.v1":
        raise PacketError("protocol shape")
    literals = {
        "workflow_id": "support-routing",
        "workflow_revision": "v1",
        "locale": "pt-BR",
        "guide_path": "docs/annotation-guides/support-routing.v1.md",
        "policy_registry_path": "benchmarks/manifests/training-data-source-policies.v1.json",
        "base_state_schema": "support-routing-base-state.v1",
        "split_algorithm": "support-routing-stratified-sha256.v1",
        "normalization": "support-routing-text-normalization.v1",
        "unicode_behavior": "python-3.11-3.13-vector-locked.v1",
    }
    if any(
        type(value[key]) is not str or value[key] != expected_value
        for key, expected_value in literals.items()
    ):
        raise PacketError("protocol literals")
    labels = [
        "billing",
        "technical_support",
        "account_access",
        "subscription_cancellation",
        "order_delivery",
    ]
    outcomes = ["ambiguous", "out_of_scope", "rejected"]
    if (
        type(value["labels"]) is not list
        or any(type(item) is not str for item in value["labels"])
        or value["labels"] != labels
        or type(value["review_outcomes"]) is not list
        or any(type(item) is not str for item in value["review_outcomes"])
        or value["review_outcomes"] != outcomes
    ):
        raise PacketError("protocol order")
    if (
        value["guide_path"] != "docs/annotation-guides/support-routing.v1.md"
        or value["policy_registry_path"]
        != "benchmarks/manifests/training-data-source-policies.v1.json"
    ):
        raise PacketError("protocol paths")
    if (
        type(value["guide_sha256"]) is not str
        or not HEX64.fullmatch(value["guide_sha256"])
        or type(value["policy_registry_sha256"]) is not str
        or not HEX64.fullmatch(value["policy_registry_sha256"])
        or hashlib.sha256(GUIDE_PATH.read_bytes()).hexdigest() != value["guide_sha256"]
        or hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest() != value["policy_registry_sha256"]
    ):
        raise PacketError("protocol digest")
    split_proportions = [
        {"split": "train", "weight": 40},
        {"split": "dev", "weight": 20},
        {"split": "calibration", "weight": 20},
        {"split": "blind_test", "weight": 20},
    ]
    if (
        type(value["split_proportions"]) is not list
        or any(
            type(item) is not dict
            or set(item) != {"split", "weight"}
            or type(item["split"]) is not str
            or type(item["weight"]) is not int
            for item in value["split_proportions"]
        )
        or value["split_proportions"] != split_proportions
    ):
        raise PacketError("split proportions")
    limits = {
        "max_input_bytes": 16777216,
        "max_plan_bytes": 8388608,
        "max_states": 10000,
        "max_text_codepoints": 2000,
        "max_atoms_per_state": 8,
        "max_pair_comparisons": 50000000,
    }
    if (
        type(value["limits"]) is not dict
        or set(value["limits"]) != set(limits)
        or any(type(item) is not int for item in value["limits"].values())
        or value["limits"] != limits
    ):
        raise PacketError("limits")
    near_duplicate = {
        "unit": "word_trigram_jaccard",
        "threshold_numerator": 85,
        "threshold_denominator": 100,
    }
    if (
        type(value["near_duplicate"]) is not dict
        or set(value["near_duplicate"]) != set(near_duplicate)
        or type(value["near_duplicate"]["unit"]) is not str
        or any(
            type(value["near_duplicate"][key]) is not int
            for key in ("threshold_numerator", "threshold_denominator")
        )
        or value["near_duplicate"] != near_duplicate
    ):
        raise PacketError("near duplicate")
    if (
        value["training_authorized"] is not False
        or value["quality_claims_allowed"] is not False
        or value["publication_authorized"] is not False
    ):
        raise PacketError("authorization")
    return value, raw, hashlib.sha256(raw).hexdigest()


def _protocol() -> tuple[dict[str, object], bytes, str]:
    return validate_protocol_bytes(MANIFEST_PATH.read_bytes())


def _usage() -> None:
    print(
        "usage: python -m benchmarks.first_party_gate "
        "{validate-protocol|build-split-plan|validate-split-plan}",
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if args in (["--help"], ["-h"]):
            _usage()
            return 0
        if args == ["validate-protocol"]:
            _protocol()
            print("first-party protocol: valid")
            return 0
        if len(args) == 7 and args[0] == "build-split-plan":
            if args[1] != "--states" or args[3] != "--seed" or args[5] != "--output":
                print("ARGUMENTS_INVALID", file=sys.stderr)
                return 2
            if not HEX64.fullmatch(args[4]):
                print("ARGUMENTS_INVALID", file=sys.stderr)
                return 2
            states, source = read_states(Path(args[2]))
            _, _, protocol_sha256 = _protocol()
            plan = build_plan(
                states,
                args[4],
                protocol_sha256,
                hashlib.sha256(source).hexdigest(),
            )
            write_plan(plan, Path(args[6]))
            print("first-party split plan: written")
            return 0
        if (
            len(args) == 5
            and args[0] == "validate-split-plan"
            and args[1] == "--states"
            and args[3] == "--plan"
        ):
            states, source = read_states(Path(args[2]))
            plan, _ = parse_plan(Path(args[4]))
            seed = plan.get("seed") if isinstance(plan, dict) else None
            if not isinstance(seed, str):
                raise PacketError("invalid plan seed")
            _, _, protocol_sha256 = _protocol()
            validate_plan(
                states,
                plan,
                seed,
                protocol_sha256,
                hashlib.sha256(source).hexdigest(),
            )
            print("first-party split plan: valid")
            return 0
        print("ARGUMENTS_INVALID", file=sys.stderr)
        return 2
    except WriteConflict:
        print("WRITE_CONFLICT", file=sys.stderr)
        return 2
    except PrivacyBlocked:
        print("PRIVACY_BLOCKED", file=sys.stderr)
        return 2
    except DuplicateBlocked:
        print("DUPLICATE_BLOCKED", file=sys.stderr)
        return 2
    except ResourceLimited:
        print("RESOURCE_LIMIT", file=sys.stderr)
        return 2
    except DigestMismatch:
        print("DIGEST_MISMATCH", file=sys.stderr)
        return 2
    except PacketError:
        print(
            "SPLIT_PLAN_INVALID"
            if args and args[0] == "validate-split-plan"
            else "PROTOCOL_INVALID"
            if args == ["validate-protocol"]
            else "BASE_STATES_INVALID",
            file=sys.stderr,
        )
        return 2
    except Exception:
        print("UNEXPECTED_FAILURE", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
