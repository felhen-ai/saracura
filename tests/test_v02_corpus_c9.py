import json
import re
from pathlib import Path
from typing import Any

import pytest

from benchmarks import v02_corpus as corpus


def _raw_author(plan: dict[str, object], identity: dict[str, object]) -> dict[str, Any]:
    count = identity["option_count"]
    assert isinstance(count, int)
    gold = identity["gold_position"]
    assert isinstance(gold, int)
    return {
        "policy": "Approve exactly at 10.",
        "facts": "The fictional request total is 10.",
        "question": "Which action is eligible?",
        "options": [
            {"id": f"option_{index}", "description": f"Action {index}."} for index in range(count)
        ],
        "answer": f"option_{gold}",
        "semantic": {
            "scenario_code": identity["scenario_code"],
            "criterion_roles": identity["criterion_roles"],
        },
        "construction": {
            "option_checks": [
                {
                    "option_id": f"option_{index}",
                    "supported": index == gold,
                    "reason": "It satisfies the policy.",
                }
                for index in range(count)
            ]
        },
        "fictionality_valid": True,
        "exclusive_options_valid": True,
        "ambiguity_free": True,
    }


def test_c9_materializes_literal_policy_and_facts_and_rejects_boundaries() -> None:
    assert corpus.materialize_training_state("Regra?", "Fato.") == "Regra? Fato."
    with pytest.raises(ValueError, match="punctuation"):
        corpus.materialize_training_state("Regra", "Fato.")
    with pytest.raises(ValueError, match="240"):
        corpus.materialize_training_state("a" * 241 + ".", "Fato.")
    with pytest.raises(ValueError, match="control characters"):
        corpus.materialize_training_state("Regra.", "Fato\nnovo.")


def test_c9_raw_training_author_is_closed_and_proof_is_derived() -> None:
    plan = corpus._generate_training_plan()
    identity = plan["slots"][0]
    request = corpus.role_request(plan, identity["slot_id"], "training_author")
    assert "Prefer a policy length of 180 Unicode code points" in request[1]["content"]
    raw = _raw_author(plan, identity)
    raw_bytes = json.dumps(raw, ensure_ascii=False).encode()
    schema = corpus.native_decoder_schema("training_author", identity)
    assert "minLength" not in schema["properties"]["policy"]
    assert "maxLength" not in schema["properties"]["policy"]
    parsed = corpus.parse_role_response(raw_bytes, plan, identity["slot_id"], "training_author")
    case = corpus.case_from_author(parsed, plan, identity["slot_id"])
    assert case["state"] == "Approve exactly at 10. The fictional request total is 10."
    envelope = corpus.role_envelope(
        parsed,
        plan,
        identity["slot_id"],
        "training_author",
        case,
        {name: True for name in corpus.LOCAL_GATES},
    )
    assert envelope["construction"]["rule_quote"] == raw["policy"]
    for forbidden in ("state", "rule_quote"):
        invalid = dict(raw)
        if forbidden == "state":
            invalid[forbidden] = case["state"]
        else:
            invalid["construction"] = {
                "rule_quote": "x.",
                "option_checks": raw["construction"]["option_checks"],
            }
        with pytest.raises(ValueError, match="closed"):
            corpus.parse_role_response(
                json.dumps(invalid).encode(), plan, identity["slot_id"], "training_author"
            )


@pytest.mark.parametrize(
    ("policy", "accepted"),
    [
        ("é" * 238 + ".", True),
        ("😀" * 239 + ".", True),
        ("😀" * 240 + ".", False),
    ],
)
def test_c9_policy_unicode_bound_is_identical_in_native_schema_and_parser(
    policy: str, accepted: bool
) -> None:
    plan = corpus._generate_training_plan()
    identity = plan["slots"][0]
    prompt_policy = corpus._role_response_schema("training_author", identity)["properties"][
        "policy"
    ]
    schema = corpus.native_decoder_schema("training_author", identity)
    native_policy = schema["properties"]["policy"]
    assert prompt_policy["maxLength"] == 240
    assert "minLength" not in native_policy
    assert "maxLength" not in native_policy
    assert native_policy["pattern"] == r'^[^\u0000-\u001F\u007F-\u009F"\\]{0,239}[.!?]$'
    assert bool(re.fullmatch(native_policy["pattern"], policy)) is accepted
    raw = _raw_author(plan, identity)
    raw["policy"] = policy
    encoded = json.dumps(raw, ensure_ascii=False).encode()
    if accepted:
        parsed = corpus.parse_role_response(encoded, plan, identity["slot_id"], "training_author")
        assert corpus.case_from_author(parsed, plan, identity["slot_id"])["state"].startswith(
            policy
        )
    else:
        with pytest.raises(ValueError, match="240"):
            corpus.parse_role_response(encoded, plan, identity["slot_id"], "training_author")


@pytest.mark.parametrize(
    "field,value",
    [
        ("policy", "Use option_1."),
        ("facts", "The code is threshold_approval."),
        ("facts", "The internal label is option_\u0661."),
        ("question", "Does matches_rule apply?"),
    ],
)
def test_c9_reserved_labels_are_rejected_only_in_public_text(field: str, value: str) -> None:
    plan = corpus._generate_training_plan()
    identity = plan["slots"][0]
    raw = _raw_author(plan, identity)
    raw[field] = value
    with pytest.raises(ValueError, match="reserved"):
        corpus.case_from_author(raw, plan, identity["slot_id"])
    raw = _raw_author(plan, identity)
    raw["construction"]["option_checks"][0]["reason"] = "option_1 is internal proof only."
    assert corpus.case_from_author(raw, plan, identity["slot_id"])["state"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("policy", "Use option_1."),
        ("facts", "A regra é option_1."),
        ("question", "Does matches_rule apply?"),
    ],
)
def test_c9_reserved_identifiers_use_unicode_word_boundaries(field: str, value: str) -> None:
    plan = corpus._generate_training_plan()
    identity = plan["slots"][0]
    raw = _raw_author(plan, identity)
    raw[field] = value
    with pytest.raises(ValueError, match="reserved"):
        corpus.case_from_author(raw, plan, identity["slot_id"])
    raw = _raw_author(plan, identity)
    raw["facts"] = "éoption_1 e pré_matches_rule são palavras fictícias."
    raw["question"] = "preoption_1post e x_matches_rule_y são palavras fictícias?"
    assert corpus.case_from_author(raw, plan, identity["slot_id"])["state"]


def test_c9_unicode_and_ordinary_words_are_valid_and_sealed_is_unchanged() -> None:
    plan = corpus._generate_training_plan()
    identity = plan["slots"][0]
    raw = _raw_author(plan, identity)
    raw["policy"] = "Aprovar igualdade até café."
    raw["facts"] = "A palavra finance descreve um cenário fictício."
    assert corpus.case_from_author(raw, plan, identity["slot_id"])["state"].startswith("Aprovar")
    sealed = corpus._generate_sealed_plan()
    sealed_identity = sealed["identities"][0]
    assert corpus._role_response_schema("sealed_author", sealed_identity)["required"] == sorted(
        {"state", "question", "options", "target", *corpus._JUDGMENT_FIELDS}
    )
    training_ids = {row["slot_id"] for row in plan["slots"]}
    assert training_ids.isdisjoint({row["identity_id"] for row in sealed["identities"]})


def test_c9_components_are_create_only_and_bind_literal_case_and_response(tmp_path: Path) -> None:
    plan = corpus._generate_training_plan()
    identity = plan["slots"][0]
    raw = _raw_author(plan, identity)
    case = corpus.case_from_author(raw, plan, identity["slot_id"])
    content = json.dumps(raw, ensure_ascii=False).encode()
    corpus._store_author_components(tmp_path, identity["slot_id"], raw, case, content)
    components = corpus._read_author_components(tmp_path, identity["slot_id"], case)
    assert components["response_sha256"] == corpus._sha256(content)
    assert components["case_digest"] == corpus.case_digest(case)
    altered = dict(raw)
    altered["facts"] = "Changed fictional facts."
    with pytest.raises(ValueError, match="mismatch"):
        corpus._store_author_components(tmp_path, identity["slot_id"], altered, case, content)
