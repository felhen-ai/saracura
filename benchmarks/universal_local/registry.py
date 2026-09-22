"""Closed Phase 4C.2 registry and acquisition-contract validation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import rfc8785

ROOT = Path(__file__).parents[2]
REGISTRY_PATH = ROOT / "benchmarks" / "manifests" / "decision-backend-candidates.v2.json"
V1_PATH = ROOT / "benchmarks" / "manifests" / "decision-backend-candidates.v1.json"
V1_DIGEST = "587f745dabffacfb4ba5fa676b88fb1b3768db701ed2121fd0007531ed6adf71"
SCHEMA = "decision-backend-candidates.v2"
DEPENDENCY_CONTRACT = "universal-local.v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
ID = re.compile(r"^[a-z][a-z0-9-]{2,63}$")

ROOT_FIELDS = {
    "schema_version",
    "reviewed_at",
    "supersedes_registry_sha256",
    "predecessor_map",
    "candidates",
}
IDENTITY_FIELDS = {
    "id",
    "display_name",
    "architecture_class",
    "role",
    "source_type",
    "source_url",
    "source_revision",
    "source_license",
    "license_evidence_url",
    "model_id",
    "model_revision",
    "execution_boundary",
    "data_transmission",
    "claimed_local_mac",
    "claimed_dynamic_choices",
    "claimed_ptbr",
    "claimed_question_conditioned",
    "claim_evidence_state",
    "exclusion_reason",
    "notes",
}
NEW_FIELDS = {
    "model_source_url",
    "model_license",
    "model_license_evidence_url",
    "base_model_id",
    "base_model_revision",
    "base_model_license",
    "base_model_license_evidence_url",
    "tokenizer_provenance",
    "training_data_provenance",
    "licensing_disposition",
    "loader_family",
    "acquisition_state",
    "weight_format",
    "dependency_contract",
    "source_weight_dtypes",
    "files",
    "acquisition_contract_digest",
    "conformance_state",
    "conformance_vector_sha256",
    "architecture_attribution",
}
CANDIDATE_FIELDS = IDENTITY_FIELDS | NEW_FIELDS
ALLOWED_EXTENSIONS = {".json", ".safetensors"}
ELIGIBLE = {"laya-multilingual", "mdeberta-nli"}
CANDIDATE_ORDER = (
    "saracura-compiled",
    "laya-multilingual",
    "von-option-marker",
    "mdeberta-nli",
    "qwen-system-one",
    "typesafe-jev",
    "diffusiongemma-openjev",
)
# These independently reviewed fingerprints bind every Phase 4C.2a field added
# by registry v2.  Recomputing an acquisition digest after changing a manifest
# value is therefore insufficient to make an unreviewed checkpoint pass.
NEW_FIELD_TRUTH_DIGESTS = {
    "saracura-compiled": "54a034f615947e8ebe97fdd5cde93f67cf696f79d8054cabefe59cc28564421e",
    "laya-multilingual": "3d794a6fe90cf975edbbe188ad5d6158598be6c345b50e471e411448491c51a7",
    "von-option-marker": "52fce00bc144dc676fa86c31e5a99e64e74fa8aed7762cf125a227aa9f0e5c77",
    "mdeberta-nli": "7d00b884dd1e132b7c93e0a7814b14e981a8656f8af9b33b21f8d9cbf855c2f9",
    "qwen-system-one": "52fce00bc144dc676fa86c31e5a99e64e74fa8aed7762cf125a227aa9f0e5c77",
    "typesafe-jev": "5139a58a6f7dfed37add1aebf98c10555771195920f3ba0afec8376010dccbfa",
    "diffusiongemma-openjev": "f870d738f65e5bea5572fd466785262c3e3e0821afe30f9b0177decc83140712",
}


@dataclass(frozen=True)
class RegistryFile:
    path: str
    bytes: int
    sha256: str
    role: str


@dataclass(frozen=True)
class Candidate:
    data: dict[str, Any]

    @property
    def id(self) -> str:
        return cast(str, self.data["id"])

    def __getattr__(self, name: str) -> Any:
        try:
            return self.data[name]
        except KeyError as error:
            raise AttributeError(name) from error

    @property
    def files(self) -> tuple[RegistryFile, ...]:
        return tuple(RegistryFile(**item) for item in self.data["files"])

    @property
    def total_bytes(self) -> int:
        return sum(item.bytes for item in self.files)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("registry contains duplicate keys")
        result[key] = value
    return result


def _load_json(raw: bytes) -> dict[str, Any]:
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON number")),
    )
    if not isinstance(value, dict):
        raise ValueError("registry root must be an object")
    return value


def _url(value: Any, *, name: str) -> None:
    if not isinstance(value, str) or not value.startswith("https://"):
        raise ValueError(f"{name} must be an HTTPS URL")


def _validate_file(raw: Any) -> None:
    if not isinstance(raw, dict) or set(raw) != {"path", "bytes", "sha256", "role"}:
        raise ValueError("registry file entry is not closed")
    path = raw["path"]
    if (
        not isinstance(path, str)
        or not path
        or Path(path).is_absolute()
        or ".." in Path(path).parts
        or "\\" in path
        or Path(path).suffix not in ALLOWED_EXTENSIONS
    ):
        raise ValueError("registry file path is unsafe")
    if not isinstance(raw["bytes"], int) or isinstance(raw["bytes"], bool) or raw["bytes"] <= 0:
        raise ValueError("registry file size is invalid")
    if not isinstance(raw["sha256"], str) or not HEX64.fullmatch(raw["sha256"]):
        raise ValueError("registry file digest is invalid")
    if not isinstance(raw["role"], str) or not raw["role"]:
        raise ValueError("registry file role is invalid")


def _acquisition_subset(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: candidate[key]
        for key in (
            "id",
            "model_id",
            "model_revision",
            "weight_format",
            "source_weight_dtypes",
            "loader_family",
            "files",
        )
    } | {"dependency_contract": DEPENDENCY_CONTRACT}


def acquisition_contract_digest(candidate: Candidate | dict[str, Any]) -> str:
    data = candidate.data if isinstance(candidate, Candidate) else candidate
    return hashlib.sha256(rfc8785.dumps(_acquisition_subset(data))).hexdigest()


def _new_field_truth_digest(candidate: dict[str, Any]) -> str:
    return hashlib.sha256(
        rfc8785.dumps({key: candidate[key] for key in sorted(NEW_FIELDS)})
    ).hexdigest()


def _validate_candidate(candidate: Any, v1_by_id: dict[str, dict[str, Any]]) -> Candidate:
    if not isinstance(candidate, dict) or set(candidate) != CANDIDATE_FIELDS:
        raise ValueError("candidate entry is not closed")
    cid = candidate["id"]
    if not isinstance(cid, str) or not ID.fullmatch(cid):
        raise ValueError("candidate id is invalid")
    if cid not in (
        *ELIGIBLE,
        "saracura-compiled",
        "von-option-marker",
        "qwen-system-one",
        "typesafe-jev",
        "diffusiongemma-openjev",
    ):
        raise ValueError("unexpected candidate id")
    for key in ("source_url", "license_evidence_url"):
        _url(candidate[key], name=key)
    if candidate["source_revision"] is not None and not HEX40.fullmatch(
        candidate["source_revision"]
    ):
        raise ValueError("source revision is invalid")
    if candidate["model_revision"] is not None and not HEX40.fullmatch(candidate["model_revision"]):
        raise ValueError("model revision is invalid")
    if candidate["model_id"] is not None and (
        not isinstance(candidate["model_id"], str) or "/" not in candidate["model_id"]
    ):
        raise ValueError("model id is invalid")
    for key in (
        "model_source_url",
        "model_license_evidence_url",
        "base_model_license_evidence_url",
    ):
        if candidate[key] is not None:
            _url(candidate[key], name=key)
    if candidate["base_model_revision"] is not None and not HEX40.fullmatch(
        candidate["base_model_revision"]
    ):
        raise ValueError("base revision is invalid")
    if candidate["acquisition_state"] == "eligible":
        if cid not in ELIGIBLE:
            raise ValueError("only selected candidates may be eligible")
        if candidate["dependency_contract"] != DEPENDENCY_CONTRACT:
            raise ValueError("eligible dependency contract mismatch")
        if candidate["weight_format"] != "safetensors" or candidate["loader_family"] not in {
            "laya_option_marker",
            "mdeberta_nli",
        }:
            raise ValueError("eligible loader or weight format mismatch")
        if (
            not isinstance(candidate["source_weight_dtypes"], dict)
            or not candidate["source_weight_dtypes"]
        ):
            raise ValueError("eligible dtype map is required")
        if any(
            not isinstance(k, str) or not isinstance(v, int) or v <= 0
            for k, v in candidate["source_weight_dtypes"].items()
        ):
            raise ValueError("eligible dtype map is invalid")
        files = candidate["files"]
        if not isinstance(files, list) or not files:
            raise ValueError("eligible file allowlist is required")
        for item in files:
            _validate_file(item)
        paths = [item["path"] for item in files]
        if len(paths) != len(set(paths)):
            raise ValueError("candidate file paths must be unique")
        if not isinstance(candidate["acquisition_contract_digest"], str) or not HEX64.fullmatch(
            candidate["acquisition_contract_digest"]
        ):
            raise ValueError("eligible acquisition digest is invalid")
        if acquisition_contract_digest(candidate) != candidate["acquisition_contract_digest"]:
            raise ValueError("acquisition contract digest mismatch")
        if (
            candidate["conformance_state"] != "pending"
            or candidate["conformance_vector_sha256"] is not None
        ):
            raise ValueError("Phase 4C.2a requires pending conformance")
        if candidate["architecture_attribution"] != "positive_compatibility_only":
            raise ValueError("eligible attribution is invalid")
        for key in (
            "model_source_url",
            "model_license",
            "model_license_evidence_url",
            "base_model_id",
            "base_model_revision",
            "base_model_license",
            "base_model_license_evidence_url",
            "tokenizer_provenance",
            "training_data_provenance",
            "licensing_disposition",
        ):
            if candidate[key] is None:
                raise ValueError(f"eligible provenance field missing: {key}")
    else:
        if any(
            candidate[key] is not None
            for key in (
                "model_source_url",
                "model_license",
                "model_license_evidence_url",
                "base_model_id",
                "base_model_revision",
                "base_model_license",
                "base_model_license_evidence_url",
                "tokenizer_provenance",
                "training_data_provenance",
                "licensing_disposition",
                "source_weight_dtypes",
                "files",
                "acquisition_contract_digest",
            )
        ):
            raise ValueError("non-eligible candidate exposes acquisition data")
        if (
            candidate["dependency_contract"] is not None
            or candidate["loader_family"] != "not_executable"
        ):
            raise ValueError("non-eligible candidate execution fields are invalid")
        if (
            candidate["conformance_state"] != "not_applicable"
            or candidate["conformance_vector_sha256"] is not None
        ):
            raise ValueError("non-eligible conformance fields are invalid")
        if candidate["architecture_attribution"] != "not_applicable":
            raise ValueError("non-eligible attribution is invalid")
    predecessor = "poorjev-nli" if cid == "mdeberta-nli" else cid
    if predecessor not in v1_by_id:
        raise ValueError("candidate is missing from v1 predecessor registry")
    old = v1_by_id[predecessor]
    for key in IDENTITY_FIELDS - {"id", "display_name", "model_id", "model_revision"}:
        if candidate[key] != old[key]:
            raise ValueError(f"shared candidate field changed: {key}")
    if _new_field_truth_digest(candidate) != NEW_FIELD_TRUTH_DIGESTS[cid]:
        raise ValueError("candidate v2 truth table mismatch")
    return Candidate(candidate)


def load_registry(raw: bytes | None = None) -> tuple[Candidate, ...]:
    payload = _load_json(REGISTRY_PATH.read_bytes() if raw is None else raw)
    if set(payload) != ROOT_FIELDS or payload["schema_version"] != SCHEMA:
        raise ValueError("registry root is not the closed v2 shape")
    if payload["reviewed_at"] != "2026-09-22":
        raise ValueError("registry review date mismatch")
    if payload["supersedes_registry_sha256"] != V1_DIGEST:
        raise ValueError("registry predecessor digest mismatch")
    if payload["predecessor_map"] != {"poorjev-nli": "mdeberta-nli"}:
        raise ValueError("registry predecessor map mismatch")
    if not isinstance(payload["candidates"], list) or len(payload["candidates"]) != 7:
        raise ValueError("registry candidate count mismatch")
    v1_raw = V1_PATH.read_bytes()
    if hashlib.sha256(v1_raw).hexdigest() != V1_DIGEST:
        raise ValueError("registry predecessor bytes mismatch")
    v1 = _load_json(v1_raw)
    v1_by_id = {item["id"]: item for item in v1["candidates"]}
    result = tuple(_validate_candidate(item, v1_by_id) for item in payload["candidates"])
    if len({item.id for item in result}) != len(result):
        raise ValueError("candidate ids must be unique")
    if tuple(item.id for item in result) != CANDIDATE_ORDER:
        raise ValueError("registry candidate identities mismatch")
    return result


def registry_bytes() -> bytes:
    return REGISTRY_PATH.read_bytes()


def registry_digest() -> str:
    return hashlib.sha256(registry_bytes()).hexdigest()


def validate_registry() -> tuple[Candidate, ...]:
    return load_registry()


def get_candidate(candidate_id: str) -> Candidate:
    for candidate in load_registry():
        if candidate.id == candidate_id:
            return candidate
    raise KeyError("unknown universal-local candidate")


def file_url(candidate: Candidate, path: str) -> str:
    allowed = {item.path for item in candidate.files}
    if path not in allowed:
        raise ValueError("file is not in the reviewed allowlist")
    return (
        f"https://huggingface.co/{candidate.model_id}/resolve/{candidate.model_revision}/"
        f"{quote(path, safe='/')}"
    )
