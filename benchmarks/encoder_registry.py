"""Closed, reviewed registry for the optional encoder research lane."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

REGISTRY_PATH = Path(__file__).parent / "manifests" / "encoder-candidates.v1.json"
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Identifier = Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9-]*$")]


class Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RegistryFile(Closed):
    path: Annotated[str, Field(min_length=1, max_length=160)]
    bytes: Annotated[int, Field(gt=0)]
    sha256: Sha256

    @model_validator(mode="after")
    def safe_path(self) -> RegistryFile:
        p = Path(self.path)
        if p.is_absolute() or ".." in p.parts or p.name != self.path:
            raise ValueError("registry file path must be a relative leaf")
        if p.suffix not in {".json", ".model", ".safetensors"}:
            raise ValueError("registry file extension is not allowlisted")
        if self.path.endswith(".bin") or ".pkl" in self.path:
            raise ValueError("unsafe weight path")
        return self


class Candidate(Closed):
    id: Identifier
    repository: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")]
    revision: Revision
    license: Literal["MIT", "Apache-2.0"]
    role: Identifier
    disposition: Literal["blocked", "eligible"]
    reason: Annotated[str, Field(min_length=1, max_length=500)]
    reviewed_at: Annotated[str, Field(pattern=r"^2026-09-21$")]
    files: tuple[RegistryFile, ...]
    model_type: Literal["xlm-roberta", "bert"] | None = None
    architecture: Literal["XLMRobertaModel", "BertModel"] | None = None
    hidden_width: Literal[384, 768] | None = None
    loader: Literal["xlm-roberta", "bert"] | None = None
    tokenizer: Literal["XLMRobertaTokenizerFast", "PreTrainedTokenizerFast"] | None = None

    @model_validator(mode="after")
    def validate_files(self) -> Candidate:
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("candidate file paths must be unique")
        if self.disposition == "blocked":
            if self.files:
                raise ValueError("blocked candidate cannot expose an allowlist")
        else:
            if sum(item.path == "model.safetensors" for item in self.files) != 1:
                raise ValueError("eligible candidate needs exactly one safetensors file")
            if not (
                self.model_type
                and self.architecture
                and self.hidden_width
                and self.loader
                and self.tokenizer
            ):
                raise ValueError("eligible candidate needs loader metadata")
            expected = {
                "xlm-roberta-base": (
                    "xlm-roberta",
                    "XLMRobertaModel",
                    768,
                    "xlm-roberta",
                    "XLMRobertaTokenizerFast",
                ),
                "multilingual-minilm-l12": (
                    "bert",
                    "BertModel",
                    384,
                    "bert",
                    "PreTrainedTokenizerFast",
                ),
            }.get(self.id)
            if (
                expected is None
                or (
                    self.model_type,
                    self.architecture,
                    self.hidden_width,
                    self.loader,
                    self.tokenizer,
                )
                != expected
            ):
                raise ValueError("candidate loader metadata is not the reviewed combination")
        return self

    @property
    def total_bytes(self) -> int:
        return sum(item.bytes for item in self.files)


class Registry(Closed):
    schema_version: Literal["encoder-candidates.v1"]
    candidates: tuple[Candidate, ...]

    @model_validator(mode="after")
    def unique_ids(self) -> Registry:
        ids = [candidate.id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate ids must be unique")
        return self


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("registry contains duplicate keys")
        result[key] = value
    return result


def registry_bytes() -> bytes:
    return REGISTRY_PATH.read_bytes()


def registry_digest() -> str:
    return hashlib.sha256(registry_bytes()).hexdigest()


def load_registry(raw: bytes | None = None) -> Registry:
    payload = registry_bytes() if raw is None else raw
    decoded = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(decoded, dict):
        raise ValueError("registry root must be an object")
    # JSON validation performs the deliberate list-to-tuple conversion while
    # the duplicate-key hook above still guarantees a closed object shape.
    return Registry.model_validate_json(json.dumps(decoded))


def get_candidate(candidate_id: str) -> Candidate:
    for candidate in load_registry().candidates:
        if candidate.id == candidate_id:
            return candidate
    raise KeyError("unknown encoder candidate")


def validate_registry() -> Registry:
    registry = load_registry()
    if registry_digest() == "":
        raise ValueError("registry digest unavailable")
    return registry
