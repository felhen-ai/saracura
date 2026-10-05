from __future__ import annotations

import re
from pathlib import Path

REQUIRED_FRONTMATTER_FIELDS = {
    "title",
    "kind",
    "area",
    "project",
    "collection",
    "owner",
    "status",
    "canonical",
    "globalRef",
    "reviewCadenceDays",
    "lastReviewedAt",
    "sourceRefs",
    "related",
    "supersedes",
    "supersededBy",
    "sensitivity",
}


def test_canonical_markdown_has_structural_frontmatter() -> None:
    root = Path(__file__).parents[1]
    markdown_files = sorted(
        path
        for path in root.rglob("*.md")
        if not any(part.startswith(".") for part in path.relative_to(root).parts)
        and "dist" not in path.relative_to(root).parts
    )
    assert markdown_files

    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n"), f"{path}: missing frontmatter opener"
        _, frontmatter, _ = text.split("---", maxsplit=2)
        fields = set(re.findall(r"^([A-Za-z][A-Za-z0-9]*):", frontmatter, re.MULTILINE))
        assert fields == REQUIRED_FRONTMATTER_FIELDS, f"{path}: frontmatter fields drifted"
        relative = path.relative_to(root).as_posix()
        assert re.search(rf"^canonical: {re.escape(relative)}$", frontmatter, re.MULTILINE)
        assert re.search(r"^sensitivity: public$", frontmatter, re.MULTILINE)


def test_shadow_public_boundary_is_documented_in_spec_and_agents_policy() -> None:
    root = Path(__file__).parents[1]
    spec = (root / "docs/action/specs/phase4f1-shadow-evaluation-and-email-reference.md").read_text(
        encoding="utf-8"
    )
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")

    assert "shadow-decide" in spec
    assert "shadow-evaluate" in spec
    assert "--encoder-snapshot" in spec
    assert "--training-capsule" in spec
    assert "automation_allowed=false" in spec
    assert "Gmail" in spec and "Outlook" in spec and "IMAP" in spec
    assert "Phase 4F.2" in spec

    normalized_spec = " ".join(spec.split())
    for phrase in (
        "ranking weights are never named confidence",
        "does not connect",
    ):
        assert phrase in normalized_spec
    # ADR 0005 moved the product boundary; AGENTS.md states it and keeps the runtime invariants.
    assert "ADR 0005" in agents
    assert "open base models, benchmark first" in agents
    assert "Teacher agreement is reported as agreement, not accuracy" in agents
    assert "Safety invariants" in agents
    assert "Never add telemetry, remote fallback, arbitrary model paths" in agents


def test_phase4f1_smoke_result_is_bound_and_conservative_in_both_readmes() -> None:
    root = Path(__file__).parents[1]
    report = (root / "docs/action/phase4f1-shadow-evaluation-result.md").read_text(encoding="utf-8")

    for binding in (
        "b6a50c51a38db49d09a310f115dbb31ecc5b675b",
        "fefe634cb4da9aa2b8afa91de84e1d284a4996f26c62e09c562c8cefc2131110",
        "dded5e4bea3c97a4dc3d7e86945a78b81df1bb4675ef3a0a3e9e158a1575c678",
        "a7497280d6faaaa6209c80bf61049992c607aa0bd0cb5c2a1164088d164be50b",
        "phase4e-saracura-ranker.v1.f1d72c34cc535ddefbf7e24cf45d1be6e0aee40ba881228b4edd092d78640d5e",
    ):
        assert binding in report
    for aggregate in (
        "8 predictions and 8 feedback records",
        "7 labeled, 1 skipped, and 0 unreviewed",
        "1 of 7 labeled records (14.2857%)",
        "73 tokens",
        "55 tokens under the 128-token context limit",
        "one runner/backend lifecycle",
        "0 errors",
        "canary text absent",
    ):
        assert aggregate in report
    for conservative_claim in (
        "not a model-quality evaluation",
        "must not be interpreted as",
        "strictly read-only",
        "with human review",
        "not training",
    ):
        assert conservative_claim in report
    for private_value in ("/Volumes/", "/Users/", "Felhen", "checkpoint.safetensors"):
        assert private_value not in report


def test_open_base_positioning_and_readme_parity() -> None:
    root = Path(__file__).parents[1]
    english = (root / "README.md").read_text(encoding="utf-8")
    portuguese = (root / "docs/README.pt-BR.md").read_text(encoding="utf-8")
    adr = (root / "docs/decisions/0005-open-base-benchmark-first.md").read_text(encoding="utf-8")
    superseded = (root / "docs/decisions/0004-model-first-product-direction.md").read_text(
        encoding="utf-8"
    )
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    claude = (root / "CLAUDE.md").read_text(encoding="utf-8")

    benchmark = "https://huggingface.co/datasets/felhen-ai/ptbr-typed-decisions-bench"
    model = "https://huggingface.co/felhen-ai/saracura-ptbr-v0"
    for document in (english, portuguese):
        assert benchmark in document
        assert model in document
        assert "0005-open-base-benchmark-first.md" in document
        assert "research/bench.py" in document
        # Published numbers stay identical in both front doors.
        for figure in ("27,6%", "39,2%", "44,6%", "64,0%", "66,2%", "68,5%"):
            assert figure.replace(",", ".") in english
            assert figure in portuguese
        assert "balanced accuracy" in english and "acurácia balanceada" in portuguese

    assert "supersedes:\n  - docs/decisions/0004-model-first-product-direction.md" in adr
    assert "Open base models, benchmark first" in adr
    assert "weights are published, rows are not" in adr
    assert "status: historical" in superseded
    assert "0005-open-base-benchmark-first.md" in superseded

    # AGENTS.md and CLAUDE.md carry the same instructions; only the canonical path differs.
    def normalize(text: str) -> str:
        for marker in (
            "canonical: AGENTS.md",
            "canonical: CLAUDE.md",
            "qmd://saracura/AGENTS.md",
            "qmd://saracura/CLAUDE.md",
        ):
            text = text.replace(marker, "")
        return text

    assert normalize(agents) == normalize(claude)
