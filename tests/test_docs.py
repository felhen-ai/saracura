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


def test_shadow_public_boundary_is_documented_in_both_languages_and_agents_policy() -> None:
    root = Path(__file__).parents[1]
    english = (root / "README.md").read_text(encoding="utf-8")
    portuguese = (root / "docs/README.pt-BR.md").read_text(encoding="utf-8")
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")

    for document, phase_marker in (
        (english, "Phase 4F.2"),
        (portuguese, "Fase 4F.2"),
    ):
        assert "shadow-decide" in document
        assert "shadow-evaluate" in document
        assert "--encoder-snapshot" in document
        assert "--training-capsule" in document
        assert "automation_allowed=false" in document
        assert "Gmail" in document and "Outlook" in document and "IMAP" in document
        assert phase_marker in document

    normalized_english = " ".join(english.split())
    normalized_portuguese = " ".join(portuguese.split())
    for phrase in (
        "ranking weights are not confidence",
        "must not be used to train",
        "does not connect",
    ):
        assert phrase in normalized_english
    for phrase in (
        "pesos ordenados de ranking não são confiança",
        "não deve ser reutilizado para treino",
        "não se conecta",
    ):
        assert phrase in normalized_portuguese
    assert "read-only evaluator" in agents
    assert "no Gmail, Outlook, IMAP" in agents
    assert "automation_allowed=false" in agents
