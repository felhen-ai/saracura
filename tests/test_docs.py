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
