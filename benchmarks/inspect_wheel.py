"""Reject weights and executable artifacts from a built wheel."""

from __future__ import annotations

import sys
import tarfile
import zipfile

FORBIDDEN = (".bin", ".pkl", ".pickle", ".so", ".dylib", ".dll", ".pyc", ".pt", ".pth")
REGISTRY_ENTRY = "saracura/encoder-candidates.v1.json"


def _allowed_entry(name: str) -> bool:
    if name == REGISTRY_ENTRY:
        return True
    if name.startswith("saracura/"):
        return name.endswith((".py", "/py.typed"))
    if ".dist-info/" in name:
        suffix = name.split(".dist-info/", 1)[1]
        return suffix in {"METADATA", "WHEEL", "entry_points.txt", "RECORD"} or suffix == (
            "licenses/LICENSE"
        )
    return False


def main() -> int:
    for name in sys.argv[1:]:
        if name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(name, "r:gz") as archive:
                entries = archive.getnames()
                bad = [
                    item
                    for item in entries
                    if "/.venv" in item or "/.artifacts" in item or item.endswith(FORBIDDEN)
                ]
                if bad:
                    raise SystemExit("forbidden sdist entries")
            continue
        with zipfile.ZipFile(name) as archive:
            entries = archive.namelist()
            unexpected = [item for item in entries if not _allowed_entry(item)]
            if unexpected:
                raise SystemExit("unexpected wheel entries")
            if entries.count(REGISTRY_ENTRY) != 1:
                raise SystemExit("encoder registry missing from wheel")
            bad = [item for item in entries if item.lower().endswith(FORBIDDEN)]
            if bad:
                raise SystemExit("forbidden wheel entries: " + ",".join(bad))
    print("wheel contains no weights or executable binaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
