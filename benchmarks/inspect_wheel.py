"""Reject weights and executable artifacts from a built wheel."""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import PurePosixPath

FORBIDDEN = (".bin", ".pkl", ".pickle", ".so", ".dylib", ".dll", ".pyc", ".pt", ".pth", ".jsonl")
REGISTRY_ENTRY = "saracura/encoder-candidates.v1.json"
PACKET_BASENAMES = {
    "packet.json",
    "families.jsonl",
    "records.jsonl",
    "reviews.jsonl",
    "adjudications.jsonl",
    "split-plan.json",
    "audit-report.json",
}


def _forbidden_artifact(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith((*FORBIDDEN, ".safetensors", ".onnx")) or (
        PurePosixPath(lowered).name in PACKET_BASENAMES
    )


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
    if len(sys.argv) < 3:
        raise SystemExit("at least one wheel and one sdist are required")
    saw_wheel = saw_sdist = False
    for name in sys.argv[1:]:
        if name.endswith((".tar.gz", ".tgz")):
            saw_sdist = True
            with tarfile.open(name, "r:gz") as archive:
                entries = archive.getnames()
                bad = [
                    item
                    for item in entries
                    if "/.venv" in item or "/.artifacts" in item or _forbidden_artifact(item)
                ]
                if bad:
                    raise SystemExit("forbidden sdist entries")
            continue
        with zipfile.ZipFile(name) as archive:
            saw_wheel = True
            entries = archive.namelist()
            unexpected = [item for item in entries if not _allowed_entry(item)]
            if unexpected:
                raise SystemExit("unexpected wheel entries")
            if entries.count(REGISTRY_ENTRY) != 1:
                raise SystemExit("encoder registry missing from wheel")
            bad = [item for item in entries if _forbidden_artifact(item)]
            if bad:
                raise SystemExit("forbidden wheel entries: " + ",".join(bad))
    if not saw_wheel or not saw_sdist:
        raise SystemExit("one concrete wheel and one concrete sdist are required")
    print("wheel and sdist contain no weights, data artifacts, or executable binaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
