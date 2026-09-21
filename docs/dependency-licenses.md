---
title: Saracura runtime dependency licenses
kind: reference
area: development
project: saracura
collection: saracura
owner: saracura-maintainers
status: current
canonical: docs/dependency-licenses.md
globalRef: qmd://saracura/docs/dependency-licenses.md
reviewCadenceDays: 90
lastReviewedAt: 2026-09-21
sourceRefs:
  - https://github.com/pydantic/pydantic
  - https://github.com/trailofbits/rfc8785.py
related:
  - pyproject.toml
  - uv.lock
  - LICENSE
supersedes: []
supersededBy: []
sensitivity: public
---

# Runtime dependency licenses

This snapshot covers the runtime dependency graph resolved by `uv.lock`. Build and development tools are not shipped in the Saracura wheel.

| Package | Resolved version | License | Role |
| --- | --- | --- | --- |
| `pydantic` | 2.13.5 | MIT | closed runtime contracts and validation |
| `pydantic-core` | 2.46.5 | MIT | transitive Pydantic runtime |
| `annotated-types` | 0.8.0 | MIT | transitive Pydantic annotations |
| `typing-inspection` | 0.4.4 | MIT | transitive Pydantic inspection |
| `typing-extensions` | 4.16.0 | PSF-2.0 | transitive typing compatibility |
| `rfc8785` | 0.1.4 | Apache-2.0 | JSON Canonicalization Scheme implementation |

The lockfile is authoritative for resolved versions. This table must be reviewed whenever the runtime dependency graph changes and does not replace the upstream license texts or notices.
