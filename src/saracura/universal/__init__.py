"""Phase 4E universal-ranker contracts registered by the saracura-universal backend.

The package contains deterministic schemas, math, and checkpoint validation.
The installed ``SaracuraUniversalBackend`` binds verified MiniLM embeddings and
the sealed Phase 4E projection checkpoint.  The backend is opt-in, requires
private verified artifacts, and returns only uncalibrated abstained research-safe
rankings.  Calibration, automation and public model publication remain out of scope.
"""

from saracura.universal.checkpoint import (
    CHECKPOINT_ARCHITECTURE_REVISION,
    CheckpointDescriptor,
    CheckpointTensor,
    validate_checkpoint_descriptor,
)
from saracura.universal.ranker import (
    RankerParameters,
    SaracuraUniversalRanker,
    VectorProjection,
)
from saracura.universal.rendering import (
    CapacityError,
    RenderedTask,
    render_task,
    validate_rendered_capacity,
)
from saracura.universal.tasks import (
    SARACURA_UNIVERSAL_WORKFLOW_ID,
    SARACURA_UNIVERSAL_WORKFLOW_REVISION,
    UniversalTask,
)

__all__ = [
    "CHECKPOINT_ARCHITECTURE_REVISION",
    "SARACURA_UNIVERSAL_WORKFLOW_ID",
    "SARACURA_UNIVERSAL_WORKFLOW_REVISION",
    "CapacityError",
    "CheckpointDescriptor",
    "CheckpointTensor",
    "RankerParameters",
    "RenderedTask",
    "SaracuraUniversalRanker",
    "UniversalTask",
    "VectorProjection",
    "render_task",
    "validate_checkpoint_descriptor",
    "validate_rendered_capacity",
]
