"""Planned Phase 4E universal-ranker contracts with no runtime registration.

The package intentionally contains deterministic schemas and math only.  It
does not load an encoder, checkpoint, optional ML dependency, or backend.
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
