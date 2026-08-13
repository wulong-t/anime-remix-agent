"""Image-First execution contracts and Phase 2 infrastructure.

Phase 1 of the Round 1..10 frozen design (2026-08-11): these documents are
independent research contracts.  They do not modify the legacy Timeline 1.9
schema and are not consumed by the legacy Renderer.
"""

from anime_remix.services.execution.adapter import (
    QwenImage30Adapter,
    QwenImage30ProAdapter,
    QwenImageEditAdapter,
    StubImageExecutor,
)
from anime_remix.services.execution.artifact_store import (
    ArtifactStore,
    register_artifact,
)
from anime_remix.services.execution.dashscope_executor import (
    DashScopeQwenExecutor,
)
from anime_remix.services.execution.execution_ledger import (
    parse_ledger_record,
)
from anime_remix.services.execution.first_frame_composer import (
    FirstFrameComposeResult,
    run_first_frame_composition,
)
from anime_remix.services.execution.generated_shot_pipeline import (
    GeneratedShotInputsDocument,
    GeneratedShotRunResult,
    parse_generated_shot_inputs,
    run_generated_shot_pipeline,
)
from anime_remix.services.execution.handoff_frame_composer import (
    HandoffFrameComposeResult,
    run_handoff_frame_composition,
)
from anime_remix.services.execution.layout_plan import (
    parse_layout_plan,
)
from anime_remix.services.execution.ledger_writer import (
    LedgerWriter,
)
from anime_remix.services.execution.local_lora_executor import (
    LocalLoraExecutor,
    LocalLoraStackAdapter,
)
from anime_remix.services.execution.orchestrator import (
    run_compose_keyframe,
)
from anime_remix.services.execution.prepared_component_composer import (
    PreparedComponentComposeResult,
    run_prepared_component_composition,
)
from anime_remix.services.execution.reference_package import (
    parse_reference_package,
)
from anime_remix.services.execution.replicate_executor import (
    ReplicateQwenExecutor,
)
from anime_remix.services.execution.resolver import (
    Resolver,
)
from anime_remix.services.execution.shot_keyframe_runner import (
    ShotKeyframeRunResult,
    run_shot_keyframes,
)
from anime_remix.services.execution.shot_spec import (
    parse_shot_spec,
)

__all__ = [
    "ArtifactStore",
    "DashScopeQwenExecutor",
    "FirstFrameComposeResult",
    "GeneratedShotInputsDocument",
    "GeneratedShotRunResult",
    "HandoffFrameComposeResult",
    "LedgerWriter",
    "LocalLoraExecutor",
    "LocalLoraStackAdapter",
    "PreparedComponentComposeResult",
    "QwenImage30Adapter",
    "QwenImage30ProAdapter",
    "QwenImageEditAdapter",
    "ReplicateQwenExecutor",
    "Resolver",
    "ShotKeyframeRunResult",
    "StubImageExecutor",
    "parse_generated_shot_inputs",
    "parse_layout_plan",
    "parse_ledger_record",
    "parse_reference_package",
    "parse_shot_spec",
    "register_artifact",
    "run_compose_keyframe",
    "run_first_frame_composition",
    "run_generated_shot_pipeline",
    "run_handoff_frame_composition",
    "run_prepared_component_composition",
    "run_shot_keyframes",
]
