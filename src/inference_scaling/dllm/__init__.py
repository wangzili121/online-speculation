"""Diffusion language-model sampling, weighting, and alignment utilities."""

from inference_scaling.dllm.config import (
    DiffusionBlockBeamConfig,
    DiffusionISConfig,
    DiffusionMHConfig,
    DiffusionPowerMHConfig,
    DiffusionSamplingConfig,
    VRPOSamplingConfig,
    diffusion_decision_stage_lengths,
)
from inference_scaling.dllm.types import (
    DiffusionBackend,
    DiffusionGenerationRequest,
    DiffusionSample,
    DiffusionTraceStep,
    DiffusionTrajectoryScoreRequest,
)
from inference_scaling.dllm.replay import (
    DiffusionReplayCandidate,
    DiffusionReplayHistory,
    DiffusionReplayRecord,
    DiffusionReplaySelection,
    build_diffusion_replay_history,
    select_diffusion_candidates_with_replay,
)
from inference_scaling.dllm.vrpo import (
    VRPOMaskPlan,
    VRPOMaskSample,
    VRPOPreferenceEstimate,
    estimate_masked_elbo,
    estimate_vrpo_preference_loss,
    sample_vrpo_mask_plan,
)
from inference_scaling.dllm.dynamic_is import (
    DynamicDiffusionDraw,
    DynamicDiffusionResult,
    DynamicDiffusionStep,
    draw_defensive_diffusion_candidates,
    run_dynamic_diffusion_is,
)

__all__ = [
    "DiffusionBackend",
    "DiffusionBlockBeamConfig",
    "DiffusionGenerationRequest",
    "DiffusionISConfig",
    "DiffusionMHConfig",
    "DiffusionPowerMHConfig",
    "DiffusionReplayCandidate",
    "DiffusionReplayHistory",
    "DiffusionReplayRecord",
    "DiffusionReplaySelection",
    "DiffusionSample",
    "DiffusionSamplingConfig",
    "DiffusionTraceStep",
    "DiffusionTrajectoryScoreRequest",
    "DynamicDiffusionDraw",
    "DynamicDiffusionResult",
    "DynamicDiffusionStep",
    "VRPOSamplingConfig",
    "VRPOMaskPlan",
    "VRPOMaskSample",
    "VRPOPreferenceEstimate",
    "estimate_masked_elbo",
    "estimate_vrpo_preference_loss",
    "diffusion_decision_stage_lengths",
    "build_diffusion_replay_history",
    "draw_defensive_diffusion_candidates",
    "sample_vrpo_mask_plan",
    "run_dynamic_diffusion_is",
    "select_diffusion_candidates_with_replay",
]
