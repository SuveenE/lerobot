"""Minimal pre/post processors for OpenVLA-OFT.

OpenVLA-OFT handles its own image preprocessing (resize, center crop, VLA processor)
and action unnormalization internally. These processors only handle:
- Preprocessor: key renaming, batch dimension, device placement
- Postprocessor: device placement (move to CPU)
"""

from __future__ import annotations

from typing import Any

import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_openvla_oft import OpenVLAOFTConfig


def make_openvla_oft_pre_post_processors(
    config: OpenVLAOFTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    preprocessor_overrides: dict[str, Any] | None = None,
    postprocessor_overrides: dict[str, Any] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create minimal pre/post processor pipelines for OpenVLA-OFT.

    No normalization or unnormalization is applied since OpenVLA-OFT
    handles these internally.
    """
    preprocessor_overrides = preprocessor_overrides or {}
    postprocessor_overrides = postprocessor_overrides or {}

    rename_cfg = preprocessor_overrides.get("rename_observations_processor", {})
    rename_map = rename_cfg.get("rename_map", {})

    pre_device_cfg = preprocessor_overrides.get("device_processor", {})
    pre_device = pre_device_cfg.get("device", config.device)

    post_device_cfg = postprocessor_overrides.get("device_processor", {})
    post_device = post_device_cfg.get("device", "cpu")

    input_steps = [
        RenameObservationsProcessorStep(rename_map=rename_map),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=pre_device),
    ]

    output_steps = [
        DeviceProcessorStep(device=post_device),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
