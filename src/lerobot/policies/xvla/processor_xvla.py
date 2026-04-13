# ------------------------------------------------------------------------------
# Copyright 2025 The HuggingFace Inc. team and 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    EnvTransition,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_xvla import XVLAConfig
from .utils import rotate6d_to_axis_angle

OBS_PREFIX = "observation."

IMAGENET_STATS = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def make_xvla_pre_post_processors(
    config: XVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    features = {**config.input_features, **config.output_features}
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding=config.pad_language_to,
            padding_side=config.tokenizer_padding_side,
        ),
        XVLAImageToFloatProcessorStep(),
        XVLAImageNetNormalizeProcessorStep(),
        XVLAAddDomainIdProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features=features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
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


@dataclass
class LiberoProcessorStep(ObservationProcessorStep):
    """Processes LIBERO observations into the LeRobot format."""

    def _process_observation(self, observation):
        processed_obs = observation.copy()
        for key in list(processed_obs.keys()):
            if key.startswith(f"{OBS_IMAGES}."):
                img = processed_obs[key]
                if key == f"{OBS_IMAGES}.image":
                    img = torch.flip(img, dims=[2, 3])
                processed_obs[key] = img

        robot_state_str = OBS_PREFIX + "robot_state"
        if robot_state_str in processed_obs:
            robot_state = processed_obs.pop(robot_state_str)

            eef_pos = robot_state["eef"]["pos"]
            eef_mat = robot_state["eef"]["mat"]
            eef_rot6d = self._mat_to_rotate6d(eef_mat)
            extra = torch.zeros((eef_pos.shape[0], 1), dtype=torch.float32, device=eef_pos.device)
            proprio_state = torch.cat((eef_pos, eef_rot6d, extra), dim=-1)
            state = torch.cat((proprio_state, torch.zeros_like(proprio_state)), dim=-1)
            state = state.float()
            if state.dim() == 1:
                state = state.unsqueeze(0)
            processed_obs[OBS_STATE] = state
        return processed_obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {}
        for ft, feats in features.items():
            if ft != PipelineFeatureType.STATE:
                new_features[ft] = feats.copy()
        state_feats = {}
        state_feats[OBS_STATE] = PolicyFeature(
            key=OBS_STATE,
            shape=(20,),
            dtype="float32",
        )
        new_features[PipelineFeatureType.STATE] = state_feats
        return new_features

    def _mat_to_rotate6d(self, rot_mats: torch.Tensor) -> torch.Tensor:
        if not isinstance(rot_mats, torch.Tensor):
            raise TypeError(f"mat_to_rot6d expects a torch.Tensor, got {type(rot_mats)}")
        if rot_mats.ndim != 3 or rot_mats.shape[1:] != (3, 3):
            raise ValueError(f"mat_to_rot6d expects shape (B, 3, 3), got {tuple(rot_mats.shape)}")
        rot_mats = rot_mats.to(torch.float32)
        col1 = rot_mats[:, :3, 0]
        col2 = rot_mats[:, :3, 1]
        rot6d = torch.cat([col1, col2], dim=-1)
        return rot6d

    def observation(self, observation):
        return self._process_observation(observation)


@dataclass
@ProcessorStepRegistry.register(name="xvla_image_scale")
class XVLAImageScaleProcessorStep(ProcessorStep):
    """Scale image observations by 255 to convert from [0, 1] to [0, 255] range."""

    image_keys: list[str] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        if obs is None:
            return new_transition
        obs = obs.copy()

        keys_to_scale = self.image_keys
        if keys_to_scale is None:
            keys_to_scale = [k for k in obs if k.startswith(OBS_IMAGES)]

        for key in keys_to_scale:
            if key in obs and isinstance(obs[key], torch.Tensor):
                obs[key] = obs[key] * 255

        new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {"image_keys": self.image_keys}


@dataclass
@ProcessorStepRegistry.register(name="xvla_image_to_float")
class XVLAImageToFloatProcessorStep(ProcessorStep):
    """Convert image observations from [0, 255] to [0, 1] range."""

    image_keys: list[str] | None = None
    validate_range: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        if obs is None:
            return new_transition
        obs = obs.copy()

        keys_to_convert = self.image_keys
        if keys_to_convert is None:
            keys_to_convert = [k for k in obs if k.startswith(OBS_IMAGES)]

        for key in keys_to_convert:
            if key in obs and isinstance(obs[key], torch.Tensor):
                tensor = obs[key]
                min_val = tensor.min().item()
                max_val = tensor.max().item()

                if max_val <= 1.0:
                    obs[key] = tensor.float()
                    continue

                if self.validate_range and (min_val < 0.0 or max_val > 255.0):
                    raise ValueError(
                        f"Image '{key}' has values outside [0, 255] range: "
                        f"min={min_val:.4f}, max={max_val:.4f}."
                    )
                obs[key] = tensor.float() / 255.0

        new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {"image_keys": self.image_keys, "validate_range": self.validate_range}


@dataclass
@ProcessorStepRegistry.register(name="xvla_imagenet_normalize")
class XVLAImageNetNormalizeProcessorStep(ProcessorStep):
    """Normalize image observations using ImageNet statistics."""

    image_keys: list[str] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        if obs is None:
            return new_transition
        obs = obs.copy()

        keys_to_normalize = self.image_keys
        if keys_to_normalize is None:
            keys_to_normalize = [k for k in obs if k.startswith(OBS_IMAGES)]

        for key in keys_to_normalize:
            if key in obs and isinstance(obs[key], torch.Tensor):
                tensor = obs[key]
                min_val = tensor.min().item()
                max_val = tensor.max().item()
                if min_val < 0.0 or max_val > 1.0:
                    raise ValueError(
                        f"Image '{key}' has values outside [0, 1] range: "
                        f"min={min_val:.4f}, max={max_val:.4f}."
                    )
                mean = torch.tensor(IMAGENET_STATS["mean"], device=tensor.device, dtype=tensor.dtype)
                std = torch.tensor(IMAGENET_STATS["std"], device=tensor.device, dtype=tensor.dtype)
                # Shape mean/std as (3, 1, 1, ...) to broadcast over spatial dims
                # Channel dim is at index -3 for (..., C, H, W) tensors
                shape = [1] * tensor.dim()
                shape[-3] = 3
                mean = mean.view(shape)
                std = std.view(shape)
                obs[key] = (tensor - mean) / std

        new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {"image_keys": self.image_keys}


@dataclass
@ProcessorStepRegistry.register(name="xvla_add_domain_id")
class XVLAAddDomainIdProcessorStep(ProcessorStep):
    """Add domain_id to complementary data."""

    domain_id: int = 0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        comp = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        comp = {} if comp is None else comp.copy()

        obs = new_transition.get(TransitionKey.OBSERVATION, {})
        batch_size = 1
        if obs:
            for v in obs.values():
                if isinstance(v, torch.Tensor):
                    batch_size = v.shape[0]
                    break

        comp["domain_id"] = torch.tensor([int(self.domain_id)] * batch_size, dtype=torch.long)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return new_transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {"domain_id": self.domain_id}


@dataclass
@ProcessorStepRegistry.register(name="xvla_rotation_6d_to_axis_angle")
class XVLARotation6DToAxisAngleProcessorStep(ProcessorStep):
    """Convert 6D rotation representation to axis-angle and reorganize action dimensions."""

    expected_action_dim: int = 10

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)

        if action is None or not isinstance(action, torch.Tensor):
            return new_transition

        device = action.device
        dtype = action.dtype
        action_np = action.cpu().numpy()

        target_eef = action_np[:, :3]
        rotation_6d = action_np[:, 3:9]
        target_act = action_np[:, 9:10]

        target_axis = rotate6d_to_axis_angle(rotation_6d)
        action_np = np.concatenate([target_eef, target_axis, target_act], axis=-1)
        action_np[:, -1] = np.where(action_np[:, -1] > 0.5, 1.0, -1.0)

        action = torch.from_numpy(action_np).to(device=device, dtype=dtype)
        new_transition[TransitionKey.ACTION] = action
        return new_transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {"expected_action_dim": self.expected_action_dim}
