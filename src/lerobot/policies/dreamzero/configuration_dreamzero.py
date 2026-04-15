"""Configuration for the DreamZero policy wrapper.

DreamZero (https://github.com/dreamzero0/dreamzero) is a World Action Model that
jointly predicts actions and videos, achieving strong zero-shot performance on
unseen tasks.

This config wraps the connection parameters needed to communicate with an external
DreamZero WebSocket inference server through lerobot's async inference system.
All model loading, preprocessing, and unnormalization is handled by the server.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("dreamzero")
@dataclass
class DreamZeroConfig(PreTrainedConfig):
    """Configuration class for the DreamZero external-server policy wrapper."""

    # Action chunking
    chunk_size: int = 24
    n_action_steps: int = 24

    # --- External server settings ---
    server_url: str = "ws://localhost:5000"

    # Maps lerobot camera keys to the observation dict keys expected by
    # the DreamZero WebSocket server (roboarena format).
    server_image_key_map: dict[str, str] = field(
        default_factory=lambda: {
            "top": "observation/exterior_image_0_left",
            "left": "observation/exterior_image_1_left",
            "right": "observation/wrist_image_left",
        }
    )

    # Maps DreamZero server state key names to [start, end] index ranges
    # into the lerobot observation.state packed vector.
    # YAM bimanual default: left_joint(6) + left_gripper(1) + right_joint(6) + right_gripper(1) = 14
    server_state_key_map: dict[str, list[int]] = field(
        default_factory=lambda: {
            "observation/left_joint_pos": [0, 6],
            "observation/left_gripper_pos": [6, 7],
            "observation/right_joint_pos": [7, 13],
            "observation/right_gripper_pos": [13, 14],
        }
    )

    # Fallback dimensions when metadata is unavailable.
    action_dim: int = 14
    state_dim: int = 14

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.chunk_size <= 0:
            raise ValueError("`chunk_size` must be strictly positive.")

    def validate_features(self) -> None:
        pass

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> list[int] | None:
        return None

    def get_optimizer_preset(self):
        raise NotImplementedError("DreamZero training is not supported through lerobot.")

    def get_scheduler_preset(self):
        raise NotImplementedError("DreamZero training is not supported through lerobot.")
