"""Configuration for the OpenVLA-OFT policy wrapper.

OpenVLA-OFT (https://github.com/moojink/openvla-oft) is a fine-tuned Vision-Language-Action
model that uses an L1 regression or diffusion action head for continuous action prediction,
with optional proprioception injection and FiLM conditioning.

This config wraps the connection parameters needed to communicate with an external
openvla-oft ``deploy.py`` FastAPI server through lerobot's async inference system.
All model loading, preprocessing, and unnormalization is handled by the server.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("openvla_oft")
@dataclass
class OpenVLAOFTConfig(PreTrainedConfig):
    """Configuration class for the OpenVLA-OFT external-server policy wrapper."""

    # Action chunking
    chunk_size: int = 25
    n_action_steps: int = 25

    # Number of camera images the model expects
    num_images_in_input: int = 2
    image_size: int = 224

    # Which robot camera to use as the primary (first) image slot.
    primary_image_key: str = "left"

    # Auxiliary camera keys (order must match training)
    wrist_image_keys: list[str] = field(default_factory=lambda: ["right"])

    # Key in dataset_statistics.json for action dim detection.
    # If empty, auto-detected from the checkpoint's statistics file.
    unnorm_key: str = ""

    # --- External server settings ---
    server_url: str = "http://0.0.0.0:8777/act"

    # Maps lerobot camera keys to the observation dict keys expected by
    # deploy.py's get_vla_action (e.g. {"left": "full_image", "right": "wrist_image_left"}).
    server_image_key_map: dict[str, str] = field(
        default_factory=lambda: {"left": "full_image", "right": "wrist_image_left"}
    )

    # Fallback dimensions when dataset_statistics.json is unavailable.
    action_dim: int = 14
    proprio_dim: int = 14

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
        raise NotImplementedError("OpenVLA-OFT training is not supported through lerobot.")

    def get_scheduler_preset(self):
        raise NotImplementedError("OpenVLA-OFT training is not supported through lerobot.")
