"""Configuration for the OpenVLA-OFT policy wrapper.

OpenVLA-OFT (https://github.com/moojink/openvla-oft) is a fine-tuned Vision-Language-Action
model that uses an L1 regression or diffusion action head for continuous action prediction,
with optional proprioception injection and FiLM conditioning.

This config wraps the OpenVLA-OFT inference parameters so the model can be used through
lerobot's async inference system. All preprocessing and unnormalization is handled
internally by OpenVLA-OFT, so normalization mappings are set to IDENTITY.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("openvla_oft")
@dataclass
class OpenVLAOFTConfig(PreTrainedConfig):
    """Configuration class for the OpenVLA-OFT policy wrapper."""

    # Action chunking.
    # num_actions_chunk must match the value used during training (see
    # prismatic/vla/constants.py in the openvla-oft repo). ALOHA default is 25
    # (at 25 Hz) or 50 (at 50 Hz). This is critical for correct action head
    # reshaping.
    num_actions_chunk: int = 25
    chunk_size: int = 25
    n_action_steps: int = 25

    # OpenVLA-OFT model parameters
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_proprio: bool = True
    use_film: bool = False
    num_images_in_input: int = 3
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    image_size: int = 224

    # Proprio/action normalization type. Must match training.
    # "bounds" = normalize to [-1, 1] using min/max (ALOHA default)
    # "bounds_q99" = normalize using q01/q99 percentiles (LIBERO default)
    proprio_normalization_type: str = "bounds"

    # Which robot camera to use as the primary (third-person) view.
    primary_image_key: str = "top"

    # Wrist / auxiliary camera keys in the order they should be concatenated
    # after the primary image. Order must match training.
    wrist_image_keys: list[str] = field(default_factory=lambda: ["left", "right"])

    # Key in dataset_statistics.json for action unnormalization.
    # If empty, auto-detected from the checkpoint.
    unnorm_key: str = ""

    # Diffusion-specific (only when use_diffusion=True)
    num_diffusion_steps_train: int = 100
    num_diffusion_steps_inference: int = 10

    # LoRA rank (used when use_film=True to apply LoRA before FiLM wrapping)
    lora_rank: int = 32

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.use_l1_regression and self.use_diffusion:
            raise ValueError("Cannot use both L1 regression and diffusion action head.")
        if not self.use_l1_regression and not self.use_diffusion:
            raise ValueError("Either use_l1_regression or use_diffusion must be True.")
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
