"""Configuration for the M policy wrapper.

The M policy is a thin HTTP proxy around an external inference server. The
lerobot ``PolicyServer`` forwards observations to the external server over
HTTP and returns the predicted actions. All model loading, image
preprocessing, and action (un)normalization is handled by the external
server.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("m")
@dataclass
class MConfig(PreTrainedConfig):
    """Configuration class for the M external-server policy wrapper."""

    # Action chunking
    chunk_size: int = 30
    n_action_steps: int = 30

    # Number of camera images the model expects
    num_images_in_input: int = 2

    # Client-side resize applied before sending images to the server.
    # Format: (height, width). Images are always resized to this shape prior
    # to the HTTP POST; the server therefore always receives frames at a
    # known, consistent resolution.
    server_input_size: tuple[int, int] = (180, 320)

    # Which robot camera to use as the primary (first) image slot.
    primary_image_key: str = "top"

    # Auxiliary camera keys (order must match training)
    wrist_image_keys: list[str] = field(default_factory=lambda: ["right"])

    # --- External server settings ---
    # The external inference server is typically reached over HTTPS (e.g. a
    # tunneled/public endpoint). Override via lerobot_config.json for your
    # deployment.
    server_url: str = "https://localhost:8777/act"

    # Maps lerobot camera keys to the observation dict keys expected by the
    # external server.
    server_image_key_map: dict[str, str] = field(
        default_factory=lambda: {
            "top": "external_cam",
            "right": "wrist_cam",
        }
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

        # JSON / dict-based overrides deserialize tuples as lists; normalize.
        if isinstance(self.server_input_size, list):
            self.server_input_size = tuple(self.server_input_size)
        if (
            not isinstance(self.server_input_size, tuple)
            or len(self.server_input_size) != 2
            or not all(isinstance(v, int) and v > 0 for v in self.server_input_size)
        ):
            raise ValueError(
                "`server_input_size` must be a (height, width) tuple of two "
                f"positive ints, got {self.server_input_size!r}"
            )

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
        raise NotImplementedError("M policy training is not supported through lerobot.")

    def get_scheduler_preset(self):
        raise NotImplementedError("M policy training is not supported through lerobot.")
