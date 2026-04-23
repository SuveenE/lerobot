"""Configuration for the Cosmos policy wrapper.

Cosmos Policy (https://github.com/NVlabs/cosmos-policy) is a 2B-parameter diffusion
transformer fine-tuned from NVIDIA Cosmos-Predict2-2B-Video2World for bimanual robot
manipulation. It jointly predicts action chunks, future images, and value estimates.

This config wraps the connection parameters needed to communicate with an external
``cosmos_policy.experiments.robot.aloha.deploy`` FastAPI server through lerobot's
async inference system. All model loading, preprocessing, and unnormalization is
handled by the server.

Schema verified against the canonical reference release
``nvidia/Cosmos-Policy-ALOHA-Predict2-2B``:
- Control frequency: **25 Hz** (NOT 30 or 50 Hz)
- Image size: 224x224
- Action chunk: 50 timesteps (2 seconds @ 25 Hz)
- Action / proprio dim: 14 (7 per arm: 6 joints + 1 gripper)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode

logger = logging.getLogger(__name__)

# Environment variable that, if set, overrides ``CosmosConfig.server_url`` at
# construction time. Useful for launching multiple lerobot policy servers
# (e.g. ``python -m lerobot.async_inference.policy_server ...``), each pointing
# to a different cosmos_policy ``deploy.py`` endpoint without having to maintain
# a separate ``lerobot_config.json`` per process. Mirrors ``LEROBOT_M_SERVER_URL``
# in the ``m`` policy.
SERVER_URL_ENV_VAR = "LEROBOT_COSMOS_SERVER_URL"


@PreTrainedConfig.register_subclass("cosmos")
@dataclass
class CosmosConfig(PreTrainedConfig):
    """Configuration class for the Cosmos external-server policy wrapper."""

    # Action chunking (native chunk is 50 @ 25 Hz = 2s of actions).
    chunk_size: int = 50
    n_action_steps: int = 50

    # Number of camera images the model expects (primary + N wrist views).
    num_images_in_input: int = 3
    image_size: int = 224

    # Which lerobot camera key is the primary third-person view.
    primary_image_key: str = "top"

    # Auxiliary wrist-mounted camera keys (order must match training).
    wrist_image_keys: list[str] = field(default_factory=lambda: ["left", "right"])

    # Fallback dimensions when dataset_statistics.json is unavailable.
    action_dim: int = 14
    proprio_dim: int = 14

    # --- External server settings ---
    server_url: str = "http://0.0.0.0:8777/act"

    # Maps lerobot camera keys -> the observation dict keys expected by
    # cosmos_policy/experiments/robot/aloha/deploy.py.
    server_image_key_map: dict[str, str] = field(
        default_factory=lambda: {
            "top": "primary_image",
            "left": "left_wrist_image",
            "right": "right_wrist_image",
        }
    )

    # Task description forwarded to deploy.py (must be an exact key in the
    # precomputed T5 embeddings pickle used to launch deploy.py). If empty,
    # the string from ``batch["task"]`` / ``observation.language_tokens`` is used.
    task_description: str = ""

    # HTTP request timeout (seconds). Cosmos diffusion inference on H100 is <1s.
    request_timeout: float = 30.0

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
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"`n_action_steps` ({self.n_action_steps}) must not exceed "
                f"`chunk_size` ({self.chunk_size})."
            )

        # Environment-variable override for ``server_url``. When set, it takes
        # precedence over both the dataclass default and any value loaded from
        # ``lerobot_config.json``. Enables running multiple policy server
        # processes that each target a different cosmos ``deploy.py`` endpoint.
        env_url = os.environ.get(SERVER_URL_ENV_VAR)
        if env_url:
            if env_url != self.server_url:
                logger.info(
                    f"Overriding server_url from {SERVER_URL_ENV_VAR}: "
                    f"{self.server_url!r} -> {env_url!r}"
                )
            self.server_url = env_url

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
        raise NotImplementedError("Cosmos Policy training is not supported through lerobot.")

    def get_scheduler_preset(self):
        raise NotImplementedError("Cosmos Policy training is not supported through lerobot.")
