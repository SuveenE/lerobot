"""Cosmos Policy wrapper for lerobot async inference.

Delegates inference to NVIDIA's ``cosmos_policy/experiments/robot/aloha/deploy.py``
FastAPI server over HTTP. The model itself is loaded and run by ``deploy.py``;
this wrapper only translates between the lerobot batch format and the JSON-numpy
payload that ``deploy.py`` expects.

The API contract (verified against the upstream ``deploy.py`` at
https://github.com/NVlabs/cosmos-policy):

    POST /act
    Body: {"encoded": json_numpy.dumps(observation)}
    observation: {
        "task_description":    str,            # must be a key in the T5 cache
        "primary_image":       np.uint8 HWC,   # top-down camera
        "left_wrist_image":    np.uint8 HWC,   # left wrist camera
        "right_wrist_image":   np.uint8 HWC,   # right wrist camera
        "proprio":             np.float32,     # shape (14,)
    }
    Response: json_numpy-encoded {
        "actions":                  np.ndarray,  # shape (chunk_size, action_dim)
        "future_image_predictions": np.ndarray,  # ignored
        "value_prediction":         float,       # ignored
    }

No heavy Cosmos Policy dependencies (torch.distributed.checkpoint, imaginaire,
cosmos_predict, t5) are imported on the lerobot side.

This is an inference-only wrapper. Training through lerobot is not supported.
"""

from __future__ import annotations

import builtins
import json
import logging
import os
from collections import deque
from pathlib import Path
from typing import TypeVar

import numpy as np
import requests
import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_IMAGES, OBS_LANGUAGE_TOKENS, OBS_STATE

from .configuration_cosmos import CosmosConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="CosmosPolicy")


class CosmosPolicy(PreTrainedPolicy):
    """Thin HTTP proxy around Cosmos Policy for lerobot's async inference system.

    Delegates all inference to an external ``cosmos_policy`` ``deploy.py``
    FastAPI server.  The server handles model loading, image resize,
    proprio normalization, action unnormalization, T5 text embedding lookup,
    and diffusion sampling natively.

    This policy only:
    - Converts lerobot tensor batches to the numpy/dict format ``deploy.py`` expects
    - POSTs to ``/act`` with a ``json_numpy``-encoded payload
    - Converts the JSON response back to a ``(1, chunk_size, action_dim)`` Tensor
    """

    config_class = CosmosConfig
    name = "cosmos"

    def __init__(self, config: CosmosConfig, **kwargs):
        super().__init__(config)
        self.config: CosmosConfig = config
        self._action_queue: deque[Tensor] = deque()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        **kwargs,
    ) -> T:
        """Initialise the policy proxy.

        If a ``lerobot_config.json`` file exists in the pretrained path, its
        keys override ``CosmosConfig`` defaults. This lets deployment-
        specific settings (``server_url``, ``server_image_key_map``,
        ``task_description``, etc.) live alongside the checkpoint without
        requiring extra CLI arguments on the robot client.

        No Cosmos Policy model weights are loaded locally -- the external
        server handles that.
        """
        model_path = str(pretrained_name_or_path)

        if config is None:
            config = cls._load_config_with_overrides(model_path)

        instance = cls(config)

        logger.info(
            f"Cosmos Policy external-server mode | server={config.server_url} | "
            f"key_map={config.server_image_key_map} | "
            f"task_description={config.task_description!r}"
        )

        instance._populate_features(config)

        # Best-effort reachability check (don't fail if server isn't up yet)
        try:
            resp = requests.get(
                config.server_url.replace("/act", "/docs"), timeout=3
            )
            logger.info(f"External server reachable (HTTP {resp.status_code})")
        except Exception:
            logger.warning(
                f"Could not reach external server at {config.server_url}. "
                "Make sure cosmos_policy deploy.py is running before sending observations."
            )

        instance.eval()
        return instance

    @classmethod
    def _load_config_with_overrides(cls, model_path: str) -> CosmosConfig:
        """Create a CosmosConfig, applying overrides from ``lerobot_config.json``.

        If ``model_path`` is a local directory containing ``lerobot_config.json``,
        its key-value pairs override the dataclass defaults.  Example file::

            {
                "server_url": "http://localhost:8777/act",
                "server_image_key_map": {
                    "top":   "primary_image",
                    "left":  "left_wrist_image",
                    "right": "right_wrist_image"
                },
                "task_description": "Stack the cubes.",
                "primary_image_key": "top",
                "wrist_image_keys": ["left", "right"],
                "num_images_in_input": 3,
                "image_size": 224,
                "chunk_size": 50,
                "n_action_steps": 50,
                "action_dim": 14,
                "proprio_dim": 14
            }
        """
        overrides: dict = {}
        if os.path.isdir(model_path):
            cfg_path = os.path.join(model_path, "lerobot_config.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path) as f:
                    overrides = json.load(f)
                logger.info(f"Loaded config overrides from {cfg_path}: {list(overrides.keys())}")
        else:
            try:
                from huggingface_hub import hf_hub_download

                cfg_path = hf_hub_download(
                    repo_id=model_path, filename="lerobot_config.json"
                )
                with open(cfg_path) as f:
                    overrides = json.load(f)
                logger.info(f"Loaded config overrides from HF Hub: {list(overrides.keys())}")
            except Exception:
                pass

        return CosmosConfig(**overrides)

    # ------------------------------------------------------------------
    # Feature metadata
    # ------------------------------------------------------------------

    def _populate_features(self, config: CosmosConfig) -> None:
        """Populate input/output features on the config for server-side processing."""
        img_shape = (3, config.image_size, config.image_size)

        if not config.input_features:
            input_features = {}
            camera_keys = ([config.primary_image_key] + list(config.wrist_image_keys))[
                : config.num_images_in_input
            ]
            for cam_key in camera_keys:
                input_features[f"{OBS_IMAGES}.{cam_key}"] = PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=img_shape,
                )
            input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(config.proprio_dim,),
            )
            config.input_features = input_features

        if not config.output_features:
            config.output_features = {
                "action": PolicyFeature(
                    type=FeatureType.ACTION,
                    shape=(config.action_dim,),
                ),
            }

    # ------------------------------------------------------------------
    # Tensor <-> numpy helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_to_numpy_image(tensor: Tensor) -> np.ndarray:
        """Convert a CHW float [0,1] tensor (optionally with leading batch dim)
        to an HWC uint8 numpy array.  ``deploy.py`` handles resize internally."""
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        img = tensor.detach().cpu()
        img = (img * 255).clamp(0, 255).to(torch.uint8)
        img = img.permute(1, 2, 0).numpy()
        return img

    def _extract_task_description(self, batch: dict[str, Tensor]) -> str:
        """Resolve the task description string to send to deploy.py.

        Priority:
        1. ``config.task_description`` (set in lerobot_config.json or CLI)
        2. ``batch["task"]`` (standard lerobot RobotClient payload)
        3. ``batch[OBS_LANGUAGE_TOKENS]`` (legacy)
        4. Empty string (will cause deploy.py to error unless "" is a T5 key)
        """
        if self.config.task_description:
            return self.config.task_description

        candidate: object = ""
        if "task" in batch:
            candidate = batch["task"]
        elif OBS_LANGUAGE_TOKENS in batch:
            candidate = batch[OBS_LANGUAGE_TOKENS]

        if isinstance(candidate, list):
            candidate = candidate[0] if candidate else ""
        if isinstance(candidate, Tensor):
            candidate = ""
        if not isinstance(candidate, str):
            candidate = str(candidate)
        return candidate

    # ------------------------------------------------------------------
    # Inference (HTTP proxy to deploy.py)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Predict an action chunk by forwarding to the external deploy.py server.

        Converts the lerobot-format batch into the observation dict that
        ``deploy.py.get_server_action`` expects, POSTs to ``/act``, and
        converts the response back to a ``(1, chunk_size, action_dim)`` tensor.

        Requires the ``json-numpy`` package (``pip install json-numpy``).
        """
        try:
            import json_numpy
        except ImportError as e:
            raise ImportError(
                "json-numpy is required for the cosmos external server mode. "
                "Install it with: pip install json-numpy"
            ) from e

        task_description = self._extract_task_description(batch)

        observation: dict = {"task_description": task_description}

        for lerobot_cam_key, server_obs_key in self.config.server_image_key_map.items():
            batch_key = f"{OBS_IMAGES}.{lerobot_cam_key}"
            if batch_key in batch:
                observation[server_obs_key] = self._tensor_to_numpy_image(batch[batch_key])

        missing = [
            v
            for v in self.config.server_image_key_map.values()
            if v not in observation
        ]
        if missing:
            raise KeyError(
                f"Cosmos Policy deploy.py expects images {missing} but they were not "
                f"found in the lerobot batch.  Present batch keys: {sorted(batch.keys())}.  "
                f"Check that your camera configuration exposes "
                f"{list(self.config.server_image_key_map.keys())} and that server_image_key_map "
                f"is correct."
            )

        if OBS_STATE in batch:
            state_tensor = batch[OBS_STATE]
            if state_tensor.ndim == 2:
                state_tensor = state_tensor.squeeze(0)
            observation["proprio"] = (
                state_tensor.detach().cpu().float().numpy().astype(np.float32)
            )
        else:
            raise KeyError(
                f"Cosmos Policy deploy.py requires proprio state under batch[{OBS_STATE!r}], "
                f"but it was not found.  Present batch keys: {sorted(batch.keys())}."
            )

        payload = json_numpy.dumps(observation)
        try:
            resp = requests.post(
                self.config.server_url,
                json={"encoded": payload},
                timeout=self.config.request_timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Cosmos Policy deploy.py request failed: {e}\n"
                f"URL: {self.config.server_url}\n"
                f"Task: {task_description!r}\n"
                "Tip: deploy.py returns the literal string 'error' (HTTP 200) when it "
                "fails internally; check the deploy.py logs for the real traceback."
            ) from e

        raw_response = resp.json()

        if isinstance(raw_response, str):
            if raw_response == "error":
                raise RuntimeError(
                    "Cosmos Policy deploy.py returned the sentinel string 'error'. "
                    "Inspect the deploy.py logs for the underlying exception. "
                    f"Most common cause: task_description {task_description!r} is not a key "
                    "in the T5 embeddings pickle passed via --t5_text_embeddings_path."
                )
            decoded = json_numpy.loads(raw_response)
        else:
            decoded = raw_response

        if isinstance(decoded, dict) and "actions" in decoded:
            action = decoded["actions"]
        else:
            action = decoded

        action = np.asarray(action)
        action = torch.from_numpy(action).float()
        if action.ndim == 2:
            action = action.unsqueeze(0)
        elif action.ndim == 1:
            action = action.unsqueeze(0).unsqueeze(0)

        return action

    # ------------------------------------------------------------------
    # Standard PreTrainedPolicy interface
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            for i in range(min(actions.shape[1], self.config.n_action_steps)):
                self._action_queue.append(actions[0, i])

        return self._action_queue.popleft()

    def reset(self):
        self._action_queue.clear()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError(
            "Cosmos Policy training is not supported through lerobot. "
            "Use the NVlabs/cosmos-policy repo directly for fine-tuning."
        )

    def get_optim_params(self) -> dict:
        raise NotImplementedError(
            "Cosmos Policy training is not supported through lerobot."
        )
