"""OpenVLA-OFT policy wrapper for lerobot async inference.

Delegates inference to the original openvla-oft ``deploy.py`` FastAPI server
over HTTP.  The model itself is loaded and run by ``deploy.py``; this wrapper
only handles the lerobot <-> deploy.py observation/action translation.

No heavy dependencies (prismatic, transformers, peft, tensorflow) are needed
on the lerobot side.

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

from .configuration_openvla_oft import OpenVLAOFTConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="OpenVLAOFTPolicy")


def _detect_unnorm_key(dataset_stats: dict) -> str:
    """Auto-detect the unnorm_key from dataset_statistics.json.

    Looks for the first key that isn't a well-known base-model pretraining dataset.
    Falls back to the first key if none found.
    """
    keys = list(dataset_stats.keys())
    if len(keys) == 1:
        return keys[0]

    BASE_MODEL_DATASETS = {
        "austin_buds_dataset_converted_externally_to_rlds",
        "austin_sailor_dataset_converted_externally_to_rlds",
        "austin_sirius_dataset_converted_externally_to_rlds",
        "berkeley_autolab_ur5",
        "berkeley_cable_routing",
        "bridge_orig",
        "bridge_dataset",
        "dlr_sara_pour_converted_externally_to_rlds",
        "fractal20220817_data",
        "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
        "jaco_play",
        "kuka",
        "nyu_door_opening_surprising_effectiveness",
        "nyu_franka_play_dataset_converted_externally_to_rlds",
        "roboturk",
        "stanford_hydra_dataset_converted_externally_to_rlds",
        "taco_play",
        "toto",
        "ucsd_kitchen_dataset_converted_externally_to_rlds",
        "utaustin_mutex",
        "viola",
    }

    for key in keys:
        if key not in BASE_MODEL_DATASETS:
            return key

    return keys[0]


class OpenVLAOFTPolicy(PreTrainedPolicy):
    """Thin HTTP proxy around OpenVLA-OFT for lerobot's async inference system.

    Delegates all inference to an external openvla-oft ``deploy.py`` FastAPI
    server.  The server handles model loading, image preprocessing,
    proprioception normalization, and action unnormalization natively.

    This policy only:
    - Converts lerobot tensor batches to the numpy/dict format deploy.py expects
    - POSTs to ``/act``
    - Converts the JSON response back to a torch Tensor
    """

    config_class = OpenVLAOFTConfig
    name = "openvla_oft"

    def __init__(self, config: OpenVLAOFTConfig, **kwargs):
        super().__init__(config)
        self.config: OpenVLAOFTConfig = config
        self._action_queue: deque[Tensor] = deque()
        self._unnorm_key: str = ""

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        **kwargs,
    ) -> T:
        """Initialise the policy proxy.

        Loads ``dataset_statistics.json`` (if available at the model path) to
        determine action/proprio dimensions for feature metadata.  No VLA
        model weights are loaded -- the external server handles that.
        """
        model_path = str(pretrained_name_or_path)

        if config is None:
            config = OpenVLAOFTConfig()

        instance = cls(config)

        logger.info(
            f"OpenVLA-OFT external-server mode | server={config.server_url} | "
            f"key_map={config.server_image_key_map}"
        )

        is_local = os.path.isdir(model_path)
        instance._load_dataset_stats(model_path, is_local)
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
                "Make sure deploy.py is running before sending observations."
            )

        instance.eval()
        return instance

    # ------------------------------------------------------------------
    # Feature metadata
    # ------------------------------------------------------------------

    def _populate_features(self, config: OpenVLAOFTConfig) -> None:
        """Populate input/output features on the config for server-side processing."""
        img_shape = (3, config.image_size, config.image_size)

        if not config.input_features:
            input_features = {}
            camera_keys = ([config.primary_image_key] + list(config.wrist_image_keys))[
                : config.num_images_in_input
            ]
            for cam_key in camera_keys:
                input_features[f"{OBS_IMAGES}.{cam_key}"] = PolicyFeature(
                    type=FeatureType.VISUAL, shape=img_shape,
                )
            input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE, shape=(self._get_proprio_dim(),),
            )
            config.input_features = input_features

        if not config.output_features:
            config.output_features = {
                "action": PolicyFeature(
                    type=FeatureType.ACTION, shape=(self._get_action_dim(),),
                ),
            }

    def _get_action_dim(self) -> int:
        norm_stats = getattr(self, "_norm_stats", None)
        if norm_stats and self._unnorm_key in norm_stats:
            stats = norm_stats[self._unnorm_key]
            if "action" in stats and "min" in stats["action"]:
                return len(stats["action"]["min"])
        return self.config.action_dim

    def _get_proprio_dim(self) -> int:
        norm_stats = getattr(self, "_norm_stats", None)
        if norm_stats and self._unnorm_key in norm_stats:
            stats = norm_stats[self._unnorm_key]
            if "proprio" in stats and "min" in stats["proprio"]:
                return len(stats["proprio"]["min"])
        return self.config.proprio_dim

    # ------------------------------------------------------------------
    # Dataset statistics (for dimension detection only)
    # ------------------------------------------------------------------

    def _load_dataset_stats(self, model_path: str, is_local: bool) -> None:
        """Load dataset_statistics.json to determine action/proprio dims."""
        if is_local:
            stats_path = os.path.join(model_path, "dataset_statistics.json")
        else:
            try:
                from huggingface_hub import hf_hub_download

                stats_path = hf_hub_download(
                    repo_id=model_path, filename="dataset_statistics.json"
                )
            except Exception:
                logger.warning("No dataset_statistics.json found. Using config defaults for dims.")
                self._norm_stats = {}
                return

        if os.path.isfile(stats_path):
            with open(stats_path) as f:
                self._norm_stats = json.load(f)

            if self.config.unnorm_key:
                self._unnorm_key = self.config.unnorm_key
            else:
                self._unnorm_key = _detect_unnorm_key(self._norm_stats)
                logger.info(f"Auto-detected unnorm_key: {self._unnorm_key}")
        else:
            logger.warning(f"dataset_statistics.json not found at {stats_path}")
            self._norm_stats = {}

    # ------------------------------------------------------------------
    # Tensor <-> numpy helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_to_numpy_image(tensor: Tensor) -> np.ndarray:
        """Convert a CHW float [0,1] tensor to HWC uint8 numpy array."""
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        img = tensor.detach().cpu()
        img = (img * 255).clamp(0, 255).to(torch.uint8)
        img = img.permute(1, 2, 0).numpy()
        return img

    # ------------------------------------------------------------------
    # Inference (HTTP proxy to deploy.py)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Predict an action chunk by forwarding to the external deploy.py server.

        Converts the lerobot-format batch into the observation dict that
        deploy.py's ``get_vla_action`` expects, POSTs to ``/act``, and
        converts the response back to a ``(1, chunk_size, action_dim)`` tensor.

        Requires the ``json-numpy`` package (``pip install json-numpy``).
        """
        try:
            import json_numpy
        except ImportError as e:
            raise ImportError(
                "json-numpy is required for external server mode. "
                "Install it with: pip install json-numpy"
            ) from e

        # -- Extract task instruction --
        task_label = ""
        if "task" in batch:
            task_label = batch["task"]
        elif OBS_LANGUAGE_TOKENS in batch:
            task_label = batch[OBS_LANGUAGE_TOKENS]
        if isinstance(task_label, list):
            task_label = task_label[0] if task_label else ""
        if isinstance(task_label, Tensor):
            task_label = ""
        if not isinstance(task_label, str):
            task_label = str(task_label)

        # -- Build observation dict for deploy.py --
        observation: dict = {"instruction": task_label}

        for lerobot_cam_key, server_obs_key in self.config.server_image_key_map.items():
            batch_key = f"{OBS_IMAGES}.{lerobot_cam_key}"
            if batch_key in batch:
                observation[server_obs_key] = self._tensor_to_numpy_image(batch[batch_key])

        if OBS_STATE in batch:
            state_tensor = batch[OBS_STATE]
            if state_tensor.ndim == 2:
                state_tensor = state_tensor.squeeze(0)
            observation["state"] = state_tensor.detach().cpu().float().numpy()

        # -- POST to external server --
        payload = json_numpy.dumps(observation)
        try:
            resp = requests.post(
                self.config.server_url,
                json={"encoded": payload},
                timeout=30,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(
                f"External OpenVLA-OFT server request failed: {e}\n"
                f"URL: {self.config.server_url}"
            ) from e

        # -- Parse response --
        raw_response = resp.json()
        if isinstance(raw_response, str):
            action_list = json_numpy.loads(raw_response)
        else:
            action_list = raw_response

        if isinstance(action_list, list):
            action = np.stack([np.asarray(a) for a in action_list], axis=0)
        else:
            action = np.asarray(action_list)

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
            "OpenVLA-OFT training is not supported through lerobot. "
            "Use the openvla-oft repo directly for fine-tuning."
        )

    def get_optim_params(self) -> dict:
        raise NotImplementedError(
            "OpenVLA-OFT training is not supported through lerobot."
        )
