"""DreamZero policy wrapper for lerobot async inference.

Delegates inference to an external DreamZero WebSocket server
(``socket_test_optimized_AR.py``) over the roboarena WebSocket protocol.
The model itself is loaded and run by the DreamZero distributed server;
this wrapper only handles the lerobot <-> WebSocket observation/action
translation.

No heavy dependencies (groot, wan, deepspeed, torch.distributed) are
needed on the lerobot side.

This is an inference-only wrapper. Training through lerobot is not supported.
"""

from __future__ import annotations

import builtins
import json
import logging
import os
import uuid
from collections import deque
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_IMAGES, OBS_LANGUAGE_TOKENS, OBS_STATE

from .configuration_dreamzero import DreamZeroConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="DreamZeroPolicy")


class DreamZeroPolicy(PreTrainedPolicy):
    """Thin WebSocket proxy around DreamZero for lerobot's async inference system.

    Delegates all inference to an external DreamZero distributed WebSocket
    server.  The server handles model loading, image preprocessing, frame
    accumulation, and action denormalization natively.

    This policy only:
    - Converts lerobot tensor batches to the numpy/dict format the server expects
    - Sends observations via WebSocket (msgpack)
    - Converts the msgpack response back to a torch Tensor
    """

    config_class = DreamZeroConfig
    name = "dreamzero"

    def __init__(self, config: DreamZeroConfig, **kwargs):
        super().__init__(config)
        self.config: DreamZeroConfig = config
        self._action_queue: deque[Tensor] = deque()
        self._ws = None
        self._packer = None
        self._session_id: str = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        **kwargs,
    ) -> T:
        """Initialise the policy proxy.

        No model weights are loaded -- the external DreamZero server handles
        that.  If a ``lerobot_config.json`` file exists in the pretrained
        path, its keys override ``DreamZeroConfig`` defaults.
        """
        model_path = str(pretrained_name_or_path)

        if config is None:
            config = cls._load_config_with_overrides(model_path)

        instance = cls(config)

        logger.info(
            f"DreamZero external-server mode | server={config.server_url} | "
            f"image_key_map={config.server_image_key_map}"
        )

        instance._populate_features(config)

        # Best-effort reachability check
        instance._try_connect()

        instance.eval()
        return instance

    @classmethod
    def _load_config_with_overrides(cls, model_path: str) -> DreamZeroConfig:
        """Create a DreamZeroConfig, applying overrides from ``lerobot_config.json``.

        If ``model_path`` is a local directory containing ``lerobot_config.json``,
        its key-value pairs override the dataclass defaults.  Example file::

            {
                "server_url": "ws://10.0.0.1:5000",
                "server_image_key_map": {
                    "top": "observation/exterior_image_0_left",
                    "left": "observation/exterior_image_1_left",
                    "right": "observation/wrist_image_left"
                },
                "server_state_key_map": {
                    "observation/left_joint_pos": [0, 6],
                    "observation/left_gripper_pos": [6, 7],
                    "observation/right_joint_pos": [7, 13],
                    "observation/right_gripper_pos": [13, 14]
                },
                "chunk_size": 24,
                "n_action_steps": 24,
                "action_dim": 14,
                "state_dim": 14
            }
        """
        overrides: dict = {}
        if os.path.isdir(model_path):
            cfg_path = os.path.join(model_path, "lerobot_config.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path) as f:
                    overrides = json.load(f)
                logger.info(f"Loaded config overrides from {cfg_path}: {list(overrides.keys())}")

        return DreamZeroConfig(**overrides)

    # ------------------------------------------------------------------
    # Feature metadata
    # ------------------------------------------------------------------

    def _populate_features(self, config: DreamZeroConfig) -> None:
        """Populate input/output features on the config for server-side processing."""
        if not config.input_features:
            input_features = {}
            for cam_key in config.server_image_key_map:
                input_features[f"{OBS_IMAGES}.{cam_key}"] = PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, 360, 640),
                )
            input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE, shape=(config.state_dim,),
            )
            config.input_features = input_features

        if not config.output_features:
            config.output_features = {
                "action": PolicyFeature(
                    type=FeatureType.ACTION, shape=(config.action_dim,),
                ),
            }

    # ------------------------------------------------------------------
    # WebSocket connection
    # ------------------------------------------------------------------

    def _try_connect(self) -> None:
        """Best-effort WebSocket connection to the DreamZero server."""
        try:
            import websockets.sync.client as ws_client
            from openpi_client import msgpack_numpy

            self._ws = ws_client.connect(
                self.config.server_url,
                compression=None,
                max_size=None,
                ping_interval=60,
                ping_timeout=600,
            )
            metadata = msgpack_numpy.unpackb(self._ws.recv())
            self._packer = msgpack_numpy.Packer()
            logger.info(f"Connected to DreamZero server. Metadata: {metadata}")
        except Exception as e:
            logger.warning(
                f"Could not connect to DreamZero server at {self.config.server_url}: {e}. "
                "Make sure the DreamZero server is running before sending observations."
            )
            self._ws = None
            self._packer = None

    def _ensure_connected(self) -> None:
        """Ensure we have a live WebSocket connection, reconnecting if needed."""
        if self._ws is None:
            self._try_connect()
        if self._ws is None:
            raise RuntimeError(
                f"Cannot connect to DreamZero server at {self.config.server_url}. "
                "Is socket_test_optimized_AR.py running?"
            )

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
    # Inference (WebSocket proxy to DreamZero server)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Predict an action chunk by forwarding to the external DreamZero server.

        Converts the lerobot-format batch into the observation dict that
        the DreamZero roboarena server expects, sends it via WebSocket, and
        converts the response back to a ``(1, chunk_size, action_dim)`` tensor.

        Requires ``websockets`` and ``msgpack-numpy`` packages.
        """
        from openpi_client import msgpack_numpy

        self._ensure_connected()

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

        # -- Build observation dict for DreamZero server --
        observation: dict = {}

        # Images: convert CHW float tensors to HWC uint8 numpy
        for lerobot_cam_key, server_obs_key in self.config.server_image_key_map.items():
            batch_key = f"{OBS_IMAGES}.{lerobot_cam_key}"
            if batch_key in batch:
                observation[server_obs_key] = self._tensor_to_numpy_image(batch[batch_key])

        # State: split packed vector into individual DreamZero state keys
        if OBS_STATE in batch:
            state_tensor = batch[OBS_STATE]
            if state_tensor.ndim == 2:
                state_tensor = state_tensor.squeeze(0)
            state_np = state_tensor.detach().cpu().float().numpy()

            for server_key, (start, end) in self.config.server_state_key_map.items():
                observation[server_key] = state_np[start:end].astype(np.float32)

        # Language prompt and session tracking
        observation["prompt"] = task_label
        observation["session_id"] = self._session_id

        # Roboarena protocol: set endpoint to "infer"
        observation["endpoint"] = "infer"

        # -- Send via WebSocket --
        try:
            data = self._packer.pack(observation)
            self._ws.send(data)
            response = self._ws.recv()
            if isinstance(response, str):
                raise RuntimeError(f"Error from DreamZero server:\n{response}")
            action = msgpack_numpy.unpackb(response)
        except Exception as e:
            # Connection may have dropped; clear so next call reconnects
            self._ws = None
            self._packer = None
            raise RuntimeError(
                f"DreamZero server request failed: {e}\n"
                f"URL: {self.config.server_url}"
            ) from e

        # -- Parse response --
        if isinstance(action, dict):
            # Server may return a dict like {"action.joint_position": ..., "action.gripper_position": ...}
            arrays = []
            for k in sorted(action.keys()):
                v = action[k]
                if isinstance(v, np.ndarray):
                    if v.ndim == 1:
                        v = v.reshape(-1, 1) if v.shape[0] == action.get("_N", v.shape[0]) else v.reshape(1, -1)
                    arrays.append(v)
            if arrays:
                action = np.concatenate(arrays, axis=-1)
            else:
                action = np.zeros((1, self.config.action_dim), dtype=np.float32)

        if isinstance(action, np.ndarray):
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
        self._session_id = str(uuid.uuid4())

        # Send reset to DreamZero server to clear frame buffers / session state
        if self._ws is not None and self._packer is not None:
            try:
                reset_obs = {"endpoint": "reset"}
                self._ws.send(self._packer.pack(reset_obs))
                self._ws.recv()
                logger.info("Sent reset to DreamZero server")
            except Exception as e:
                logger.warning(f"Failed to send reset to DreamZero server: {e}")
                self._ws = None
                self._packer = None

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError(
            "DreamZero training is not supported through lerobot. "
            "Use the dreamzero repo directly for fine-tuning."
        )

    def get_optim_params(self) -> dict:
        raise NotImplementedError(
            "DreamZero training is not supported through lerobot."
        )
