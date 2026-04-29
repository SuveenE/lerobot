"""M policy wrapper for lerobot async inference.

Delegates inference to an external HTTP server over a simple ``POST /act``
JSON contract. The model itself is loaded and run by the external server;
this wrapper only handles the lerobot <-> server observation/action
translation.

No heavy ML dependencies (transformers, peft, etc.) are needed on the
lerobot side. This is an inference-only wrapper. Training through lerobot
is not supported.
"""

from __future__ import annotations

import builtins
import json
import logging
import os
import time
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

from .configuration_m import MConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="MPolicy")


class MPolicy(PreTrainedPolicy):
    """Thin HTTP proxy policy for lerobot's async inference system.

    Delegates all inference to an external FastAPI-style server. The server
    handles model loading, image preprocessing, proprioception normalization,
    and action unnormalization natively.

    This policy only:
    - Converts lerobot tensor batches to the numpy/dict format the server expects
    - POSTs a raw ``json_numpy``-serialized body to ``{server_url}``
    - Converts the JSON response back to a torch Tensor
    """

    config_class = MConfig
    name = "m"

    def __init__(self, config: MConfig, **kwargs):
        super().__init__(config)
        self.config: MConfig = config
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

        Loads optional ``lerobot_config.json`` overrides from the pretrained
        path (local directory or HF Hub). No model weights are loaded -- the
        external server handles that.
        """
        model_path = str(pretrained_name_or_path)

        if config is None:
            config = cls._load_config_with_overrides(model_path)

        instance = cls(config)

        logger.info(
            f"M policy external-server mode | server={config.server_url} | "
            f"key_map={config.server_image_key_map}"
        )

        instance._populate_features(config)

        # Best-effort reachability check (don't fail if server isn't up yet).
        # We simply GET the configured ``server_url``; any HTTP response --
        # including 404 or 405 for a POST-only route -- is sufficient proof
        # the server is reachable. We don't assume any particular route name
        # (e.g. ``/act``), so this works for arbitrary endpoints.
        try:
            resp = requests.get(config.server_url, timeout=3)
            logger.info(f"External server reachable (HTTP {resp.status_code})")
        except Exception:
            logger.warning(
                f"Could not reach external server at {config.server_url}. "
                "Make sure the server is running before sending observations."
            )

        instance.eval()
        return instance

    @classmethod
    def _load_config_with_overrides(cls, model_path: str) -> MConfig:
        """Create an MConfig, applying overrides from ``lerobot_config.json``.

        If ``model_path`` is a local directory containing ``lerobot_config.json``,
        its key-value pairs override the dataclass defaults. Example file
        matching the MolmoAct YAM 3-view contract::

            {
                "server_url": "https://policy9000.ngrok.app/act",
                "server_image_key_map": {
                    "left":  "left_cam",
                    "top":   "top_cam",
                    "right": "right_cam"
                },
                "primary_image_key": "top",
                "wrist_image_keys": ["left", "right"],
                "num_images_in_input": 3,
                "action_dim": 14,
                "proprio_dim": 14,
                "chunk_size": 30,
                "n_action_steps": 30,
                "server_input_size": [360, 640]
            }

        ``server_input_size`` should match the native resolution of your
        robot cameras. ``PolicyServer`` resizes every frame to this shape
        *before* it reaches this wrapper, so a mismatched value causes an
        unwanted bilinear resample.
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

        return MConfig(**overrides)

    # ------------------------------------------------------------------
    # Feature metadata
    # ------------------------------------------------------------------

    def _populate_features(self, config: MConfig) -> None:
        """Populate input/output features on the config for server-side processing.

        The shape declared here drives the ``PolicyServer``'s client-side
        image resize in ``prepare_raw_observation`` -- every robot frame
        is bilinearly resampled to ``server_input_size`` before it ever
        reaches this wrapper's ``predict_action_chunk``. To avoid that
        resize, set ``server_input_size`` to match your robot cameras'
        native resolution in ``lerobot_config.json``.
        """
        img_h, img_w = int(config.server_input_size[0]), int(config.server_input_size[1])
        img_shape = (3, img_h, img_w)

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
                type=FeatureType.STATE, shape=(config.proprio_dim,),
            )
            config.input_features = input_features

        if not config.output_features:
            config.output_features = {
                "action": PolicyFeature(
                    type=FeatureType.ACTION, shape=(config.action_dim,),
                ),
            }

    # ------------------------------------------------------------------
    # Tensor <-> numpy helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_to_numpy_image(
        tensor: Tensor,
        resize_hw: tuple[int, int],
    ) -> np.ndarray:
        """Convert a CHW float [0,1] tensor to HWC uint8 numpy array.

        Images are bilinear-antialias resized to ``resize_hw`` (height,
        width) if they're not already that shape. In practice incoming
        frames are *already* at ``resize_hw`` because ``PolicyServer``
        resizes them to ``input_features[...].shape`` upstream, so this
        is usually a no-op -- it's kept as a safety net for direct
        callers that bypass ``PolicyServer``.
        """
        img = tensor.detach().cpu()
        if img.ndim == 3:
            img = img.unsqueeze(0)

        target_h, target_w = int(resize_hw[0]), int(resize_hw[1])
        if img.shape[-2:] != (target_h, target_w):
            img = torch.nn.functional.interpolate(
                img.float(),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

        img = img.squeeze(0)
        img = (img * 255).clamp(0, 255).to(torch.uint8)
        img = img.permute(1, 2, 0).numpy()
        return img

    # ------------------------------------------------------------------
    # Inference (HTTP proxy to external server)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Predict an action chunk by forwarding to the external server.

        Converts the lerobot-format batch into the observation dict that the
        external server expects, POSTs to ``server_url``, and converts the
        response back to a ``(1, chunk_size, action_dim)`` tensor.

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

        # -- Build observation dict for the external server --
        observation: dict = {
            "instruction": task_label,
            "timestamp": time.time(),
        }

        # Track what we pulled out of the batch so we can log which cameras
        # were actually found client-side. "mapped" means the lerobot batch
        # contained the expected key and we forwarded it to the server;
        # "skipped" means ``server_image_key_map`` referenced a camera that
        # isn't in the batch (typo in the config, wrong ``--robot.cameras``,
        # etc.) -- those never reach the server.
        #
        # Each value of ``server_image_key_map`` may be a single server key
        # or a list of server keys. The latter lets one robot camera feed
        # multiple server slots (e.g. ``"left": ["left_cam", "right_cam"]``
        # when the external server requires 3 views but the robot has 2
        # physical cameras). The actual uint8 image is computed once and
        # shared by reference between duplicated slots.
        resize_hw = self.config.server_input_size
        mapped_cams: dict[str, list[str]] = {}
        skipped_cams: dict[str, list[str]] = {}
        incoming_image_shapes: dict[str, tuple] = {}
        for lerobot_cam_key, server_obs_keys in self.config.server_image_key_map.items():
            if isinstance(server_obs_keys, str):
                server_obs_keys_list = [server_obs_keys]
            else:
                server_obs_keys_list = list(server_obs_keys)

            batch_key = f"{OBS_IMAGES}.{lerobot_cam_key}"
            if batch_key in batch:
                src_tensor = batch[batch_key]
                incoming_image_shapes[batch_key] = tuple(src_tensor.shape)
                img_np = self._tensor_to_numpy_image(src_tensor, resize_hw=resize_hw)
                for server_obs_key in server_obs_keys_list:
                    observation[server_obs_key] = img_np
                mapped_cams[lerobot_cam_key] = server_obs_keys_list
            else:
                skipped_cams[lerobot_cam_key] = server_obs_keys_list

        all_image_batch_keys = [k for k in batch if k.startswith(f"{OBS_IMAGES}.")]

        if OBS_STATE in batch:
            state_tensor = batch[OBS_STATE]
            if state_tensor.ndim == 2:
                state_tensor = state_tensor.squeeze(0)
            # Match ``molmoact.py::send_request`` which forwards
            # ``obs["joint_positions"]`` as-is (float64 by numpy default).
            # Some MolmoAct servers are strict about dtype and will reject
            # float32 payloads with a generic 500 error.
            observation["state"] = (
                state_tensor.detach().cpu().double().numpy().astype(np.float64)
            )

        # -- Serialize first so we can log the on-the-wire payload size --
        serialized = json_numpy.dumps(observation)
        payload_bytes = len(serialized.encode("utf-8")) if isinstance(serialized, str) else len(serialized)

        def _fmt_bytes(n: int) -> str:
            for unit in ("B", "KB", "MB", "GB"):
                if n < 1024 or unit == "GB":
                    return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
                n /= 1024
            return f"{n:.1f}GB"

        # -- Client-side sanity log --
        # Fires every inference at DEBUG so it's easy to silence, but at
        # WARNING when nothing was mapped (the payload would reach the
        # server with no images at all -- almost always a config bug).
        log_msg = (
            f"M payload built | mapped={mapped_cams} | skipped={skipped_cams} | "
            f"batch_image_keys={all_image_batch_keys} | "
            f"incoming_image_shapes={incoming_image_shapes} | "
            f"resize_hw={tuple(resize_hw)} | "
            f"payload_size={_fmt_bytes(payload_bytes)} ({payload_bytes} bytes)"
        )
        if not mapped_cams:
            logger.warning(
                f"No camera keys from server_image_key_map matched the "
                f"observation batch -- the external server will receive NO "
                f"images. {log_msg}"
            )
        else:
            logger.debug(log_msg)

        # -- POST to external server (raw json_numpy body) --
        try:
            resp = requests.post(
                self.config.server_url,
                headers={"Content-Type": "application/json"},
                data=serialized,
                timeout=30,
            )
        except requests.RequestException as e:
            raise RuntimeError(
                f"External M server request failed: {e}\n"
                f"URL: {self.config.server_url}"
            ) from e

        if not resp.ok:
            # Include the server-side response body so the client log shows
            # the actual traceback / error from the inference server, plus a
            # summary of the payload we sent and what arrived from the
            # observation batch, so shape/dtype mismatches and missing camera
            # keys are obvious without having to attach a debugger.
            payload_summary = {
                k: (
                    f"ndarray shape={v.shape} dtype={v.dtype}"
                    if isinstance(v, np.ndarray)
                    else type(v).__name__
                )
                for k, v in observation.items()
            }
            body_preview = resp.text[:2000] if resp.text else "<empty body>"
            raise RuntimeError(
                f"External M server returned HTTP {resp.status_code} for "
                f"{self.config.server_url}\n"
                f"Payload keys/shapes: {payload_summary}\n"
                f"Payload size: {_fmt_bytes(payload_bytes)} ({payload_bytes} bytes)\n"
                f"Batch image keys seen: {all_image_batch_keys}\n"
                f"Cameras mapped -> server: {mapped_cams}\n"
                f"Cameras in config but NOT in batch: {skipped_cams}\n"
                f"Incoming image tensor shapes: {incoming_image_shapes}\n"
                f"Server response body:\n{body_preview}"
            )

        # -- Parse response --
        # ``json_numpy.loads`` is a strict superset of ``json.loads``: it
        # transparently decodes ndarray-encoded dicts of the form
        # ``{"__ndarray__": ..., "dtype": ..., "shape": ...}`` while still
        # handling plain JSON (lists/dicts) unchanged. Using it avoids the
        # need for a global ``json_numpy.patch()`` side effect.
        raw_response = json_numpy.loads(resp.text)
        if isinstance(raw_response, str):
            raw_response = json_numpy.loads(raw_response)

        if isinstance(raw_response, dict):
            if "actions" not in raw_response:
                raise RuntimeError(
                    f"External M server response missing 'actions' key. "
                    f"Got keys: {list(raw_response.keys())}"
                )
            action = np.asarray(raw_response["actions"])
        elif isinstance(raw_response, list):
            action = np.stack([np.asarray(a) for a in raw_response], axis=0)
        else:
            action = np.asarray(raw_response)

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
            "M policy training is not supported through lerobot. "
            "Use the external server repo directly for fine-tuning."
        )

    def get_optim_params(self) -> dict:
        raise NotImplementedError(
            "M policy training is not supported through lerobot."
        )
