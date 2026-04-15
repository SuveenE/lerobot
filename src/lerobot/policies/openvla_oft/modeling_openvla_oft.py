"""OpenVLA-OFT policy wrapper for lerobot async inference.

Wraps the OpenVLA-OFT model (https://github.com/moojink/openvla-oft) behind
lerobot's PreTrainedPolicy interface, enabling use via the async inference
robot_client / policy_server system.

This is an inference-only wrapper. Training through lerobot is not supported.
"""

from __future__ import annotations

import builtins
import glob
import json
import logging
import os
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

from .configuration_openvla_oft import OpenVLAOFTConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="OpenVLAOFTPolicy")

OPENVLA_IMAGE_SIZE = 224


def _find_checkpoint_file(directory: str, pattern: str) -> str | None:
    """Find a checkpoint file matching a pattern in a directory."""
    matches = glob.glob(os.path.join(directory, f"{pattern}--*_checkpoint.pt"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        matches.sort()
        return matches[-1]
    return None


def _load_component_state_dict(checkpoint_path: str) -> dict[str, Tensor]:
    """Load a component's state dict, stripping DDP 'module.' prefix if present."""
    state_dict = torch.load(checkpoint_path, weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.removeprefix("module.")] = v
    return new_state_dict


def _resolve_checkpoint_dir(pretrained_name_or_path: str) -> tuple[str, bool]:
    """Resolve a pretrained name or path to a local directory.

    Returns (local_dir, is_hub) tuple.
    """
    if os.path.isdir(pretrained_name_or_path):
        return pretrained_name_or_path, False
    return pretrained_name_or_path, True


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
    """Wrapper around OpenVLA-OFT for inference within lerobot's async inference system.

    This policy wraps the OpenVLA-OFT model, which consists of:
    - A VLA backbone (openvla-7b based on LLaMA-2)
    - An MLP action head (L1 regression or diffusion)
    - An optional proprioception projector
    - Optional LoRA adapters and FiLM conditioning

    The model handles its own image preprocessing and action unnormalization.
    """

    config_class = OpenVLAOFTConfig
    name = "openvla_oft"

    def __init__(self, config: OpenVLAOFTConfig, **kwargs):
        super().__init__(config)
        self.config: OpenVLAOFTConfig = config
        self._action_queue: deque[Tensor] = deque()

        # These are populated by from_pretrained
        self.vla = None
        self.vla_processor = None
        self.action_head = None
        self.proprio_projector = None
        self.noisy_action_projector = None
        self._unnorm_key: str = ""
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        **kwargs,
    ) -> T:
        """Load an OpenVLA-OFT checkpoint.

        This completely overrides the base class loading since OpenVLA-OFT
        uses AutoModelForVision2Seq + separate component checkpoints rather
        than a single model.safetensors file.
        """
        from huggingface_hub import hf_hub_download

        model_path = str(pretrained_name_or_path)

        if config is None:
            config = OpenVLAOFTConfig()

        instance = cls(config)
        is_local = os.path.isdir(model_path)

        logger.info(f"Loading OpenVLA-OFT from {model_path}...")

        # 1. Load dataset statistics FIRST so we can determine action/proprio dims.
        instance._load_dataset_stats(model_path, is_local)

        # 2. Patch prismatic constants BEFORE any prismatic or VLA model imports.
        #    The VLA backbone reads NUM_ACTIONS_CHUNK and ACTION_DIM from
        #    prismatic.vla.constants to determine how many action tokens to
        #    generate. If these are wrong (LIBERO defaults: ACTION_DIM=7,
        #    NUM_ACTIONS_CHUNK=8), the hidden states will have the wrong size
        #    and the action head reshape will fail.
        instance._configure_prismatic_constants(config)

        # 3. Load the base VLA model (now with correct constants)
        from transformers import AutoModelForVision2Seq, AutoProcessor

        vla = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=config.load_in_8bit,
            load_in_4bit=config.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # 3b. Re-patch constants in any modules that were freshly imported
        #     during from_pretrained (modeling_prismatic, action_heads, etc.)
        instance._configure_prismatic_constants(config)

        # 4. Apply LoRA adapter if present
        has_lora = False
        if is_local:
            has_lora = os.path.isdir(os.path.join(model_path, "lora_adapter"))
        else:
            try:
                hf_hub_download(repo_id=model_path, filename="lora_adapter/adapter_config.json")
                has_lora = True
            except Exception:
                pass

        if has_lora:
            logger.info("Applying LoRA adapter...")
            from peft import PeftModel

            lora_path = os.path.join(model_path, "lora_adapter") if is_local else model_path
            if is_local:
                vla = PeftModel.from_pretrained(vla, lora_path)
            else:
                lora_config_path = hf_hub_download(
                    repo_id=model_path, filename="lora_adapter/adapter_config.json"
                )
                lora_adapter_path = hf_hub_download(
                    repo_id=model_path, filename="lora_adapter/adapter_model.safetensors"
                )
                lora_dir = os.path.dirname(lora_config_path)
                vla = PeftModel.from_pretrained(vla, lora_dir)

            vla = vla.merge_and_unload()
            logger.info("LoRA adapter merged.")

        # 5. Apply FiLM if configured
        if config.use_film:
            logger.info("Applying FiLM conditioning to vision backbone...")
            vla = instance._apply_film(vla, model_path, is_local, config)

        # 6. Set number of images and move to device
        vla.vision_backbone.set_num_images_in_input(config.num_images_in_input)
        vla.eval()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not config.load_in_8bit and not config.load_in_4bit:
            vla = vla.to(device)

        instance.vla = vla
        instance._device = device

        # 7. Attach dataset stats to VLA (needed for predict_action unnormalization)
        instance._attach_norm_stats_to_vla()

        # 8. Load processor
        instance.vla_processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        # 9. Load action head
        instance._load_action_head(model_path, is_local, config)

        # 10. Load proprio projector
        if config.use_proprio:
            instance._load_proprio_projector(model_path, is_local, config)

        # 10b. Load noisy action projector (required for diffusion)
        if config.use_diffusion:
            instance._load_noisy_action_projector(model_path, is_local, config)

        # 11. Populate input/output features for server-side processing
        instance._populate_features(config)

        camera_order = [config.primary_image_key] + list(config.wrist_image_keys)
        logger.info(
            f"OpenVLA-OFT loaded successfully. unnorm_key={instance._unnorm_key}, "
            f"num_images={config.num_images_in_input}, "
            f"camera_order={camera_order} (slot 0=primary)"
        )

        instance.eval()
        return instance

    def _populate_features(self, config: OpenVLAOFTConfig) -> None:
        """Populate input/output features on the config for server-side processing.

        Uses the actual camera names (primary_image_key + wrist_image_keys) so
        that robot camera keys match policy keys directly, avoiding any rename.
        """
        img_shape = (3, config.image_size, config.image_size)

        if not config.input_features:
            input_features = {}
            camera_keys = [config.primary_image_key] + list(config.wrist_image_keys)
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
            if "action" in stats and "mean" in stats["action"]:
                return len(stats["action"]["mean"])
        return 14

    def _get_proprio_dim(self) -> int:
        norm_stats = getattr(self, "_norm_stats", None)
        if norm_stats and self._unnorm_key in norm_stats:
            stats = norm_stats[self._unnorm_key]
            if "proprio" in stats and "mean" in stats["proprio"]:
                return len(stats["proprio"]["mean"])
        return 14

    def _configure_prismatic_constants(self, config: OpenVLAOFTConfig) -> None:
        """Patch prismatic.vla.constants to match this checkpoint's training config.

        The prismatic constants module auto-detects the robot platform from
        sys.argv (looking for 'aloha', 'libero', etc.). When running through
        lerobot, it defaults to LIBERO (ACTION_DIM=7, NUM_ACTIONS_CHUNK=8),
        which is wrong for bimanual robots. We patch the constants module
        directly so that subsequent imports of action heads and projectors
        get the correct values.
        """
        import prismatic.vla.constants as prismatic_constants

        action_dim = self._get_action_dim()
        proprio_dim = self._get_proprio_dim()

        prismatic_constants.ACTION_DIM = action_dim
        prismatic_constants.PROPRIO_DIM = proprio_dim
        prismatic_constants.NUM_ACTIONS_CHUNK = config.num_actions_chunk

        norm_type = config.proprio_normalization_type.lower()
        if norm_type == "bounds":
            prismatic_constants.ACTION_PROPRIO_NORMALIZATION_TYPE = (
                prismatic_constants.NormalizationType.BOUNDS
            )
        elif norm_type == "bounds_q99":
            prismatic_constants.ACTION_PROPRIO_NORMALIZATION_TYPE = (
                prismatic_constants.NormalizationType.BOUNDS_Q99
            )
        else:
            raise ValueError(
                f"Unknown proprio_normalization_type: {config.proprio_normalization_type}. "
                "Must be 'bounds' or 'bounds_q99'."
            )

        # Also patch any already-imported modules that used `from` imports,
        # since `from X import Y` creates a local copy that our module-level
        # patch won't reach.
        import sys
        for mod_name, mod in sys.modules.items():
            if mod is None:
                continue
            if "action_head" in mod_name or "modeling_prismatic" in mod_name:
                if hasattr(mod, "NUM_ACTIONS_CHUNK"):
                    mod.NUM_ACTIONS_CHUNK = config.num_actions_chunk
                if hasattr(mod, "ACTION_DIM"):
                    mod.ACTION_DIM = action_dim
                if hasattr(mod, "PROPRIO_DIM"):
                    mod.PROPRIO_DIM = proprio_dim

        logger.info(
            f"Configured prismatic constants: ACTION_DIM={action_dim}, "
            f"PROPRIO_DIM={proprio_dim}, NUM_ACTIONS_CHUNK={config.num_actions_chunk}, "
            f"NORMALIZATION_TYPE={norm_type}"
        )

    def _load_dataset_stats(self, model_path: str, is_local: bool) -> None:
        """Load dataset_statistics.json (stored on self until VLA is ready)."""
        from huggingface_hub import hf_hub_download

        if is_local:
            stats_path = os.path.join(model_path, "dataset_statistics.json")
        else:
            try:
                stats_path = hf_hub_download(
                    repo_id=model_path, filename="dataset_statistics.json"
                )
            except Exception:
                logger.warning("No dataset_statistics.json found. Action unnormalization may fail.")
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

    def _attach_norm_stats_to_vla(self) -> None:
        """Attach loaded norm_stats to the VLA model for predict_action unnormalization."""
        if hasattr(self, "_norm_stats") and self._norm_stats:
            self.vla.norm_stats = self._norm_stats

    def _load_action_head(self, model_path: str, is_local: bool, config: OpenVLAOFTConfig) -> None:
        """Load the action head (L1 regression or diffusion)."""
        from huggingface_hub import HfApi, hf_hub_download

        llm_dim = self.vla.llm_dim
        action_dim = self._get_action_dim()

        if config.use_l1_regression:
            from prismatic.models.action_heads import L1RegressionActionHead

            action_head = L1RegressionActionHead(
                input_dim=llm_dim, hidden_dim=llm_dim, action_dim=action_dim,
            )
        elif config.use_diffusion:
            from prismatic.models.action_heads import DiffusionActionHead

            action_head = DiffusionActionHead(
                input_dim=llm_dim, hidden_dim=llm_dim, action_dim=action_dim,
                num_diffusion_steps_train=config.num_diffusion_steps_train,
            )
            action_head.noise_scheduler.set_timesteps(config.num_diffusion_steps_inference)
        else:
            raise ValueError("Either use_l1_regression or use_diffusion must be True.")

        action_head = action_head.to(torch.bfloat16).to(self._device)
        action_head.eval()

        # Find and load checkpoint
        checkpoint_path = self._find_component_checkpoint(
            model_path, is_local, "action_head"
        )
        if checkpoint_path:
            state_dict = _load_component_state_dict(checkpoint_path)
            action_head.load_state_dict(state_dict)
            logger.info(f"Loaded action head from {checkpoint_path}")
        else:
            logger.warning("No action head checkpoint found.")

        self.action_head = action_head

    def _load_proprio_projector(self, model_path: str, is_local: bool, config: OpenVLAOFTConfig) -> None:
        """Load the proprioception projector."""
        llm_dim = self.vla.llm_dim
        proprio_dim = self._get_proprio_dim()

        from prismatic.models.projectors import ProprioProjector

        proprio_projector = ProprioProjector(
            llm_dim=llm_dim, proprio_dim=proprio_dim,
        )
        proprio_projector = proprio_projector.to(torch.bfloat16).to(self._device)
        proprio_projector.eval()

        checkpoint_path = self._find_component_checkpoint(
            model_path, is_local, "proprio_projector"
        )
        if checkpoint_path:
            state_dict = _load_component_state_dict(checkpoint_path)
            proprio_projector.load_state_dict(state_dict)
            logger.info(f"Loaded proprio projector from {checkpoint_path}")
        else:
            logger.warning("No proprio projector checkpoint found.")

        self.proprio_projector = proprio_projector

    def _load_noisy_action_projector(
        self, model_path: str, is_local: bool, config: OpenVLAOFTConfig
    ) -> None:
        """Load the noisy action projector (required for diffusion-based prediction)."""
        llm_dim = self.vla.llm_dim

        from prismatic.models.projectors import NoisyActionProjector

        noisy_action_projector = NoisyActionProjector(llm_dim=llm_dim)
        noisy_action_projector = noisy_action_projector.to(torch.bfloat16).to(self._device)
        noisy_action_projector.eval()

        checkpoint_path = self._find_component_checkpoint(
            model_path, is_local, "noisy_action_projector"
        )
        if checkpoint_path:
            state_dict = _load_component_state_dict(checkpoint_path)
            noisy_action_projector.load_state_dict(state_dict)
            logger.info(f"Loaded noisy action projector from {checkpoint_path}")
        else:
            logger.warning("No noisy action projector checkpoint found.")

        self.noisy_action_projector = noisy_action_projector

    def _apply_film(self, vla, model_path: str, is_local: bool, config: OpenVLAOFTConfig):
        """Apply FiLM conditioning to the VLA vision backbone.

        Returns the unwrapped model (not the PeftModel wrapper) with FiLM applied,
        matching the original openvla-oft `_apply_film_to_vla`.
        """
        from peft import LoraConfig, get_peft_model
        from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone

        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=min(config.lora_rank, 16),
            lora_dropout=0.0,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)

        new_vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.vision_backbone, llm_dim=vla.llm_dim,
        )
        vla.model.vision_backbone = new_vision_backbone

        checkpoint_path = self._find_component_checkpoint(
            model_path, is_local, "vision_backbone"
        )
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, weights_only=True)
            vla.model.vision_backbone.load_state_dict(state_dict)

        vla = vla.model
        vla.vision_backbone = vla.vision_backbone.to(torch.bfloat16)

        return vla

    def _find_component_checkpoint(
        self, model_path: str, is_local: bool, component: str
    ) -> str | None:
        """Find a component checkpoint file, either locally or on HF Hub."""
        from huggingface_hub import HfApi, hf_hub_download

        if is_local:
            return _find_checkpoint_file(model_path, component)

        # Search HF Hub for matching file
        try:
            api = HfApi()
            files = api.list_repo_files(model_path)
            matches = [f for f in files if f.startswith(f"{component}--") and f.endswith("_checkpoint.pt")]
            if matches:
                matches.sort()
                return hf_hub_download(repo_id=model_path, filename=matches[-1])
        except Exception as e:
            logger.warning(f"Could not find {component} checkpoint on Hub: {e}")

        return None

    def _tensor_to_numpy_image(self, tensor: Tensor) -> np.ndarray:
        """Convert a CHW float [0,1] tensor to HWC uint8 numpy array."""
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        img = tensor.detach().cpu()
        img = (img * 255).clamp(0, 255).to(torch.uint8)
        img = img.permute(1, 2, 0).numpy()
        return img

    def _normalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        """Normalize proprioception using the model's norm_stats."""
        stats = self.vla.norm_stats[self._unnorm_key]["proprio"]
        norm_type = self.config.proprio_normalization_type.lower()

        if norm_type == "bounds":
            mask = np.array(stats.get("mask", np.ones_like(stats["min"], dtype=bool)))
            high, low = np.array(stats["max"]), np.array(stats["min"])
        elif norm_type == "bounds_q99":
            mask = np.array(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)))
            high, low = np.array(stats["q99"]), np.array(stats["q01"])
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

        return np.clip(
            np.where(
                mask,
                2 * (proprio - low) / (high - low + 1e-8) - 1,
                proprio,
            ),
            a_min=-1.0,
            a_max=1.0,
        )

    @staticmethod
    def _resize_image_for_policy(img: np.ndarray, resize_size: int) -> np.ndarray:
        """Resize image to match training pipeline, including JPEG encode/decode roundtrip.

        Matches the original openvla-oft `resize_image_for_policy` which goes through
        TF JPEG codec to reproduce the compression artifacts seen during RLDS training.
        """
        import tensorflow as tf

        img = tf.image.encode_jpeg(img)
        img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
        img = tf.image.resize(img, (resize_size, resize_size), method="lanczos3", antialias=True)
        img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
        return img.numpy()

    @staticmethod
    def _center_crop_image(image: np.ndarray) -> np.ndarray:
        """Center crop with scale 0.9, matching the original TF-based crop_and_resize.

        Matches the original openvla-oft `center_crop_image` / `crop_and_resize`.
        """
        import tensorflow as tf

        crop_scale = 0.9
        image_tf = tf.convert_to_tensor(image)
        orig_dtype = image_tf.dtype
        image_tf = tf.image.convert_image_dtype(image_tf, tf.float32)

        if image_tf.shape.ndims == 3:
            image_tf = tf.expand_dims(image_tf, axis=0)

        new_edge = tf.clip_by_value(tf.sqrt(crop_scale), 0, 1)
        offset = (1 - new_edge) / 2
        bounding_boxes = tf.reshape(
            tf.stack([offset, offset, offset + new_edge, offset + new_edge]),
            shape=(1, 4),
        )
        image_tf = tf.image.crop_and_resize(
            image_tf, bounding_boxes, [0], (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE)
        )
        image_tf = image_tf[0]
        image_tf = tf.clip_by_value(image_tf, 0, 1)
        image_tf = tf.image.convert_image_dtype(image_tf, orig_dtype, saturate=True)
        return image_tf.numpy()

    def _prepare_images(self, images: list[np.ndarray]) -> list:
        """Prepare images for VLA input: resize, optionally center crop, convert to PIL.

        Uses the same TF-based pipeline as the original openvla-oft for exact
        distribution matching (JPEG roundtrip, TF lanczos3 resize, TF crop_and_resize).
        """
        from PIL import Image

        processed = []
        for img in images:
            assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[-1] == 3

            if img.shape[:2] != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE):
                img = self._resize_image_for_policy(img, OPENVLA_IMAGE_SIZE)

            if self.config.center_crop:
                img = self._center_crop_image(img)

            pil_img = Image.fromarray(img).convert("RGB")
            processed.append(pil_img)
        return processed

    @torch.inference_mode()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Predict an action chunk from a lerobot-formatted observation batch.

        Converts lerobot tensors to the format expected by OpenVLA-OFT, runs
        inference, and returns the action chunk as a (1, T, action_dim) tensor.
        """
        self.eval()

        # Extract task description (may be a string, list of strings, or tensor)
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

        # Look up images by the configured camera names to guarantee correct ordering.
        primary_key = f"{OBS_IMAGES}.{self.config.primary_image_key}"
        wrist_keys = [f"{OBS_IMAGES}.{k}" for k in self.config.wrist_image_keys]

        if primary_key not in batch:
            available = [k for k in batch if k.startswith(OBS_IMAGES)]
            raise ValueError(
                f"Primary image key '{primary_key}' not in batch. "
                f"Available image keys: {available}"
            )

        # Convert tensors to numpy images: primary first, then wrist in config order
        all_np_images = [self._tensor_to_numpy_image(batch[primary_key])]
        for wk in wrist_keys:
            if wk in batch:
                all_np_images.append(self._tensor_to_numpy_image(batch[wk]))

        # Limit to configured number of images
        all_np_images = all_np_images[: self.config.num_images_in_input]

        # Prepare images (resize, crop, to PIL)
        all_pil_images = self._prepare_images(all_np_images)

        # Build VLA prompt
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

        # Process primary image through VLA processor
        primary_image = all_pil_images.pop(0)
        inputs = self.vla_processor(prompt, primary_image).to(
            self._device, dtype=torch.bfloat16
        )

        # Process additional wrist images if any
        if all_pil_images:
            wrist_inputs_list = [
                self.vla_processor(prompt, wrist_img).to(
                    self._device, dtype=torch.bfloat16
                )
                for wrist_img in all_pil_images
            ]
            primary_pixels = inputs["pixel_values"]
            wrist_pixels = [wi["pixel_values"] for wi in wrist_inputs_list]
            inputs["pixel_values"] = torch.cat(
                [primary_pixels] + wrist_pixels, dim=1
            )

        # Process proprioception
        proprio = None
        if self.config.use_proprio and OBS_STATE in batch:
            state_tensor = batch[OBS_STATE]
            if state_tensor.ndim == 2:
                state_tensor = state_tensor.squeeze(0)
            state_np = state_tensor.detach().cpu().float().numpy()
            proprio = self._normalize_proprio(state_np)

        # Run VLA inference
        action, _ = self.vla.predict_action(
            **inputs,
            unnorm_key=self._unnorm_key,
            do_sample=False,
            proprio=proprio,
            proprio_projector=self.proprio_projector,
            noisy_action_projector=self.noisy_action_projector,
            action_head=self.action_head,
            use_film=self.config.use_film,
        )

        # action is a numpy array of shape (chunk_size, action_dim) or list of arrays
        if isinstance(action, list):
            action = np.stack(action, axis=0)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()

        # Ensure shape is (1, chunk_size, action_dim)
        if action.ndim == 2:
            action = action.unsqueeze(0)
        elif action.ndim == 1:
            action = action.unsqueeze(0).unsqueeze(0)

        return action

    @torch.inference_mode()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            # actions is (1, chunk_size, action_dim); queue individual steps
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
