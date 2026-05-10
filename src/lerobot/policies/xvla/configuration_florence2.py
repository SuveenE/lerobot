#!/usr/bin/env python

# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Florence2VisionConfig(PretrainedConfig):
    model_type = "davit"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        drop_path_rate=0.1,
        patch_size=None,
        patch_stride=None,
        patch_padding=None,
        patch_prenorm=None,
        enable_checkpoint=False,
        dim_embed=None,
        num_heads=None,
        num_groups=None,
        depths=None,
        window_size=12,
        projection_dim=1024,
        visual_temporal_embedding=None,
        image_pos_embed=None,
        image_feature_source=None,
        **kwargs,
    ):
        self.drop_path_rate = drop_path_rate
        self.patch_size = patch_size if patch_size is not None else [7, 3, 3, 3]
        self.patch_stride = patch_stride if patch_stride is not None else [4, 2, 2, 2]
        self.patch_padding = patch_padding if patch_padding is not None else [3, 1, 1, 1]
        self.patch_prenorm = patch_prenorm if patch_prenorm is not None else [False, True, True, True]
        self.enable_checkpoint = enable_checkpoint
        self.dim_embed = dim_embed if dim_embed is not None else [256, 512, 1024, 2048]
        self.num_heads = num_heads if num_heads is not None else [8, 16, 32, 64]
        self.num_groups = num_groups if num_groups is not None else [8, 16, 32, 64]
        self.depths = depths if depths is not None else [1, 1, 9, 1]
        self.window_size = window_size
        self.projection_dim = projection_dim

        if visual_temporal_embedding is None:
            visual_temporal_embedding = {
                "type": "COSINE",
                "max_temporal_embeddings": 100,
            }
        self.visual_temporal_embedding = visual_temporal_embedding

        if image_pos_embed is None:
            image_pos_embed = {
                "type": "learned_abs_2d",
                "max_pos_embeddings": 1000,
            }
        self.image_pos_embed = image_pos_embed

        self.image_feature_source = (
            image_feature_source
            if image_feature_source is not None
            else ["spatial_avg_pool", "temporal_avg_pool"]
        )

        super().__init__(**kwargs)


class Florence2LanguageConfig(PretrainedConfig):
    model_type = "florence2_language"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
        vocab_size=51289,
        max_position_embeddings=1024,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        use_cache=True,
        num_labels=3,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        is_encoder_decoder=True,
        decoder_start_token_id=2,
        forced_eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding

        super().__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )

        if not hasattr(self, "forced_bos_token_id"):
            self.forced_bos_token_id = None
        if self.forced_bos_token_id is None and kwargs.get("force_bos_token_to_be_generated", False):
            self.forced_bos_token_id = self.bos_token_id
            warnings.warn(
                f"Please make sure the config includes `forced_bos_token_id={self.bos_token_id}` in future versions. "
                "The config can simply be saved and uploaded again to be fixed.",
                stacklevel=2,
            )


class Florence2Config(PretrainedConfig):
    model_type = "florence2"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        vocab_size=51289,
        projection_dim=1024,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        if vision_config is not None:
            vision_config = Florence2VisionConfig(**vision_config)
        self.vision_config = vision_config

        self.text_config = text_config
        if text_config is not None:
            self.text_config = Florence2LanguageConfig(**text_config)

        super().__init__(**kwargs)
