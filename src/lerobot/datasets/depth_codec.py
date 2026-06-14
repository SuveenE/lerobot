#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
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
"""Lossless depth compression for dataset depth columns via the ``rvlcodec`` codec.

Single-channel ``uint16`` depth maps are compressed with zdepth (RVL + zstd) and
the resulting bytes are stored directly in the Parquet column, instead of being
encoded as 16-bit PNGs. This is markedly faster to encode/decode and compresses
better, while staying fully lossless (``mode=6``).

The codec is decoupled behind these helpers so the rest of the dataset code only
deals with ``bytes``. ``rvlcodec`` ships as a platform-specific (linux-x86_64)
wheel and is therefore an optional dependency: it is only required when a dataset
actually records or reads rvl-encoded depth.
"""

import numpy as np

try:
    import rvlcodec

    _RVL_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover - import availability is environment dependent
    rvlcodec = None
    _RVL_IMPORT_ERROR = e

# Marker stored in a feature dict (``info.json``) to flag rvl-encoded depth columns.
RVL_ENCODING = "rvl"

# zdepth encode mode. 6 == Lossless16: round-trips any uint16 value exactly.
RVL_LOSSLESS_MODE = 6

# zdepth processes the image in 8x8 tiles, so both dimensions must be multiples of 8.
RVL_BLOCK = 8

_INSTALL_HINT = (
    "The 'rvlcodec' package is required to encode/decode rvl-compressed depth but is not "
    "available in this environment. Install the platform wheel (linux-x86_64, CPython>=3.11) "
    "or build it from source (https://github.com/cortexairobot/rvlcodec)."
)


def is_rvl_available() -> bool:
    """Return True if the ``rvlcodec`` extension can be imported."""
    return rvlcodec is not None


def _require_rvl() -> None:
    if rvlcodec is None:
        raise ImportError(_INSTALL_HINT) from _RVL_IMPORT_ERROR


def is_depth_feature(feature: dict) -> bool:
    """Return True if a LeRobot feature describes a single-channel depth map."""
    if feature.get("dtype") not in ("image", "video"):
        return False
    shape = feature.get("shape")
    return bool(shape) and tuple(shape)[-1] == 1


def is_rvl_depth_feature(feature: dict) -> bool:
    """Return True if a feature is a depth map stored as rvl-compressed bytes."""
    return is_depth_feature(feature) and feature.get("encoding") == RVL_ENCODING


def _to_depth_2d(depth: np.ndarray) -> np.ndarray:
    """Squeeze a depth array to a C-contiguous 2D ``uint16`` (H, W) array."""
    arr = np.asarray(depth)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected a single-channel depth map, got array with shape {np.asarray(depth).shape}."
        )

    if arr.dtype != np.uint16:
        if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer):
            if arr.min() < 0 or arr.max() > np.iinfo(np.uint16).max:
                raise ValueError(
                    "Depth values must fit in uint16 (0..65535) to be losslessly compressed, "
                    f"but range is [{arr.min()}, {arr.max()}]."
                )
            arr = arr.astype(np.uint16)
        else:
            raise ValueError(f"Unsupported depth dtype {arr.dtype}; expected an integer or float type.")

    h, w = arr.shape
    if h % RVL_BLOCK != 0 or w % RVL_BLOCK != 0:
        raise ValueError(
            f"zdepth requires both depth dimensions to be multiples of {RVL_BLOCK}, but got "
            f"(height={h}, width={w}). Configure the camera to a supported resolution."
        )

    return np.ascontiguousarray(arr)


def encode_depth(depth: np.ndarray) -> bytes:
    """Losslessly compress a single-channel depth map to zdepth ``bytes``.

    Accepts (H, W), (H, W, 1) or (1, H, W) ``uint16`` arrays.
    """
    _require_rvl()
    arr = _to_depth_2d(depth)
    return rvlcodec.compress_zdepth(arr, mode=RVL_LOSSLESS_MODE)


def decode_depth(data: bytes) -> np.ndarray:
    """Decompress zdepth ``bytes`` back into a 2D ``uint16`` (H, W) depth map.

    The width and height are read from the frame header.
    """
    _require_rvl()
    return rvlcodec.decompress_zdepth(bytes(data))
