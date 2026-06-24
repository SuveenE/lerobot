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

"""Lossless RVL depth-map compression for LeRobot datasets.

Depth maps are 16-bit single-channel images (raw millimeters). By default they
are stored as lossless 16-bit PNGs (`dtype: "image"`), which is bulky. This
module wraps the `rvlcodec` extension (https://github.com/cortexairobot/rvlcodec)
to compress each depth frame with the RVL ("Fast Lossless Depth Image
Compression", Wilson 2018) codec instead.

Encoded frames are stored as a binary column in the dataset parquet (feature
`dtype: "depth_rvl"`). Each frame is self-describing: an 8-byte little-endian
header carries `(height, width)` so the frame can be decoded for random access
at train time without needing the feature metadata.

The `rvlcodec` package is an optional dependency (it is a Rust extension and
requires a Rust toolchain to build). It is imported lazily so that nothing in
this module is needed unless depth RVL recording/reading is actually used.
"""

import struct

import numpy as np

# LeRobot feature dtype used for RVL-compressed depth columns.
DEPTH_RVL = "depth_rvl"

# 8-byte little-endian header: (height, width) as uint32.
_HEADER = struct.Struct("<II")


def _import_rvlcodec():
    try:
        import rvlcodec
    except ImportError as e:
        raise ImportError(
            "The 'rvlcodec' package is required to record or read depth with the RVL codec "
            "(`--dataset.depth_codec=rvl`). It is bundled as a git submodule. Install it with a "
            "Rust toolchain available, e.g. `uv pip install ./rvlcodec` or `pip install ./rvlcodec` "
            "(or `pip install -e '.[depth]'`)."
        ) from e
    return rvlcodec


def _as_hw_uint16(depth: np.ndarray) -> np.ndarray:
    """Squeeze a depth frame to a contiguous (H, W) uint16 array.

    Accepts (H, W), (H, W, 1) or (1, H, W) layouts.
    """
    arr = np.squeeze(np.asarray(depth))
    if arr.ndim != 2:
        raise ValueError(
            f"Expected a single-channel depth map, got array with shape {np.asarray(depth).shape}."
        )
    if arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)
    return np.ascontiguousarray(arr)


def encode_depth_rvl(depth: np.ndarray) -> bytes:
    """Compress a single depth frame to RVL bytes (with an (H, W) header)."""
    rvlcodec = _import_rvlcodec()
    arr = _as_hw_uint16(depth)
    height, width = arr.shape
    payload = bytes(rvlcodec.compress_rvl(arr))
    return _HEADER.pack(height, width) + payload


def decode_depth_rvl(data: bytes) -> np.ndarray:
    """Decompress RVL bytes back into a single (H, W) uint16 depth frame."""
    rvlcodec = _import_rvlcodec()
    height, width = _HEADER.unpack_from(data, 0)
    payload = bytes(data[_HEADER.size :])
    arr = rvlcodec.decompress_rvl(payload, height, width)
    return np.asarray(arr, dtype=np.uint16)


def is_depth_image_feature(feature: dict) -> bool:
    """Whether a LeRobot feature is a single-channel depth `image` column."""
    return feature.get("dtype") == "image" and tuple(feature.get("shape", ()))[-1:] == (1,)


def convert_depth_features_to_rvl(features: dict[str, dict]) -> dict[str, dict]:
    """Return a copy of `features` with depth `image` columns switched to RVL.

    Single-channel `image` features (depth maps) are re-typed to `depth_rvl`;
    all other features are left untouched. Used when `--dataset.depth_codec=rvl`.
    """
    converted: dict[str, dict] = {}
    for key, feature in features.items():
        if is_depth_image_feature(feature):
            feature = {**feature, "dtype": DEPTH_RVL}
        converted[key] = feature
    return converted
