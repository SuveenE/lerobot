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

"""
Validates a LeRobot dataset for potential training issues, particularly checking
that video frame timestamps don't exceed video boundaries.

This script helps catch issues like:
- Frame index out of bounds errors during training
- Timestamp/FPS mismatches between metadata and actual video content
- Off-by-one errors at video boundaries

Example usage:
```shell
# Validate a local dataset
python -m lerobot.scripts.lerobot_validate_dataset --repo-id /path/to/dataset

# Validate a HuggingFace dataset
python -m lerobot.scripts.lerobot_validate_dataset --repo-id lerobot/aloha_sim_transfer_cube_human

# Quick validation (skip deep frame-by-frame check)
python -m lerobot.scripts.lerobot_validate_dataset --repo-id /path/to/dataset --quick
```
"""

import argparse
import importlib
import logging
from pathlib import Path

import av
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_video_frame_info(video_path: str | Path) -> dict:
    """Get video frame count and FPS using PyAV and optionally torchcodec."""
    video_path = str(video_path)
    info = {
        "path": video_path,
        "pyav_num_frames": None,
        "pyav_fps": None,
        "pyav_duration_s": None,
        "torchcodec_num_frames": None,
        "torchcodec_fps": None,
    }

    # Get info using PyAV
    try:
        with av.open(video_path) as container:
            video_stream = container.streams.video[0]
            # Note: frames attribute may require decoding to get accurate count
            info["pyav_fps"] = float(video_stream.base_rate)
            if video_stream.duration is not None:
                info["pyav_duration_s"] = float(video_stream.duration * video_stream.time_base)
            else:
                info["pyav_duration_s"] = float(container.duration / av.time_base)
            # Calculate expected frames from duration
            info["pyav_num_frames"] = int(info["pyav_duration_s"] * info["pyav_fps"])
    except Exception as e:
        logger.warning(f"PyAV could not read {video_path}: {e}")

    # Get info using torchcodec if available
    if importlib.util.find_spec("torchcodec"):
        try:
            from torchcodec.decoders import VideoDecoder

            decoder = VideoDecoder(video_path)
            metadata = decoder.metadata
            info["torchcodec_num_frames"] = metadata.num_frames
            info["torchcodec_fps"] = metadata.average_fps
        except Exception as e:
            logger.warning(f"torchcodec could not read {video_path}: {e}")

    return info


def validate_video_frame_bounds(
    video_info: dict,
    from_timestamp: float,
    to_timestamp: float,
    episode_idx: int,
    video_key: str,
) -> list[dict]:
    """
    Validate that timestamps won't produce out-of-bounds frame indices.

    Returns a list of issues found.
    """
    issues = []

    num_frames = video_info.get("torchcodec_num_frames") or video_info.get("pyav_num_frames")
    fps = video_info.get("torchcodec_fps") or video_info.get("pyav_fps")

    if num_frames is None or fps is None:
        issues.append({
            "type": "missing_metadata",
            "episode": episode_idx,
            "video_key": video_key,
            "message": f"Could not determine frame count or FPS for video",
            "severity": "warning",
        })
        return issues

    # Check the boundary frame index
    max_frame_idx = round(to_timestamp * fps)

    if max_frame_idx >= num_frames:
        issues.append({
            "type": "frame_index_out_of_bounds",
            "episode": episode_idx,
            "video_key": video_key,
            "from_timestamp": from_timestamp,
            "to_timestamp": to_timestamp,
            "max_frame_idx": max_frame_idx,
            "num_frames": num_frames,
            "fps": fps,
            "message": (
                f"Episode {episode_idx}, video '{video_key}': "
                f"max frame index {max_frame_idx} >= num_frames {num_frames}. "
                f"This will cause 'Invalid frame index' errors during training."
            ),
            "severity": "error",
        })

    # Check for potential off-by-one at the boundary
    # Even if max_frame_idx == num_frames - 1, check if there's risk with rounding
    last_valid_ts = (num_frames - 1) / fps
    if to_timestamp > last_valid_ts:
        # Calculate how close we are to the boundary
        margin = to_timestamp - last_valid_ts
        if margin > 0 and max_frame_idx < num_frames:
            issues.append({
                "type": "boundary_risk",
                "episode": episode_idx,
                "video_key": video_key,
                "to_timestamp": to_timestamp,
                "last_valid_ts": last_valid_ts,
                "margin_s": margin,
                "message": (
                    f"Episode {episode_idx}, video '{video_key}': "
                    f"to_timestamp {to_timestamp:.6f}s exceeds last valid timestamp "
                    f"{last_valid_ts:.6f}s by {margin:.6f}s. May cause issues with some decoders."
                ),
                "severity": "warning",
            })

    return issues


def validate_dataset(
    repo_id: str,
    root: str | Path | None = None,
    quick: bool = False,
    sample_rate: float = 0.1,
) -> dict:
    """
    Validate a LeRobot dataset for potential training issues.

    Args:
        repo_id: Repository ID or local path to the dataset
        root: Optional root path override
        quick: If True, skip deep frame-by-frame validation
        sample_rate: Fraction of samples to check in deep validation (0.0 to 1.0)

    Returns:
        Dictionary with validation results
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    logger.info(f"Loading dataset: {repo_id}")

    # Load the dataset
    dataset = LeRobotDataset(repo_id, root=root)

    results = {
        "repo_id": repo_id,
        "num_episodes": dataset.meta.total_episodes,
        "num_frames": dataset.meta.total_frames,
        "video_keys": list(dataset.meta.video_keys),
        "issues": [],
        "videos_checked": 0,
        "samples_checked": 0,
    }

    logger.info(f"Dataset has {results['num_episodes']} episodes, {results['num_frames']} frames")
    logger.info(f"Video keys: {results['video_keys']}")

    if not dataset.meta.video_keys:
        logger.info("No video keys found - nothing to validate")
        return results

    # Cache video info to avoid repeated reads
    video_info_cache = {}

    # Validate each episode's video metadata
    logger.info("Validating video frame boundaries...")
    for ep_idx, episode in enumerate(dataset.meta.episodes):
        for video_key in dataset.meta.video_keys:
            video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, video_key)

            if not video_path.exists():
                results["issues"].append({
                    "type": "missing_video",
                    "episode": ep_idx,
                    "video_key": video_key,
                    "path": str(video_path),
                    "message": f"Video file not found: {video_path}",
                    "severity": "error",
                })
                continue

            # Get or cache video info
            video_path_str = str(video_path)
            if video_path_str not in video_info_cache:
                video_info_cache[video_path_str] = get_video_frame_info(video_path)
                results["videos_checked"] += 1

            video_info = video_info_cache[video_path_str]

            # Get episode timestamps
            from_ts_key = f"videos/{video_key}/from_timestamp"
            to_ts_key = f"videos/{video_key}/to_timestamp"

            from_timestamp = episode.get(from_ts_key, 0.0)
            to_timestamp = episode.get(to_ts_key)

            if to_timestamp is None:
                results["issues"].append({
                    "type": "missing_timestamp",
                    "episode": ep_idx,
                    "video_key": video_key,
                    "message": f"Missing to_timestamp for episode {ep_idx}, video '{video_key}'",
                    "severity": "warning",
                })
                continue

            # Validate frame bounds
            issues = validate_video_frame_bounds(
                video_info, from_timestamp, to_timestamp, ep_idx, video_key
            )
            results["issues"].extend(issues)

        # Progress update every 100 episodes
        if (ep_idx + 1) % 100 == 0:
            logger.info(f"Checked {ep_idx + 1}/{results['num_episodes']} episodes...")

    # Deep validation: actually try to load samples
    if not quick and len(results["issues"]) == 0:
        logger.info(f"Running deep validation (sampling {sample_rate*100:.0f}% of frames)...")
        num_samples = max(1, int(len(dataset) * sample_rate))
        indices = torch.randperm(len(dataset))[:num_samples].tolist()

        for i, idx in enumerate(indices):
            try:
                _ = dataset[idx]
                results["samples_checked"] += 1
            except RuntimeError as e:
                if "Invalid frame index" in str(e):
                    results["issues"].append({
                        "type": "runtime_frame_error",
                        "sample_idx": idx,
                        "message": str(e),
                        "severity": "error",
                    })
                else:
                    raise

            if (i + 1) % 100 == 0:
                logger.info(f"Deep validation: checked {i + 1}/{num_samples} samples...")

    return results


def print_results(results: dict) -> None:
    """Print validation results in a readable format."""
    print("\n" + "=" * 70)
    print("DATASET VALIDATION RESULTS")
    print("=" * 70)

    print(f"\nDataset: {results['repo_id']}")
    print(f"Episodes: {results['num_episodes']}")
    print(f"Frames: {results['num_frames']}")
    print(f"Videos checked: {results['videos_checked']}")
    print(f"Samples checked: {results['samples_checked']}")

    errors = [i for i in results["issues"] if i.get("severity") == "error"]
    warnings = [i for i in results["issues"] if i.get("severity") == "warning"]

    if not results["issues"]:
        print("\n✅ No issues found! Dataset is ready for training.")
    else:
        if errors:
            print(f"\n❌ ERRORS ({len(errors)}):")
            print("-" * 50)
            for issue in errors:
                print(f"  • {issue['message']}")
                if issue["type"] == "frame_index_out_of_bounds":
                    print(f"    Frame {issue['max_frame_idx']} requested, but video has {issue['num_frames']} frames")

        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            print("-" * 50)
            for issue in warnings:
                print(f"  • {issue['message']}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Validate a LeRobot dataset for potential training issues"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID or local path to the dataset",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Optional root path override",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip deep frame-by-frame validation",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.1,
        help="Fraction of samples to check in deep validation (default: 0.1)",
    )

    args = parser.parse_args()

    results = validate_dataset(
        repo_id=args.repo_id,
        root=args.root,
        quick=args.quick,
        sample_rate=args.sample_rate,
    )

    print_results(results)

    # Exit with error code if there are errors
    errors = [i for i in results["issues"] if i.get("severity") == "error"]
    if errors:
        exit(1)


if __name__ == "__main__":
    main()
