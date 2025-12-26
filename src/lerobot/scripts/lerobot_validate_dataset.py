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
- Missing video files

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
    to_timestamp: float,
    episode_idx: int,
    video_key: str,
    dataset_fps: int,
) -> list[dict]:
    """
    Validate that timestamps won't produce out-of-bounds frame indices.

    The to_timestamp represents the absolute end position of this episode
    within the video file. Multiple episodes may be concatenated in a single
    video file, so we check if to_timestamp exceeds the video duration.

    Returns a list of issues found.
    """
    issues = []

    num_frames = video_info.get("torchcodec_num_frames") or video_info.get("pyav_num_frames")
    video_fps = video_info.get("torchcodec_fps") or video_info.get("pyav_fps")
    video_duration_s = video_info.get("pyav_duration_s")

    if num_frames is None or video_fps is None or video_duration_s is None:
        issues.append({
            "type": "missing_metadata",
            "episode": episode_idx,
            "video_key": video_key,
            "message": f"Could not determine frame count, FPS, or duration for video",
            "severity": "warning",
        })
        return issues

    # Primary check: Does the to_timestamp exceed video duration?
    # Add small tolerance for floating point issues
    tolerance_s = 0.1
    if to_timestamp > video_duration_s + tolerance_s:
        # Calculate frame indices for informational purposes
        max_frame_idx = round(to_timestamp * video_fps)
        issues.append({
            "type": "timestamp_exceeds_video_duration",
            "episode": episode_idx,
            "video_key": video_key,
            "to_timestamp": to_timestamp,
            "video_duration_s": video_duration_s,
            "max_frame_idx": max_frame_idx,
            "num_frames": num_frames,
            "video_fps": video_fps,
            "dataset_fps": dataset_fps,
            "message": (
                f"Episode {episode_idx}, video '{video_key}': "
                f"to_timestamp {to_timestamp:.3f}s > video duration {video_duration_s:.3f}s. "
                f"This may cause frame index errors during training."
            ),
            "severity": "error",
        })

    return issues


def validate_dataset(
    repo_id: str,
    root: str | Path | None = None,
    quick: bool = False,
    sample_rate: float = 1.0,
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

    # Group episodes by their video file to validate each file once
    video_file_episodes: dict[str, list[tuple[int, float, float]]] = {}

    logger.info("Collecting video file information...")
    for ep_idx in range(dataset.meta.total_episodes):
        # Access episode data correctly from HuggingFace dataset
        episode = dataset.meta.episodes[ep_idx]

        for video_key in dataset.meta.video_keys:
            try:
                video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, video_key)
            except (KeyError, TypeError) as e:
                results["issues"].append({
                    "type": "missing_video_metadata",
                    "episode": ep_idx,
                    "video_key": video_key,
                    "message": f"Could not get video path for episode {ep_idx}, video '{video_key}': {e}",
                    "severity": "error",
                })
                continue

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

            # Get timestamps for this episode's video
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

            video_path_str = str(video_path)
            if video_path_str not in video_file_episodes:
                video_file_episodes[video_path_str] = []
            video_file_episodes[video_path_str].append((ep_idx, from_timestamp, to_timestamp))

    # Validate each video file
    logger.info(f"Validating {len(video_file_episodes)} unique video files...")
    logger.info(f"Dataset FPS: {dataset.meta.fps}")

    # Debug: show sample of video files
    for i, (path, eps) in enumerate(list(video_file_episodes.items())[:3]):
        logger.info(f"  Sample video {i}: {path}")
        logger.info(f"    Episodes: {len(eps)}, max_to_ts: {max(to_ts for _, _, to_ts in eps):.3f}s")

    dataset_fps = dataset.meta.fps

    for video_path_str, episodes_info in video_file_episodes.items():
        # Get video info (cached)
        if video_path_str not in video_info_cache:
            video_info_cache[video_path_str] = get_video_frame_info(video_path_str)
            results["videos_checked"] += 1

        video_info = video_info_cache[video_path_str]

        # Find the maximum to_timestamp for this video file
        max_to_timestamp = max(to_ts for _, _, to_ts in episodes_info)
        max_episode_idx = next(ep_idx for ep_idx, _, to_ts in episodes_info if to_ts == max_to_timestamp)

        # Extract video_key from path
        video_key = Path(video_path_str).parent.name

        # Debug output for first few files
        if results["videos_checked"] <= 5:
            logger.info(f"  Checking: {video_path_str}")
            logger.info(f"    Episodes in this file: {len(episodes_info)}")
            logger.info(f"    Max to_timestamp: {max_to_timestamp:.3f}s")
            logger.info(f"    Video duration: {video_info.get('pyav_duration_s', 'N/A')}s")
            logger.info(f"    Video num_frames: {video_info.get('torchcodec_num_frames') or video_info.get('pyav_num_frames')}")
            logger.info(f"    Video fps: {video_info.get('torchcodec_fps') or video_info.get('pyav_fps')}")

        # Validate that max timestamp doesn't exceed video duration
        issues = validate_video_frame_bounds(
            video_info,
            max_to_timestamp,
            max_episode_idx,
            video_key,
            dataset_fps,
        )
        results["issues"].extend(issues)

    # Progress update
    logger.info(f"Checked {results['videos_checked']} video files")

    # Deep validation: actually try to load samples
    if not quick:
        errors_before = len([i for i in results["issues"] if i.get("severity") == "error"])
        if errors_before == 0:
            logger.info(f"Running deep validation (sampling {sample_rate*100:.0f}% of frames)...")
            num_samples = max(1, int(len(dataset) * sample_rate))
            indices = torch.randperm(len(dataset))[:num_samples].tolist()

            failed_indices = []
            for i, idx in enumerate(indices):
                try:
                    _ = dataset[idx]
                    results["samples_checked"] += 1
                except (RuntimeError, AssertionError) as e:
                    error_msg = str(e)
                    # Try to get episode info for this sample
                    try:
                        episode_idx = dataset.hf_dataset[idx]["episode_index"]
                        frame_idx = dataset.hf_dataset[idx]["frame_index"]
                    except Exception:
                        episode_idx = "unknown"
                        frame_idx = "unknown"
                    
                    failed_indices.append(idx)
                    
                    if "Invalid frame index" in error_msg:
                        error_type = "runtime_frame_error"
                    elif "tolerance" in error_msg.lower():
                        error_type = "tolerance_error"
                    else:
                        error_type = "load_error"
                    
                    results["issues"].append({
                        "type": error_type,
                        "sample_idx": idx,
                        "episode_idx": episode_idx,
                        "frame_idx": frame_idx,
                        "message": error_msg[:200],  # Truncate long messages
                        "severity": "error",
                    })

                if (i + 1) % 500 == 0:
                    logger.info(f"Deep validation: checked {i + 1}/{num_samples} samples, {len(failed_indices)} failures...")
            
            if failed_indices:
                logger.info(f"Deep validation complete: {len(failed_indices)} failed samples out of {num_samples}")
        else:
            logger.info(f"Skipping deep validation due to {errors_before} errors found in metadata check")

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

    repo_id = results['repo_id']
    
    if not results["issues"]:
        print("\n✅ No issues found! Dataset is ready for training.")
    else:
        if errors:
            print(f"\n❌ ERRORS ({len(errors)}):")
            print("-" * 50)
            
            # Group errors by type
            error_types = {}
            for issue in errors:
                error_type = issue.get("type", "unknown")
                if error_type not in error_types:
                    error_types[error_type] = []
                error_types[error_type].append(issue)
            
            for error_type, type_errors in error_types.items():
                print(f"\n  [{error_type}] ({len(type_errors)} errors):")
                for issue in type_errors[:10]:  # Show first 10 of each type
                    if issue.get("episode_idx") is not None:
                        print(f"    • Sample {issue.get('sample_idx')}: Episode {issue.get('episode_idx')}, Frame {issue.get('frame_idx')}")
                        print(f"      {issue['message'][:100]}")
                    elif issue["type"] == "timestamp_exceeds_video_duration":
                        print(f"    • {issue['message']}")
                        print(f"      Video has {issue['num_frames']} frames at {issue['video_fps']:.1f} fps, dataset expects {issue['dataset_fps']} fps")
                    else:
                        print(f"    • {issue['message']}")
                if len(type_errors) > 10:
                    print(f"    ... and {len(type_errors) - 10} more {error_type} errors")
            
            # Print failed samples in CSV format for easy copy-paste
            failed_samples = [i for i in errors if i.get("sample_idx") is not None]
            if failed_samples:
                print("\n  FAILED SAMPLES (CSV format):")
                print("  dataset,episode,frame,sample_idx,error_type")
                for issue in failed_samples:
                    print(f"  {repo_id},{issue.get('episode_idx', 'N/A')},{issue.get('frame_idx', 'N/A')},{issue.get('sample_idx', 'N/A')},{issue.get('type', 'unknown')}")

        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            print("-" * 50)
            for issue in warnings[:10]:  # Show first 10 warnings
                print(f"  • {issue['message']}")
            if len(warnings) > 10:
                print(f"  ... and {len(warnings) - 10} more warnings")

    print("\n" + "=" * 70)


def write_errors_to_file(results: dict, output_file: str) -> None:
    """Write errors to a CSV file for easy analysis."""
    import csv
    
    repo_id = results['repo_id']
    errors = [i for i in results["issues"] if i.get("severity") == "error"]
    
    # Check if file exists to determine if we need to write header
    file_exists = Path(output_file).exists()
    
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow(['dataset', 'episode', 'frame', 'sample_idx', 'error_type', 'message'])
        
        for issue in errors:
            writer.writerow([
                repo_id,
                issue.get('episode_idx', 'N/A'),
                issue.get('frame_idx', 'N/A'),
                issue.get('sample_idx', 'N/A'),
                issue.get('type', 'unknown'),
                issue.get('message', '')[:200].replace('\n', ' '),  # Truncate and remove newlines
            ])


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
        default=1.0,
        help="Fraction of samples to check in deep validation (default: 1.0 = all samples)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to write errors CSV file (e.g., errors.csv)",
    )

    args = parser.parse_args()

    results = validate_dataset(
        repo_id=args.repo_id,
        root=args.root,
        quick=args.quick,
        sample_rate=args.sample_rate,
    )

    print_results(results)

    # Write errors to CSV file if specified
    errors = [i for i in results["issues"] if i.get("severity") == "error"]
    if args.output_file and errors:
        write_errors_to_file(results, args.output_file)
        print(f"\n📁 Errors written to: {args.output_file}")

    # Exit with error code if there are errors
    if errors:
        exit(1)


if __name__ == "__main__":
    main()
