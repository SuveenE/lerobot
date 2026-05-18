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
Helper to read camera calibration from Intel RealSense devices.

Examples:

```shell
# Print color intrinsics for all RealSense cameras using the default 640x360 @ 30 FPS profile.
lerobot-get-camera-calibration

# Print calibration for one camera at a specific color resolution.
lerobot-get-camera-calibration 0123456789 --width 1280 --height 720

# Include depth intrinsics/extrinsics and save the JSON payload.
lerobot-get-camera-calibration --include-depth --output-file outputs/realsense_calibration.json
```
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_IMAGE_SIZE = (640, 360)
DEFAULT_FPS = 30


def positive_int(value: str) -> int:
    """Parse an argparse integer that must be positive."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value}.")
    return parsed


def import_pyrealsense2() -> Any:
    """Import pyrealsense2 with a helpful install message."""
    try:
        import pyrealsense2 as rs  # type: ignore  # TODO: add type stubs for pyrealsense2
    except Exception as e:
        raise ImportError(
            "pyrealsense2 is required to read RealSense calibration. "
            "Install it with `pip install 'lerobot[intelrealsense]'`."
        ) from e

    return rs


def enum_name(value: Any) -> str:
    """Return a stable string for pyrealsense2 enum values."""
    return str(getattr(value, "name", value))


def get_device_info(rs: Any, device: Any) -> dict[str, Any]:
    """Extract user-facing metadata from a RealSense device."""
    fields = {
        "name": rs.camera_info.name,
        "id": rs.camera_info.serial_number,
        "firmware_version": rs.camera_info.firmware_version,
        "usb_type_descriptor": rs.camera_info.usb_type_descriptor,
        "physical_port": rs.camera_info.physical_port,
        "product_id": rs.camera_info.product_id,
        "product_line": rs.camera_info.product_line,
    }

    info: dict[str, Any] = {"type": "RealSense"}
    for key, field in fields.items():
        if device.supports(field):
            info[key] = device.get_info(field)

    return info


def find_realsense_cameras(rs: Any) -> list[dict[str, Any]]:
    """Find all connected RealSense cameras."""
    context = rs.context()
    devices = context.query_devices()
    return [get_device_info(rs, device) for device in devices]


def select_realsense_cameras(
    cameras: list[dict[str, Any]], camera_identifier: str | None
) -> list[dict[str, Any]]:
    """Select all cameras or one camera by serial number or unique name."""
    if camera_identifier is None:
        return cameras

    matches = [
        camera
        for camera in cameras
        if camera_identifier in {str(camera.get("id")), str(camera.get("name"))}
    ]

    if not matches:
        available = [f"{camera.get('id')} ({camera.get('name', 'unknown')})" for camera in cameras]
        raise ValueError(
            f"No RealSense camera found for '{camera_identifier}'. Available cameras: {available}"
        )

    if len(matches) > 1:
        serial_numbers = [camera.get("id") for camera in matches]
        raise ValueError(
            f"Multiple RealSense cameras matched '{camera_identifier}'. "
            f"Use a serial number instead. Matching serial numbers: {serial_numbers}"
        )

    return matches


def intrinsics_to_dict(intrinsics: Any) -> dict[str, Any]:
    """Convert pyrealsense2 intrinsics to JSON-serializable values."""
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "model": enum_name(intrinsics.model),
        "coeffs": [float(coeff) for coeff in intrinsics.coeffs],
    }


def video_stream_profile_to_dict(profile: Any) -> dict[str, Any]:
    """Return stream metadata and intrinsics for a RealSense video stream."""
    video_profile = profile.as_video_stream_profile()
    return {
        "stream_type": video_profile.stream_name(),
        "format": enum_name(video_profile.format()),
        "fps": video_profile.fps(),
        "intrinsics": intrinsics_to_dict(video_profile.get_intrinsics()),
    }


def extrinsics_to_dict(extrinsics: Any) -> dict[str, list[float]]:
    """Convert pyrealsense2 extrinsics to JSON-serializable values."""
    return {
        "rotation": [float(value) for value in extrinsics.rotation],
        "translation": [float(value) for value in extrinsics.translation],
    }


def get_depth_scale(profile: Any) -> float | None:
    """Return the active depth sensor scale in meters per depth unit, if available."""
    try:
        return float(profile.get_device().first_depth_sensor().get_depth_scale())
    except RuntimeError:
        return None


def get_camera_calibration(
    rs: Any,
    camera: dict[str, Any],
    width: int,
    height: int,
    fps: int,
    include_depth: bool,
) -> dict[str, Any]:
    """Start a RealSense pipeline and read calibration for its active stream profile."""
    serial_number = str(camera["id"])
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

    if include_depth:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    logger.info(f"Reading calibration from RealSense camera {serial_number} at {width}x{height} @ {fps} FPS")
    pipeline_started = False

    try:
        profile = pipeline.start(config)
        pipeline_started = True

        color_profile = profile.get_stream(rs.stream.color)
        calibration: dict[str, Any] = {
            "camera": camera,
            "requested_profile": {
                "width": width,
                "height": height,
                "fps": fps,
                "color_format": "rgb8",
                "include_depth": include_depth,
            },
            "color": video_stream_profile_to_dict(color_profile),
        }

        if include_depth:
            depth_profile = profile.get_stream(rs.stream.depth)
            calibration["depth"] = video_stream_profile_to_dict(depth_profile)
            depth_scale = get_depth_scale(profile)
            if depth_scale is not None:
                calibration["depth"]["scale_m_per_unit"] = depth_scale
            calibration["extrinsics"] = {
                "color_to_depth": extrinsics_to_dict(color_profile.get_extrinsics_to(depth_profile)),
                "depth_to_color": extrinsics_to_dict(depth_profile.get_extrinsics_to(color_profile)),
            }

        return calibration
    finally:
        if pipeline_started:
            pipeline.stop()


def get_camera_calibrations(
    camera_identifier: str | None,
    width: int,
    height: int,
    fps: int,
    include_depth: bool,
) -> list[dict[str, Any]]:
    """Read calibration from selected RealSense cameras."""
    rs = import_pyrealsense2()
    cameras = find_realsense_cameras(rs)

    if not cameras:
        logger.warning("No RealSense cameras were detected.")
        return []

    selected_cameras = select_realsense_cameras(cameras, camera_identifier)
    return [
        get_camera_calibration(rs, camera, width, height, fps, include_depth)
        for camera in selected_cameras
    ]


def save_or_print_calibrations(calibrations: list[dict[str, Any]], output_file: Path | None) -> None:
    """Write calibration JSON to a file and always print it to stdout."""
    payload = {"cameras": calibrations}
    json_payload = json.dumps(payload, indent=2)

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(f"{json_payload}\n")
        logger.info(f"Saved calibration to {output_file}")

    print(json_payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read factory calibration from Intel RealSense cameras.")
    parser.add_argument(
        "camera_identifier",
        type=str,
        nargs="?",
        default=None,
        help="Optional RealSense serial number or unique camera name. Reads all RealSense cameras if omitted.",
    )
    parser.add_argument(
        "--width",
        type=positive_int,
        default=DEFAULT_IMAGE_SIZE[0],
        help=f"Requested color stream width. Default: {DEFAULT_IMAGE_SIZE[0]}.",
    )
    parser.add_argument(
        "--height",
        type=positive_int,
        default=DEFAULT_IMAGE_SIZE[1],
        help=f"Requested color stream height. Default: {DEFAULT_IMAGE_SIZE[1]}.",
    )
    parser.add_argument(
        "--fps",
        type=positive_int,
        default=DEFAULT_FPS,
        help=f"Requested stream FPS. Default: {DEFAULT_FPS}.",
    )
    parser.add_argument(
        "--include-depth",
        action="store_true",
        help="Also enable depth and report depth intrinsics plus color/depth extrinsics.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional JSON file to write calibration results to.",
    )
    args = parser.parse_args()

    try:
        calibrations = get_camera_calibrations(
            camera_identifier=args.camera_identifier,
            width=args.width,
            height=args.height,
            fps=args.fps,
            include_depth=args.include_depth,
        )
    except (ImportError, RuntimeError, ValueError) as e:
        logger.error(e)
        raise SystemExit(1) from e

    save_or_print_calibrations(calibrations, args.output_file)


if __name__ == "__main__":
    main()
