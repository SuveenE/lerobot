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
Hardware-reset Intel RealSense cameras.

RealSense devices occasionally get stuck in a bad state (frame timeouts, "failed
to set power state", "device or resource busy", or a hang during connect()).
Power-cycling the USB device via `hardware_reset()` usually recovers them without
physically unplugging the camera.

Examples:

Reset every connected RealSense camera:

```shell
lerobot-reset-cameras
```

Reset only specific cameras by serial number:

```shell
lerobot-reset-cameras --serial-number 0123456789 --serial-number 9876543210
```

List connected cameras without resetting:

```shell
lerobot-reset-cameras --list
```
"""

import argparse
import logging
import time

from lerobot.utils.import_utils import require_package

logger = logging.getLogger(__name__)


def reset_realsense_cameras(serial_numbers: list[str] | None = None, settle_time_s: float = 2.0) -> int:
    """
    Issues a hardware reset to connected Intel RealSense cameras.

    Args:
        serial_numbers: Optional list of serial numbers to reset. If None or empty,
            all detected RealSense cameras are reset.
        settle_time_s: Time in seconds to wait after issuing resets so the devices
            can re-enumerate on the USB bus before the script exits.

    Returns:
        The number of cameras that were successfully reset.

    Raises:
        ImportError: If pyrealsense2 is not installed.
    """
    require_package("pyrealsense2", extra="intelrealsense")
    import pyrealsense2 as rs

    context = rs.context()
    devices = context.query_devices()

    if len(devices) == 0:
        logger.warning("No RealSense cameras were detected.")
        return 0

    requested = {str(sn) for sn in serial_numbers} if serial_numbers else None
    reset_count = 0

    for device in devices:
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)

        if requested is not None and serial not in requested:
            continue

        try:
            logger.info(f"Resetting {name} (serial: {serial})...")
            device.hardware_reset()
            reset_count += 1
        except Exception as e:
            logger.error(f"Failed to reset {name} (serial: {serial}): {e}")

    if requested is not None:
        detected = {device.get_info(rs.camera_info.serial_number) for device in devices}
        for serial in requested - detected:
            logger.warning(f"Requested serial '{serial}' was not found among connected cameras.")

    if reset_count > 0:
        logger.info(
            f"Issued reset to {reset_count} camera(s). Waiting {settle_time_s:.1f}s for re-enumeration..."
        )
        time.sleep(settle_time_s)
        logger.info("Done. Cameras should be available again shortly.")

    return reset_count


def list_realsense_cameras() -> None:
    """Prints the serial number and name of every connected RealSense camera."""
    require_package("pyrealsense2", extra="intelrealsense")
    import pyrealsense2 as rs

    context = rs.context()
    devices = context.query_devices()

    if len(devices) == 0:
        logger.warning("No RealSense cameras were detected.")
        return

    print("\n--- Connected RealSense Cameras ---")
    for i, device in enumerate(devices):
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        print(f"Camera #{i}: {name} (serial: {serial})")
    print("-" * 35)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Hardware-reset connected Intel RealSense cameras.")
    parser.add_argument(
        "--serial-number",
        action="append",
        dest="serial_numbers",
        default=None,
        help="Serial number of a camera to reset. Can be passed multiple times. "
        "If omitted, all connected RealSense cameras are reset.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List connected RealSense cameras and exit without resetting.",
    )
    parser.add_argument(
        "--settle-time-s",
        type=float,
        default=2.0,
        help="Time to wait after resetting for cameras to re-enumerate. Default: 2 seconds.",
    )
    args = parser.parse_args()

    if args.list:
        list_realsense_cameras()
        return

    reset_realsense_cameras(serial_numbers=args.serial_numbers, settle_time_s=args.settle_time_s)


if __name__ == "__main__":
    main()
