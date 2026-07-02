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

A `hardware_reset()` drops the camera off the USB bus and it takes a few seconds
to re-enumerate. This script waits for the camera(s) to come back before exiting,
so it is safe to run `lerobot-record` immediately afterwards.

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
import time

from lerobot.utils.import_utils import require_package


def _query_serials(rs) -> dict[str, str]:
    """Returns a {serial_number: name} mapping of currently connected RealSense cameras."""
    context = rs.context()
    serials = {}
    for device in context.query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        serials[serial] = name
    return serials


def reset_realsense_cameras(
    serial_numbers: list[str] | None = None,
    wait_for_reconnect: bool = True,
    timeout_s: float = 30.0,
) -> int:
    """
    Issues a hardware reset to connected Intel RealSense cameras.

    Args:
        serial_numbers: Optional list of serial numbers to reset. If None or empty,
            all detected RealSense cameras are reset.
        wait_for_reconnect: If True, poll the USB bus until every reset camera has
            re-enumerated (or `timeout_s` elapses) before returning. This makes it
            safe to launch recording immediately afterwards.
        timeout_s: Maximum time in seconds to wait for cameras to re-enumerate.

    Returns:
        The number of cameras that were successfully reset.

    Raises:
        ImportError: If pyrealsense2 is not installed.
    """
    require_package("pyrealsense2", extra="intelrealsense")
    import pyrealsense2 as rs

    connected = _query_serials(rs)

    if not connected:
        print("No RealSense cameras were detected.")
        return 0

    requested = {str(sn) for sn in serial_numbers} if serial_numbers else set(connected)

    missing = requested - set(connected)
    for serial in missing:
        print(f"WARNING: requested serial '{serial}' was not found among connected cameras.")

    targets = requested & set(connected)
    if not targets:
        print("No matching cameras to reset.")
        return 0

    reset_serials = set()
    # Re-query each loop: device handles are invalidated as soon as a reset is issued.
    for serial in list(targets):
        try:
            context = rs.context()
            for device in context.query_devices():
                if device.get_info(rs.camera_info.serial_number) != serial:
                    continue
                name = device.get_info(rs.camera_info.name)
                print(f"Resetting {name} (serial: {serial})...")
                device.hardware_reset()
                reset_serials.add(serial)
                break
        except Exception as e:
            print(f"ERROR: failed to reset serial '{serial}': {e}")

    if not reset_serials:
        return 0

    if wait_for_reconnect:
        print(f"Waiting up to {timeout_s:.0f}s for {len(reset_serials)} camera(s) to re-enumerate...")
        # Give the kernel a moment to actually drop the devices before we start polling.
        time.sleep(2.0)
        deadline = time.time() + timeout_s
        pending = set(reset_serials)
        while pending and time.time() < deadline:
            available = set(_query_serials(rs))
            recovered = pending & available
            for serial in recovered:
                print(f"  Camera {serial} is back.")
            pending -= recovered
            if pending:
                time.sleep(0.5)

        if pending:
            print(
                f"WARNING: {len(pending)} camera(s) did not reappear within {timeout_s:.0f}s: "
                f"{sorted(pending)}. They may need a few more seconds or a physical replug."
            )
        else:
            print("All reset cameras are back online.")

    print(f"Done. Reset {len(reset_serials)} camera(s).")
    return len(reset_serials)


def list_realsense_cameras() -> None:
    """Prints the serial number and name of every connected RealSense camera."""
    require_package("pyrealsense2", extra="intelrealsense")
    import pyrealsense2 as rs

    connected = _query_serials(rs)

    if not connected:
        print("No RealSense cameras were detected.")
        return

    print("\n--- Connected RealSense Cameras ---")
    for i, (serial, name) in enumerate(connected.items()):
        print(f"Camera #{i}: {name} (serial: {serial})")
    print("-" * 35)


def main():
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
        "--no-wait",
        action="store_true",
        help="Do not wait for cameras to re-enumerate after resetting.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=30.0,
        help="Max time to wait for cameras to re-enumerate after reset. Default: 30 seconds.",
    )
    args = parser.parse_args()

    if args.list:
        list_realsense_cameras()
        return

    reset_realsense_cameras(
        serial_numbers=args.serial_numbers,
        wait_for_reconnect=not args.no_wait,
        timeout_s=args.timeout_s,
    )


if __name__ == "__main__":
    main()
