#!/usr/bin/env python

import argparse
import sys
import time

import numpy as np


def _print_frame_info(title: str, frame: np.ndarray) -> None:
    print(f"{title}: type={type(frame).__name__}, dtype={frame.dtype}, shape={frame.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test depth camera and print RGB/depth frame info",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "\nExamples:\n"
            "  # RealSense\n"
            "  python examples/test_depth_camera.py --type intelrealsense --serial_number_or_name 123456789 \\\n"
            "      --width 640 --height 480 --fps 30\n\n"
            "  # Orbbec\n"
            "  python examples/test_depth_camera.py --type orbbec --serial_number_or_name 123456789 \\\n"
            "      --width 640 --height 480 --fps 30\n"
        ),
    )
    parser.add_argument("--type", required=True, choices=["intelrealsense", "orbbec"], help="Camera type")
    parser.add_argument("--serial_number_or_name", required=True, help="Serial number or device name")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--timeout_ms", type=int, default=500, help="Wait timeout for async read (ms)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.type == "intelrealsense":
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
        from lerobot.cameras.realsense.camera_realsense import RealSenseCamera

        cfg = RealSenseCameraConfig(
            serial_number_or_name=args.serial_number_or_name,
            fps=args.fps,
            width=args.width,
            height=args.height,
            use_depth=True,
        )
        cam = RealSenseCamera(cfg)
    elif args.type == "orbbec":
        from lerobot.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig
        from lerobot.cameras.orbbec.camera_orbbec import OrbbecCamera

        cfg = OrbbecCameraConfig(
            serial_number_or_name=args.serial_number_or_name,
            fps=args.fps,
            width=args.width,
            height=args.height,
            use_depth=True,
        )
        cam = OrbbecCamera(cfg)
    else:
        print(f"Unsupported type: {args.type}")
        return 2

    try:
        print(f"Connecting to {cam} ...")
        cam.connect(warmup=True)
        print("Connected.")

        # Give the background thread a moment on some backends
        time.sleep(0.1)

        rgb, depth = cam.async_read_depth(timeout_ms=args.timeout_ms)
        _print_frame_info("RGB", rgb)
        _print_frame_info("Depth", depth)

        # Basic sanity
        if depth.dtype != np.uint16:
            print(f"Warning: depth dtype is {depth.dtype}, expected uint16")

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            print(f"Warning: unexpected RGB shape {rgb.shape}, expected (H, W, 3)")

    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        try:
            cam.disconnect()
            print("Disconnected.")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


