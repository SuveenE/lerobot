#!/usr/bin/env python

import argparse
import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

"""
example test commands:
# Single Orbbec device
python test_depth_camera.py --type orbbec --devices 0 --width 640 --height 480 --fps 30

# Multiple Orbbec devices
python test_depth_camera.py --type orbbec --devices 0 1 --width 640 --height 480 --fps 30

# Single RealSense device
python test_depth_camera.py --type intelrealsense --devices 123456789 --width 640 --height 480 --fps 30

# Multiple RealSense devices
python test_depth_camera.py --type intelrealsense --devices 123456789 987654321 --width 640 --height 480 --fps 30
"""

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')


def _print_frame_info(title: str, frame: np.ndarray) -> None:
    print(f"{title}: type={type(frame).__name__}, dtype={frame.dtype}, shape={frame.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test depth camera and print RGB/depth frame info. Supports multiple devices.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "\nExamples:\n"
            "  # Single RealSense\n"
            "  python examples/test_depth_camera.py --type intelrealsense --devices 123456789 \\\n"
            "      --width 640 --height 480 --fps 30\n\n"
            "  # Multiple RealSense devices\n"
            "  python examples/test_depth_camera.py --type intelrealsense --devices 123456789 987654321 \\\n"
            "      --width 640 --height 480 --fps 30\n\n"
            "  # Single Orbbec (by index)\n"
            "  python examples/test_depth_camera.py --type orbbec --devices 0 \\\n"
            "      --width 640 --height 480 --fps 30\n\n"
            "  # Multiple Orbbec devices (by index)\n"
            "  python examples/test_depth_camera.py --type orbbec --devices 0 1 \\\n"
            "      --width 640 --height 480 --fps 30\n"
        ),
    )
    parser.add_argument("--type", required=True, choices=["intelrealsense", "orbbec"], help="Camera type")
    parser.add_argument("--devices", required=True, nargs="+", help="Device identifier(s): serial number (RealSense) or index (Orbbec). Can specify multiple devices.")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--timeout_ms", type=int, default=500, help="Wait timeout for async read (ms)")
    parser.add_argument("--output_dir", type=str, default="outputs/test_depth", help="Output directory for saved images")
    return parser.parse_args()


def test_single_device(camera_type: str, device_id: str, args: argparse.Namespace, device_index: int) -> bool:
    """Test a single camera device and save its output.
    
    Args:
        camera_type: "intelrealsense" or "orbbec"
        device_id: Device identifier (serial number for RealSense, index for Orbbec)
        args: Command line arguments
        device_index: Index for naming output files when testing multiple devices
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing device {device_index + 1}: {device_id} ({camera_type})")
    print(f"{'='*60}")
    
    try:
        if camera_type == "intelrealsense":
            from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
            from lerobot.cameras.realsense.camera_realsense import RealSenseCamera

            cfg = RealSenseCameraConfig(
                serial_number_or_name=device_id,
                fps=args.fps,
                width=args.width,
                height=args.height,
                use_depth=True,
            )
            cam = RealSenseCamera(cfg)
        elif camera_type == "orbbec":
            from lerobot.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig
            from lerobot.cameras.orbbec.camera_orbbec import OrbbecCamera

            # For Orbbec, convert device_id to int if it's numeric (index)
            try:
                index_or_path = int(device_id)
            except ValueError:
                # It's a path or serial number string
                index_or_path = device_id

            cfg = OrbbecCameraConfig(
                index_or_path=index_or_path,
                fps=args.fps,
                width=args.width,
                height=args.height,
                use_depth=True,
            )
            cam = OrbbecCamera(cfg)
        else:
            print(f"Unsupported type: {camera_type}")
            return False

        print(f"Connecting to {cam} ...")
        cam.connect(warmup=True)
        print("Connected.")

        rgb, depth = cam.async_read_depth(timeout_ms=args.timeout_ms)
        _print_frame_info("RGB", rgb)
        _print_frame_info("Depth", depth)

        # Basic sanity checks
        if depth.dtype != np.uint16:
            print(f"Warning: depth dtype is {depth.dtype}, expected uint16")

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            print(f"Warning: unexpected RGB shape {rgb.shape}, expected (H, W, 3)")

        # Save images with device-specific names
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a safe filename from device_id
        safe_device_id = str(device_id).replace("/", "_").replace(":", "_")
        
        # Save RGB (convert to BGR for OpenCV)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        rgb_path = output_dir / f"{camera_type}_{safe_device_id}_rgb.png"
        cv2.imwrite(str(rgb_path), rgb_bgr)
        print(f"Saved RGB image to: {rgb_path}")
        
        # Save depth visualization using matplotlib grayscale colormap
        depth_path = output_dir / f"{camera_type}_{safe_device_id}_depth.png"
        plt.imsave(depth_path, depth, cmap='Greys_r')
        print(f"Saved depth image to: {depth_path}")
        
        # Also save raw depth values as 16-bit PNG for precise data
        depth_raw_path = output_dir / f"{camera_type}_{safe_device_id}_depth_raw.png"
        cv2.imwrite(str(depth_raw_path), depth)
        print(f"Saved raw depth (uint16) to: {depth_raw_path}")

        cam.disconnect()
        print("Disconnected.")
        print(f"✓ Device {device_id} tested successfully")
        return True

    except Exception as e:
        print(f"✗ Error testing device {device_id}: {e}")
        import traceback
        traceback.print_exc()
        try:
            cam.disconnect()
        except Exception:
            pass
        return False


def test_multiple_cameras(camera_type: str, device_ids: list[str], args: argparse.Namespace) -> int:
    """Test multiple cameras by connecting to all, then capturing from all in parallel.
    
    This simulates real-world multi-camera robotics usage where all cameras need to be
    active simultaneously.
    
    Args:
        camera_type: "intelrealsense" or "orbbec"
        device_ids: List of device identifiers
        args: Command line arguments
        
    Returns:
        0 if all succeed, 1 if any fail
    """
    print(f"\n{'='*60}")
    print(f"Testing {len(device_ids)} {camera_type} camera(s) in parallel")
    print(f"Devices: {', '.join(device_ids)}")
    print(f"{'='*60}")
    
    cameras = []
    results = []
    
    try:
        # Step 1: Connect to all cameras
        print(f"\n[1/3] Connecting to {len(device_ids)} camera(s)...")
        for idx, device_id in enumerate(device_ids):
            print(f"\n  Device {idx + 1}/{len(device_ids)}: {device_id}")
            try:
                if camera_type == "intelrealsense":
                    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
                    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera

                    cfg = RealSenseCameraConfig(
                        serial_number_or_name=device_id,
                        fps=args.fps,
                        width=args.width,
                        height=args.height,
                        use_depth=True,
                    )
                    print(f"RealSenseCameraConfig: {cfg}")
                    cam = RealSenseCamera(cfg)
                elif camera_type == "orbbec":
                    from lerobot.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig
                    from lerobot.cameras.orbbec.camera_orbbec import OrbbecCamera

                    # For Orbbec, convert device_id to int if it's numeric (index)
                    try:
                        index_or_path = int(device_id)
                    except ValueError:
                        index_or_path = device_id

                    cfg = OrbbecCameraConfig(
                        index_or_path=index_or_path,
                        fps=args.fps,
                        width=args.width,
                        height=args.height,
                        use_depth=True,
                    )
                    cam = OrbbecCamera(cfg)
                else:
                    print(f"  ✗ Unsupported camera type: {camera_type}")
                    return 1
                
                print(f"  Connecting to {cam}...")
                cam.connect(warmup=True)
                print(f"  ✓ Connected to {device_id}")
                cameras.append((device_id, cam, idx))
                
            except Exception as e:
                print(f"  ✗ Failed to connect to {device_id}: {e}")
                import traceback
                traceback.print_exc()
                results.append((device_id, False))
        
        if not cameras:
            print("\n✗ No cameras connected successfully")
            return 1
        
        print(f"\n✓ Successfully connected to {len(cameras)}/{len(device_ids)} camera(s)")
        
        # Step 2: Capture frames from all cameras (in parallel/quick succession)
        print(f"\n[2/3] Capturing frames from {len(cameras)} camera(s)...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for device_id, cam, idx in cameras:
            print(f"\n  Device {idx + 1}: {device_id}")
            try:
                rgb, depth = cam.async_read_depth(timeout_ms=args.timeout_ms)
                
                _print_frame_info(f"    RGB", rgb)
                _print_frame_info(f"    Depth", depth)
                
                # Basic sanity checks
                if depth.dtype != np.uint16:
                    print(f"    Warning: depth dtype is {depth.dtype}, expected uint16")
                
                if rgb.ndim != 3 or rgb.shape[2] != 3:
                    print(f"    Warning: unexpected RGB shape {rgb.shape}, expected (H, W, 3)")
                
                # Save images
                safe_device_id = str(device_id).replace("/", "_").replace(":", "_")
                
                # Save RGB (convert to BGR for OpenCV)
                rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                rgb_path = output_dir / f"{camera_type}_{safe_device_id}_rgb.png"
                cv2.imwrite(str(rgb_path), rgb_bgr)
                print(f"    Saved RGB: {rgb_path}")
                
                # Save depth visualization using matplotlib grayscale colormap
                depth_path = output_dir / f"{camera_type}_{safe_device_id}_depth.png"
                processed_depth = cv2.normalize(depth, None, 0, 255, norm_type=cv2.NORM_MINMAX)
                plt.imsave(depth_path, processed_depth, cmap='Greys_r')
                print(f"    Saved depth: {depth_path}")
                
                # Save raw depth
                depth_raw_path = output_dir / f"{camera_type}_{safe_device_id}_depth_raw.png"
                cv2.imwrite(str(depth_raw_path), depth)
                print(f"    Saved raw depth: {depth_raw_path}")
                
                print(f"  ✓ Captured from {device_id}")
                results.append((device_id, True))
                
            except Exception as e:
                print(f"  ✗ Error capturing from {device_id}: {e}")
                import traceback
                traceback.print_exc()
                results.append((device_id, False))
        
        # Step 3: Disconnect all cameras
        print(f"\n[3/3] Disconnecting {len(cameras)} camera(s)...")
        for device_id, cam, idx in cameras:
            try:
                cam.disconnect()
                print(f"  ✓ Disconnected {device_id}")
            except Exception as e:
                print(f"  ! Error disconnecting {device_id}: {e}")
        
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup
        print("\nAttempting cleanup...")
        for device_id, cam, _ in cameras:
            try:
                cam.disconnect()
                print(f"  Cleaned up {device_id}")
            except Exception:
                pass
        
        return 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    successful = sum(1 for _, success in results if success)
    total = len(results)
    print(f"Tested {total} device(s): {successful} successful, {total - successful} failed")
    
    for device_id, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {device_id}")
    
    return 0 if all(success for _, success in results) else 1


def main() -> int:
    args = parse_args()

    # For multi-camera robotics use case: connect to all, capture from all, then disconnect all
    if len(args.devices) > 1:
        return test_multiple_cameras(args.type, args.devices, args)
    else:
        # Single device: use simpler sequential test
        success = test_single_device(args.type, args.devices[0], args, 0)
        return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())


