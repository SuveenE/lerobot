#!/usr/bin/env python

"""Find RealSense cameras and print their serial numbers/names for use with lerobot."""

try:
    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
except ImportError as e:
    print(f"Error importing RealSense camera module: {e}")
    print("Make sure pyrealsense2 is installed: pip install pyrealsense2")
    exit(1)


def find_realsense_cameras():
    """Find and display all connected RealSense cameras."""
    try:
        cameras = RealSenseCamera.find_cameras()
    except Exception as e:
        print(f"Error finding RealSense cameras: {e}")
        print("\nNote: On macOS, you may need to run with sudo permissions.")
        return

    if len(cameras) == 0:
        print("No RealSense devices detected.")
        return

    print(f"Found {len(cameras)} RealSense device(s):\n")
    
    for i, cam_info in enumerate(cameras):
        serial = cam_info.get("id", "")
        name = cam_info.get("name", "Unknown")
        firmware = cam_info.get("firmware_version", "Unknown")
        product_line = cam_info.get("product_line", "Unknown")
        
        print(f"Device {i}:")
        print(f"  Name:           '{name}'")
        print(f"  Serial Number:  '{serial}'")
        print(f"  Product Line:   {product_line}")
        print(f"  Firmware:       {firmware}")
        
        # Display default stream profile if available
        if "default_stream_profile" in cam_info:
            profile = cam_info["default_stream_profile"]
            print(f"  Default Profile:")
            print(f"    Stream Type: {profile.get('stream_type', 'N/A')}")
            print(f"    Format:      {profile.get('format', 'N/A')}")
            print(f"    Resolution:  {profile.get('width', 'N/A')}x{profile.get('height', 'N/A')}")
            print(f"    FPS:         {profile.get('fps', 'N/A')}")
        
        print(f"\nTo use this camera, run:")
        print(f"  python examples/test_depth_camera.py --type intelrealsense --devices {serial} \\")
        print(f"      --width 640 --height 480 --fps 30")
        print()


if __name__ == "__main__":
    find_realsense_cameras()
