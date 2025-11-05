#!/usr/bin/env python
"""
Script to find and test all cameras (1 Orbbec + 2 RealSense).
Captures RGB and depth frames from all cameras and saves them to disk.
"""

import pyorbbecsdk as ob
import pyrealsense2 as rs
import numpy as np
import cv2
import os
from typing import Optional, List, Tuple

# Configuration parameters
ORBBEC_DEVICE_INDEX = 0  # Orbbec is always index 0
WIDTH = 640
HEIGHT = 480
FPS = 30
NUM_FRAMES = 1  # Number of frames to capture from each camera


def find_realsense_cameras() -> List[str]:
    """Find all RealSense cameras and return their serial numbers."""
    context = rs.context()
    devices = context.query_devices()
    
    if len(devices) == 0:
        print("No RealSense devices found")
        return []
    
    serial_numbers = []
    print(f"\nFound {len(devices)} RealSense device(s):")
    for i, device in enumerate(devices):
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        serial_numbers.append(serial)
        print(f"  RealSense {i}: {name} (Serial: {serial})")
    
    return serial_numbers


def find_orbbec_camera() -> Optional[ob.Device]:
    """Find Orbbec camera at index 0."""
    context = ob.Context()
    device_list = context.query_devices()
    
    if device_list.get_count() == 0:
        print("No Orbbec devices found")
        return None
    
    if ORBBEC_DEVICE_INDEX >= device_list.get_count():
        print(f"Orbbec device index {ORBBEC_DEVICE_INDEX} not available (found {device_list.get_count()} device(s))")
        return None
    
    device = device_list.get_device_by_index(ORBBEC_DEVICE_INDEX)
    
    try:
        info = device.get_device_info()
        name = info.name() if hasattr(info, "name") else "Orbbec"
        serial = info.serial_number() if hasattr(info, "serial_number") else ""
        print(f"\nFound Orbbec camera: {name} (Serial: {serial})")
    except Exception:
        print(f"\nFound Orbbec camera at index {ORBBEC_DEVICE_INDEX}")
    
    return device


def frame_to_rgb_image(frame: ob.VideoFrame) -> np.ndarray:
    """Convert Orbbec frame to RGB format."""
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    
    if color_format == ob.OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        return image
    elif color_format == ob.OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == ob.OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2RGB_YUYV)
    elif color_format == ob.OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == ob.OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2RGB_UYVY)
    else:
        print(f"Warning: Unsupported color format: {color_format}, attempting default conversion")
        return np.resize(data, (height, width, 3))


def capture_orbbec_frames(device: ob.Device, output_dir: str, num_frames: int = NUM_FRAMES):
    """Capture RGB and depth frames from Orbbec camera."""
    print(f"\n=== Capturing from Orbbec camera ===")
    
    # Create pipeline
    pipeline = ob.Pipeline(device)
    
    # Get stream profile lists
    color_profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
    depth_profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
    
    if color_profile_list is None or color_profile_list.get_count() == 0:
        print("Error: No color stream profiles available for Orbbec")
        return
    
    if depth_profile_list is None or depth_profile_list.get_count() == 0:
        print("Error: No depth stream profiles available for Orbbec")
        return
    
    # Try to get specific profiles matching requested parameters
    color_profile = None
    depth_profile = None
    
    try:
        color_profile = color_profile_list.get_video_stream_profile(
            WIDTH, HEIGHT, ob.OBFormat.MJPG, FPS
        )
        print(f"Using MJPG color profile: {WIDTH}x{HEIGHT}@{FPS}fps")
    except Exception:
        try:
            color_profile = color_profile_list.get_video_stream_profile(
                WIDTH, 0, ob.OBFormat.ANY, FPS
            )
            print(f"Using ANY format color profile: {WIDTH}x*@{FPS}fps")
        except Exception:
            color_profile = color_profile_list.get_default_video_stream_profile()
            print("Using default color profile")
    
    try:
        depth_profile = depth_profile_list.get_video_stream_profile(
            WIDTH, HEIGHT, ob.OBFormat.Y12, FPS
        )
        print(f"Using Y12 depth profile: {WIDTH}x{HEIGHT}@{FPS}fps")
    except Exception:
        try:
            depth_profile = depth_profile_list.get_video_stream_profile(
                WIDTH, HEIGHT, ob.OBFormat.Y11, FPS
            )
            print(f"Using Y11 depth profile: {WIDTH}x{HEIGHT}@{FPS}fps")
        except Exception:
            depth_profile = depth_profile_list.get_default_video_stream_profile()
            print("Using default depth profile")
    
    # Configure streams
    config = ob.Config()
    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)
    
    # Start pipeline
    pipeline.start(config)
    print("Orbbec pipeline started successfully!")
    
    # Warmup
    print("Warming up (30 frames)...")
    for i in range(30):
        frameset = pipeline.wait_for_frames(1000)
        if frameset is None:
            print(f"Warning: No frameset received during warmup frame {i}")
    
    print("Warmup complete. Capturing frames...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Capture frames
    for frame_num in range(num_frames):
        frameset = pipeline.wait_for_frames(1000)
        if frameset is None:
            print(f"  Frame {frame_num}: No frameset received, skipping...")
            continue
        
        # Process color frame
        color_frame = frameset.get_color_frame()
        if color_frame is not None:
            rgb_image = frame_to_rgb_image(color_frame)
            rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            rgb_filename = f"{output_dir}/orbbec-rgb-{frame_num}.png"
            cv2.imwrite(rgb_filename, rgb_bgr)
            print(f"  Frame {frame_num}: Saved RGB: {rgb_filename}")
        
        # Process depth frame
        depth_frame = frameset.get_depth_frame()
        if depth_frame is not None:
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            depth_scale = depth_frame.get_depth_scale()
            
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
            depth_data_mm = depth_data.astype(np.float32) * depth_scale
            depth_image = depth_data_mm.astype(np.uint16)
            
            # Save raw depth
            raw_depth_filename = f"{output_dir}/orbbec-depth-raw-{frame_num}.png"
            cv2.imwrite(raw_depth_filename, depth_image)
            
            # Save colorized depth
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            colorized_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            colorized_filename = f"{output_dir}/orbbec-depth-colorized-{frame_num}.png"
            cv2.imwrite(colorized_filename, colorized_depth)
            
            # Save RGB-encoded depth
            if depth_image.dtype == np.uint16:
                high_byte = (depth_image >> 8).astype(np.uint8)
                low_byte = (depth_image & 0xFF).astype(np.uint8)
                zero_channel = np.zeros_like(high_byte, dtype=np.uint8)
                rgb_encoded_depth = np.stack([high_byte, low_byte, zero_channel], axis=-1)
                custom_rgb_filename = f"{output_dir}/orbbec-depth-rgb-encoded-{frame_num}.png"
                cv2.imwrite(custom_rgb_filename, rgb_encoded_depth)
            
            print(f"  Frame {frame_num}: Saved depth files")
    
    # Stop pipeline
    pipeline.stop()
    print("Orbbec pipeline stopped.")


def capture_realsense_frames(serial_number: str, output_dir: str, num_frames: int = NUM_FRAMES):
    """Capture RGB and depth frames from RealSense camera."""
    print(f"\n=== Capturing from RealSense camera (Serial: {serial_number}) ===")
    
    # Configure streams
    config = rs.config()
    config.enable_device(serial_number)
    
    COLOR_FORMAT = rs.format.rgb8
    DEPTH_FORMAT = rs.format.z16
    
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, COLOR_FORMAT, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, DEPTH_FORMAT, FPS)
    
    # Start pipeline
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    print(f"RealSense pipeline started successfully!")
    
    # Get actual stream profiles
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    print(f"  Color: {color_stream.width()}x{color_stream.height()}@{color_stream.fps()}fps")
    print(f"  Depth: {depth_stream.width()}x{depth_stream.height()}@{depth_stream.fps()}fps")
    
    # Create colorizer for depth visualization
    colorizer = rs.colorizer()
    
    # Warmup
    print("Warming up (30 frames)...")
    for i in range(30):
        pipeline.wait_for_frames()
    
    print("Warmup complete. Capturing frames...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Capture frames
    for frame_num in range(num_frames):
        frameset = pipeline.wait_for_frames()
        
        # Process color frame
        color_frame = frameset.get_color_frame()
        if color_frame:
            width = color_frame.get_width()
            height = color_frame.get_height()
            frame_format = color_frame.get_profile().format()
            frame_data = np.asanyarray(color_frame.get_data())
            
            if frame_format == rs.format.rgb8:
                rgb_image = np.resize(frame_data, (height, width, 3))
            elif frame_format == rs.format.yuyv:
                yuyv_image = np.resize(frame_data, (height, width, 2))
                rgb_image = cv2.cvtColor(yuyv_image, cv2.COLOR_YUV2RGB_YUYV)
            else:
                rgb_image = np.resize(frame_data, (height, width, 3))
            
            rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            rgb_filename = f"{output_dir}/realsense-{serial_number}-rgb-{frame_num}.png"
            cv2.imwrite(rgb_filename, rgb_bgr)
            print(f"  Frame {frame_num}: Saved RGB: {rgb_filename}")
        
        # Process depth frame
        depth_frame = frameset.get_depth_frame()
        if depth_frame:
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            raw_depth_data = np.asanyarray(depth_frame.get_data())
            raw_depth_image = np.resize(raw_depth_data, (height, width))
            
            depth_units = depth_frame.get_units()
            distance_matrix_m = raw_depth_image.astype(np.float32) * depth_units
            distance_matrix_mm = distance_matrix_m * 1000.0
            depth_image = distance_matrix_mm.astype(np.uint16)
            
            # Save raw depth
            depth_filename = f"{output_dir}/realsense-{serial_number}-depth-{frame_num}.png"
            cv2.imwrite(depth_filename, depth_image)
            
            # Save colorized depth
            colorized_depth = colorizer.colorize(depth_frame)
            colorized_data = np.asanyarray(colorized_depth.get_data())
            colorized_filename = f"{output_dir}/realsense-{serial_number}-depth-colorized-{frame_num}.png"
            cv2.imwrite(colorized_filename, colorized_data)
            
            # Save RGB-encoded depth
            if depth_image.dtype == np.uint16:
                high_byte = (depth_image >> 8).astype(np.uint8)
                low_byte = (depth_image & 0xFF).astype(np.uint8)
                zero_channel = np.zeros_like(high_byte, dtype=np.uint8)
                rgb_encoded_depth = np.stack([high_byte, low_byte, zero_channel], axis=-1)
                custom_rgb_filename = f"{output_dir}/realsense-{serial_number}-depth-rgb-encoded-{frame_num}.png"
                cv2.imwrite(custom_rgb_filename, rgb_encoded_depth)
            
            print(f"  Frame {frame_num}: Saved depth files")
    
    # Stop pipeline
    pipeline.stop()
    print(f"RealSense pipeline stopped (Serial: {serial_number}).")


def main():
    """Main function to find all cameras and capture frames."""
    print("=" * 60)
    print("Finding and testing all cameras")
    print("=" * 60)
    
    # Find RealSense cameras
    realsense_serials = find_realsense_cameras()
    
    if len(realsense_serials) < 2:
        print(f"\nWarning: Expected 2 RealSense cameras, found {len(realsense_serials)}")
        if len(realsense_serials) == 0:
            print("No RealSense cameras available. Skipping RealSense capture.")
    
    # Find Orbbec camera
    orbbec_device = find_orbbec_camera()
    
    if orbbec_device is None:
        print("No Orbbec camera available. Skipping Orbbec capture.")
    
    print("\n" + "=" * 60)
    print("Starting capture process")
    print("=" * 60)
    
    # Capture from Orbbec
    if orbbec_device is not None:
        try:
            capture_orbbec_frames(orbbec_device, "outputs/test/all_cameras/orbbec", NUM_FRAMES)
        except Exception as e:
            print(f"Error capturing from Orbbec: {e}")
    
    # Capture from RealSense cameras
    for i, serial in enumerate(realsense_serials[:2]):  # Only capture from first 2 RealSense cameras
        try:
            capture_realsense_frames(serial, f"outputs/test/all_cameras/realsense_{i}", NUM_FRAMES)
        except Exception as e:
            print(f"Error capturing from RealSense {serial}: {e}")
    
    print("\n" + "=" * 60)
    print("Capture complete!")
    print("=" * 60)
    print(f"\nOutput directories:")
    if orbbec_device is not None:
        print(f"  Orbbec: outputs/test/all_cameras/orbbec/")
    for i, serial in enumerate(realsense_serials[:2]):
        print(f"  RealSense {i} ({serial}): outputs/test/all_cameras/realsense_{i}/")


if __name__ == "__main__":
    main()

