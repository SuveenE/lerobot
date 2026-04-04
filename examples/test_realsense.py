#!/usr/bin/env python
"""
Minimal script to test pyrealsense2 for RGB and Depth capture.

Python version of the C++ save-to-disk example from Intel RealSense SDK.
Captures frames and saves them to disk, following the official example pattern.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Configuration parameters (as variables, not flags)
SERIAL_NUMBER = "218622277937"  # Set to specific serial number string, or None to use first device
WIDTH = 424
HEIGHT = 240
FPS = 30

# Color format options:
# - rs.format.rgb8: RGB8 [24 bits] - RECOMMENDED: Simple, accurate colors, no conversion needed
# - rs.format.yuyv: YUY2 [16 bits] - Requires manual conversion, can have color issues
# - rs.format.bgr8: BGR8 [24 bits] - Direct BGR (OpenCV format), no conversion needed
COLOR_FORMAT = rs.format.rgb8  # RGB8 - simplest and most reliable
DEPTH_FORMAT = rs.format.z16   # Z [16 bits] as per datasheet

# Initialize RealSense context
context = rs.context()
devices = context.query_devices()

if len(devices) == 0:
    raise RuntimeError("No RealSense devices found")

# Select device
if SERIAL_NUMBER:
    device = None
    for d in devices:
        if d.get_info(rs.camera_info.serial_number) == SERIAL_NUMBER:
            device = d
            break
    if device is None:
        raise RuntimeError(f"Device with serial number {SERIAL_NUMBER} not found")
else:
    device = devices[0]

serial_number = device.get_info(rs.camera_info.serial_number)
print(f"Using device: {device.get_info(rs.camera_info.name)}")
print(f"Serial number: {serial_number}")

# Configure streams
config = rs.config()
config.enable_device(serial_number)

# Enable color stream
config.enable_stream(
    rs.stream.color,
    WIDTH,
    HEIGHT,
    COLOR_FORMAT,
    FPS
)

# Enable depth stream
config.enable_stream(
    rs.stream.depth,
    WIDTH,
    HEIGHT,
    DEPTH_FORMAT,
    FPS
)

# Start pipeline (like pipe.start() in C++)
pipeline = rs.pipeline()
profile = pipeline.start(config)

print(f"\nPipeline started successfully!")
print(f"Capturing RGB ({WIDTH}x{HEIGHT}@30fps) and Depth ({WIDTH}x{HEIGHT}@30fps)")

# Get actual stream profiles to verify
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()

print(f"\nActual stream profiles:")
print(f"  Color: {color_stream.width()}x{color_stream.height()}@{color_stream.fps()}fps, format={color_stream.format()}")
print(f"  Depth: {depth_stream.width()}x{depth_stream.height()}@{depth_stream.fps()}fps, format={depth_stream.format()}")

# Declare colorizer for depth visualization (like rs2::colorizer in C++)
colorizer = rs.colorizer()

# Capture 30 frames to give autoexposure, etc. a chance to settle (like C++ example)
print(f"\nWarming up (30 frames)...")
for i in range(30):
    pipeline.wait_for_frames()

print(f"Warmup complete. Capturing frames...")

# Create output directory
output_dir = "outputs/test/realsense"
os.makedirs(output_dir, exist_ok=True)

# Capture frames and save them (like the C++ example)
for frame_num in range(10):
    # Wait for frames (like pipe.wait_for_frames() in C++)
    frameset = pipeline.wait_for_frames()
    
    # Get color and depth frames directly from frameset (cleaner approach)
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()
    
    # Process color frame
    if color_frame:
        width = color_frame.get_width()
        height = color_frame.get_height()
        frame_format = color_frame.get_profile().format()
        frame_data = np.asanyarray(color_frame.get_data())
        
        print(f"  Frame {frame_num}, Color stream: "
              f"Size: {width}x{height}, Format: {frame_format}, "
              f"Data shape: {frame_data.shape}")
        
        # Convert to RGB if needed and save
        if frame_format == rs.format.rgb8:
            rgb_image = np.resize(frame_data, (height, width, 3))
        elif frame_format == rs.format.yuyv:
            # Handle YUY2 conversion if needed
            yuyv_image = np.resize(frame_data, (height, width, 2))
            rgb_image = cv2.cvtColor(yuyv_image, cv2.COLOR_YUV2RGB_YUYV)
        else:
            rgb_image = np.resize(frame_data, (height, width, 3))
        
        # Save RGB image (convert to BGR for OpenCV imwrite)
        rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        rgb_filename = f"{output_dir}/rs-rgb-{frame_num}.png"
        cv2.imwrite(rgb_filename, rgb_bgr)
        print(f"    Saved RGB: {rgb_filename}")
    
    # Process depth frame
    if depth_frame:
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        raw_depth_data = np.asanyarray(depth_frame.get_data())
        
        # Reshape depth data to 2D array (height, width)
        raw_depth_image = np.resize(raw_depth_data, (height, width))
        
        # Get depth units (conversion factor from raw pixels to meters)
        # Typically 0.001 for RealSense (raw pixel * 0.001 = distance in meters)
        depth_units = depth_frame.get_units()
        
        # Convert raw depth matrix to distance matrix (in meters)
        # Similar to Orbbec: depth_data * depth_scale = distance in millimeters
        # For RealSense: raw_pixel * depth_units = distance in meters
        distance_matrix_m = raw_depth_image.astype(np.float32) * depth_units
        
        # Convert to millimeters for storage and display (like Orbbec)
        distance_matrix_mm = distance_matrix_m * 1000.0
        
        # Convert back to uint16 for saving (preserve precision, values in millimeters)
        depth_image = distance_matrix_mm.astype(np.uint16)
        
        # Get center point coordinates
        center_x = width // 2
        center_y = height // 2
        
        # Get distance at center point from distance matrix (in meters)
        center_distance_m = distance_matrix_m[center_y, center_x]
        center_distance_mm = distance_matrix_mm[center_y, center_x]
        
        # Also verify using get_distance() API for comparison
        center_distance_m_api = depth_frame.get_distance(center_x, center_y)
        
        print(f"  Frame {frame_num}, Depth stream: "
              f"Size: {width}x{height}, Format: {depth_frame.get_profile().format()}, "
              f"Units: {depth_units}, Data shape: {depth_image.shape}, dtype: {depth_image.dtype}")
        print(f"    Center point ({center_x}, {center_y}) distance: "
              f"{center_distance_m:.3f} m ({center_distance_mm:.1f} mm) [from distance matrix]")
        print(f"    Verified with get_distance(): {center_distance_m_api:.3f} m")
        
        # Save distance matrix as depth (16-bit PNG, values in millimeters)
        depth_filename = f"{output_dir}/rs-depth-{frame_num}.png"
        cv2.imwrite(depth_filename, depth_image)
        print(f"    Saved depth (distance matrix in mm): {depth_filename}")
        
        # Colorize depth using RealSense colorizer (like color_map.process(frame) in C++)
        colorized_depth = colorizer.colorize(depth_frame)
        colorized_data = np.asanyarray(colorized_depth.get_data())
        
        # Save colorized depth (RGB encoded)
        colorized_filename = f"{output_dir}/rs-depth-colorized-{frame_num}.png"
        cv2.imwrite(colorized_filename, colorized_data)
        print(f"    Saved colorized depth: {colorized_filename}")
        
        # Also save custom RGB-encoded depth (matching image_writer.py encoding)
        # Encode 16-bit depth into RGB channels to preserve precision:
        # - R channel = high byte (upper 8 bits)
        # - G channel = low byte (lower 8 bits)
        # - B channel = 0 (unused)
        # This is lossless encoding that preserves full 16-bit precision
        # Note: depth_image now contains distance values in millimeters
        if depth_image.dtype == np.uint16:
            high_byte = (depth_image >> 8).astype(np.uint8)  # Upper 8 bits
            low_byte = (depth_image & 0xFF).astype(np.uint8)  # Lower 8 bits
            zero_channel = np.zeros_like(high_byte, dtype=np.uint8)  # B channel = 0
            # Stack as RGB: (H, W, 3)
            rgb_encoded_depth = np.stack([high_byte, low_byte, zero_channel], axis=-1)
            custom_rgb_filename = f"{output_dir}/rs-depth-rgb-encoded-{frame_num}.png"
            cv2.imwrite(custom_rgb_filename, rgb_encoded_depth)
            print(f"    Saved RGB-encoded depth (distance in mm): {custom_rgb_filename}")
        else:
            print(f"    Warning: Depth dtype is {depth_image.dtype}, expected uint16 for RGB encoding")

# Stop pipeline
pipeline.stop()
print(f"\nPipeline stopped. Test complete!")
