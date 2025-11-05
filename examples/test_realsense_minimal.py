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
output_dir = "output/test_depth/realsense"
os.makedirs(output_dir, exist_ok=True)

# Capture frames and save them (like the C++ example)
for frame_num in range(10):
    # Wait for frames (like pipe.wait_for_frames() in C++)
    frames = pipeline.wait_for_frames()
    
    # Process each frame in the frameset
    for frame in frames:
        # Check if it's a video frame (like vf = frame.as<rs2::video_frame>() in C++)
        if frame.is_video_frame():
            vf = frame.as_video_frame()
            stream_type = vf.get_profile().stream_type()
            width = vf.get_width()
            height = vf.get_height()
            
            # Process frames and save RGB, raw depth, and colorized depth
            if stream_type == rs.stream.color:
                # Color frame - get data directly
                frame_data = np.asanyarray(vf.get_data())
                frame_format = vf.get_profile().format()
                
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
                
            elif stream_type == rs.stream.depth:
                # Depth frame - save both raw and colorized versions
                raw_depth_data = np.asanyarray(vf.get_data())
                
                print(f"  Frame {frame_num}, Depth stream: "
                      f"Size: {width}x{height}, Format: {vf.get_profile().format()}, "
                      f"Data shape: {raw_depth_data.shape}, dtype: {raw_depth_data.dtype}")
                
                # Save raw depth (16-bit PNG)
                raw_depth_filename = f"{output_dir}/rs-depth-raw-{frame_num}.png"
                cv2.imwrite(raw_depth_filename, raw_depth_data)
                print(f"    Saved raw depth: {raw_depth_filename}")
                
                # Colorize depth using RealSense colorizer (like color_map.process(frame) in C++)
                colorized_depth = colorizer.colorize(vf)
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
                if raw_depth_data.dtype == np.uint16:
                    high_byte = (raw_depth_data >> 8).astype(np.uint8)  # Upper 8 bits
                    low_byte = (raw_depth_data & 0xFF).astype(np.uint8)  # Lower 8 bits
                    zero_channel = np.zeros_like(high_byte, dtype=np.uint8)  # B channel = 0
                    # Stack as RGB: (H, W, 3)
                    rgb_encoded_depth = np.stack([high_byte, low_byte, zero_channel], axis=-1)
                    custom_rgb_filename = f"{output_dir}/rs-depth-rgb-encoded-{frame_num}.png"
                    cv2.imwrite(custom_rgb_filename, rgb_encoded_depth)
                    print(f"    Saved RGB-encoded depth: {custom_rgb_filename}")
                else:
                    print(f"    Warning: Depth dtype is {raw_depth_data.dtype}, expected uint16 for RGB encoding")

# Stop pipeline
pipeline.stop()
print(f"\nPipeline stopped. Test complete!")
