#!/usr/bin/env python
"""
Test script to decode RGB-encoded depth images back to uint16 depth frames.

This script reads RGB-encoded depth images (created by test_realsense.py and test_orbbec.py)
and decodes them back to the original uint16 depth values, then calculates and prints
the center point distance in millimeters.

RGB Encoding Format:
- R channel = high byte (upper 8 bits)
- G channel = low byte (lower 8 bits)
- B channel = 0 (unused)

Decoding: depth = (R << 8) | G = (R * 256) + G

Example usage:
python examples/test_decode_rgb_depth.py outputs/test/realsense/rs-depth-rgb-encoded-0.png

"""

import argparse
import os
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np


def decode_rgb_depth_image(rgb_image_path: Union[str, Path]) -> np.ndarray:
    """
    Decode RGB-encoded depth image back to uint16 depth values.
    
    Args:
        rgb_image_path: Path to RGB-encoded depth image (PNG)
        
    Returns:
        np.ndarray: uint16 depth array (H, W) with distance values in millimeters
    """
    # Load RGB image
    rgb_image = cv2.imread(str(rgb_image_path), cv2.IMREAD_COLOR)
    if rgb_image is None:
        raise ValueError(f"Failed to load image: {rgb_image_path}")
    
    # OpenCV loads as BGR, convert to RGB
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    
    # Extract R and G channels (already uint8 [0-255])
    r_channel = rgb_image[:, :, 0].astype(np.uint16)  # High byte
    g_channel = rgb_image[:, :, 1].astype(np.uint16)  # Low byte
    
    # Reconstruct uint16 depth: depth = (high_byte << 8) | low_byte
    # Equivalent to: depth = (high_byte * 256) + low_byte
    depth = (r_channel << 8) | g_channel
    
    return depth


def calculate_center_distance(depth_image: np.ndarray) -> Tuple[float, float]:
    """
    Calculate distance at center point of depth image.
    
    Args:
        depth_image: uint16 depth array (H, W) with distance values in millimeters
        
    Returns:
        tuple: (distance_mm, distance_m) - distance in millimeters and meters
    """
    height, width = depth_image.shape
    center_x = width // 2
    center_y = height // 2
    
    center_distance_mm = float(depth_image[center_y, center_x])
    center_distance_m = center_distance_mm / 1000.0
    
    return center_distance_mm, center_distance_m


def test_decode_images(image_paths: list[Union[str, Path]], camera_type: str = "unknown") -> None:
    """
    Test decoding multiple RGB-encoded depth images.
    
    Args:
        image_paths: List of paths to RGB-encoded depth images
        camera_type: Type of camera ("realsense", "orbbec", or "unknown")
    """
    print(f"\n{'='*70}")
    print(f"Testing RGB-encoded depth decoding ({camera_type})")
    print(f"{'='*70}\n")
    
    for i, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            print(f"  Skipping {image_path}: File not found")
            continue
        
        try:
            # Decode RGB-encoded depth image
            depth_image = decode_rgb_depth_image(image_path)
            
            # Calculate center point distance
            center_distance_mm, center_distance_m = calculate_center_distance(depth_image)
            
            # Get image statistics
            height, width = depth_image.shape
            nonzero_mask = depth_image > 0
            if nonzero_mask.any():
                min_depth = depth_image[nonzero_mask].min()
                max_depth = depth_image[nonzero_mask].max()
                mean_depth = depth_image[nonzero_mask].mean()
            else:
                min_depth = max_depth = mean_depth = 0
            
            print(f"  Image {i+1}: {os.path.basename(image_path)}")
            print(f"    Size: {width}x{height}")
            print(f"    Depth range: {min_depth:.1f} - {max_depth:.1f} mm")
            print(f"    Mean depth: {mean_depth:.1f} mm")
            print(f"    Center point ({width//2}, {height//2}) distance: "
                  f"{center_distance_m:.3f} m ({center_distance_mm:.1f} mm)")
            print()
            
        except Exception as e:
            print(f"  Error processing {image_path}: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Decode RGB-encoded depth images and print center point distance"
    )
    parser.add_argument(
        "image_paths",
        nargs="+",
        help="Paths to RGB-encoded depth images (PNG files)"
    )
    parser.add_argument(
        "--camera-type",
        choices=["realsense", "orbbec", "unknown"],
        default="unknown",
        help="Type of camera (for display purposes)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for RGB-encoded depth images from test scripts"
    )
    
    args = parser.parse_args()
    
    # If output directory is provided, try to find RGB-encoded depth images
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if output_dir.exists():
            # Look for RGB-encoded depth images
            if "realsense" in str(output_dir).lower():
                pattern = "rs-depth-rgb-encoded-*.png"
                camera_type = "realsense"
            elif "orbbec" in str(output_dir).lower():
                pattern = "ob-depth-rgb-encoded-*.png"
                camera_type = "orbbec"
            else:
                # Try both patterns
                realsense_images = list(output_dir.glob("rs-depth-rgb-encoded-*.png"))
                orbbec_images = list(output_dir.glob("ob-depth-rgb-encoded-*.png"))
                if realsense_images:
                    test_decode_images(sorted(realsense_images), "realsense")
                if orbbec_images:
                    test_decode_images(sorted(orbbec_images), "orbbec")
                return
            
            image_paths = sorted(output_dir.glob(pattern))
            if image_paths:
                test_decode_images(image_paths, camera_type)
                return
            else:
                print(f"No RGB-encoded depth images found matching pattern: {pattern}")
    
    # Process provided image paths
    if args.image_paths:
        test_decode_images(args.image_paths, args.camera_type)


if __name__ == "__main__":
    # Example usage without arguments: test common output directories
    import sys
    
    if len(sys.argv) == 1:
        # Try to find images in common output directories
        common_dirs = [
            "outputs/test/realsense",
            "outputs/test/orbbec",
            "output/test_depth/realsense",
            "output/test_depth/orbbec",
        ]
        
        found_any = False
        for output_dir in common_dirs:
            if os.path.exists(output_dir):
                print(f"\nFound output directory: {output_dir}")
                # Determine camera type from directory name
                if "realsense" in output_dir:
                    pattern = "rs-depth-rgb-encoded-*.png"
                    camera_type = "realsense"
                elif "orbbec" in output_dir:
                    pattern = "ob-depth-rgb-encoded-*.png"
                    camera_type = "orbbec"
                else:
                    continue
                
                image_paths = sorted(Path(output_dir).glob(pattern))
                if image_paths:
                    test_decode_images(image_paths, camera_type)
                    found_any = True
        
        if not found_any:
            print("\nNo RGB-encoded depth images found in common output directories.")
            print("\nUsage examples:")
            print("  # Decode specific images:")
            print("  python test_decode_rgb_depth.py outputs/test/realsense/rs-depth-rgb-encoded-0.png")
            print("  python test_decode_rgb_depth.py outputs/test/orbbec/ob-depth-rgb-encoded-0.png")
            print("\n  # Decode all images in a directory:")
            print("  python test_decode_rgb_depth.py --output-dir outputs/test/realsense")
            print("  python test_decode_rgb_depth.py --output-dir outputs/test/orbbec")
    else:
        main()

