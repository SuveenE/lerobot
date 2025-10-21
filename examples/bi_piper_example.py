#!/usr/bin/env python3

"""
Example script showing how to use the BiPiper robot for recording data.

This script demonstrates the configuration and usage of the BiPiper robot
with the LeRobot framework, including optional depth data collection.

Basic Usage (RGB only):
    python -m lerobot.record \
        --robot.type=bi_piper \
        --robot.left_arm_can_port=can_0 \
        --robot.right_arm_can_port=can_1 \
        --robot.id=arm \
        --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, right: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, left: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 30}}" \
        --dataset.repo_id=your_username/bimanual-piper-dataset \
        --dataset.num_episodes=10 \
        --dataset.single_task="Pick and place task" \
        --dataset.episode_time_s=30 \
        --dataset.reset_time_s=10

With Depth Data Collection (RealSense):
    python -m lerobot.record \
        --robot.type=bi_piper \
        --robot.left_arm_can_port=can_0 \
        --robot.right_arm_can_port=can_1 \
        --robot.id=arm \
        --robot.cameras="{front: {type: intelrealsense, serial_number_or_name: '123456789', width: 640, height: 480, fps: 30, use_depth: true}, right: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
        --dataset.repo_id=your_username/bimanual-piper-depth-dataset \
        --dataset.num_episodes=10 \
        --dataset.single_task="Pick and place task" \
        --dataset.episode_time_s=30 \
        --dataset.reset_time_s=10

Requirements:
    - Install piper_sdk: pip install piper_sdk
    - Install lerobot with piper support: pip install -e ".[piper]"
    - For depth: Install pyrealsense2: pip install pyrealsense2
    - Connect Piper arms to CAN ports (can_0 and can_1)
    - Connect cameras to the specified indices

Notes:
    - Orbbec Gemini cameras can use OpenCV for RGB (via UVC support)
    - For Orbbec depth, you would need to implement an Orbbec camera driver (optional)
    - RealSense D405 cameras support both RGB and depth with use_depth=True
    - Depth data is stored as separate arrays with suffix "_depth" (e.g., front_depth)
"""

from lerobot.robots.bi_piper.config_bi_piper import BiPiperConfig


def create_bi_piper_config():
    """Create a sample BiPiper configuration"""

    # Basic configuration for BiPiper robot
    config = BiPiperConfig(
        type="bi_piper",
        left_arm_can_port="can_0",  # Adjust to your actual CAN port
        right_arm_can_port="can_1",  # Adjust to your actual CAN port
        cameras={
            "front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
            "right": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
            "top": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
            "left": {"type": "opencv", "index_or_path": 3, "width": 640, "height": 480, "fps": 30},
        },
    )

    return config


def create_bi_piper_config_with_depth():
    """Create a sample BiPiper configuration with depth capture enabled"""

    # Configuration with depth capture for RealSense cameras
    # Note: Orbbec Gemini cameras can use OpenCV for RGB (via UVC)
    # For Orbbec depth, you would need to implement an Orbbec camera driver
    config = BiPiperConfig(
        type="bi_piper",
        left_arm_can_port="can_0",  # Adjust to your actual CAN port
        right_arm_can_port="can_1",  # Adjust to your actual CAN port
        cameras={
            # RealSense camera with depth enabled
            "front": {
                "type": "intelrealsense",
                "serial_number_or_name": "123456789",  # Replace with actual serial number
                "width": 640,
                "height": 480,
                "fps": 30,
                "use_depth": True  # Enable depth capture
            },
            # Orbbec Gemini cameras can use OpenCV for RGB (UVC support)
            "right": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
            "top": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
            "left": {"type": "opencv", "index_or_path": 3, "width": 640, "height": 480, "fps": 30},
        },
    )

    return config


if __name__ == "__main__":
    # Example usage - Basic configuration
    print("=" * 70)
    print("Basic BiPiper Configuration (RGB only)")
    print("=" * 70)
    config = create_bi_piper_config()
    print(f"  Left arm CAN port: {config.left_arm_can_port}")
    print(f"  Right arm CAN port: {config.right_arm_can_port}")
    print(f"  Number of cameras: {len(config.cameras)}")
    print(f"  Camera names: {list(config.cameras.keys())}")

    print("\n" + "=" * 70)
    print("BiPiper Configuration with Depth Enabled")
    print("=" * 70)
    config_depth = create_bi_piper_config_with_depth()
    print(f"  Left arm CAN port: {config_depth.left_arm_can_port}")
    print(f"  Right arm CAN port: {config_depth.right_arm_can_port}")
    print(f"  Number of cameras: {len(config_depth.cameras)}")
    print(f"  Camera names: {list(config_depth.cameras.keys())}")

    # Check which cameras have depth enabled
    depth_cameras = [
        name for name, cfg in config_depth.cameras.items()
        if hasattr(cfg, 'use_depth') and cfg.use_depth
    ]
    if depth_cameras:
        print(f"  Cameras with depth enabled: {depth_cameras}")

    print("\n" + "=" * 70)
    print("To record data with this robot, run:")
    print("=" * 70)
    print("Basic (RGB only):")
    print(
        "  python -m lerobot.record --robot.type=bi_piper \\\n"
        "    --robot.left_arm_can_port=can_0 \\\n"
        "    --robot.right_arm_can_port=can_1 ..."
    )
    print("\nWith depth:")
    print(
        "  python -m lerobot.record --robot.type=bi_piper \\\n"
        "    --robot.cameras=\"{front: {type: intelrealsense, serial_number_or_name: 'YOUR_SN', use_depth: true}}\" ..."
    )
