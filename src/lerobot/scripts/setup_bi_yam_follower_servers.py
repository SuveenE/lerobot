#!/usr/bin/env python3
"""
Helper script to launch the bimanual Yam follower arm servers for use with LeRobot.

This script starts two follower arm servers:
- Right follower arm server on port 1234
- Left follower arm server on port 1235

Usage:
    python -m lerobot.scripts.setup_bi_yam_follower_servers
"""

from lerobot.scripts.setup_bi_yam_server_utils import run_server_group


def main():
    """Launch only the Yam follower arm servers."""
    run_server_group(
        group_name="follower",
        required_interfaces=["can_follower_r", "can_follower_l"],
        server_configs=[
            {
                "can_channel": "can_follower_r",
                "gripper": "linear_4310",
                "mode": "follower",
                "server_port": 1234,
                "use_encoder_server": False,
            },
            {
                "can_channel": "can_follower_l",
                "gripper": "linear_4310",
                "mode": "follower",
                "server_port": 1235,
                "use_encoder_server": False,
            },
        ],
        setup_lines=[
            "Right follower arm: localhost:1234",
            "Left follower arm:  localhost:1235",
        ],
        usage_lines=[
            "--robot.type=bi_yam_follower",
            "--robot.left_arm_port=1235",
            "--robot.right_arm_port=1234",
        ],
    )


if __name__ == "__main__":
    main()
