#!/usr/bin/env python3
"""
Helper script to launch the bimanual Yam leader arm servers for use with LeRobot.

This script starts two leader arm servers:
- Right leader arm server on port 5001
- Left leader arm server on port 5002

Usage:
    python -m lerobot.scripts.setup_bi_yam_leader_servers
"""

from lerobot.scripts.setup_bi_yam_server_utils import run_server_group


def main():
    """Launch only the Yam leader arm servers."""
    run_server_group(
        group_name="leader",
        required_interfaces=["can_leader_r", "can_leader_l"],
        server_configs=[
            {
                "can_channel": "can_leader_r",
                "gripper": "yam_teaching_handle",
                "mode": "follower",
                "server_port": 5001,
                "use_encoder_server": True,
                "transport": "udp",
            },
            {
                "can_channel": "can_leader_l",
                "gripper": "yam_teaching_handle",
                "mode": "follower",
                "server_port": 5002,
                "use_encoder_server": True,
                "transport": "udp",
            },
        ],
        setup_lines=[
            "Right leader arm: localhost:5001 (UDP)",
            "Left leader arm:  localhost:5002 (UDP)",
        ],
        usage_lines=[
            "--teleop.type=bi_yam_leader",
            "--teleop.transport=udp",
            "--teleop.left_arm_port=5002",
            "--teleop.right_arm_port=5001",
        ],
    )


if __name__ == "__main__":
    main()
