#!/usr/bin/env python3
"""
Helper script to launch the bimanual Yam leader arm servers for use with LeRobot.

This script starts two leader arm servers:
- Right leader arm server on port 5001
- Left leader arm server on port 5002

Usage:
    python -m lerobot.scripts.setup_bi_yam_leader_servers                  # portal/TCP (default)
    python -m lerobot.scripts.setup_bi_yam_leader_servers --transport udp  # UDP push streaming
"""

import argparse

from lerobot.scripts.setup_bi_yam_server_utils import run_server_group


def main():
    """Launch only the Yam leader arm servers."""
    parser = argparse.ArgumentParser(description="Launch bimanual Yam leader arm servers for LeRobot.")
    parser.add_argument(
        "--transport",
        choices=["portal", "udp"],
        default="portal",
        help="Transport for exposing leader state: 'portal' (TCP RPC, default) or 'udp' (push streaming).",
    )
    args = parser.parse_args()

    transport_note = " (UDP)" if args.transport == "udp" else ""
    usage_lines = ["--teleop.type=bi_yam_leader"]
    if args.transport == "udp":
        usage_lines.append("--teleop.transport=udp")
    usage_lines += [
        "--teleop.left_arm_port=5002",
        "--teleop.right_arm_port=5001",
    ]

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
                "transport": args.transport,
            },
            {
                "can_channel": "can_leader_l",
                "gripper": "yam_teaching_handle",
                "mode": "follower",
                "server_port": 5002,
                "use_encoder_server": True,
                "transport": args.transport,
            },
        ],
        setup_lines=[
            f"Right leader arm: localhost:5001{transport_note}",
            f"Left leader arm:  localhost:5002{transport_note}",
        ],
        usage_lines=usage_lines,
    )


if __name__ == "__main__":
    main()
