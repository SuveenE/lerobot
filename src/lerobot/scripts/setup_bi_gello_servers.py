#!/usr/bin/env python3
"""
Helper script to launch bimanual GELLO leader + YAM follower servers for use with LeRobot.

This script starts four server processes:
- Two YAM follower arm servers (ports 1234 and 1235) - uses CAN interfaces
- Two GELLO leader arm servers (ports 6001 and 6002) - uses USB serial ports

The follower servers will be controlled by LeRobot's bi_yam_follower robot.
The GELLO leader servers expose the leader arm positions to LeRobot's bi_gello_leader teleoperator.

Expected interfaces:
- can_follower_r: Right YAM follower arm (CAN)
- can_follower_l: Left YAM follower arm (CAN)
- /dev/ttyUSB0: Right GELLO leader arm (USB serial)
- /dev/ttyUSB1: Left GELLO leader arm (USB serial)

Usage:
    python -m lerobot.scripts.setup_bi_gello_servers

    # Or with custom interfaces:
    python -m lerobot.scripts.setup_bi_gello_servers \
        --left_gello_port /dev/ttyUSB1 --right_gello_port /dev/ttyUSB0 \
        --left_follower_can can_follower_l --right_follower_can can_follower_r

Requirements:
    - LeRobot installed with yam support: pip install -e ".[yam]"
    - i2rt library (installed automatically with the above command)
    - CAN interfaces configured and available (for YAM followers)
    - USB serial ports available (for GELLO leaders)
    - Proper permissions to access devices
"""

import argparse
import os
import subprocess
import sys
import time


def check_can_interface(interface):
    """Check if a CAN interface exists and is available."""
    try:
        result = subprocess.run(
            ["ip", "link", "show", interface], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            return False

        if "state UP" in result.stdout or "state UNKNOWN" in result.stdout:
            return True
        else:
            print(f"Warning: CAN interface {interface} exists but is not UP")
            return False

    except Exception as e:
        print(f"Error checking CAN interface {interface}: {e}")
        return False


def check_serial_port(port):
    """Check if a serial port exists."""
    if os.path.exists(port):
        return True
    else:
        print(f"Warning: Serial port {port} does not exist")
        return False


def check_all_interfaces(
    left_gello_port: str,
    right_gello_port: str,
    left_follower_can: str,
    right_follower_can: str,
):
    """Check if all required interfaces exist."""
    missing = []

    # Check CAN interfaces for YAM followers
    for interface in [right_follower_can, left_follower_can]:
        if not check_can_interface(interface):
            missing.append(f"CAN: {interface}")

    # Check serial ports for GELLO leaders
    for port in [right_gello_port, left_gello_port]:
        if not check_serial_port(port):
            missing.append(f"Serial: {port}")

    if missing:
        raise RuntimeError(f"Missing or unavailable interfaces: {', '.join(missing)}")

    print("✓ All interfaces are available")
    return True


def find_i2rt_script():
    """Find the i2rt minimum_gello.py script from the installed package."""
    try:
        import i2rt

        i2rt_path = os.path.dirname(i2rt.__file__)
        script_path = os.path.join(os.path.dirname(i2rt_path), "scripts", "minimum_gello.py")
        if os.path.exists(script_path):
            return script_path
    except ImportError as err:
        raise RuntimeError(
            "Could not import i2rt. Please install it separately:\n"
            "  cd i2rt && pip install -e . && cd ..\n"
            "Then install LeRobot: pip install -e '.[yam]'"
        ) from err

    raise RuntimeError(
        "Could not find i2rt minimum_gello.py script. "
        "The i2rt installation may be incomplete."
    )


def launch_yam_follower_process(can_channel: str, gripper: str, server_port: int):
    """Launch a single server process for a YAM follower arm (CAN-based)."""
    try:
        script_path = find_i2rt_script()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    cmd = [
        sys.executable,
        script_path,
        "--can_channel",
        can_channel,
        "--gripper",
        gripper,
        "--mode",
        "follower",
        "--server_port",
        str(server_port),
    ]

    print(f"Starting YAM follower server: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(cmd)
        return process
    except Exception as e:
        print(f"Error starting process for {can_channel}: {e}")
        return None


def launch_gello_dynamixel_process(
    serial_port: str,
    server_port: int,
    motor_type: str = "xl330-m288",
    has_gripper: bool = True,
):
    """Launch a single server process for a Dynamixel GELLO leader arm (USB serial)."""
    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gello_server_dynamixel.py"
    )
    if not os.path.exists(script_path):
        print(f"Error: Dynamixel GELLO server script not found at {script_path}")
        sys.exit(1)

    cmd = [
        sys.executable,
        script_path,
        "--port",
        serial_port,
        "--server_port",
        str(server_port),
        "--motor_type",
        motor_type,
        "--has_gripper",
        str(has_gripper),
    ]

    print(f"Starting GELLO leader server: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(cmd)
        return process
    except Exception as e:
        print(f"Error starting process for {serial_port}: {e}")
        return None


def main():
    """Main function to launch all server processes."""
    parser = argparse.ArgumentParser(
        description="Launch bimanual GELLO leader + YAM follower servers"
    )

    # GELLO leader arguments (USB serial)
    parser.add_argument(
        "--left_gello_port",
        default="/dev/ttyUSB1",
        help="Serial port for left GELLO leader arm (e.g., /dev/ttyUSB1)",
    )
    parser.add_argument(
        "--right_gello_port",
        default="/dev/ttyUSB0",
        help="Serial port for right GELLO leader arm (e.g., /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--left_gello_server_port",
        type=int,
        default=6001,
        help="Server port for left GELLO leader arm",
    )
    parser.add_argument(
        "--right_gello_server_port",
        type=int,
        default=6002,
        help="Server port for right GELLO leader arm",
    )
    parser.add_argument(
        "--gello_motor_type",
        default="xl330-m288",
        help="Dynamixel motor type for GELLO arms (e.g., xl330-m288, xl330-m077)",
    )
    parser.add_argument(
        "--gello_no_gripper",
        action="store_true",
        help="GELLO arms don't have a gripper motor",
    )

    # YAM follower arguments (CAN)
    parser.add_argument(
        "--left_follower_can",
        default="can_follower_l",
        help="CAN interface for left YAM follower arm",
    )
    parser.add_argument(
        "--right_follower_can",
        default="can_follower_r",
        help="CAN interface for right YAM follower arm",
    )
    parser.add_argument(
        "--left_follower_port",
        type=int,
        default=1235,
        help="Server port for left YAM follower arm",
    )
    parser.add_argument(
        "--right_follower_port",
        type=int,
        default=1234,
        help="Server port for right YAM follower arm",
    )
    parser.add_argument(
        "--follower_gripper",
        default="linear_4310",
        help="Gripper type for YAM follower arms",
    )

    args = parser.parse_args()

    processes = []

    try:
        # Check interfaces
        print("Checking interfaces...")
        check_all_interfaces(
            args.left_gello_port,
            args.right_gello_port,
            args.left_follower_can,
            args.right_follower_can,
        )

        # Launch YAM follower servers first (CAN-based)
        print("\nLaunching YAM follower server processes (CAN)...")
        follower_configs = [
            # Right YAM follower arm
            {
                "can_channel": args.right_follower_can,
                "gripper": args.follower_gripper,
                "server_port": args.right_follower_port,
            },
            # Left YAM follower arm
            {
                "can_channel": args.left_follower_can,
                "gripper": args.follower_gripper,
                "server_port": args.left_follower_port,
            },
        ]

        for config in follower_configs:
            process = launch_yam_follower_process(**config)
            if process:
                processes.append(process)
                print(
                    f"✓ Started YAM follower process {process.pid} for {config['can_channel']} "
                    f"on port {config['server_port']}"
                )
            else:
                raise RuntimeError(f"Failed to start process for {config['can_channel']}")

        # Launch GELLO leader servers (USB serial Dynamixel)
        print("\nLaunching GELLO leader server processes (USB serial)...")
        gello_configs = [
            # Right GELLO leader arm
            {
                "serial_port": args.right_gello_port,
                "server_port": args.right_gello_server_port,
                "motor_type": args.gello_motor_type,
                "has_gripper": not args.gello_no_gripper,
            },
            # Left GELLO leader arm
            {
                "serial_port": args.left_gello_port,
                "server_port": args.left_gello_server_port,
                "motor_type": args.gello_motor_type,
                "has_gripper": not args.gello_no_gripper,
            },
        ]

        for config in gello_configs:
            process = launch_gello_dynamixel_process(**config)
            if process:
                processes.append(process)
                print(
                    f"✓ Started GELLO leader process {process.pid} for {config['serial_port']} "
                    f"on port {config['server_port']}"
                )
            else:
                raise RuntimeError(f"Failed to start process for {config['serial_port']}")

        print(f"\n✓ Successfully launched {len(processes)} server processes")
        print("\nServer setup:")
        print(f"  - Right YAM follower arm:  localhost:{args.right_follower_port} (CAN: {args.right_follower_can})")
        print(f"  - Left YAM follower arm:   localhost:{args.left_follower_port} (CAN: {args.left_follower_can})")
        print(f"  - Right GELLO leader arm:  localhost:{args.right_gello_server_port} (Serial: {args.right_gello_port})")
        print(f"  - Left GELLO leader arm:   localhost:{args.left_gello_server_port} (Serial: {args.left_gello_port})")
        print("\nYou can now use lerobot-record with:")
        print("  --robot.type=bi_yam_follower")
        print(f"  --robot.left_arm_port={args.left_follower_port}")
        print(f"  --robot.right_arm_port={args.right_follower_port}")
        print("  --teleop.type=bi_gello_leader")
        print(f"  --teleop.left_arm_port={args.left_gello_server_port}")
        print(f"  --teleop.right_arm_port={args.right_gello_server_port}")
        print("\nPress Ctrl+C to stop all server processes")

        # Wait for processes and handle termination
        try:
            while True:
                # Check if any process has died
                for i, process in enumerate(processes):
                    if process.poll() is not None:
                        print(f"\nProcess {process.pid} has terminated")
                        processes.pop(i)
                        break

                if not processes:
                    print("All processes have terminated")
                    break

                time.sleep(1)

        except KeyboardInterrupt:
            print("\n\nReceived Ctrl+C, terminating all server processes...")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    finally:
        # Clean up: terminate all running processes
        for process in processes:
            try:
                print(f"Terminating process {process.pid}...")
                process.terminate()

                # Wait up to 5 seconds for graceful termination
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Force killing process {process.pid}...")
                    process.kill()
                    process.wait()

            except Exception as e:
                print(f"Error terminating process {process.pid}: {e}")

        print("All server processes terminated")


if __name__ == "__main__":
    main()
