#!/usr/bin/env python3
"""
Shared utilities for launching Yam RPC server processes.
"""

import os
import subprocess
import sys
import time


def check_can_interface(interface: str) -> bool:
    """Check if a CAN interface exists and is available."""
    try:
        result = subprocess.run(
            ["ip", "link", "show", interface], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            return False

        if "state UP" in result.stdout or "state UNKNOWN" in result.stdout:
            return True

        print(f"Warning: CAN interface {interface} exists but is not UP")
        return False
    except Exception as e:
        print(f"Error checking CAN interface {interface}: {e}")
        return False


def check_all_can_interfaces(required_interfaces: list[str]) -> bool:
    """Check if all required CAN interfaces exist."""
    missing_interfaces = [interface for interface in required_interfaces if not check_can_interface(interface)]

    if missing_interfaces:
        raise RuntimeError(f"Missing or unavailable CAN interfaces: {', '.join(missing_interfaces)}")

    print("✓ All CAN interfaces are available")
    return True


def find_i2rt_script() -> str:
    """Find the i2rt minimum_gello.py script from the installed package."""
    try:
        import i2rt

        i2rt_path = os.path.dirname(i2rt.__file__)
        script_path = os.path.join(os.path.dirname(i2rt_path), "scripts", "minimum_gello.py")
        if os.path.exists(script_path):
            return script_path
    except ImportError:
        raise RuntimeError(
            "Could not import i2rt. Please install it separately:\n"
            "  cd i2rt && pip install -e . && cd ..\n"
            "Then install LeRobot: pip install -e '.[yam]'"
        )

    raise RuntimeError(
        "Could not find i2rt minimum_gello.py script. "
        "The i2rt installation may be incomplete."
    )


def launch_server_process(
    can_channel: str, gripper: str, mode: str, server_port: int, use_encoder_server: bool = False
):
    """Launch a single server process for a Yam arm."""
    if use_encoder_server:
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yam_server_with_encoder.py")
        if not os.path.exists(script_path):
            print(f"Error: Enhanced server script not found at {script_path}")
            sys.exit(1)
    else:
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
        mode,
        "--server_port",
        str(server_port),
    ]

    server_type = "Enhanced (Encoder)" if use_encoder_server else "Standard"
    print(f"Starting [{server_type}]: {' '.join(cmd)}")

    try:
        return subprocess.Popen(cmd)
    except Exception as e:
        print(f"Error starting process for {can_channel}: {e}")
        return None


def run_server_group(
    *,
    group_name: str,
    required_interfaces: list[str],
    server_configs: list[dict],
    setup_lines: list[str],
    usage_lines: list[str],
) -> None:
    """Launch a group of Yam server processes and keep them alive until interrupted."""
    processes = []

    try:
        print("Checking CAN interfaces...")
        check_all_can_interfaces(required_interfaces)

        print("\nLaunching server processes...")
        for config in server_configs:
            process = launch_server_process(**config)
            if process:
                processes.append(process)
                print(f"✓ Started process {process.pid} for {config['can_channel']} on port {config['server_port']}")
            else:
                raise RuntimeError(f"Failed to start process for {config['can_channel']}")

        print(f"\n✓ Successfully launched {len(processes)} {group_name} server processes")
        print("\nServer setup:")
        for line in setup_lines:
            print(f"  - {line}")

        print("\nYou can now use lerobot-record with:")
        for line in usage_lines:
            print(f"  {line}")

        print("\nPress Ctrl+C to stop all server processes")

        try:
            while True:
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
        for process in processes:
            try:
                print(f"Terminating process {process.pid}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Force killing process {process.pid}...")
                    process.kill()
                    process.wait()
            except Exception as e:
                print(f"Error terminating process {process.pid}: {e}")

        print("All server processes terminated")
