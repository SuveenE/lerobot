#!/usr/bin/env python3
"""
GELLO arm server for Dynamixel-based GELLO arms (USB serial connection).

This script wraps the DynamixelMotorsBus to expose GELLO leader arm state
through the portal RPC interface, allowing LeRobot to read joint positions
from the GELLO device.

Usage:
    python -m lerobot.scripts.gello_server_dynamixel --port /dev/ttyUSB0 --server_port 6001
"""

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import portal
import tyro

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.dynamixel import DynamixelMotorsBus

DEFAULT_SERVER_PORT = 6001

# Common Dynamixel baud rates to try
BAUDRATES_TO_TRY = [1000000, 57600, 115200, 2000000, 3000000, 4000000]


class DynamixelGelloRobot:
    """
    Wrapper around DynamixelMotorsBus for GELLO leader arms.
    Exposes joint positions through a simple interface for the portal server.
    """

    def __init__(
        self,
        port: str,
        motor_type: str = "xl330-m288",
        joint_ids: list[int] | None = None,
        gripper_id: int | None = 7,
        joint_offsets: list[float] | None = None,
        joint_signs: list[float] | None = None,
    ):
        """
        Initialize the Dynamixel GELLO robot.

        Args:
            port: Serial port path (e.g., /dev/ttyUSB0)
            motor_type: Dynamixel motor model
            joint_ids: List of joint motor IDs (default: [1, 2, 3, 4, 5, 6])
            gripper_id: Gripper motor ID (default: 7, None for no gripper)
            joint_offsets: Offsets to add to joint positions (default: [π, π, π, π, π, π])
            joint_signs: Signs to multiply joint positions (default: [1, -1, 1, 1, 1, 1])
        """
        self._port = port
        self._motor_type = motor_type

        # Default GELLO configuration
        if joint_ids is None:
            joint_ids = [1, 2, 3, 4, 5, 6]
        if joint_offsets is None:
            joint_offsets = [math.pi] * 6
        if joint_signs is None:
            joint_signs = [1.0, -1.0, 1.0, 1.0, 1.0, 1.0]

        self._joint_ids = joint_ids
        self._gripper_id = gripper_id
        self._joint_offsets = np.array(joint_offsets, dtype=np.float32)
        self._joint_signs = np.array(joint_signs, dtype=np.float32)

        # Create motor configuration
        motor_names = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
        motors = {}
        for name, motor_id in zip(motor_names, joint_ids):
            motors[name] = Motor(motor_id, motor_type, MotorNormMode.DEGREES)

        if gripper_id is not None:
            motors["gripper"] = Motor(gripper_id, motor_type, MotorNormMode.DEGREES)
            motor_names.append("gripper")

        self._bus = DynamixelMotorsBus(port=port, motors=motors)
        self._motor_names = motor_names
        self._joint_names = [n for n in motor_names if n != "gripper"]

    def connect(self) -> None:
        """Connect to the motors, trying different baud rates."""
        print(f"Connecting to GELLO on {self._port}...")

        # Try different baud rates
        for baudrate in BAUDRATES_TO_TRY:
            try:
                print(f"  Trying baud rate {baudrate}...")
                self._bus.default_baudrate = baudrate
                self._bus.connect()
                print(f"  ✓ Connected at baud rate {baudrate}")

                # Disable torque so the arm can be moved freely (leader mode)
                self._bus.disable_torque()
                print(f"Connected to GELLO on {self._port}")
                return
            except Exception as e:
                # If connection failed, try to disconnect and try next baud rate
                try:
                    self._bus.disconnect()
                except Exception:
                    pass
                if baudrate == BAUDRATES_TO_TRY[-1]:
                    raise RuntimeError(
                        f"Could not connect to GELLO on {self._port}. "
                        f"Tried baud rates: {BAUDRATES_TO_TRY}. "
                        f"Last error: {e}"
                    )

    def disconnect(self) -> None:
        """Disconnect from the motors."""
        self._bus.disconnect()
        print(f"Disconnected from GELLO on {self._port}")

    def num_dofs(self) -> int:
        """Get the number of degrees of freedom."""
        return len(self._motor_names)

    def get_joint_pos(self) -> np.ndarray:
        """Get current joint positions (raw, in degrees)."""
        positions = self._bus.sync_read("Present_Position")
        pos_array = np.array([positions[name] for name in self._motor_names], dtype=np.float32)
        return pos_array

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Command joint positions (not used for leader arms)."""
        pass

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        """Command joint state (not used for leader arms)."""
        pass

    def get_observations(self) -> Dict[str, np.ndarray]:
        """
        Get current observations including joint positions.
        
        Returns joint positions in radians with offsets and signs applied,
        matching the GELLO software output format.
        """
        positions = self._bus.sync_read("Present_Position")

        # Get joint positions in degrees and convert to radians
        joint_pos_deg = np.array(
            [positions[name] for name in self._joint_names],
            dtype=np.float32,
        )
        joint_pos_rad = np.deg2rad(joint_pos_deg)

        # Apply signs and offsets: output = pos * sign + offset
        joint_pos_mapped = joint_pos_rad * self._joint_signs + self._joint_offsets

        obs = {"joint_pos": joint_pos_mapped}

        if self._gripper_id is not None:
            # Gripper position in degrees, normalized to 0-1 range
            gripper_deg = positions["gripper"]
            # GELLO gripper typically ranges from ~-34 to ~25 degrees
            # Normalize to 0-1 (closed to open)
            gripper_normalized = (gripper_deg + 34.1875) / (25.3125 + 34.1875)
            gripper_normalized = np.clip(gripper_normalized, 0.0, 1.0)
            obs["gripper_pos"] = np.array([gripper_normalized], dtype=np.float32)

        return obs


class ServerRobot:
    """A simple portal server for the robot."""

    def __init__(self, robot: DynamixelGelloRobot, port: int):
        self._robot = robot
        self._server = portal.Server(port)
        print(f"Dynamixel GELLO Server binding to port {port}")

        self._server.bind("num_dofs", self._robot.num_dofs)
        self._server.bind("get_joint_pos", self._robot.get_joint_pos)
        self._server.bind("command_joint_pos", self._robot.command_joint_pos)
        self._server.bind("command_joint_state", self._robot.command_joint_state)
        self._server.bind("get_observations", self._robot.get_observations)

    def serve(self) -> None:
        """Start serving."""
        self._server.start()


@dataclass
class Args:
    # Serial port for the GELLO arm
    port: str = "/dev/ttyUSB0"

    # Server port for portal RPC
    server_port: int = DEFAULT_SERVER_PORT

    # Dynamixel motor type
    motor_type: str = "xl330-m288"


def main(args: Args) -> None:
    """Main function to start the Dynamixel GELLO server."""
    # Create the robot with default GELLO configuration
    robot = DynamixelGelloRobot(
        port=args.port,
        motor_type=args.motor_type,
        # Uses default GELLO config:
        # joint_ids=[1,2,3,4,5,6], gripper_id=7
        # joint_offsets=[π,π,π,π,π,π]
        # joint_signs=[1,-1,1,1,1,1]
    )

    # Connect (will try multiple baud rates)
    robot.connect()

    # Start the server
    server = ServerRobot(robot, args.server_port)

    print(f"\n{'='*60}")
    print("Dynamixel GELLO Server Started")
    print(f"  Serial Port: {args.port}")
    print(f"  Motor Type: {args.motor_type}")
    print(f"  Joint IDs: [1, 2, 3, 4, 5, 6]")
    print(f"  Gripper ID: 7")
    print(f"  Server Port: {args.server_port}")
    print(f"{'='*60}\n")

    try:
        server.serve()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main(tyro.cli(Args))
