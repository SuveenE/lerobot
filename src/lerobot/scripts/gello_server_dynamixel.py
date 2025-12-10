#!/usr/bin/env python3
"""
GELLO arm server for Dynamixel-based GELLO arms (USB serial connection).

This script wraps the DynamixelMotorsBus to expose GELLO leader arm state
through the portal RPC interface, allowing LeRobot to read joint positions
from the GELLO device.

Unlike the CAN-based GELLO server, this uses USB serial communication
with Dynamixel motors (e.g., XL330, XM430, etc.).

Usage:
    python -m lerobot.scripts.gello_server_dynamixel --port /dev/ttyUSB0 --server_port 6001

    # With custom motor type:
    python -m lerobot.scripts.gello_server_dynamixel --port /dev/ttyUSB0 --motor_type xl330-m288
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import portal
import tyro

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.dynamixel import DynamixelMotorsBus, OperatingMode

DEFAULT_SERVER_PORT = 6001


class DynamixelGelloRobot:
    """
    Wrapper around DynamixelMotorsBus for GELLO leader arms.
    Exposes joint positions through a simple interface for the portal server.
    """

    def __init__(
        self,
        port: str,
        motor_type: str = "xl330-m288",
        motor_ids: list[int] | None = None,
        has_gripper: bool = True,
    ):
        """
        Initialize the Dynamixel GELLO robot.

        Args:
            port: Serial port path (e.g., /dev/ttyUSB0)
            motor_type: Dynamixel motor model (e.g., xl330-m288, xl330-m077, xm430-w350)
            motor_ids: List of motor IDs (default: [1, 2, 3, 4, 5, 6] for 6-DOF + gripper)
            has_gripper: Whether the arm has a gripper motor
        """
        self._port = port
        self._motor_type = motor_type
        self._has_gripper = has_gripper

        # Default motor IDs for 6-DOF arm + gripper
        if motor_ids is None:
            motor_ids = [1, 2, 3, 4, 5, 6] if has_gripper else [1, 2, 3, 4, 5]

        self._motor_ids = motor_ids

        # Create motor configuration
        motor_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
        if has_gripper:
            motor_names.append("gripper")

        motors = {}
        for i, (name, motor_id) in enumerate(zip(motor_names, motor_ids)):
            if name == "gripper":
                motors[name] = Motor(motor_id, motor_type, MotorNormMode.RANGE_0_100)
            else:
                motors[name] = Motor(motor_id, motor_type, MotorNormMode.RANGE_M100_100)

        self._bus = DynamixelMotorsBus(port=port, motors=motors)
        self._motor_names = motor_names

    def connect(self) -> None:
        """Connect to the motors."""
        self._bus.connect()
        # Disable torque so the arm can be moved freely (leader mode)
        self._bus.disable_torque()
        # Set operating mode to extended position for all motors
        for motor in self._motor_names:
            if motor != "gripper":
                self._bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)
        print(f"Connected to GELLO on {self._port}")

    def disconnect(self) -> None:
        """Disconnect from the motors."""
        self._bus.disconnect()
        print(f"Disconnected from GELLO on {self._port}")

    def num_dofs(self) -> int:
        """Get the number of degrees of freedom."""
        return len(self._motor_names)

    def get_joint_pos(self) -> np.ndarray:
        """Get current joint positions."""
        positions = self._bus.sync_read("Present_Position")
        # Return as numpy array in motor order
        pos_array = np.array([positions[name] for name in self._motor_names], dtype=np.float32)
        return pos_array

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Command joint positions (not used for leader arms, but required for interface)."""
        # Leader arms are passive - we don't command them
        pass

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        """Command joint state (not used for leader arms)."""
        pass

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get current observations including joint positions."""
        positions = self._bus.sync_read("Present_Position")

        # Separate joint positions and gripper
        joint_pos = np.array(
            [positions[name] for name in self._motor_names if name != "gripper"],
            dtype=np.float32,
        )

        obs = {"joint_pos": joint_pos}

        if self._has_gripper:
            # Gripper position (0-100 range normalized)
            gripper_pos = positions["gripper"]
            obs["gripper_pos"] = np.array([gripper_pos / 100.0], dtype=np.float32)

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

    # Motor IDs (comma-separated, e.g., "1,2,3,4,5,6")
    motor_ids: str | None = None

    # Whether the arm has a gripper motor
    has_gripper: bool = True


def main(args: Args) -> None:
    """Main function to start the Dynamixel GELLO server."""
    # Parse motor IDs if provided
    motor_ids = None
    if args.motor_ids:
        motor_ids = [int(x.strip()) for x in args.motor_ids.split(",")]

    # Create the robot
    robot = DynamixelGelloRobot(
        port=args.port,
        motor_type=args.motor_type,
        motor_ids=motor_ids,
        has_gripper=args.has_gripper,
    )

    # Connect
    robot.connect()

    # Start the server
    server = ServerRobot(robot, args.server_port)

    print(f"\n{'='*60}")
    print("Dynamixel GELLO Server Started")
    print(f"  Serial Port: {args.port}")
    print(f"  Motor Type: {args.motor_type}")
    print(f"  Has Gripper: {args.has_gripper}")
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

