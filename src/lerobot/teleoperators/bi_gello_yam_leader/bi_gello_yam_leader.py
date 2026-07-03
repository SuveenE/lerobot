#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from functools import cached_property

import numpy as np

from ..teleoperator import Teleoperator
from .config_bi_gello_yam_leader import BiGelloYamLeaderConfig

logger = logging.getLogger(__name__)

# Each GELLO leader exposes 6 arm joints plus 1 normalized gripper value.
DOFS_PER_ARM = 7


class GelloLeaderArm:
    """Reads a single GELLO leader directly from its U2D2 device.

    Wraps the gello_software `GelloAgent`, which connects to the Dynamixel
    chain (torque off) and returns calibrated joint positions plus a normalized
    gripper value in [0, 1] via `get_joint_state()`.
    """

    def __init__(
        self,
        port: str,
        joint_ids: list[int],
        joint_offsets: list[float],
        joint_signs: list[float],
        gripper_config: list[float],
        start_joints: list[float],
        baudrate: int = 57600,
    ):
        self.port = port
        self._joint_ids = joint_ids
        self._joint_offsets = joint_offsets
        self._joint_signs = joint_signs
        self._gripper_config = gripper_config
        self._start_joints = start_joints
        self._baudrate = baudrate
        self._agent = None

    def connect(self) -> None:
        """Build the GELLO agent, which opens the serial port and calibrates offsets."""
        try:
            from gello.agents.gello_agent import DynamixelRobotConfig, GelloAgent
        except ImportError as e:
            raise ImportError(
                "The 'gello' package is required for bi_gello_yam_leader. Install it with "
                "`pip install -e /path/to/gello_software` in this environment."
            ) from e

        logger.info(f"Connecting to GELLO leader on {self.port}")
        dynamixel_config = DynamixelRobotConfig(
            joint_ids=tuple(self._joint_ids),
            joint_offsets=list(self._joint_offsets),
            joint_signs=tuple(self._joint_signs),
            gripper_config=(
                int(self._gripper_config[0]),
                self._gripper_config[1],
                self._gripper_config[2],
            ),
        )
        self._agent = GelloAgent(
            port=self.port,
            dynamixel_config=dynamixel_config,
            start_joints=np.array(self._start_joints),
        )
        logger.info(f"Connected to GELLO leader on {self.port}")

    @property
    def is_connected(self) -> bool:
        return self._agent is not None

    def get_joint_pos(self) -> np.ndarray:
        """Return calibrated joint positions (6 arm joints + normalized gripper)."""
        if self._agent is None:
            raise RuntimeError("GELLO leader not connected")
        # GelloAgent.act ignores its observation argument and returns the current
        # calibrated joint state directly from the Dynamixel driver.
        return np.asarray(self._agent.act({}), dtype=float)

    def disconnect(self) -> None:
        # The GELLO Dynamixel driver has no explicit close; dropping the reference
        # releases the serial handle. Torque was never enabled on the leaders.
        self._agent = None


class BiGelloYamLeader(Teleoperator):
    """Bimanual GELLO leader teleoperator for YAM follower arms.

    This is a drop-in replacement for `bi_yam_leader` on the teleoperation side.
    It reads two GELLO leaders directly over USB and produces the exact same
    action keys consumed by `bi_yam_follower`:

        left_joint_0.pos ... left_joint_5.pos, left_gripper.pos
        right_joint_0.pos ... right_joint_5.pos, right_gripper.pos
    """

    config_class = BiGelloYamLeaderConfig
    name = "bi_gello_yam_leader"

    def __init__(self, config: BiGelloYamLeaderConfig):
        super().__init__(config)
        self.config = config

        self.left_arm = GelloLeaderArm(
            port=config.left_port,
            joint_ids=config.left_joint_ids,
            joint_offsets=config.left_joint_offsets,
            joint_signs=config.left_joint_signs,
            gripper_config=config.left_gripper_config,
            start_joints=config.start_joints,
            baudrate=config.baudrate,
        )
        self.right_arm = GelloLeaderArm(
            port=config.right_port,
            joint_ids=config.right_joint_ids,
            joint_offsets=config.right_joint_offsets,
            joint_signs=config.right_joint_signs,
            gripper_config=config.right_gripper_config,
            start_joints=config.start_joints,
            baudrate=config.baudrate,
        )

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Per-arm joint + gripper position features, matching bi_yam_follower."""
        features: dict[str, type] = {}
        for side in ("left", "right"):
            for i in range(DOFS_PER_ARM):
                if i == DOFS_PER_ARM - 1:
                    features[f"{side}_gripper.pos"] = float
                else:
                    features[f"{side}_joint_{i}.pos"] = float
        return features

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        """GELLO leaders are passive and do not accept feedback."""
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect to both GELLO leaders.

        Args:
            calibrate: Unused; GELLO calibration is supplied via the config.
        """
        logger.info("Connecting to bimanual GELLO leaders")
        self.left_arm.connect()
        self.right_arm.connect()
        logger.info("Successfully connected to bimanual GELLO leaders")

    @property
    def is_calibrated(self) -> bool:
        """GELLO calibration is config-driven, so connection implies calibration."""
        return self.is_connected

    def calibrate(self) -> None:
        """No-op: GELLO offsets/signs/gripper come from the config."""
        pass

    def configure(self) -> None:
        """No runtime configuration needed for passive GELLO leaders."""
        pass

    def setup_motors(self) -> None:
        """No motor setup needed for passive GELLO leaders."""
        pass

    def _populate_arm_action(self, action: dict[str, float], side: str, joint_pos: np.ndarray) -> None:
        for i, pos in enumerate(joint_pos):
            if i == len(joint_pos) - 1:
                action[f"{side}_gripper.pos"] = float(pos)
            else:
                action[f"{side}_joint_{i}.pos"] = float(pos)

    def get_action(self) -> dict[str, float]:
        """Read both GELLO leaders and return follower-compatible joint targets."""
        action: dict[str, float] = {}
        self._populate_arm_action(action, "left", self.left_arm.get_joint_pos())
        self._populate_arm_action(action, "right", self.right_arm.get_joint_pos())
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """GELLO leaders are passive devices and do not support feedback."""
        pass

    def disconnect(self) -> None:
        logger.info("Disconnecting from bimanual GELLO leaders")
        self.left_arm.disconnect()
        self.right_arm.disconnect()
        logger.info("Disconnected from bimanual GELLO leaders")
