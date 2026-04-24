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
import time
from functools import cached_property
from typing import Any

import numpy as np
import portal

from lerobot.cameras.utils import make_cameras_from_configs

from ..robot import Robot
from .config_bi_yam_linear_bot import BiYamLinearBotConfig

logger = logging.getLogger(__name__)


class YamArmClient:
    """Client interface for a single Yam arm using the portal RPC framework."""

    def __init__(self, port: int, host: str = "localhost"):
        self.port = port
        self.host = host
        self._client = None

    def connect(self):
        logger.info(f"Connecting to Yam arm server at {self.host}:{self.port}")
        self._client = portal.Client(f"{self.host}:{self.port}")
        logger.info(f"Successfully connected to Yam arm server at {self.host}:{self.port}")

    def disconnect(self):
        if self._client is not None:
            logger.info(f"Disconnecting from Yam arm server at {self.host}:{self.port}")
            self._client = None

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    def num_dofs(self) -> int:
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.num_dofs().result()

    def get_joint_pos(self) -> np.ndarray:
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.get_joint_pos().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        if self._client is None:
            raise RuntimeError("Client not connected")
        self._client.command_joint_pos(joint_pos)

    def get_observations(self) -> dict[str, np.ndarray]:
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.get_observations().result()


class BiYamLinearBot(Robot):
    """
    Linear Bot: bimanual Yam arms + FlowBase mobile base + optional linear rail.

    This robot extends the bimanual Yam arm setup with a FlowBase holonomic
    mobile base and an optional linear rail lift module.  All actuator state
    and commanded values are exposed as flat float features so that the
    standard LeRobot dataset pipeline can record them.
    """

    config_class = BiYamLinearBotConfig
    name = "bi_yam_linear_bot"

    def __init__(self, config: BiYamLinearBotConfig):
        super().__init__(config)
        self.config = config

        self.left_arm = YamArmClient(port=config.left_arm_port, host=config.arm_server_host)
        self.right_arm = YamArmClient(port=config.right_arm_port, host=config.arm_server_host)

        self.cameras = make_cameras_from_configs(config.cameras)

        self._left_dofs = None
        self._right_dofs = None

        self._flow_base_client = None

    # ------------------------------------------------------------------
    # Feature declarations
    # ------------------------------------------------------------------

    @property
    def _arm_ft(self) -> dict[str, type]:
        if self._left_dofs is None or self._right_dofs is None:
            left_dofs = 7
            right_dofs = 7
        else:
            left_dofs = self._left_dofs
            right_dofs = self._right_dofs

        features: dict[str, type] = {}
        for i in range(left_dofs):
            if left_dofs == 7 and i == left_dofs - 1:
                features["left_gripper.pos"] = float
            else:
                features[f"left_joint_{i}.pos"] = float

        for i in range(right_dofs):
            if right_dofs == 7 and i == right_dofs - 1:
                features["right_gripper.pos"] = float
            else:
                features[f"right_joint_{i}.pos"] = float

        return features

    @property
    def _base_obs_ft(self) -> dict[str, type]:
        if self.config.x_only_mode:
            return {"base.x": float}

        ft: dict[str, type] = {
            "base.x": float,
            "base.y": float,
            "base.theta": float,
        }
        if self.config.with_linear_rail:
            ft.update({
                "rail.position": float,
                "rail.velocity": float,
                "rail.upper_limit": float,
                "rail.lower_limit": float,
            })
        ft.update({
            "base.cmd.x.vel": float,
            "base.cmd.y.vel": float,
            "base.cmd.theta.vel": float,
        })
        if self.config.with_linear_rail:
            ft["rail.cmd.vel"] = float
        return ft

    @property
    def _base_action_ft(self) -> dict[str, type]:
        if self.config.x_only_mode:
            return {"base.x.vel": float}

        ft: dict[str, type] = {
            "base.x.vel": float,
            "base.y.vel": float,
            "base.theta.vel": float,
        }
        if self.config.with_linear_rail:
            ft["rail.vel"] = float
        return ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._arm_ft, **self._base_obs_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**self._arm_ft, **self._base_action_ft}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return (
            self.left_arm.is_connected
            and self.right_arm.is_connected
            and self._flow_base_client is not None
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        logger.info("Connecting to Linear Bot")

        self.left_arm.connect()
        self.right_arm.connect()

        self._left_dofs = self.left_arm.num_dofs()
        self._right_dofs = self.right_arm.num_dofs()
        logger.info(f"Left arm DOFs: {self._left_dofs}, Right arm DOFs: {self._right_dofs}")

        from i2rt.flow_base.flow_base_client import FlowBaseClient

        self._flow_base_client = FlowBaseClient(
            host=self.config.flow_base_host,
            with_linear_rail=self.config.with_linear_rail,
        )
        # Stop the heartbeat thread that FlowBaseClient starts automatically.
        # It sends zeros at 50 Hz which would override joystick input on the
        # FlowBase controller.  We send explicit velocity commands in
        # send_action() only when the action dict contains base keys (i.e.
        # during policy deployment, not during teleoperation).
        self._flow_base_client.running = False
        if self._flow_base_client._thread.is_alive():
            self._flow_base_client._thread.join(timeout=1.0)
        logger.info(
            f"Connected to FlowBase at {self.config.flow_base_host} "
            f"(linear rail: {self.config.with_linear_rail})"
        )

        for cam in self.cameras.values():
            cam.connect()

        logger.info("Successfully connected to Linear Bot")

    @property
    def is_calibrated(self) -> bool:
        return self.is_connected

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def setup_motors(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        obs_dict: dict[str, Any] = {}

        # --- arms ---
        left_obs = self.left_arm.get_observations()
        left_joint_pos = left_obs["joint_pos"]
        left_has_gripper = "gripper_pos" in left_obs
        if left_has_gripper:
            left_joint_pos = np.concatenate([left_joint_pos, left_obs["gripper_pos"]])

        for i, pos in enumerate(left_joint_pos):
            if left_has_gripper and i == len(left_joint_pos) - 1:
                obs_dict["left_gripper.pos"] = pos
            else:
                obs_dict[f"left_joint_{i}.pos"] = pos

        right_obs = self.right_arm.get_observations()
        right_joint_pos = right_obs["joint_pos"]
        right_has_gripper = "gripper_pos" in right_obs
        if right_has_gripper:
            right_joint_pos = np.concatenate([right_joint_pos, right_obs["gripper_pos"]])

        for i, pos in enumerate(right_joint_pos):
            if right_has_gripper and i == len(right_joint_pos) - 1:
                obs_dict["right_gripper.pos"] = pos
            else:
                obs_dict[f"right_joint_{i}.pos"] = pos

        # --- FlowBase odometry ---
        odometry = self._flow_base_client.get_odometry()
        translation = odometry["translation"]
        rotation = odometry["rotation"]
        obs_dict["base.x"] = float(translation[0])
        if not self.config.x_only_mode:
            obs_dict["base.y"] = float(translation[1])
            obs_dict["base.theta"] = float(rotation)

        # --- Linear rail ---
        # In x_only_mode the rail is parked externally and never commanded by
        # the robot; we skip the rail RPC entirely to save a round-trip and
        # leave rail fields out of the dataset.
        if self.config.with_linear_rail and not self.config.x_only_mode:
            rail = self._flow_base_client.get_linear_rail_state()
            obs_dict["rail.position"] = float(rail["position"])
            obs_dict["rail.velocity"] = float(rail["velocity"])
            obs_dict["rail.upper_limit"] = 1.0 if rail.get("upper_limit_triggered") else 0.0
            obs_dict["rail.lower_limit"] = 1.0 if rail.get("lower_limit_triggered") else 0.0

        # --- Resolved command (captures joystick and/or remote input) ---
        resolved = self._flow_base_client.get_current_command()
        vel = resolved["velocity"]
        if self.config.x_only_mode:
            # Stash base.cmd.x.vel in the raw obs dict so teleop_action_from_obs
            # can copy it into action.base.x.vel. This key is intentionally NOT
            # declared in observation_features, so build_dataset_frame will drop
            # it from the saved dataset.
            obs_dict["base.cmd.x.vel"] = float(vel[0])
        else:
            obs_dict["base.cmd.x.vel"] = float(vel[0])
            obs_dict["base.cmd.y.vel"] = float(vel[1])
            obs_dict["base.cmd.theta.vel"] = float(vel[2])
            if self.config.with_linear_rail:
                obs_dict["rail.cmd.vel"] = float(vel[3]) if len(vel) > 3 else 0.0

        # --- cameras ---
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # --- arms ---
        left_action = []
        for i in range(self._left_dofs):
            key = (
                "left_gripper.pos"
                if (self._left_dofs == 7 and i == self._left_dofs - 1)
                else f"left_joint_{i}.pos"
            )
            if key in action:
                left_action.append(action[key])

        right_action = []
        for i in range(self._right_dofs):
            key = (
                "right_gripper.pos"
                if (self._right_dofs == 7 and i == self._right_dofs - 1)
                else f"right_joint_{i}.pos"
            )
            if key in action:
                right_action.append(action[key])

        if self.config.left_arm_max_relative_target is not None:
            left_current = self.left_arm.get_joint_pos()
            left_action = self._clip_relative_target(
                np.array(left_action), left_current, self.config.left_arm_max_relative_target
            )

        if self.config.right_arm_max_relative_target is not None:
            right_current = self.right_arm.get_joint_pos()
            right_action = self._clip_relative_target(
                np.array(right_action), right_current, self.config.right_arm_max_relative_target
            )

        if len(left_action) > 0:
            self.left_arm.command_joint_pos(np.array(left_action))

        if len(right_action) > 0:
            self.right_arm.command_joint_pos(np.array(right_action))

        # --- FlowBase velocity ---
        # Only send velocity commands when the action dict actually contains
        # base keys.  During teleoperation the leader only produces arm keys,
        # so we must NOT send zeros -- that would override joystick input on
        # the FlowBase controller (its remote-command timeout keeps the
        # joystick blocked while valid remote commands arrive).
        has_base_action = "base.x.vel" in action or "base.y.vel" in action or "base.theta.vel" in action
        if has_base_action:
            base_vel = np.array([
                action.get("base.x.vel", 0.0),
                action.get("base.y.vel", 0.0),
                action.get("base.theta.vel", 0.0),
            ])

            # Policy outputs are physical velocities (m/s, rad/s) recorded
            # from get_current_command, but the FlowBase controller expects
            # normalised [-1, 1] commands and scales internally by max_vel /
            # lift_max_vel.  Divide here to avoid double-scaling.
            base_max = np.array(self.config.base_max_vel)
            base_vel_norm = base_vel / np.where(base_max != 0, base_max, 1.0)

            if self.config.x_only_mode:
                # Send a 3D base velocity with y=theta=0; the rail is parked
                # externally and never commanded from here.
                vel_cmd = base_vel_norm
            elif self.config.with_linear_rail:
                rail_vel = action.get("rail.vel", 0.0)
                rail_max = self.config.rail_max_vel if self.config.rail_max_vel != 0 else 1.0
                rail_vel_norm = rail_vel / rail_max
                vel_cmd = np.concatenate([base_vel_norm, [rail_vel_norm]])
            else:
                vel_cmd = base_vel_norm

            # Send directly via the portal RPC client (the heartbeat thread
            # is stopped to avoid sending unwanted zeros, so we bypass
            # FlowBaseClient.set_target_velocity which only updates an
            # internal dict read by the heartbeat).
            self._flow_base_client.client.set_target_velocity(
                {"target_velocity": vel_cmd, "frame": "local"}
            ).result()

        return action

    # ------------------------------------------------------------------
    # Teleop helpers
    # ------------------------------------------------------------------

    def teleop_action_from_obs(self, obs: dict[str, Any]) -> dict[str, float]:
        if self.config.x_only_mode:
            # base.cmd.x.vel is present in the raw obs dict (stashed by
            # get_observation) even though it is not a declared observation
            # feature, so we can use it as the action fallback during
            # teleoperation when the leader only produces arm joint targets.
            return {"base.x.vel": float(obs.get("base.cmd.x.vel", 0.0))}

        fallback: dict[str, float] = {
            "base.x.vel": float(obs.get("base.cmd.x.vel", 0.0)),
            "base.y.vel": float(obs.get("base.cmd.y.vel", 0.0)),
            "base.theta.vel": float(obs.get("base.cmd.theta.vel", 0.0)),
        }
        if self.config.with_linear_rail:
            fallback["rail.vel"] = float(obs.get("rail.cmd.vel", 0.0))
        return fallback

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clip_relative_target(
        self, target: np.ndarray, current: np.ndarray, max_relative: float | dict[str, float]
    ) -> np.ndarray:
        if isinstance(max_relative, dict):
            clipped = target.copy()
            for i in range(len(target)):
                key = f"joint_{i}.pos"
                if key in max_relative:
                    max_delta = max_relative[key]
                    clipped[i] = np.clip(target[i], current[i] - max_delta, current[i] + max_delta)
            return clipped
        return np.clip(target, current - max_relative, current + max_relative)

    def disconnect(self):
        logger.info("Disconnecting from Linear Bot")

        self.left_arm.disconnect()
        self.right_arm.disconnect()

        if self._flow_base_client is not None:
            self._flow_base_client.close()
            self._flow_base_client = None

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info("Disconnected from Linear Bot")
