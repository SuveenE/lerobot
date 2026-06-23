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
        """Disconnect from the arm server.

        `portal.Client` runs a background socket thread (plus an OS pipe) that is only
        torn down by `Client.close()`. Previously we just dropped the reference, which
        leaked the thread/socket/fds on every disconnect and could wedge process
        teardown after recording. We close it explicitly with a bounded timeout so a
        stuck connection can't block shutdown.
        """
        if self._client is not None:
            logger.info(f"Disconnecting from Yam arm server at {self.host}:{self.port}")
            try:
                self._client.close(timeout=2.0)
            except Exception as e:
                logger.warning(f"Error closing Yam arm client at {self.host}:{self.port}: {e}")
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
        # Linear-rail motor rad <-> linear meter conversion factor, captured from
        # the FlowBase controller's homing calibration. Used to expose the rail
        # in meters / m/s instead of motor rad / rad/s. None until connected.
        self._meters_per_rad: float | None = None

    # ------------------------------------------------------------------
    # Feature declarations
    # ------------------------------------------------------------------

    @property
    def _arm_ft(self) -> dict[str, type]:
        return {
            **self._build_per_arm_features("left", "pos"),
            **self._build_per_arm_features("right", "pos"),
        }

    def _build_per_arm_features(self, side: str, suffix: str) -> dict[str, type]:
        """Build a flat per-motor feature dict for one arm, keyed `{side}_{joint|gripper}.{suffix}`.

        7 DOFs are assumed to be 6 joints + gripper; otherwise all motors are named
        `joint_i`. Positions (`.pos`) and torques (`.eff`) share this layout so they
        line up 1-to-1 by motor.
        """
        dofs_attr = self._left_dofs if side == "left" else self._right_dofs
        dofs = dofs_attr if dofs_attr is not None else 7

        features: dict[str, type] = {}
        for i in range(dofs):
            if dofs == 7 and i == dofs - 1:
                features[f"{side}_gripper.{suffix}"] = float
            else:
                features[f"{side}_joint_{i}.{suffix}"] = float
        return features

    @property
    def _base_obs_ft(self) -> dict[str, type]:
        # Units: base.x/base.y in m, base.theta in rad (the only angular state),
        # rail.position in m, rail.velocity in m/s, base.cmd.*.vel in m/s
        # (theta in rad/s), rail.cmd.vel in m/s.
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
        # Units: base.x.vel/base.y.vel in m/s, base.theta.vel in rad/s,
        # rail.vel in m/s.
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

    @property
    def extra_dataset_features(self) -> dict[str, dict]:
        """Per-arm torque columns that bypass the standard hw_to_dataset_features lumping.

        Returns empty when `config.record_torques` is False. When enabled, emits
        `observation.left_torques` and `observation.right_torques` columns built
        from the flat `.eff` keys produced by `get_observation()`. Consumed by
        `lerobot.scripts.lerobot_record` via
        `getattr(robot, "extra_dataset_features", {})`.
        """
        features: dict[str, dict] = {}

        if getattr(self.config, "record_torques", False):
            for side in ("left", "right"):
                names = list(self._build_per_arm_features(side, "eff").keys())
                features[f"observation.{side}_torques"] = {
                    "dtype": "float32",
                    "shape": (len(names),),
                    "names": names,
                }

        # Depth maps are recorded as their own single-channel `image` columns
        # (lossless 16-bit PNG in millimeters), separate from the 3-channel color
        # `observation.images.<cam>` columns. Gated per-camera on `use_depth`.
        for cam_key, cam_cfg in self.config.cameras.items():
            if not getattr(cam_cfg, "use_depth", False):
                continue
            if cam_cfg.height is None or cam_cfg.width is None:
                raise ValueError(
                    f"Camera '{cam_key}' has use_depth=True but height/width are unset; "
                    "set them explicitly to record depth."
                )
            features[f"observation.images.{cam_key}_depth"] = {
                "dtype": "image",
                "shape": (cam_cfg.height, cam_cfg.width, 1),
                "names": ["height", "width", "channels"],
            }

        return features

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

        # If any step below fails partway, tear down whatever already opened a
        # `portal.Client` (each one runs a non-daemon socket thread that keeps
        # auto-reconnecting). Leaving these dangling holds stale connections on
        # the arm servers, which is a common cause of flaky / "arm dropped"
        # behaviour on the next start.
        try:
            self.left_arm.connect()
            self.right_arm.connect()

            # First real RPCs: these block until the servers actually respond,
            # so they double as a connection health-check.
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

            # Cache the rail's meters_per_rad calibration so we can report and
            # command the rail in linear units (m / m/s). The controller derives
            # it during startup homing; it's constant for the session.
            #
            # We fail hard if the rail is enabled but uncalibrated: recording would
            # otherwise silently fall back to motor rad under the rail.* keys while
            # claiming meters, producing a dataset with mislabeled units. Better to
            # stop now than to discover the corruption after a recording session.
            if self.config.with_linear_rail:
                try:
                    rail_state = self._flow_base_client.get_linear_rail_state()
                except Exception as e:
                    raise RuntimeError(
                        "Failed to read linear rail calibration from the FlowBase controller; "
                        "cannot record the rail in meters. Check that the flow_base_controller "
                        "is running and reachable."
                    ) from e

                self._meters_per_rad = rail_state.get("meters_per_rad")
                if self._meters_per_rad is None:
                    raise RuntimeError(
                        "FlowBase linear rail is not calibrated (meters_per_rad is None). "
                        "Restart the flow_base_controller so its startup limit-switch homing "
                        "can calibrate the rail, or run with --robot.with_linear_rail=false. "
                        "Recording would otherwise save rail.position/velocity in motor rad "
                        "while labeling them as meters."
                    )
                logger.info(f"Linear rail meters_per_rad = {self._meters_per_rad:.6f}")

            logger.info(
                f"Connected to FlowBase at {self.config.flow_base_host} "
                f"(linear rail: {self.config.with_linear_rail})"
            )

            for cam in self.cameras.values():
                cam.connect()
        except Exception:
            logger.exception("Linear Bot connection failed; cleaning up partial connections")
            self.disconnect()
            raise

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
        self._populate_arm_obs(obs_dict, side="left", arm_obs=left_obs)

        right_obs = self.right_arm.get_observations()
        self._populate_arm_obs(obs_dict, side="right", arm_obs=right_obs)

        # --- FlowBase odometry ---
        odometry = self._flow_base_client.get_odometry()
        translation = odometry["translation"]
        rotation = odometry["rotation"]
        obs_dict["base.x"] = float(translation[0])
        obs_dict["base.y"] = float(translation[1])
        obs_dict["base.theta"] = float(rotation)

        # --- Linear rail ---
        # The controller exposes both motor-space (rad / rad/s) and calibrated
        # linear-space (m / m/s) values. We record the linear ones so the rail
        # matches the rest of the metric state space; only base.theta stays rad.
        if self.config.with_linear_rail:
            rail = self._flow_base_client.get_linear_rail_state()
            mpr = rail.get("meters_per_rad")
            if mpr is not None:
                self._meters_per_rad = mpr
            pos_linear = rail.get("position_linear")
            vel_linear = rail.get("velocity_linear")
            # Fall back to motor units only if the rail is uncalibrated (linear
            # fields are None); in normal operation homing has already run.
            obs_dict["rail.position"] = float(pos_linear) if pos_linear is not None else float(rail["position"])
            obs_dict["rail.velocity"] = float(vel_linear) if vel_linear is not None else float(rail["velocity"])
            obs_dict["rail.upper_limit"] = 1.0 if rail.get("upper_limit_triggered") else 0.0
            obs_dict["rail.lower_limit"] = 1.0 if rail.get("lower_limit_triggered") else 0.0

        # --- Resolved command (captures joystick and/or remote input) ---
        resolved = self._flow_base_client.get_current_command()
        vel = resolved["velocity"]
        obs_dict["base.cmd.x.vel"] = float(vel[0])
        obs_dict["base.cmd.y.vel"] = float(vel[1])
        obs_dict["base.cmd.theta.vel"] = float(vel[2])
        if self.config.with_linear_rail:
            # The resolved command rail velocity is already in m/s: the controller
            # scales the gamepad by lift_max_vel_ms and passes remote commands
            # through unchanged. Cap it so the recorded action stays within
            # ±rail_max_vel_mps, matching the m/s rail.velocity observation.
            rail_cmd_mps = float(vel[3]) if len(vel) > 3 else 0.0
            cap = self.config.rail_max_vel_mps
            obs_dict["rail.cmd.vel"] = float(np.clip(rail_cmd_mps, -cap, cap))

        # --- cameras ---
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            # For depth-enabled cameras, peek the depth frame captured alongside the
            # color frame just read (no extra wait). Stored as (H, W, 1) uint16 mm.
            if getattr(cam, "use_depth", False):
                depth = cam.read_depth_latest()
                obs_dict[f"{cam_key}_depth"] = depth[..., np.newaxis]
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def _populate_arm_obs(self, obs_dict: dict[str, Any], side: str, arm_obs: dict[str, np.ndarray]) -> None:
        """Emit flat `.pos` (and optionally `.eff`) keys for one arm from a single RPC response.

        The i2rt server returns joint and gripper components separately; we concatenate
        them so a 7-DOF (6 joints + gripper) arm yields keys 0..5 + `gripper`. Torques
        come from `joint_eff` / `gripper_eff` (motor effort feedback over CAN) and are
        only emitted when `config.record_torques=True`.
        """
        joint_pos = arm_obs["joint_pos"]
        has_gripper = "gripper_pos" in arm_obs

        if has_gripper:
            joint_pos = np.concatenate([joint_pos, arm_obs["gripper_pos"]])

        for i, pos in enumerate(joint_pos):
            if has_gripper and i == len(joint_pos) - 1:
                obs_dict[f"{side}_gripper.pos"] = pos
            else:
                obs_dict[f"{side}_joint_{i}.pos"] = pos

        if not getattr(self.config, "record_torques", False):
            return

        joint_eff = arm_obs.get("joint_eff")
        if joint_eff is None:
            return

        if has_gripper and "gripper_eff" in arm_obs:
            joint_eff = np.concatenate([joint_eff, arm_obs["gripper_eff"]])

        for i, eff in enumerate(joint_eff):
            if has_gripper and i == len(joint_eff) - 1:
                obs_dict[f"{side}_gripper.eff"] = eff
            else:
                obs_dict[f"{side}_joint_{i}.eff"] = eff

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

            # Base policy outputs are physical velocities (m/s, rad/s) recorded
            # from get_current_command, but the FlowBase controller expects the
            # base axes as normalised [-1, 1] commands that it scales internally
            # by max_vel. Divide here to avoid double-scaling. (The rail is sent
            # in physical m/s instead; see below.)
            base_max = np.array(self.config.base_max_vel)
            base_vel_norm = base_vel / np.where(base_max != 0, base_max, 1.0)

            if self.config.with_linear_rail:
                # The controller takes the rail command in physical m/s and converts
                # it to motor rad/s server-side via meters_per_rad, so we send
                # rail.vel straight through after capping at ±rail_max_vel_mps. Note
                # the mixed command vector: base axes are normalised, the rail is
                # physical m/s.
                rail_vel_mps = action.get("rail.vel", 0.0)
                cap = self.config.rail_max_vel_mps
                rail_vel_mps = float(np.clip(rail_vel_mps, -cap, cap))
                vel_cmd = np.concatenate([base_vel_norm, [rail_vel_mps]])
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
    # Reset / homing
    # ------------------------------------------------------------------

    def reset_odometry(self, events: dict | None = None) -> None:
        """Zero the FlowBase odometry so the next episode starts at the origin.

        The FlowBase controller integrates wheel odometry continuously for the
        lifetime of its process and never resets on its own, so without this
        every episode would start at whatever pose the base drifted to during
        the previous episode + reset window. The recorder calls this just
        before each recorded episode so `base.x` / `base.y` / `base.theta`
        always begin at 0.

        `events` is accepted (and ignored) so this can be used as a drop-in
        per-episode hook alongside `move_to_initial_position`.
        """
        if self._flow_base_client is None:
            logger.warning("Cannot reset odometry: FlowBase client not connected")
            return
        try:
            self._flow_base_client.reset_odometry()
            logger.info("FlowBase odometry reset to origin")
        except Exception as e:
            logger.warning(f"Failed to reset FlowBase odometry: {e}")

    def _send_rail_velocity(self, vel_mps: float) -> None:
        """Command the linear rail at `vel_mps` (m/s, +up) leaving the base still.

        The FlowBaseClient heartbeat thread is stopped in `connect()` (so it
        can't spam zeros over joystick input), which means
        `FlowBaseClient.set_linear_rail_velocity` -- which only mutates the dict
        the heartbeat reads -- would never actually be sent. So, exactly like
        `send_action`, we push a full 4-DOF command straight through the portal
        RPC client: base axes are normalised (0 here = stationary) and the rail
        is physical m/s, capped at ±rail_max_vel_mps.
        """
        cap = self.config.rail_max_vel_mps
        vel_mps = float(np.clip(vel_mps, -cap, cap))
        vel_cmd = np.array([0.0, 0.0, 0.0, vel_mps])
        self._flow_base_client.client.set_target_velocity(
            {"target_velocity": vel_cmd, "frame": "local"}
        ).result()

    def _rail_position_m(self, rail_state: dict[str, Any]) -> float:
        """Current rail height in meters, falling back to motor rad if uncalibrated."""
        pos_linear = rail_state.get("position_linear")
        return float(pos_linear) if pos_linear is not None else float(rail_state["position"])

    def move_to_initial_height(self, events: dict | None = None) -> None:
        """Drive the linear rail to `config.rail_initial_height_m` (absolute, m).

        Called at the start of every recorded episode (alongside
        `reset_odometry`) so each episode begins at the same working height.
        The move is a closed-loop P-controller on rail velocity -- identical in
        spirit to `move_linear_rail_to` in flow_base_client.py -- so it converges
        from either direction and stops on limit switches.

        SAFETY: the rail seeks an absolute height and may descend. The intended
        workflow is that the operator parks the base somewhere the arms can
        safely lower during the preceding reset window (where the arms are also
        homed by `move_to_initial_position`); the rail then descends here when
        the next episode starts. The move aborts cleanly on `events["exit_early"]`
        (the keyboard right-arrow) and on a wall-clock timeout.

        No-op when the rail is disabled or `rail_initial_height_m` is unset.
        """
        if not self.config.with_linear_rail or self.config.rail_initial_height_m is None:
            return
        if self._flow_base_client is None:
            logger.warning("Cannot move rail to initial height: FlowBase client not connected")
            return

        target = float(self.config.rail_initial_height_m)
        kp = self.config.rail_move_kp
        max_accel = self.config.rail_move_max_accel_mps2
        tolerance = self.config.rail_move_tolerance_m
        timeout = self.config.rail_move_timeout_s

        # Match the rail's startup calibration/homing speed: that routine drives
        # the motor at a constant `rail_move_motor_speed_rad_s` (rad/s), which we
        # convert to linear m/s via the meters_per_rad calibration so the height
        # move travels at the same physical speed. Fall back to the static m/s
        # cap if the calibration factor isn't available yet, and always bound by
        # the rail's hard m/s cap.
        if self._meters_per_rad is not None:
            max_speed = abs(self.config.rail_move_motor_speed_rad_s * self._meters_per_rad)
        else:
            max_speed = self.config.rail_move_max_speed_mps
        max_speed = min(max_speed, self.config.rail_max_vel_mps)

        def _should_exit() -> bool:
            return events is not None and events.get("exit_early", False)

        # Slew-rate limit the commanded velocity so the rail eases in/out instead
        # of stepping straight to the speed cap (which is what makes it jerky).
        # `prev_vel` is the last command we sent; we let it change by at most
        # `max_accel * dt` toward the new desired velocity each control step.
        prev_vel = 0.0
        last_cmd_t = time.perf_counter()

        def _ramp_toward(desired: float) -> float:
            nonlocal prev_vel, last_cmd_t
            now = time.perf_counter()
            dt = now - last_cmd_t
            last_cmd_t = now
            max_dv = max_accel * dt
            prev_vel += float(np.clip(desired - prev_vel, -max_dv, max_dv))
            self._send_rail_velocity(prev_vel)
            return prev_vel

        logger.info(
            f"Moving linear rail to initial height {target:.4f} m "
            f"(speed cap {max_speed:.4f} m/s, accel {max_accel:.4f} m/s^2)"
        )
        start_t = time.perf_counter()
        try:
            while not _should_exit():
                rail_state = self._flow_base_client.get_linear_rail_state()
                pos = self._rail_position_m(rail_state)
                error = target - pos
                if abs(error) < tolerance:
                    break
                if rail_state.get("upper_limit_triggered") and error > 0:
                    logger.warning("Rail hit upper limit during height move; stopping.")
                    break
                if rail_state.get("lower_limit_triggered") and error < 0:
                    logger.warning("Rail hit lower limit during height move; stopping.")
                    break
                if time.perf_counter() - start_t > timeout:
                    logger.warning(
                        f"Rail height move timed out after {timeout:.0f}s at {pos:.4f} m "
                        f"(target {target:.4f} m); continuing."
                    )
                    break
                # P-controller velocity, speed-capped, then acceleration-limited.
                desired = float(np.clip(kp * error, -max_speed, max_speed))
                _ramp_toward(desired)
                time.sleep(0.05)
        finally:
            # Smoothly ramp the rail down to a stop rather than slamming the
            # velocity to zero (the abrupt stop is itself a source of jerk). The
            # ramp is bounded so a comms failure can't trap us here, and we
            # always issue a final hard zero so the rail can't keep creeping.
            try:
                ramp_start = time.perf_counter()
                while abs(prev_vel) > 1e-4 and time.perf_counter() - ramp_start < 2.0:
                    _ramp_toward(0.0)
                    time.sleep(0.05)
                self._send_rail_velocity(0.0)
            except Exception as e:
                logger.warning(f"Failed to stop rail after height move: {e}")
            time.sleep(0.2)

        # Consume the exit flag so an abort here doesn't leak into the episode.
        if events is not None and events.get("exit_early", False):
            events["exit_early"] = False

    def _initial_arm_target(self, dofs: int) -> np.ndarray:
        """Home joint vector for one arm: all joints at 0.0, gripper open (1.0).

        Mirrors the hardcoded home pose used by the async-inference
        `RobotClient`. For a 7-DOF arm (6 joints + gripper) the last entry is
        the gripper.
        """
        target = np.zeros(dofs, dtype=float)
        if dofs == 7:
            target[-1] = 1.0
        return target

    def move_to_initial_position(
        self,
        events: dict | None = None,
        num_steps: int = 100,
        step_sleep: float = 0.1,
    ) -> None:
        """Slowly drive both arms back to the home position.

        Called at the start of the reset window so the arms return to a known
        home pose instead of freezing wherever the episode ended (often
        mid-air). Logic matches the async-inference
        `RobotClient._slow_move_to_position`: linear interpolation over
        `num_steps` discrete steps with a fixed `step_sleep` between them
        (default 100 x 0.1s = ~10s). After this the caller resumes the normal
        teleop `record_loop` reset so the operator can reset the environment.
        """

        def _should_exit() -> bool:
            return events is not None and events.get("exit_early", False)

        left_target = self._initial_arm_target(self._left_dofs)
        right_target = self._initial_arm_target(self._right_dofs)

        left_start = np.array(self.left_arm.get_joint_pos(), dtype=float)
        right_start = np.array(self.right_arm.get_joint_pos(), dtype=float)

        # Smooth linear interpolation to the home pose.
        for i in range(num_steps):
            if _should_exit():
                break
            blend = i / num_steps  # 0 -> ~1
            left_cmd = left_target * blend + left_start * (1.0 - blend)
            right_cmd = right_target * blend + right_start * (1.0 - blend)
            self.left_arm.command_joint_pos(left_cmd)
            self.right_arm.command_joint_pos(right_cmd)
            time.sleep(step_sleep)

        # Final exact target so we settle precisely at home.
        if not _should_exit():
            self.left_arm.command_joint_pos(left_target)
            self.right_arm.command_joint_pos(right_target)

        # Consume the exit flag so it doesn't leak into the following reset loop.
        if events is not None and events.get("exit_early", False):
            events["exit_early"] = False

    # ------------------------------------------------------------------
    # Teleop helpers
    # ------------------------------------------------------------------

    def teleop_action_from_obs(self, obs: dict[str, Any]) -> dict[str, float]:
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

        # Tear each resource down independently so a single stuck close (e.g. a
        # wedged portal client or an unreachable FlowBase) can't block the rest
        # of shutdown and leave the process hanging with no output.
        for name, closer in (
            ("left arm", self.left_arm.disconnect),
            ("right arm", self.right_arm.disconnect),
        ):
            try:
                closer()
            except Exception as e:
                logger.warning(f"Error disconnecting {name}: {e}")

        if self._flow_base_client is not None:
            try:
                self._flow_base_client.close()
            except Exception as e:
                logger.warning(f"Error closing FlowBase client: {e}")
            self._flow_base_client = None

        for cam_key, cam in self.cameras.items():
            try:
                cam.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting camera {cam_key}: {e}")

        logger.info("Disconnected from Linear Bot")
