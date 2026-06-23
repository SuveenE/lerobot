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

import math
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("bi_yam_linear_bot")
@dataclass
class BiYamLinearBotConfig(RobotConfig):
    # Yam follower arm server ports
    left_arm_port: int = 1235
    right_arm_port: int = 1234

    # Server host for the Yam arm servers
    arm_server_host: str = "localhost"

    # FlowBase server host (may differ from arm host in a split setup)
    flow_base_host: str = "localhost"

    # Whether the FlowBase has a linear rail lift module
    with_linear_rail: bool = True

    # FlowBase controller velocity limits – must match the values used by the
    # running flow_base_controller so that policy outputs (physical velocities)
    # are correctly normalised to the [-1, 1] range the controller expects.
    # Base limits align with FlowBaseClient DEFAULT_MAX_VEL_{X,Y,THETA}.
    base_max_vel: tuple[float, float, float] = (0.5, 0.5, math.pi / 2)
    # Deprecated / unused: the rail is now commanded end-to-end in physical m/s
    # (the controller converts m/s -> motor rad/s via its meters_per_rad
    # calibration), so this rad/s normalisation scale is no longer applied by the
    # robot. Kept only for backward compatibility with existing configs/CLIs.
    rail_max_vel: float = 7.0
    # Linear rail speed cap in m/s. Bounds both the recorded rail command and the
    # policy rail output so the rail action space stays within ±this, like
    # base_max_vel does for the base.
    #
    # SINGLE SOURCE OF TRUTH: leave this as None (the default) to inherit the cap
    # from the running flow_base_controller (its --rail-max-vel flag, surfaced via
    # the rail state's `max_vel_mps`). That way the rail speed is configured in
    # exactly one place and the gamepad scaling, motor cap, recorded action, and
    # policy-send path all agree. Set an explicit float only to OVERRIDE the
    # controller's value (rarely needed). Falls back to 0.5 if the controller does
    # not report a cap (older controller).
    rail_max_vel_mps: float | None = None

    # Target linear-rail height (meters, in the same calibrated `rail.position`
    # space recorded in the dataset) that the rail is driven to at the start of
    # every recorded episode via `move_to_initial_height()`. None disables the
    # behaviour entirely (rail is left wherever the operator parked it). The
    # move is an absolute closed-loop seek, so it works whether the rail needs
    # to go up or down to reach the target.
    rail_initial_height_m: float | None = None
    # Closed-loop parameters for the start-of-episode rail height move. The
    # commanded velocity is `clip(kp * (target - position), ±max_speed)` in m/s,
    # mirroring the `move_linear_rail_to` controller in flow_base_client.py.
    rail_move_kp: float = 1.0
    # Top speed of the height move is matched to the rail's startup
    # calibration/homing speed: `linear_rail_controller.py` drives the motor at
    # a constant `rail_speed * HOMING_SPEED_RATIO = 14.0 * 0.5 = 7.0 rad/s`. At
    # runtime we convert that to linear m/s via the rail's meters_per_rad
    # calibration so the height move travels at the same physical speed as
    # calibration. `rail_move_max_speed_mps` is only the fallback used when
    # meters_per_rad isn't known yet; the result is always bounded by
    # rail_max_vel_mps.
    rail_move_motor_speed_rad_s: float = 7.0
    rail_move_max_speed_mps: float = 0.5
    rail_move_tolerance_m: float = 0.005
    # Acceleration cap (m/s^2) for the height move: the commanded rail velocity
    # is slew-rate limited to change by at most `max_accel * dt` per control
    # step. This is what keeps the move smooth -- without it the velocity steps
    # straight from 0 to the speed cap (and back to 0), which feels jerky.
    rail_move_max_accel_mps2: float = 0.25
    # Safety timeout (s): give up the height move if it hasn't converged, so a
    # stuck/obstructed rail can't block the recording loop indefinitely.
    rail_move_timeout_s: float = 30.0

    # Optional: Maximum relative target for arm safety
    left_arm_max_relative_target: float | dict[str, float] | None = None
    right_arm_max_relative_target: float | dict[str, float] | None = None

    # When True, also record per-arm motor torques (joint_eff / gripper_eff) as
    # dedicated `observation.left_torques` and `observation.right_torques`
    # columns. Defaults to False so existing recordings / pretrained
    # positions-only policies remain unaffected. Enable via
    # `--robot.record_torques=true` on the lerobot-record CLI.
    record_torques: bool = False

    # Cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
