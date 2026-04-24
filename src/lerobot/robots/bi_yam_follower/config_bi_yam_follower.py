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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("bi_yam_follower")
@dataclass
class BiYamFollowerConfig(RobotConfig):
    # Server ports for left and right arm followers
    # These should match the ports in the bimanual_lead_follower.py script
    # Default: 1235 for left arm, 1234 for right arm
    left_arm_port: int = 1235
    right_arm_port: int = 1234

    # Server host (usually localhost for local setup)
    server_host: str = "localhost"

    # Optional: Maximum relative target for safety
    left_arm_max_relative_target: float | dict[str, float] | None = None
    right_arm_max_relative_target: float | dict[str, float] | None = None

    # Cameras (shared between both arms)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # When True, pad observation.state / action with the extra base+rail fields used by the
    # linear-bot datasets so recordings made on a plain bi_yam match that schema. All extra
    # fields are written as 0.0 except rail.position, which takes the value of `rail_position`.
    enable_linear_bot_padding: bool = False

    # Constant value written into observation.state's `rail.position` slot every frame when
    # `enable_linear_bot_padding` is True. Ignored otherwise.
    rail_position: float = 0.0

