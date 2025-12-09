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

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_gello_leader")
@dataclass
class BiGelloLeaderConfig(TeleoperatorConfig):
    # Server ports for left and right GELLO leader arms
    # These should be different from the follower ports
    # Note: You'll need to run separate server processes for the GELLO leader arms
    # that expose their state for reading (similar to i2rt minimum_gello.py)
    left_arm_port: int = 6001
    right_arm_port: int = 6002

    # Server host (usually localhost for local setup)
    server_host: str = "localhost"

    # Joint signs for GELLO to YAM mapping
    # These multiply joint positions to account for motor direction differences
    # Default for YAM: [1, -1, -1, -1, 1, 1] (from GELLO README)
    # Set to None to use 1:1 mapping (no sign changes)
    left_joint_signs: list[float] | None = field(
        default_factory=lambda: [1.0, -1.0, -1.0, -1.0, 1.0, 1.0]
    )
    right_joint_signs: list[float] | None = field(
        default_factory=lambda: [1.0, -1.0, -1.0, -1.0, 1.0, 1.0]
    )

    # Joint offsets for calibration (in radians)
    # These are added to joint positions after applying signs
    # Generate these using GELLO's scripts/generate_yam_config.py or gello_get_offset.py
    # Set to None for no offsets
    left_joint_offsets: list[float] | None = None
    right_joint_offsets: list[float] | None = None

    # Gripper inversion (if GELLO and YAM grippers have opposite conventions)
    invert_gripper: bool = False

