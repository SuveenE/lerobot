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


@TeleoperatorConfig.register_subclass("bi_gello_yam_leader")
@dataclass
class BiGelloYamLeaderConfig(TeleoperatorConfig):
    """Configuration for a bimanual GELLO leader driving YAM follower arms.

    Unlike `bi_yam_leader`, which reads two YAM teaching-handle arms over portal
    RPC servers, this teleoperator reads two GELLO leaders directly from their
    U2D2 USB serial devices using the calibrated GELLO Dynamixel logic.

    The defaults below are the calibrated values from the gello_software configs
    `configs/yam_left_hw.yaml` and `configs/yam_right_hw.yaml`. Override any of
    them from the CLI (e.g. `--teleop.left_port=...`).
    """

    # U2D2 serial device for each GELLO leader.
    left_port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAO9WA0-if00-port0"
    right_port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAO9WI5-if00-port0"

    # Dynamixel baudrate for the GELLO leaders.
    baudrate: int = 57600

    # Left GELLO calibration (from configs/yam_left_hw.yaml).
    left_joint_ids: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    left_joint_offsets: list[float] = field(
        default_factory=lambda: [3.14159, 3.14159, 3.14159, 1.5708, 6.28319, 4.71239]
    )
    left_joint_signs: list[float] = field(default_factory=lambda: [1.0, -1.0, -1.0, -1.0, 1.0, 1.0])
    # (gripper_joint_id, open_degrees, closed_degrees)
    left_gripper_config: list[float] = field(default_factory=lambda: [7, 72.2578125, 114.0578125])

    # Right GELLO calibration (from configs/yam_right_hw.yaml).
    right_joint_ids: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    right_joint_offsets: list[float] = field(
        default_factory=lambda: [3.14159, 3.14159, 1.5708, 3.14159, 3.14159, 3.14159]
    )
    right_joint_signs: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, -1.0, 1.0, 1.0])
    # (gripper_joint_id, open_degrees, closed_degrees)
    right_gripper_config: list[float] = field(default_factory=lambda: [7, 70.236328125, 112.036328125])

    # Starting pose used by GELLO to resolve joint-offset wrap-around (6 arm joints
    # + normalized gripper). Matches the gello_software hardware configs.
    start_joints: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
