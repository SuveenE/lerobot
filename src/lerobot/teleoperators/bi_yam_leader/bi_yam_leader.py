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
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from threading import Event, Lock, Thread

import numpy as np
import portal

from ..teleoperator import Teleoperator
from .config_bi_yam_leader import BiYamLeaderConfig

logger = logging.getLogger(__name__)


class YamLeaderClient:
    """Client interface for a single Yam leader arm using the portal RPC framework."""

    def __init__(self, port: int, host: str = "localhost"):
        """
        Initialize the Yam leader arm client.

        Args:
            port: Server port for the leader arm
            host: Server host address
        """
        self.port = port
        self.host = host
        self._client = None

    def connect(self):
        """Connect to the leader arm server."""
        logger.info(f"Connecting to Yam leader arm server at {self.host}:{self.port}")
        self._client = portal.Client(f"{self.host}:{self.port}")
        logger.info(f"Successfully connected to Yam leader arm server at {self.host}:{self.port}")

    def disconnect(self):
        """Disconnect from the leader arm server."""
        if self._client is not None:
            logger.info(f"Disconnecting from Yam leader arm server at {self.host}:{self.port}")
            self._client = None

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected."""
        return self._client is not None

    def num_dofs(self) -> int:
        """Get the number of degrees of freedom."""
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.num_dofs().result()

    def get_joint_pos(self) -> np.ndarray:
        """Get current joint positions from the leader arm."""
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.get_joint_pos().result()

    def get_observations(self) -> dict[str, np.ndarray]:
        """Get current observations including joint positions, velocities, etc."""
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.get_observations().result()

    @staticmethod
    def gripper_from_encoder_obs(obs: dict) -> float:
        """
        Derive gripper state from teaching handle encoder button in an observation dict.
        Returns a value between 0 (closed) and 1 (open).
        Falls back to 1.0 (open) if not available.
        """
        try:
            if "io_inputs" in obs:
                # Button pressed = closed gripper (0), not pressed = open (1)
                return 0.0 if obs["io_inputs"][0] > 0.5 else 1.0
            return 1.0
        except Exception:
            return 1.0

    def get_gripper_from_encoder(self, obs: dict | None = None) -> float:
        """
        Try to get gripper state from teaching handle encoder button.
        Returns a value between 0 (closed) and 1 (open).
        Falls back to 1.0 (open) if not available.

        Args:
            obs: Optional pre-fetched observations. When provided, avoids an extra RPC call.
        """
        if obs is not None:
            return self.gripper_from_encoder_obs(obs)

        if self._client is None:
            raise RuntimeError("Client not connected")
        try:
            return self.gripper_from_encoder_obs(self._client.get_observations().result())
        except Exception:
            return 1.0


class BiYamLeader(Teleoperator):
    """
    Bimanual Yam Arms leader (teleoperator) using the i2rt library.

    This teleoperator reads joint positions from two Yam leader arms (with teaching handles)
    and provides them as actions for the follower robot.

    Expected setup:
    - Two Yam leader arms connected via CAN interfaces with teaching handles
    - Server processes running for each leader arm in read-only mode
    - Left leader arm server on port 5002 (default)
    - Right leader arm server on port 5001 (default)

    Note: You'll need to run separate server processes for the leader arms.
    You can modify the i2rt minimum_gello.py script to create read-only
    servers that just expose the leader arm state without trying to control
    a follower.
    """

    config_class = BiYamLeaderConfig
    name = "bi_yam_leader"

    def __init__(self, config: BiYamLeaderConfig):
        super().__init__(config)
        self.config = config

        # Create clients for left and right leader arms
        self.left_arm = YamLeaderClient(port=config.left_arm_port, host=config.server_host)
        self.right_arm = YamLeaderClient(port=config.right_arm_port, host=config.server_host)

        # Store number of DOFs (will be set after connection)
        self._left_dofs = None
        self._right_dofs = None
        self._io_executor: ThreadPoolExecutor | None = None
        self._poll_thread: Thread | None = None
        self._stop_event: Event | None = None
        self._action_lock = Lock()
        self._action_ready = Event()
        self._latest_action: dict[str, float] | None = None

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define action features for both arms."""
        if self._left_dofs is None or self._right_dofs is None:
            # Default to 7 DOFs (6 joints + 1 gripper) per arm if not yet connected
            left_dofs = 7
            right_dofs = 7
        else:
            left_dofs = self._left_dofs
            right_dofs = self._right_dofs

        features = {}
        # Left arm joints and gripper
        # Assume last DOF is gripper if we have 7 DOFs
        for i in range(left_dofs):
            if left_dofs == 7 and i == left_dofs - 1:  # Last DOF is gripper
                features["left_gripper.pos"] = float
            else:
                features[f"left_joint_{i}.pos"] = float

        # Right arm joints and gripper
        # Assume last DOF is gripper if we have 7 DOFs
        for i in range(right_dofs):
            if right_dofs == 7 and i == right_dofs - 1:  # Last DOF is gripper
                features["right_gripper.pos"] = float
            else:
                features[f"right_joint_{i}.pos"] = float

        return features

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        """Yam leader arms don't support feedback."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if both leader arms are connected."""
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to both leader arm servers.

        Args:
            calibrate: Not used for Yam arms (kept for API compatibility)
        """
        logger.info("Connecting to bimanual Yam leader arms")

        # Connect to leader arm servers
        self.left_arm.connect()
        self.right_arm.connect()

        # Get number of DOFs from each arm
        self._left_dofs = self.left_arm.num_dofs()
        self._right_dofs = self.right_arm.num_dofs()

        logger.info(f"Left leader arm DOFs: {self._left_dofs}, Right leader arm DOFs: {self._right_dofs}")
        self._io_executor = ThreadPoolExecutor(max_workers=2)

        # Prime cached action before recording starts so get_action() never blocks on first poll.
        first_action = self._fetch_action_from_arms()
        with self._action_lock:
            self._latest_action = first_action
        self._action_ready.set()

        self._stop_event = Event()
        self._poll_thread = Thread(target=self._poll_action_loop, name="bi_yam_leader_poll", daemon=True)
        self._poll_thread.start()
        logger.info("Successfully connected to bimanual Yam leader arms")

    @property
    def is_calibrated(self) -> bool:
        """Yam leader arms don't require calibration in the lerobot sense."""
        return self.is_connected

    def calibrate(self) -> None:
        """Yam leader arms don't require calibration in the lerobot sense."""
        pass

    def configure(self) -> None:
        """Configure the teleoperator (not needed for Yam leader arms)."""
        pass

    def setup_motors(self) -> None:
        """Setup motors (not needed for Yam leader arms)."""
        pass

    def _action_from_arm_obs(self, side: str, arm_obs: dict[str, np.ndarray]) -> dict[str, float]:
        action_dict: dict[str, float] = {}
        joint_pos = arm_obs["joint_pos"]

        has_gripper = "gripper_pos" in arm_obs
        if has_gripper:
            joint_pos = np.concatenate([joint_pos, arm_obs["gripper_pos"]])
        else:
            gripper = YamLeaderClient.gripper_from_encoder_obs(arm_obs)
            joint_pos = np.concatenate([joint_pos, [gripper]])
            has_gripper = True

        for i, pos in enumerate(joint_pos):
            if has_gripper and i == len(joint_pos) - 1:
                action_dict[f"{side}_gripper.pos"] = float(pos)
            else:
                action_dict[f"{side}_joint_{i}.pos"] = float(pos)
        return action_dict

    def _fetch_action_from_arms(self) -> dict[str, float]:
        if self._io_executor is None:
            raise RuntimeError(f"{self} is not connected.")

        left_future = self._io_executor.submit(self.left_arm.get_observations)
        right_future = self._io_executor.submit(self.right_arm.get_observations)
        left_obs = left_future.result()
        right_obs = right_future.result()
        return {
            **self._action_from_arm_obs("left", left_obs),
            **self._action_from_arm_obs("right", right_obs),
        }

    def _poll_action_loop(self) -> None:
        """Background loop: polls remote leader arms so get_action() never blocks on RPC."""
        poll_interval_s = 1.0 / 30.0
        while self._stop_event is not None and not self._stop_event.is_set():
            start = time.perf_counter()
            try:
                action = self._fetch_action_from_arms()
                with self._action_lock:
                    self._latest_action = action
                self._action_ready.set()
            except Exception as e:
                logger.warning(f"{self} background action poll failed: {e}")

            elapsed = time.perf_counter() - start
            remaining = poll_interval_s - elapsed
            if remaining > 0 and self._stop_event is not None:
                self._stop_event.wait(timeout=remaining)

    def get_action(self) -> dict[str, float]:
        """
        Return the latest leader action polled by the background thread.

        Remote RPC latency is kept off the record-loop critical path. The returned
        action may be up to one poll interval stale when the network is slow.

        Returns:
            Dictionary with joint positions for both arms (including gripper)
        """
        if self._io_executor is None:
            raise RuntimeError(f"{self} is not connected.")

        if not self._action_ready.wait(timeout=2.0):
            raise TimeoutError(f"Timed out waiting for first leader action from {self}.")

        with self._action_lock:
            if self._latest_action is None:
                raise RuntimeError(f"Internal error: action event set but no action cached for {self}.")
            return dict(self._latest_action)

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """
        Send feedback to leader arms (not supported for Yam teaching handles).

        Args:
            feedback: Dictionary with feedback values (ignored)
        """
        # Yam teaching handles are passive devices and don't support feedback
        pass

    def disconnect(self) -> None:
        """Disconnect from both leader arms."""
        logger.info("Disconnecting from bimanual Yam leader arms")

        if self._stop_event is not None:
            self._stop_event.set()
        if self._poll_thread is not None and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
        self._poll_thread = None
        self._stop_event = None
        self._action_ready.clear()
        with self._action_lock:
            self._latest_action = None

        self.left_arm.disconnect()
        self.right_arm.disconnect()

        if self._io_executor is not None:
            self._io_executor.shutdown(wait=False, cancel_futures=True)
            self._io_executor = None

        logger.info("Disconnected from bimanual Yam leader arms")

