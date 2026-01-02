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

"""
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```

Example with dataset recording:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=bi_yam_follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --task="Pick and place cube" \
    --server_address=remote-gpu:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --dataset.enabled=true \
    --dataset.repo_id=user/eval_yam_async \
    --dataset.push_to_hub=true \
    --dataset.max_episode_seconds=30

# Keyboard controls when recording:
#   'n' + Enter: Save current episode and start new one
#   's' + Enter: Save current episode and stop recording
```

Example with reset to home position (for multi-episode recording):
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=bi_yam_follower \
    --robot.left_arm_port=1235 \
    --robot.right_arm_port=1234 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --task="Pick and place cube" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --dataset.enabled=true \
    --dataset.repo_id=user/eval_dataset \
    --dataset.reset_time_s=10 \
    --dataset.initial_position_blend_s=5

# During reset, robot smoothly moves to the hardcoded initial_position
# (defined in RobotClient.__init__) over 5 seconds, then holds there.
# Update self.initial_position in robot_client.py to your desired home position.
```
"""

import contextlib
import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.processor import make_default_processors
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    bi_yam_follower,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, is_headless

from .configs import RobotClientConfig
from .constants import SUPPORTED_ROBOTS
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

        # Dataset recording initialization
        self.dataset: LeRobotDataset | None = None
        self.dataset_features: dict | None = None
        self.keyboard_listener = None
        self.keyboard_events: dict | None = None

        if config.dataset.enabled:
            self._init_dataset_recording()

        # Hardcoded initial/home position for bi_yam robot
        # TODO: Update these values to your desired starting position
        self.initial_position: dict[str, float] = {
            "left_joint_0.pos": 0.0,
            "left_joint_1.pos": 0.0,
            "left_joint_2.pos": 0.0,
            "left_joint_3.pos": 0.0,
            "left_joint_4.pos": 0.0,
            "left_joint_5.pos": 0.0,
            "left_gripper.pos": 1.0,
            "right_joint_0.pos": 0.0,
            "right_joint_1.pos": 0.0,
            "right_joint_2.pos": 0.0,
            "right_joint_3.pos": 0.0,
            "right_joint_4.pos": 0.0,
            "right_joint_5.pos": 0.0,
            "right_gripper.pos": 1.0,
        }

    def _init_dataset_recording(self):
        """Initialize dataset recording for evaluation."""
        self.logger.info("Initializing dataset recording...")

        # Get default processors for feature transformation
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

        # Build dataset features from robot observation and action features
        self.dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(action=self.robot.action_features),
                use_videos=self.config.dataset.use_videos,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=self.robot.observation_features),
                use_videos=self.config.dataset.use_videos,
            ),
        )

        # Determine number of cameras for image writer threads
        num_cameras = len(self.robot.cameras) if hasattr(self.robot, "cameras") else 0
        num_image_writer_threads = self.config.dataset.num_image_writer_threads_per_camera * max(num_cameras, 1)

        if self.config.dataset.resume:
            # Resume recording to an existing dataset
            self.dataset = LeRobotDataset(
                self.config.dataset.repo_id,
                root=self.config.dataset.root,
                batch_encoding_size=self.config.dataset.video_encoding_batch_size,
            )

            if num_cameras > 0:
                self.dataset.start_image_writer(
                    num_processes=self.config.dataset.num_image_writer_processes,
                    num_threads=num_image_writer_threads,
                )

            # Use features from the existing dataset
            self.dataset_features = self.dataset.meta.features

            self.logger.info(f"Resuming dataset at {self.dataset.root} with {self.dataset.num_episodes} existing episodes")
        else:
            # Create a new dataset
            self.dataset = LeRobotDataset.create(
                repo_id=self.config.dataset.repo_id,
                fps=self.config.fps,
                root=self.config.dataset.root,
                robot_type=self.robot.name if hasattr(self.robot, "name") else None,
                features=self.dataset_features,
                use_videos=self.config.dataset.use_videos,
                image_writer_processes=self.config.dataset.num_image_writer_processes,
                image_writer_threads=num_image_writer_threads,
                batch_encoding_size=self.config.dataset.video_encoding_batch_size,
            )

            self.logger.info(f"Dataset created at {self.dataset.root} with features: {list(self.dataset_features.keys())}")

        # Initialize keyboard listener for episode control
        if not is_headless():
            self.keyboard_listener, self.keyboard_events = init_keyboard_listener()
            self.logger.info("Keyboard controls enabled: 'n' = save episode, 's' = stop recording")
        else:
            self.keyboard_events = {
                "exit_early": False,
                "rerecord_episode": False,
                "stop_recording": False,
            }
            self.logger.warning("Headless mode: keyboard controls not available. Use time-based episodes only.")

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client and finalize dataset if recording"""
        self.shutdown_event.set()

        # Finalize dataset recording
        if self.dataset is not None:
            try:
                # Check if there are any unsaved frames in the buffer
                if self.dataset.episode_buffer is not None and self.dataset.episode_buffer.get("size", 0) > 0:
                    self.logger.info("Saving final episode before shutdown...")
                    self._save_current_episode()

                # Encode any remaining episodes that haven't been batch encoded into videos
                if hasattr(self.dataset, "episodes_since_last_encoding") and self.dataset.episodes_since_last_encoding > 0:
                    start_ep = self.dataset.num_episodes - self.dataset.episodes_since_last_encoding
                    end_ep = self.dataset.num_episodes
                    self.logger.info(
                        f"Encoding remaining {self.dataset.episodes_since_last_encoding} episodes "
                        f"(episodes {start_ep} to {end_ep - 1}) into videos..."
                    )
                    self.dataset._batch_save_episode_video(start_ep, end_ep)
                    self.dataset.episodes_since_last_encoding = 0

                # Finalize the dataset (close parquet writers)
                self.logger.info("Finalizing dataset...")
                self.dataset.finalize()

                # Push to hub if configured
                if self.config.dataset.push_to_hub:
                    self.logger.info(f"Pushing dataset to HuggingFace Hub: {self.config.dataset.repo_id}")
                    self.dataset.push_to_hub(
                        tags=self.config.dataset.tags,
                        private=self.config.dataset.private,
                    )
                    self.logger.info("Dataset pushed to Hub successfully")

                self.logger.info(f"Dataset recording complete. Total episodes: {self.dataset.num_episodes}")

            except Exception as e:
                self.logger.error(f"Error finalizing dataset: {e}")

        # Stop keyboard listener if active
        if self.keyboard_listener is not None:
            with contextlib.suppress(Exception):
                self.keyboard_listener.stop()

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def _build_recording_frame(
        self,
        observation: RawObservation,
        action: dict[str, Any],
        task: str,
    ) -> dict:
        """Build a frame dict for dataset recording.

        Args:
            observation: Raw observation from the robot
            action: Action dict that was executed
            task: Task description string

        Returns:
            Frame dict ready for dataset.add_frame()
        """
        # Build observation frame
        obs_frame = build_dataset_frame(self.dataset_features, observation, prefix=OBS_STR)

        # Build action frame
        action_frame = build_dataset_frame(self.dataset_features, action, prefix=ACTION)

        # Combine into a single frame with task
        # Note: timestamp is automatically computed by add_frame() from frame_index / fps
        frame = {
            **obs_frame,
            **action_frame,
            "task": task,
        }

        return frame

    def _record_frame(
        self,
        observation: RawObservation,
        action: dict[str, Any],
        task: str,
    ) -> None:
        """Record a single frame to the dataset.

        Args:
            observation: Raw observation from the robot
            action: Action dict that was executed
            task: Task description string
        """
        if self.dataset is None:
            return

        try:
            frame = self._build_recording_frame(observation, action, task)
            self.dataset.add_frame(frame)
        except Exception as e:
            self.logger.error(f"Error recording frame: {e}")

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        _performed_action = self.robot.send_action(
            self._action_tensor_to_action_dict(timed_action.get_action())
        )
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def _check_episode_end(self, episode_start_time: float) -> tuple[bool, bool]:
        """Check if the current episode should end.

        Args:
            episode_start_time: The time when the current episode started

        Returns:
            Tuple of (should_save_episode, should_stop_recording)
        """
        should_save = False
        should_stop = False

        # Check keyboard events
        if self.keyboard_events is not None:
            if self.keyboard_events.get("exit_early", False):
                should_save = True
                self.keyboard_events["exit_early"] = False
                self.logger.info("Keyboard trigger: Saving episode...")

            if self.keyboard_events.get("stop_recording", False):
                should_save = True
                should_stop = True
                self.logger.info("Keyboard trigger: Stopping recording...")

        # Check time-based trigger
        max_seconds = self.config.dataset.max_episode_seconds
        if max_seconds is not None and self.dataset is not None:
            elapsed = time.perf_counter() - episode_start_time
            if elapsed >= max_seconds:
                should_save = True
                self.logger.info(f"Time trigger: Episode reached {max_seconds}s, saving...")

        return should_save, should_stop

    def _save_current_episode(self) -> None:
        """Save the current episode to the dataset."""
        if self.dataset is None:
            return

        try:
            self.dataset.save_episode()
            self.logger.info(f"Episode {self.dataset.num_episodes - 1} saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving episode: {e}")

    def _clear_action_queue(self) -> None:
        """Clear all pending actions from the queue."""
        with self.action_queue_lock:
            while not self.action_queue.empty():
                try:
                    self.action_queue.get_nowait()
                except Exception:
                    break
        self.logger.debug("Action queue cleared")

    def _sync_to_current_position(self) -> None:
        """Sync internal state to the robot's current physical position.

        This prevents the robot from jumping to the last commanded position
        after manual repositioning.
        """
        # Read current robot position
        current_obs = self.robot.get_observation()

        # Extract position values and send them as the new goal
        # This makes the robot "hold" its current position
        action_dict = {}
        for key in self.robot.action_features:
            if key in current_obs:
                action_dict[key] = current_obs[key]

        if action_dict:
            self.robot.send_action(action_dict)
            self.logger.debug(f"Synced to current position: {action_dict}")

    def _slow_move_to_position(self, target_position: dict[str, float], num_steps: int = 100, step_sleep: float = 0.03) -> None:
        """Slowly move robot from current position to target position.

        Matches i2rt's slow_move logic:
        - Discrete steps (default 100)
        - Fixed sleep between steps (default 0.03s)
        - Linear interpolation: target * (i/num_steps) + start * (1 - i/num_steps)

        Args:
            target_position: Target joint positions
            num_steps: Number of interpolation steps (default 100, like i2rt)
            step_sleep: Sleep duration between steps in seconds (default 0.03s, like i2rt)
        """
        # Get current position as starting point
        current_obs = self.robot.get_observation()
        start_positions = {}
        for key in self.robot.action_features:
            if key in current_obs:
                start_positions[key] = current_obs[key]

        # Interpolate over num_steps (like i2rt's slow_move)
        for i in range(num_steps):
            blend_factor = i / num_steps  # 0 to ~1

            blended_action = {}
            for key in self.robot.action_features:
                if key in start_positions and key in target_position:
                    start_val = start_positions[key]
                    target_val = target_position[key]
                    # Same formula as i2rt: target * (i/100) + start * (1 - i/100)
                    blended_action[key] = target_val * blend_factor + start_val * (1 - blend_factor)
                elif key in target_position:
                    blended_action[key] = target_position[key]
                elif key in start_positions:
                    blended_action[key] = start_positions[key]

            self.robot.send_action(blended_action)
            time.sleep(step_sleep)

            # Check for early exit
            if self.keyboard_events is not None and self.keyboard_events.get("exit_early", False):
                break

        # Final step: send exact target position
        self.robot.send_action(target_position)

    def _run_reset_period_initial_position(self, verbose: bool = False) -> None:
        """Run reset period by moving robot to a fixed initial/home position.

        Uses slow_move logic matching i2rt (100 steps, 0.03s sleep = ~3s move).
        Then holds at initial position for remaining reset time.
        """
        reset_time_s = self.config.dataset.reset_time_s

        if reset_time_s <= 0:
            return

        self.logger.info(f"=== RESET PERIOD ({reset_time_s}s) - MOVING TO HOME POSITION ===")
        self.logger.info("  -> Slow move to initial position (100 steps, 0.1s each = 10s)")
        self.logger.info("  -> Press 'n' + Enter to exit reset early")

        # Clear pending policy actions
        self._clear_action_queue()

        reset_start_time = time.perf_counter()

        # Phase 1: Slow move to initial position (100 steps * 0.1s = 10 seconds)
        self._slow_move_to_position(self.initial_position, num_steps=100, step_sleep=0.1)

        # Phase 2: Hold at initial position for remaining reset time
        while self.running and (time.perf_counter() - reset_start_time) < reset_time_s:
            loop_start = time.perf_counter()

            # Hold at initial position
            self.robot.send_action(self.initial_position)

            # Check if user wants to exit early
            if self.keyboard_events is not None and self.keyboard_events.get("exit_early", False):
                self.keyboard_events["exit_early"] = False
                self.logger.info("Exiting reset period early...")
                break

            # Maintain control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - loop_start)))

        self.logger.info("=== RESET COMPLETE - AT HOME POSITION, POLICY ACTIVE ===")

    def _run_reset_period_manual(self, verbose: bool = False) -> None:
        """Run reset period in manual mode - pause actions and allow manual repositioning.

        Actions are paused so the robot won't fight against manual movement.
        After reset, the robot syncs to its current position to avoid jumping.
        """
        reset_time_s = self.config.dataset.reset_time_s
        if reset_time_s <= 0:
            return

        self.logger.info(
            f"Reset period (manual mode): {reset_time_s}s\n"
            "  -> Robot motors are holding position. You can:\n"
            "     1. Manually move the robot (it will gently resist but you can overpower it)\n"
            "     2. Press 'n' + Enter to exit reset early\n"
            "  -> Position will sync before next episode starts"
        )

        # Clear pending policy actions
        self._clear_action_queue()

        reset_start_time = time.perf_counter()

        while self.running and (time.perf_counter() - reset_start_time) < reset_time_s:
            loop_start = time.perf_counter()

            # In manual mode, we DON'T execute policy actions
            # The robot holds its last position (motors still enabled with low torque)
            # User can manually overpower and reposition

            # Periodically sync to current position so robot doesn't fight back hard
            if int((time.perf_counter() - reset_start_time) * 2) % 2 == 0:  # Every 0.5s
                self._sync_to_current_position()

            # Check if user wants to exit early from reset period
            if self.keyboard_events is not None and self.keyboard_events.get("exit_early", False):
                self.keyboard_events["exit_early"] = False
                self.logger.info("Exiting reset period early...")
                break

            # Maintain control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - loop_start)))

        # Final sync to current position before resuming policy
        self._sync_to_current_position()
        self.logger.info("Reset period complete (manual mode)")

    def _run_reset_period(self, task: str, verbose: bool = False) -> None:
        """Run a reset period where the robot moves to the initial/home position.

        Smoothly moves the robot to the hardcoded initial_position over
        initial_position_blend_s seconds.

        Args:
            task: Task description string
            verbose: Whether to log verbose output
        """
        self._run_reset_period_initial_position(verbose)

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        # Episode tracking for dataset recording
        episode_start_time = time.perf_counter()
        recording_active = self.dataset is not None

        # For consistent recording: track last executed action
        _last_action: dict[str, Any] | None = None

        if recording_active:
            num_episodes_target = self.config.dataset.num_episodes
            if num_episodes_target is not None:
                self.logger.info(f"=== EPISODE {self.dataset.num_episodes} of {num_episodes_target} - POLICY ACTIVE ===")
            else:
                self.logger.info(f"=== EPISODE {self.dataset.num_episodes} - POLICY ACTIVE ===")

        while self.running:
            control_loop_start = time.perf_counter()

            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)
                _last_action = _performed_action  # Track for recording

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)

            """Control loop: (3) Recording frame to dataset (if enabled)

            For smooth recording at consistent FPS:
            - Always capture fresh observation for recording (not just when sending to server)
            - Use the most recently executed action
            - Record every loop iteration once we have valid data
            """
            if recording_active and _last_action is not None:
                # Get fresh observation for recording (independent of server send rate)
                recording_observation = self.robot.get_observation()
                self._record_frame(
                    observation=recording_observation,
                    action=_last_action,
                    task=task,
                )

            """Control loop: (4) Check for episode boundaries (keyboard or time-based)"""
            if recording_active:
                should_save, should_stop = self._check_episode_end(episode_start_time)

                if should_save:
                    self._save_current_episode()

                    # Check if we've reached the target number of episodes
                    num_episodes_target = self.config.dataset.num_episodes
                    if num_episodes_target is not None and self.dataset.num_episodes >= num_episodes_target:
                        self.logger.info(f"Reached target of {num_episodes_target} episodes. Stopping recording.")
                        should_stop = True

                    if should_stop:
                        self.logger.info("Recording complete")
                        break
                    else:
                        # Run reset period to give time to reset the environment
                        self._run_reset_period(task, verbose)

                        # Start new episode
                        episode_start_time = time.perf_counter()
                        _last_action = None  # Reset so we wait for first action of new episode
                        if num_episodes_target is not None:
                            self.logger.info(f"=== EPISODE {self.dataset.num_episodes} of {num_episodes_target} - POLICY ACTIVE ===")
                        else:
                            self.logger.info(f"=== EPISODE {self.dataset.num_episodes} - POLICY ACTIVE ===")

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)

        # Start action receiver thread
        action_receiver_thread.start()

        try:
            # The main thread runs the control loop
            client.control_loop(task=cfg.task)

        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    async_client()  # run the client
