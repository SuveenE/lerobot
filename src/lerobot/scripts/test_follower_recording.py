#!/usr/bin/env python3
"""
Test follower-only recording: connects to the bi_yam_follower (arms + cameras),
reads observations in a loop, echoes current joint state as "action" (no-op),
and saves everything to a LeRobotDataset.

This bypasses the lerobot-record requirement for a teleoperator or policy,
letting you verify camera feeds and dataset saving with just the follower hardware.

Usage (run on the follower PC):

    python -m lerobot.scripts.test_follower_recording \
      --left_arm_port=1235 \
      --right_arm_port=1234 \
      --cameras='{
        left: {"type": "intelrealsense", "index_or_path": 6, "width": 640, "height": 480, "fps": 30},
        top: {"type": "intelrealsense", "index_or_path": 12, "width": 640, "height": 480, "fps": 30},
        right: {"type": "intelrealsense", "index_or_path": 18, "width": 640, "height": 480, "fps": 30}
      }' \
      --repo_id=test-user/follower-test \
      --num_episodes=1 \
      --episode_time_s=30
"""

import argparse
import json
import logging
import time

import numpy as np

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.robots.bi_yam_follower.bi_yam_follower import BiYamFollower
from lerobot.robots.bi_yam_follower.config_bi_yam_follower import BiYamFollowerConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_cameras_json(raw: str) -> dict[str, dict]:
    """Parse the cameras JSON/YAML-ish string from the CLI (same format as lerobot-record)."""
    normalized = raw.strip()
    if not normalized.startswith("{"):
        normalized = "{" + normalized + "}"
    # Handle bare keys (YAML-style) by quoting them
    import re

    normalized = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', normalized)
    return json.loads(normalized)


def build_camera_configs(cameras_dict: dict[str, dict]) -> dict:
    """Build CameraConfig objects from parsed camera dicts."""
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    type_to_cls = {
        "intelrealsense": RealSenseCameraConfig,
        "opencv": OpenCVCameraConfig,
    }

    configs = {}
    for name, cam_params in cameras_dict.items():
        cam_type = cam_params.pop("type", "intelrealsense")
        cfg_cls = type_to_cls.get(cam_type)
        if cfg_cls is None:
            raise ValueError(f"Unknown camera type '{cam_type}'. Supported: {list(type_to_cls.keys())}")
        configs[name] = cfg_cls(**cam_params)
    return configs


def extract_action_from_obs(obs: dict, action_keys: list[str]) -> dict:
    """Extract joint/gripper position keys from observation to use as echo-back action."""
    return {k: obs[k] for k in action_keys if k in obs}


def main():
    parser = argparse.ArgumentParser(
        description="Test follower-only recording (cameras + actions, no teleop needed)"
    )
    parser.add_argument("--server_host", type=str, default="localhost")
    parser.add_argument("--left_arm_port", type=int, default=1235)
    parser.add_argument("--right_arm_port", type=int, default=1234)
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="Camera config JSON string (same format as lerobot-record --robot.cameras)",
    )
    parser.add_argument("--repo_id", type=str, default="test-user/follower-test")
    parser.add_argument("--single_task", type=str, default="Follower echo-back test")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--episode_time_s", type=float, default=30)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--display_data", action="store_true", help="Show rerun visualization")
    parser.add_argument("--no_dataset", action="store_true", help="Skip dataset saving, just print obs")
    args = parser.parse_args()

    init_logging()

    camera_configs = {}
    if args.cameras:
        cameras_dict = parse_cameras_json(args.cameras)
        camera_configs = build_camera_configs(cameras_dict)

    robot_config = BiYamFollowerConfig(
        left_arm_port=args.left_arm_port,
        right_arm_port=args.right_arm_port,
        server_host=args.server_host,
        cameras=camera_configs,
    )
    robot = BiYamFollower(robot_config)

    logger.info("Connecting to follower robot...")
    robot.connect()
    logger.info("Connected. Robot observation features: %s", list(robot.observation_features.keys()))
    logger.info("Robot action features: %s", list(robot.action_features.keys()))

    action_keys = list(robot.action_features.keys())

    if args.display_data:
        init_rerun(session_name="follower_test")

    _, _, robot_observation_processor = make_default_processors()

    dataset = None
    if not args.no_dataset:
        obs_features = create_initial_features(observation=robot.observation_features)
        act_features = create_initial_features(action=robot.action_features)
        dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=obs_features,
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=act_features,
                use_videos=True,
            ),
        )

        num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
        dataset = LeRobotDataset.create(
            args.repo_id,
            args.fps,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4 * max(num_cameras, 1),
        )
        logger.info("Dataset created: %s", args.repo_id)

    listener, events = init_keyboard_listener()

    try:
        if dataset is not None:
            with VideoEncodingManager(dataset):
                _run_episodes(robot, dataset, events, args, action_keys, robot_observation_processor)
        else:
            _run_episodes(robot, None, events, args, action_keys, robot_observation_processor)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        logger.info("Disconnecting robot...")
        robot.disconnect()
        if listener is not None:
            listener.stop()
        logger.info("Done.")


def _run_episodes(robot, dataset, events, args, action_keys, robot_observation_processor):
    for ep in range(args.num_episodes):
        if events.get("stop_recording"):
            break

        logger.info("=== Episode %d / %d ===", ep + 1, args.num_episodes)
        _record_episode(
            robot=robot,
            dataset=dataset,
            events=events,
            fps=args.fps,
            episode_time_s=args.episode_time_s,
            single_task=args.single_task,
            action_keys=action_keys,
            robot_observation_processor=robot_observation_processor,
            display_data=args.display_data,
        )

        if dataset is not None:
            dataset.save_episode()
            logger.info("Episode %d saved (%d total frames)", ep + 1, len(dataset))


def _record_episode(
    robot,
    dataset,
    events,
    fps,
    episode_time_s,
    single_task,
    action_keys,
    robot_observation_processor,
    display_data,
):
    frame_count = 0
    start_t = time.perf_counter()

    while (time.perf_counter() - start_t) < episode_time_s:
        loop_start = time.perf_counter()

        if events.get("exit_early"):
            events["exit_early"] = False
            break

        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        action = extract_action_from_obs(obs, action_keys)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(observation=obs_processed, action=action)

        frame_count += 1
        if frame_count % (fps * 5) == 0:
            elapsed = time.perf_counter() - start_t
            actual_fps = frame_count / elapsed
            joint_summary = {k: f"{v:.3f}" for k, v in action.items()}
            logger.info(
                "frame=%d  elapsed=%.1fs  fps=%.1f  joints=%s",
                frame_count, elapsed, actual_fps, joint_summary,
            )

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

    elapsed = time.perf_counter() - start_t
    logger.info(
        "Episode done: %d frames in %.1fs (%.1f fps)",
        frame_count, elapsed, frame_count / max(elapsed, 1e-6),
    )


if __name__ == "__main__":
    main()
