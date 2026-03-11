# Linear Bot with LeRobot

This guide explains how to use the Linear Bot bimanual Yam setup with LeRobot for teleoperation and data collection.

## Overview

The Linear Bot setup consists of:
- **2 follower arms** that execute actions
- **2 leader arms** with teaching handles for teleoperation
- **4 CAN interfaces** across one or more machines
- **LeRobot** for teleop, recording, and dataset writing

The setup supports both:
- a **single-host** configuration where all four arms are connected to one computer
- a **split-host** configuration where follower arms and leader arms live on different machines

## Hardware Setup

### Required CAN Interfaces

Use the following CAN interface names:
- `can_follower_r`: right follower arm
- `can_follower_l`: left follower arm
- `can_leader_r`: right leader arm
- `can_leader_l`: left leader arm

For persistent CAN interface naming, see `i2rt/doc/set_persist_id_socket_can.md`.

Verify the interfaces are up before launching any servers:

```bash
ip link show can_follower_r
ip link show can_follower_l
ip link show can_leader_r
ip link show can_leader_l
```

## Software Setup

### 1. Clone the repository

```bash
git clone --recursive https://github.com/SuveenE/lerobot.git
cd lerobot
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2. Install `i2rt`

```bash
cd i2rt
pip install -e .
cd ..
```

### 3. Install LeRobot with Yam support

```bash
pip install -e ".[yam]"
```

This installs the `portal` RPC dependency used by the Yam client/server layer.

## Launch Options

### Option A: Single host

If all leader and follower arms are connected to the same machine, use:

```bash
python -m lerobot.scripts.setup_bi_yam_servers
```

This launches:
- follower right arm on `localhost:1234`
- follower left arm on `localhost:1235`
- leader right arm on `localhost:5001`
- leader left arm on `localhost:5002`

Verify all servers are up:

```bash
python -m lerobot.scripts.check_linearbot_servers --read
```

### Option B: Split hosts

If the follower arms and leader arms are connected to different machines, use the split launchers.

On the follower machine:

```bash
python -m lerobot.scripts.setup_bi_yam_follower_servers
```

On the leader machine:

```bash
python -m lerobot.scripts.setup_bi_yam_leader_servers
```

Verify all servers are up from the leader machine:

```bash
python -m lerobot.scripts.check_linearbot_servers --follower_host <FOLLOWER_HOST_IP> --read
```

This checks TCP connectivity and reads joint positions from all four arms.

This is the recommended setup when:
- follower arms are connected to a Raspberry Pi
- leader arms are connected to a separate PC
- `lerobot-record` runs on either of those machines or on a third machine on the same network

## Recording Data

### Single-host example

```bash
lerobot-record \
  --robot.type=bi_yam_follower \
  --robot.server_host=localhost \
  --robot.left_arm_port=1235 \
  --robot.right_arm_port=1234 \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
    right: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}
  }' \
  --teleop.type=bi_yam_leader \
  --teleop.server_host=localhost \
  --teleop.left_arm_port=5002 \
  --teleop.right_arm_port=5001 \
  --dataset.repo_id=${HF_USER}/linear-bot-demo \
  --dataset.num_episodes=10 \
  --dataset.single_task="Pick and place the object" \
  --display_data=true
```

### Split-host example

If follower servers run on a Raspberry Pi and leader servers run on another PC:

```bash
lerobot-record \
  --robot.type=bi_yam_follower \
  --robot.server_host=<FOLLOWER_HOST_IP> \
  --robot.left_arm_port=1235 \
  --robot.right_arm_port=1234 \
  --robot.cameras='{
    left: {"type": "intelrealsense", "index_or_path": 6, "width": 640, "height": 480, "fps": 30},
    top: {"type": "intelrealsense", "index_or_path": 12, "width": 640, "height": 480, "fps": 30},
    right: {"type": "intelrealsense", "index_or_path": 18, "width": 640, "height": 480, "fps": 30}
  }' \
  --teleop.type=bi_yam_leader \
  --teleop.server_host=<LEADER_HOST_IP> \
  --teleop.left_arm_port=5002 \
  --teleop.right_arm_port=5001 \
  --dataset.repo_id=${HF_USER}/linear-bot-demo \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=20 \
  --dataset.single_task="Pick and place the object" \
  --display_data=true \
  --resume=false
```

## Configuration Reference

### `bi_yam_follower`

- `robot.type`: `bi_yam_follower`
- `robot.server_host`: hostname or IP of the follower server machine
- `robot.left_arm_port`: default `1235`
- `robot.right_arm_port`: default `1234`
- `robot.cameras`: camera config dictionary
- `robot.left_arm_max_relative_target`: optional safety limit
- `robot.right_arm_max_relative_target`: optional safety limit

### `bi_yam_leader`

- `teleop.type`: `bi_yam_leader`
- `teleop.server_host`: hostname or IP of the leader server machine
- `teleop.left_arm_port`: default `5002`
- `teleop.right_arm_port`: default `5001`

## Gripper Control

The teaching handles do not have physical grippers. Instead, the encoder knob is exposed by the enhanced leader server and mapped to follower gripper motion:

- encoder position controls gripper opening
- encoder button state is also exposed through observations
- follower grippers mirror the leader-side encoder state in real time

## Architecture

```text
Leader arms -> leader servers -> LeRobot teleoperator -> LeRobot robot -> follower servers -> follower arms
```

Each server process runs the Yam low-level stack through the `portal` RPC layer. In this integration, the server processes run the underlying Yam arm in `follower` mode to provide gravity compensation and expose arm state over RPC.

## Troubleshooting

### Missing CAN interfaces

```bash
ip link show | grep can
sudo ip link set can_follower_r up
sudo ip link set can_follower_l up
sudo ip link set can_leader_r up
sudo ip link set can_leader_l up
```

### Ports already in use

```bash
lsof -ti:1234 | xargs kill -9
lsof -ti:1235 | xargs kill -9
lsof -ti:5001 | xargs kill -9
lsof -ti:5002 | xargs kill -9
```

### Check server connectivity

Run the check script from the leader machine to verify all servers are reachable and read current joint positions:

```bash
# Leader servers only
python -m lerobot.scripts.check_linearbot_servers --read

# All four servers (leaders + followers)
python -m lerobot.scripts.check_linearbot_servers --follower_host <FOLLOWER_HOST_IP> --read
```

### Connection failures

Check the following:
1. The appropriate launcher script is running on each machine.
2. The hostnames or IPs passed to `robot.server_host` and `teleop.server_host` are correct.
3. Ports `1234`, `1235`, `5001`, and `5002` are reachable across the network.
4. All machines are on the same LAN or otherwise have low-latency connectivity.

### Slow control loop

If control frequency drops:
- reduce camera resolution or FPS
- check CPU usage
- avoid running heavy workloads on the same machine as camera capture or Yam RPC servers

## Manual Server Launch

If you want to launch the servers manually instead of using the helper scripts:

```bash
# Follower servers
python -m i2rt.scripts.minimum_gello \
  --can_channel can_follower_r \
  --gripper linear_4310 \
  --mode follower \
  --server_port 1234

python -m i2rt.scripts.minimum_gello \
  --can_channel can_follower_l \
  --gripper linear_4310 \
  --mode follower \
  --server_port 1235

# Leader servers
python -m lerobot.scripts.yam_server_with_encoder \
  --can_channel can_leader_r \
  --gripper yam_teaching_handle \
  --mode follower \
  --server_port 5001

python -m lerobot.scripts.yam_server_with_encoder \
  --can_channel can_leader_l \
  --gripper yam_teaching_handle \
  --mode follower \
  --server_port 5002
```

## References

- `i2rt`: Yam hardware control library
- LeRobot documentation: training, evaluation, and dataset workflows
