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
git checkout suveen/linear-bot
git submodule update --init --recursive
```

If you already cloned without submodules or on a different branch:

```bash
git fetch origin
git checkout suveen/linear-bot
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

## Recording with FlowBase and Linear Rail

The `bi_yam_linear_bot` robot type records arm state alongside FlowBase odometry
and linear rail state. The FlowBase server must be running on the same or a
reachable host before starting the recording.

### Start the FlowBase server

On the machine connected to the FlowBase CAN interface:

```bash
python i2rt/i2rt/flow_base/flow_base_controller.py --channel can_flow_base
```

### Record with `bi_yam_linear_bot` (single PC)

If all arms, FlowBase, and cameras are connected to the same machine:

```bash
# Terminal 1 — arm servers
python -m lerobot.scripts.setup_bi_yam_servers

# Terminal 2 — FlowBase server
python i2rt/i2rt/flow_base/flow_base_controller.py --channel can_flow_base

# Terminal 3 — record
lerobot-record \
  --robot.type=bi_yam_linear_bot \
  --robot.arm_server_host=localhost \
  --robot.left_arm_port=1235 \
  --robot.right_arm_port=1234 \
  --robot.flow_base_host=localhost \
  --robot.with_linear_rail=true \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
    right: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}
  }' \
  --teleop.type=bi_yam_leader \
  --teleop.server_host=localhost \
  --teleop.left_arm_port=5002 \
  --teleop.right_arm_port=5001 \
  --dataset.repo_id=${HF_USER}/linear-bot-full \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=20 \
  --dataset.single_task="Pick and place the object" \
  --display_data=true \
  --resume=false
```

Use the joystick connected to the FlowBase server machine to control base
movement and the linear rail while teleoperating the arms with the teaching
handles.

### Record with `bi_yam_linear_bot` (split hosts)

In this setup:
- **Follower PC** — connected to follower arms, FlowBase, cameras, and joystick. Runs `lerobot-record`.
- **Leader PC** — connected to leader arms (teaching handles).

Both machines must be on the same network. Replace `<LEADER_PC_IP>` below
with the actual IP of the leader machine (e.g. `192.168.1.50`).

#### Step 1 — Leader PC: start leader arm servers

```bash
# Terminal 1 on Leader PC
python -m lerobot.scripts.setup_bi_yam_leader_servers
```

This starts:
- Right leader arm on port `5001`
- Left leader arm on port `5002`

#### Step 2 — Follower PC: start follower arm servers

```bash
# Terminal 1 on Follower PC
python -m lerobot.scripts.setup_bi_yam_follower_servers
```

This starts:
- Right follower arm on port `1234`
- Left follower arm on port `1235`

#### Step 3 — Follower PC: start FlowBase server

```bash
# Terminal 2 on Follower PC
python i2rt/i2rt/flow_base/flow_base_controller.py --channel can_flow_base
```

Connect the joystick to this machine to control the base and linear rail.

#### Step 4 — Follower PC: verify all servers

```bash
python -m lerobot.scripts.check_linearbot_servers \
  --leader_host <LEADER_PC_IP> \
  --follower_host localhost \
  --read
```

#### Step 5 — Follower PC: record

```bash
# Terminal 3 on Follower PC
lerobot-record \
  --robot.type=bi_yam_linear_bot \
  --robot.arm_server_host=localhost \
  --robot.left_arm_port=1235 \
  --robot.right_arm_port=1234 \
  --robot.flow_base_host=localhost \
  --robot.with_linear_rail=true \
  --robot.cameras='{
    left: {"type": "intelrealsense", "index_or_path": 6, "width": 640, "height": 480, "fps": 30},
    top: {"type": "intelrealsense", "index_or_path": 12, "width": 640, "height": 480, "fps": 30},
    right: {"type": "intelrealsense", "index_or_path": 18, "width": 640, "height": 480, "fps": 30}
  }' \
  --teleop.type=bi_yam_leader \
  --teleop.server_host=<LEADER_PC_IP> \
  --teleop.left_arm_port=5002 \
  --teleop.right_arm_port=5001 \
  --dataset.repo_id=${HF_USER}/linear-bot-full \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=20 \
  --dataset.single_task="Pick and place the object" \
  --display_data=true \
  --resume=false
```

Since the follower arms, FlowBase, and cameras are all local to the
follower PC, only `--teleop.server_host` needs the remote IP. Everything
else points to `localhost`.

### Y-only data collection (rail parked + base Y axis only)

For tasks where the rail should sit at a fixed height and the base should
only move along its Y axis (sideways, left/right), the controller and the
LeRobot robot type both expose a coordinated "Y-only" mode. The joystick's
left analog stick X axis (or the D-pad) drives base Y velocity; X, theta,
and rail channels are hard-masked to zero. The dataset schema is trimmed
to record only `base.y` (observation) and `base.y.vel` (action) on the
base side.

The rail is held at `--rail-height` with an active position-hold P-loop
rather than the physical brake. On this hardware the brake circuit is
wired into the wheel-motor power loop, so engaging it would kill wheel
power and freeze the base.

See `Y_ONLY_RUNBOOK.md` at the repo root for a fuller end-to-end
runbook (RPi GPIO, CAN reset, four arm servers, recording, troubleshooting).

#### Step A — Start the FlowBase controller in Y-only mode

```bash
python i2rt/i2rt/flow_base/flow_base_controller.py \
  --channel can_linearbot \
  --gpio-host 172.16.0.67:8765 \
  --y-only \
  --rail-height 25.0
```

Startup sequence:
1. Auto-homing drives the rail down to the lower limit (`pos = 0`).
2. The parking loop drives the rail up to `--rail-height` (motor rad,
   same unit as `rail.position` printed by the controller).
3. With `--y-only`, the brake stays released and the rail position-hold
   loop kicks in to hold the height; wheel LEDs stay green.
4. The main loop runs with Y-only input. `X`, `theta`, and rail
   commands are masked to zero on every tick, including any incoming
   remote command.

Useful flags:
- `--rail-height <rad>`: target height in motor rad. Omit to leave the
  rail at home.
- `--y-max-vel <m/s>`: peak Y speed at full stick deflection (default
  `0.25`).
- `--y-input {auto,dpad,stick}`: input source. `auto` picks D-pad if
  the gamepad reports a hat, otherwise falls back to an analog axis.
- `--y-axis-index <int>`: analog axis index used in stick mode (default
  `0` = left-stick X, sideways). Watch the `aN:` values in the status
  line to identify your controller's mapping.
- `--y-axis-invert`, `--y-axis-deadzone <float>`: tune analog input.
- `--rail-park-max-vel <rad/s>`: caps the parking velocity (default
  `0.5`). Pass a low value (e.g. `0.05`) for a very slow crawl up to
  `--rail-height`. The parking timeout is auto-extended so this does
  not cause spurious failures.
- `--rail-kp`, `--rail-tol`, `--rail-park-timeout`, `--rail-park-settle`:
  P-loop tuning for parking.

To lower the rail back later, Ctrl+C the controller and relaunch
without `--rail-height` (and without `--y-only`). Auto-homing always
drives the rail to the lower limit on startup.

#### Step B — Record with `y_only_mode=true`

Add `--robot.y_only_mode=true` to the `lerobot-record` command. Everything
else (cameras, arm servers, repo id) stays the same:

```bash
lerobot-record \
  --robot.type=bi_yam_linear_bot \
  --robot.arm_server_host=localhost \
  --robot.left_arm_port=1235 \
  --robot.right_arm_port=1234 \
  --robot.flow_base_host=localhost \
  --robot.with_linear_rail=true \
  --robot.y_only_mode=true \
  --robot.cameras='{
    left: {"type": "intelrealsense", "serial_number_or_name": "230422272258", "width": 640, "height": 480, "fps": 30},
    top: {"type": "intelrealsense", "serial_number_or_name": "335522072330", "width": 640, "height": 480, "fps": 30},
    right: {"type": "intelrealsense", "serial_number_or_name": "130322271069", "width": 640, "height": 480, "fps": 30}
  }' \
  --teleop.type=bi_yam_leader \
  --teleop.server_host=172.16.0.89 \
  --teleop.left_arm_port=5002 \
  --teleop.right_arm_port=5001 \
  --dataset.repo_id=${HF_USER}/linear-bot-y-only \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=20 \
  --dataset.single_task="Pick and place the object" \
  --resume=false
```

When `y_only_mode=true`, the recorded fields shrink to:

**Observation state**:

| Key | Description |
|---|---|
| `left_joint_0.pos` .. `left_joint_5.pos` | Left arm joint positions |
| `left_gripper.pos` | Left gripper position |
| `right_joint_0.pos` .. `right_joint_5.pos` | Right arm joint positions |
| `right_gripper.pos` | Right gripper position |
| `base.y` | FlowBase odometry Y translation (sideways) |

**Action**:

| Key | Description |
|---|---|
| `left_joint_0.pos` .. `left_joint_5.pos` | Left arm joint targets |
| `left_gripper.pos` | Left gripper target |
| `right_joint_0.pos` .. `right_joint_5.pos` | Right arm joint targets |
| `right_gripper.pos` | Right gripper target |
| `base.y.vel` | Base Y velocity command |

No `base.x`, `base.theta`, `base.cmd.*`, or `rail.*` fields are written.
The robot still queries `base.cmd.y.vel` from the FlowBase internally to
fill in `action.base.y.vel` during teleoperation, but that key is never
serialised to the dataset.

### Recorded fields

The dataset will contain the following observation and action fields in
addition to camera images.

**Observation state** (`observation.state`):

| Key | Description |
|---|---|
| `left_joint_0.pos` .. `left_joint_5.pos` | Left arm joint positions |
| `left_gripper.pos` | Left gripper position |
| `right_joint_0.pos` .. `right_joint_5.pos` | Right arm joint positions |
| `right_gripper.pos` | Right gripper position |
| `base.x` | FlowBase odometry X translation |
| `base.y` | FlowBase odometry Y translation |
| `base.theta` | FlowBase odometry rotation |
| `rail.position` | Linear rail motor position |
| `rail.velocity` | Linear rail motor velocity |
| `rail.upper_limit` | Upper limit switch triggered (0.0 / 1.0) |
| `rail.lower_limit` | Lower limit switch triggered (0.0 / 1.0) |
| `base.cmd.x.vel` | Resolved base X velocity command (joystick or remote) |
| `base.cmd.y.vel` | Resolved base Y velocity command (joystick or remote) |
| `base.cmd.theta.vel` | Resolved base rotation velocity command (joystick or remote) |
| `rail.cmd.vel` | Resolved linear rail velocity command (joystick or remote) |

**Action** (`action`):

| Key | Description |
|---|---|
| `left_joint_0.pos` .. `left_joint_5.pos` | Left arm joint targets |
| `left_gripper.pos` | Left gripper target |
| `right_joint_0.pos` .. `right_joint_5.pos` | Right arm joint targets |
| `right_gripper.pos` | Right gripper target |
| `base.x.vel` | Base X velocity command |
| `base.y.vel` | Base Y velocity command |
| `base.theta.vel` | Base rotation velocity command |
| `rail.vel` | Linear rail velocity command |

Rail fields are omitted when `with_linear_rail=false`.

> **Note on base/rail actions during teleop:** The leader arms only produce
> arm joint targets. Base and rail velocity actions (`base.x.vel`,
> `base.y.vel`, `base.theta.vel`, `rail.vel`) are automatically filled
> from the FlowBase's resolved command observations (`base.cmd.x.vel`,
> etc.) so the dataset captures the actual joystick-driven base movement.
> This allows a trained policy to reproduce both arm and base behavior.

## Configuration Reference

### `bi_yam_linear_bot`

- `robot.type`: `bi_yam_linear_bot`
- `robot.arm_server_host`: hostname or IP of the Yam arm follower servers
- `robot.left_arm_port`: default `1235`
- `robot.right_arm_port`: default `1234`
- `robot.flow_base_host`: hostname or IP of the FlowBase server
- `robot.with_linear_rail`: default `true`
- `robot.cameras`: camera config dictionary
- `robot.left_arm_max_relative_target`: optional safety limit
- `robot.right_arm_max_relative_target`: optional safety limit

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
