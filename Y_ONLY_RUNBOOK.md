# Linear Bot — Y-only Data Collection Runbook

Step-by-step commands to collect data with the linear rail parked at a
fixed height and the base moving only along its Y axis (sideways,
left/right) via the joystick. Single-host setup (all four arms
connected to the mini-PC).

Architecture in one sentence: the linear rail is driven to a target
height on startup and held there with an active position-hold (the
brake is intentionally **not** engaged on this hardware because the
brake circuit also kills wheel power); the joystick's left analog
stick X axis (sideways) commands base Y velocity, and X/theta/rail
channels are hard-masked to zero; only `base.y` and `base.y.vel` are
written into the dataset.

## Machines involved

| Machine | What runs on it |
|---|---|
| Mini-PC | TAB 1 (CAN reset), TAB 2 (arm servers), TAB 3 (FlowBase controller), TAB 4 (lerobot-record). Cameras, joystick, and all four arms are connected here. |
| Raspberry Pi | GPIO satellite server (brake + limit-switch control). Must be running **before** TAB 3. |

## Prerequisites

- You are on the branch `suveen/x-only-linear-bot` of
  `SuveenE/lerobot`, and the `i2rt` submodule is on the matching
  `suveen/x-only-linear-bot` branch of `SuveenE/i2rt`. (The branch
  name still says `x-only` for historical reasons, but the actual
  feature it ships is Y-only sideways data collection.)
- Python venv `.lerobot` exists on the mini-PC and has `lerobot` +
  `i2rt` installed in editable mode.
- CAN interfaces named `can_linearbot`, `can_follower_r`,
  `can_follower_l`, `can_leader_r`, `can_leader_l` exist on the mini-PC.
- The Raspberry Pi has the `i2rt` repo checked out and reachable on the
  current wifi.

### Fresh checkout from scratch (mini-PC)

```bash
git clone --recursive https://github.com/SuveenE/lerobot.git
cd lerobot
git checkout suveen/x-only-linear-bot
git submodule update --init --recursive
cd i2rt && git checkout suveen/x-only-linear-bot && cd ..
```

### Updating an existing clone (mini-PC)

```bash
cd ~/lerobot
git fetch origin
git checkout suveen/x-only-linear-bot
git pull
git submodule update --init --recursive
(cd i2rt && git checkout suveen/x-only-linear-bot && git pull)
```

## Step-by-step command list

Run the RPi step first (in its own SSH session), then the four tabs on
the mini-PC in the order below.

### RPi — Start the GPIO satellite server

On the Raspberry Pi (SSH in from the mini-PC):

```bash
ssh pi@172.16.0.67   # use the RPi's current IP
cd ~/i2rt            # wherever i2rt is checked out on the RPi
python i2rt/flow_base/gpio_satellite_server.py --port 8765
```

You should see `GPIO Satellite Server starting on port 8765`. Leave
this running for the whole session — it handles the linear rail brake
and the upper/lower limit switches. `flow_base_controller.py --gpio-host
...` will refuse to start if this is not reachable.

Take note of the RPi's current IP (here `172.16.0.67`). Whenever the
wifi changes, re-check the IP and update the `--gpio-host` value in
TAB 3.

### TAB 1 (mini-PC) — Reset all CAN interfaces

```bash
cd ~/lerobot/i2rt
bash scripts/reset_all_can.sh
```

Wait for it to finish. Verify interfaces are up with
`ip link show | grep can`.

### TAB 2 (mini-PC) — Start all four arm servers

```bash
cd ~/lerobot
source .lerobot/bin/activate
python -m lerobot.scripts.setup_bi_yam_servers
```

This launches all four arms on localhost:
- Follower right arm → port 1234
- Follower left arm → port 1235
- Leader right arm → port 5001
- Leader left arm → port 5002

Verify:

```bash
python -m lerobot.scripts.check_linearbot_servers --read
```

### TAB 3 (mini-PC) — Start FlowBase in Y-only mode with rail parked

```bash
cd ~/lerobot
source .lerobot/bin/activate
python i2rt/i2rt/flow_base/flow_base_controller.py \
  --channel can_linearbot \
  --gpio-host 172.16.0.67:8765 \
  --y-only \
  --rail-height 25.0
```

Startup sequence you should see in the logs:
1. Homing starts, rail drives down, `lower_limit_triggered` fires,
   `pos = 0.0`.
2. `Parking linear rail at 25.000 rad ...`
3. P-loop runs; `pos` climbs toward 25.0.
4. `Rail parking complete at pos=25.000 rad (err=...)`.
5. `Rail hold-at-target enabled (target=25.000 rad, kp=...); brake
   stays released so wheels remain powered.` — the rail is held in
   place by a position-hold P-loop, not the brake. Wheel LEDs stay
   green.
6. `--y-only Y input source: analog axis 0 (...)` (or D-pad if your
   gamepad reports a hat).
7. Main loop: `Joystick is at rest, please check joystick`, then the
   usual `frame: local cmd: 0.0 0.0 0.0 rail: 0.0 | a0:... a1:...` line.

Useful knobs:

| Flag | Default | Meaning |
|---|---|---|
| `--rail-height <rad>` | (unset) | Target rail height in motor rad (same unit as the `rail.position` field and the controller's `pos:` log line). Omit to leave the rail at home after auto-homing. |
| `--y-max-vel <m/s>` | `0.25` | Base Y speed at full stick deflection / D-pad press. Lower this for slower sliding. |
| `--y-input {auto,dpad,stick}` | `auto` | Y input source. `auto` picks D-pad when a hat is reported, otherwise an analog axis. |
| `--y-axis-index <int>` | `0` | Joystick axis used when input is the analog stick. Default 0 = left-stick X (sideways). Use 1 for left-stick Y, 2/3 for right stick. Watch the `aN:` values in the status line to identify the right axis on your controller. |
| `--y-axis-invert` | (off) | Invert the analog axis if pushing the stick the "wrong" way drives the base in the unexpected direction. |
| `--y-axis-deadzone <float>` | `0.15` | Deadzone for the analog axis to suppress drift. |
| `--rail-kp <float>` | `6.0` | P-gain for the rail parking and rail position-hold loops. |
| `--rail-tol <rad>` | `0.05` | Position tolerance for "we are there" during initial parking. |
| `--rail-park-timeout <s>` | `20.0` | Hard timeout if parking stalls. |
| `--rail-park-settle <s>` | `0.3` | Time the rail must stay within tolerance before parking is considered complete. |
| `--no-gamepad` | (off) | Skip joystick init entirely. The RPC server still serves odometry and accepts remote velocity commands. Useful for arm-only teleop tests. |

> **Important:** update `--gpio-host` whenever the Raspberry Pi's IP
> changes (for example, joining a different wifi). The value above
> (`172.16.0.67:8765`) is the NUS Enterprise (Member) wifi address.

### TAB 4 (mini-PC) — Record

```bash
cd ~/lerobot
source .lerobot/bin/activate
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
  --teleop.server_host=localhost \
  --teleop.left_arm_port=5002 \
  --teleop.right_arm_port=5001 \
  --dataset.repo_id=HCHoongChing/260326-linear-bot-1 \
  --dataset.num_image_writer_threads_per_camera=2 \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=20 \
  --dataset.single_task="Pick and place the object" \
  --resume=false
```

#### What moves, and when

| Input | Effect |
|---|---|
| Left analog stick **left** | Base moves -Y (sideways) at up to `--y-max-vel`. Released = 0. |
| Left analog stick **right** | Base moves +Y (sideways) at up to `--y-max-vel`. Released = 0. |
| Forward/back stick motion | Ignored. |
| D-pad left/right | If `--y-input dpad` (or auto-detected), commands ±Y at `--y-max-vel`. |
| Rail axis (right stick Y) | Ignored. Rail stays at `--rail-height` via the position-hold loop. |
| `key_left_1` | Resets odometry (zeros `base.y`). |
| Leader arm teaching handles | Drive both follower arms (unchanged behavior). |

#### What gets recorded

Observation keys (per frame):

- `left_joint_0.pos` .. `left_joint_5.pos`, `left_gripper.pos`
- `right_joint_0.pos` .. `right_joint_5.pos`, `right_gripper.pos`
- `base.y` (FlowBase odometry Y translation, meters)
- The three cameras

Action keys (per frame):

- `left_joint_0.pos` .. `left_joint_5.pos`, `left_gripper.pos`
- `right_joint_0.pos` .. `right_joint_5.pos`, `right_gripper.pos`
- `base.y.vel` (m/s)

No `base.x`, `base.theta`, `base.cmd.*`, or `rail.*` fields are
written.

## When recording is done

### Lower the rail back to home

1. Ctrl+C in TAB 3 — the rail position-hold stops and the brake
   engages on the cleanup path, so the rail stays put.
2. Relaunch TAB 3 **without** `--rail-height` and **without**
   `--y-only`:

   ```bash
   python i2rt/i2rt/flow_base/flow_base_controller.py \
     --channel can_linearbot \
     --gpio-host 172.16.0.67:8765
   ```

   Auto-homing drives the rail down until the lower-limit switch
   triggers, leaving it at `pos = 0`. Ctrl+C again to exit once homed.

### Re-park at a different height later

Same procedure — Ctrl+C TAB 3, relaunch with a new `--rail-height`. The
sequence is always: auto-home → park → hold (or lock if not in
`--y-only`).

### Power off

Ctrl+C TAB 4 (record), TAB 3 (FlowBase), TAB 2 (arm servers), and the
RPi GPIO server in any order. The FlowBase cleanup path engages the
rail brake on exit, so Ctrl+C-ing TAB 3 with the rail at height is
safe.

## Troubleshooting

**TAB 3 errors with "Linear rail is enabled but no GPIO backend is
available".**
The GPIO satellite server on the RPi is not reachable. Check:
1. `gpio_satellite_server.py` is actually running on the RPi.
2. `--gpio-host <ip>:8765` on the mini-PC points to the RPi's current
   IP (re-check after any wifi change).
3. `ping <rpi-ip>` from the mini-PC works.

**Rail parking never settles / stalls at the upper limit.**
`--rail-height` is beyond the physical travel. Pick a smaller value.
The controller logs `Rail parking: upper limit hit at pos=...`, stops,
and enters the main loop where it stopped.

**Wheel LEDs are red, base won't move.**
On this rig the rail brake circuit also kills wheel-motor power.
`--y-only` is wired to release the brake after parking and use a
position-hold instead, so wheels should be green. If they're red:
- Confirm you actually passed `--y-only` on the controller command line.
- Check the E-stop.
- Look for a `Rail hold-at-target enabled ...; brake stays released`
  log line on TAB 3 — its absence means parking didn't complete and
  the brake never got released.

**Stick is the wrong axis (base moves forward/back instead of
sideways).**
The default `--y-axis-index 0` assumes left-stick X is sideways. If
your controller swaps that, watch the `a0: a1: a2: a3:` values in the
TAB 3 status line while you push the stick — pick the axis whose value
changes for sideways motion and pass it via `--y-axis-index <N>`. Add
`--y-axis-invert` if pushing left makes the base move +Y.

**`base.y` is huge / keeps accumulating.**
Press `key_left_1` on the joystick in TAB 3 to reset odometry. This
zeros `base.y` at that pose. Do this before starting a recording if
you want the trajectory relative to the start.

**FlowBase is still running but joystick input has no effect.**
A stale remote command may be holding the override. In `--y-only` mode
the controller also zeros the X/theta/rail channels of any remote
command, so the base will stop when the remote goes stale
(~250 ms timeout).

**I changed wifi and the GPIO host moved.**
Update `--gpio-host` in TAB 3 to the new Raspberry Pi IP. The rest of
the rig (arm servers, cameras, record command) is unaffected.

## References

- Detailed field tables and broader Linear Bot docs:
  `src/lerobot/robots/bi_yam_follower/README_LINEAR_BOT.md`
- FlowBase controller source (argparse block):
  `i2rt/i2rt/flow_base/flow_base_controller.py`
- Linear rail controller source:
  `i2rt/i2rt/flow_base/linear_rail_controller.py`
- GPIO satellite server (runs on the RPi):
  `i2rt/i2rt/flow_base/gpio_satellite_server.py`
