# M Policy (External Server Mode)

This policy integrates an external inference server into lerobot's async inference system by delegating inference to a FastAPI-style HTTPS server over a simple `POST <server_url>` JSON contract. The route name (e.g. `/act`, `/predict`, `/v1/infer`) is entirely up to the server — the wrapper POSTs to the exact URL configured in `MConfig.server_url` without appending or rewriting any path.

The model runs on a remote GPU host (e.g. a tunneled endpoint or a dedicated inference service). Lerobot's `PolicyServer` acts as a thin proxy — it receives observations from the `RobotClient`, forwards them to the external server over HTTPS, and returns the predicted actions. No heavy dependencies (transformers, peft, etc.) are needed on the lerobot side.

> **In a hurry?** Jump to [Quick start (tl;dr)](#quick-start-tldr) for copy-paste commands covering install, config, multi-server launch, and client startup.

## Architecture

```
Robot (cameras + arms)
  │
  │  gRPC
  ▼
PolicyServer  (VM or robot-side — lightweight HTTP proxy)
  │
  │  HTTPS POST {server_url}/act
  ▼
External inference server (remote GPU host — loads model, runs inference)
  │
  │  JSON response (actions)
  ▼
PolicyServer → gRPC → RobotClient → robot arms
```

The `PolicyServer` and the external inference server run on **different hosts**. `RobotClient` talks to `PolicyServer` over gRPC; `PolicyServer` talks to the inference server over HTTPS using the URL configured via `MConfig.server_url`.

## HTTP contract

The M policy wrapper sends a single `POST {server_url}` request per inference. The path is whatever you put in `server_url` (commonly `/act`, but arbitrary). The body is a raw `json_numpy`-serialized dict with `Content-Type: application/json`.

The defaults match the **MolmoAct YAM 3-view contract** (see `molmoact.py`):

| Key | Type | Notes |
|---|---|---|
| `instruction` | `str` | Task string from `--task` |
| `state` | `np.ndarray` shape `(proprio_dim,)` | YAM joint positions — 14 for bimanual, 7 for unimanual |
| `left_cam` | `np.ndarray` shape `(H, W, 3)` uint8 | Left wrist camera (from lerobot key `left`) |
| `top_cam` | `np.ndarray` shape `(H, W, 3)` uint8 | Front / scene camera (from lerobot key `front`) |
| `right_cam` | `np.ndarray` shape `(H, W, 3)` uint8 | Right wrist camera (from lerobot key `right`) |
| `timestamp` | `float` | Client-side wall-clock time |

The server-side key names (`left_cam`, `top_cam`, `right_cam`) are just defaults — everything is configurable through `server_image_key_map`. To target a 2-view MolmoAct2 Franka server that expects `external_cam` / `wrist_cam`, or any other layout, swap the values in the map; the wrapper logic doesn't change.

Response:

```json
{"actions": <np.ndarray shape (N, action_dim)>}
```

The wrapper reshapes this into `(1, chunk_size, action_dim)` for the lerobot pipeline.

## Quick start (tl;dr)

For the common bimanual YAM / MolmoAct case where the dataclass defaults already work (`server_input_size=[360, 640]`, `action_dim/proprio_dim=14`, camera map `left→left_cam`, `front→top_cam`, `right→right_cam`, `chunk_size=30`). Only `server_url` is deployment-specific, and we override it via env var at launch.

```bash
# 0) One-time install
pip install -e ".[all]" json-numpy

# 1) Create an empty config dir (defaults for everything)
mkdir -p ~/m-lerobot && echo '{}' > ~/m-lerobot/lerobot_config.json

# 2) Launch the external inference server(s) on your GPU host.
#    Note each URL, e.g. https://m-a.example.com/act, https://m-b.example.com/act

# 3) Start one lerobot PolicyServer per external endpoint
LEROBOT_M_SERVER_URL=https://m-a.example.com/act \
nohup python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 --port=8000 --fps=30 \
  > policy_server_a.log 2>&1 &

LEROBOT_M_SERVER_URL=https://m-b.example.com/act \
nohup python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 --port=8001 --fps=30 \
  > policy_server_b.log 2>&1 &

# Verify each is up and the env-var override was applied
tail -f policy_server_a.log
grep 'Overriding server_url' policy_server_a.log
pgrep -af lerobot.async_inference.policy_server

# 4) Start the RobotClient on the robot machine, pointing at one PolicyServer
python -m lerobot.async_inference.robot_client \
  --server_address <policy_server_host>:8000 \
  --robot.type bi_yam_follower \
  --robot.left_arm_port 1235 --robot.right_arm_port 1234 \
  --robot.cameras '{
right: {"type": "intelrealsense", "serial_number_or_name": "323622271967", "width": 640, "height": 360, "fps": 30},
left:  {"type": "intelrealsense", "serial_number_or_name": "335122271899", "width": 640, "height": 360, "fps": 30},
top:   {"type": "intelrealsense", "serial_number_or_name": "406122071208", "width": 640, "height": 360, "fps": 30}
}' \
  --task "Fold the towel." \
  --policy_type m \
  --pretrained_name_or_path ~/m-lerobot \
  --policy_device cpu \
  --actions_per_chunk 30 --chunk_size_threshold 0.5

# 5) Stop everything
pkill -f lerobot.async_inference.policy_server
```

To route the client to a different external backend, just change `--server_address` to `:8001` (server B) — no other changes needed.

For the full explanation of each step, non-default configs (swapping the wrist camera, custom `server_input_size`, etc.), and troubleshooting, continue below.

## Prerequisites

### External GPU host

- The external inference server exposing `POST /act` over HTTPS.
- `json-numpy` on the lerobot side:
  ```bash
  pip install json-numpy
  ```

### Robot machine

- lerobot installed from this branch:
  ```bash
  pip install -e ".[all]"
  pip install json-numpy
  ```

## Step 1: Start the external inference server

Start the server on your GPU host so that `POST {server_url}/act` is reachable over HTTPS from the machine running the lerobot `PolicyServer`. The server is responsible for model loading, image preprocessing, and action (un)normalization. The exact launch command depends on your server implementation.

Note the public HTTPS URL (e.g. a reverse-proxy domain or tunneled endpoint) — you'll put it in `server_url` in the next step.

## Step 2: Create a lerobot config directory (on the machine running `PolicyServer`)

Because the inference server is external, **you almost always need to set `server_url` to your actual public HTTPS endpoint**. The `MConfig` default (`https://localhost:8777/act`) is just a placeholder.

> The `--pretrained_name_or_path` CLI flag is required by lerobot's argument parser (it must be a non-empty string), but the value is only used as a lookup path for an optional `lerobot_config.json` override file. No model weights are ever loaded by the wrapper — the model lives entirely behind the HTTPS server.

Create a small directory with a `lerobot_config.json`:

```bash
mkdir -p /home/<user>/m-lerobot
```

Then create `/home/<user>/m-lerobot/lerobot_config.json` (MolmoAct YAM defaults):

```json
{
    "server_url": "https://your-m-server.example.com/act",
    "server_image_key_map": {
        "left":  "left_cam",
        "front": "top_cam",
        "right": "right_cam"
    },
    "primary_image_key": "front",
    "wrist_image_keys": ["left", "right"],
    "num_images_in_input": 3,
    "action_dim": 14,
    "proprio_dim": 14,
    "chunk_size": 30,
    "n_action_steps": 30
}
```

For a **single YAM arm** (7 DoF), set `"action_dim": 7` and `"proprio_dim": 7`. For a **2-view Franka/MolmoAct2-style server** that expects `external_cam` / `wrist_cam`, set:

```json
{
    "server_image_key_map": {"top": "external_cam", "right": "wrist_cam"},
    "primary_image_key": "top",
    "wrist_image_keys": ["right"],
    "num_images_in_input": 2,
    "action_dim": 8,
    "proprio_dim": 8,
    "server_input_size": [180, 320]
}
```

### Image resizing

`server_input_size` (`[height, width]`) has two effects that are easy to confuse:

1. It's the shape declared on `policy_image_features`, which `PolicyServer` uses to **resize every incoming robot frame via bilinear interpolation** in `prepare_raw_observation` — this happens before the wrapper ever sees the observation.
2. It's also the target shape for the wrapper's own (defensive, usually no-op) client-side resize in `_tensor_to_numpy_image`.

To forward images at **native camera resolution** (the MolmoAct YAM behaviour in `molmoact.py`), **set `server_input_size` to match your robot cameras' actual output**, e.g. `[360, 640]` for the standard YAM/DROID RealSense config (default), `[480, 640]` for standard RealSense, or `[720, 1280]` for HD. Because incoming frames will already be at that shape, both resizes are no-ops and bytes are forwarded untouched.

To *force* a smaller resolution — useful for bandwidth-constrained links or when your external server was trained on downsampled frames — set it to whatever your server expects (e.g. `[180, 320]` for a MolmoAct2 Franka server). `PolicyServer` then downsamples for you.

Any robot camera that isn't mentioned as a *key* in `server_image_key_map` is simply dropped on the wrapper side — it's never sent to the server.

### What goes in `--pretrained_name_or_path`?

| Situation | What to pass |
|---|---|
| Local dir with `lerobot_config.json` (recommended, required to set `server_url`) | Absolute path, e.g. `/home/<user>/m-lerobot` |
| `lerobot_config.json` published to an HF Hub repo | The repo ID, e.g. `<org>/m-config` |
| All defaults are acceptable (only useful if you're testing locally) | Any non-empty placeholder, e.g. `m` |

The wrapper never loads or downloads model weights — only an optional `lerobot_config.json`.

## Step 3: Start lerobot PolicyServer

Launch the policy server as a detached background process with `nohup` so it survives the shell session and writes its output to a log file:

```bash
nohup python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8000 \
  --fps=30 \
  > policy_server.log 2>&1 &
```

The command prints the background PID and returns immediately. Tail the log with `tail -f policy_server.log` to confirm it started, and stop it later with `pkill -f 'lerobot.async_inference.policy_server'` (or `kill <PID>`).

### Overriding `server_url` per process (multi-server setups)

You can override `MConfig.server_url` at launch time by setting the `LEROBOT_M_SERVER_URL` environment variable. This is the cleanest way to run **multiple lerobot `PolicyServer` processes that each point at a different external inference endpoint** without having to maintain a separate `lerobot_config.json` per process.

Precedence (highest wins):

1. `LEROBOT_M_SERVER_URL` env var
2. `server_url` in `lerobot_config.json`
3. `MConfig` dataclass default

Example — three lerobot policy servers launched as detached background processes with `nohup`, each proxying to a different external endpoint and writing logs to its own file:

```bash
# Server A -> proxies to external endpoint A (port 8000)
LEROBOT_M_SERVER_URL=https://m-server-a.example.com/act \
nohup python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 --port=8000 --fps=30 \
  > policy_server_a.log 2>&1 &

# Server B -> proxies to external endpoint B (port 8001)
LEROBOT_M_SERVER_URL=https://m-server-b.example.com/act \
nohup python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 --port=8001 --fps=30 \
  > policy_server_b.log 2>&1 &

# Server C -> proxies to external endpoint C (port 8002)
LEROBOT_M_SERVER_URL=https://m-server-c.example.com/act \
nohup python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 --port=8002 --fps=30 \
  > policy_server_c.log 2>&1 &
```

Each command prints the background PID and returns immediately. Useful follow-up commands:

```bash
# Tail any server's output
tail -f policy_server_a.log

# Confirm the env-var override was applied
grep 'Overriding server_url' policy_server_a.log
# -> Overriding server_url from LEROBOT_M_SERVER_URL: 'https://localhost:8777/act' -> 'https://m-server-a.example.com/act'

# List all running policy servers (and capture their PIDs)
pgrep -af 'lerobot.async_inference.policy_server'

# Stop a specific server (by port, by log name, or by PID)
pkill -f 'policy_server.*--port=8000'
# or: kill <PID>
```

The same `LEROBOT_M_SERVER_URL` env var also works with `docker run -e LEROBOT_M_SERVER_URL=...`, systemd unit files, tmux windows, etc.

## Step 4: Start RobotClient (robot machine)

```bash
python -m lerobot.async_inference.robot_client \
  --server_address <VM_IP>:8000 \
  --robot.type bi_yam_follower \
  --robot.left_arm_port 1235 \
  --robot.right_arm_port 1234 \
  --robot.cameras '{
right: {"type": "intelrealsense", "serial_number_or_name": "323622271967", "width": 640, "height": 360, "fps": 30},
left:  {"type": "intelrealsense", "serial_number_or_name": "335122271899", "width": 640, "height": 360, "fps": 30},
top:   {"type": "intelrealsense", "serial_number_or_name": "406122071208", "width": 640, "height": 360, "fps": 30}
}' \
  --task "Fold the towel and place it on the side." \
  --policy_type m \
  --pretrained_name_or_path /home/<user>/m-lerobot \
  --policy_device cpu \
  --actions_per_chunk 30 \
  --chunk_size_threshold 0.5 \
  --aggregate_fn_name weighted_average \
  --debug_visualize_queue_size True \
  --dataset.enabled true \
  --dataset.num_episodes 5 \
  --dataset.repo_id <org>/eval_m_<task> \
  --dataset.push_to_hub true \
  --dataset.max_episode_seconds 360 \
  --dataset.reset_time_s 15 \
  --dataset.resume false
```

Notes:
- `--policy_device cpu` — no GPU needed on the policy server side, it's just an HTTPS proxy.
- `--pretrained_name_or_path` should point to the directory containing `lerobot_config.json` on the machine running `PolicyServer`. You can pass a placeholder string only if the defaults truly match your deployment.
- Camera keys (`top`, `left`, `right`) in `--robot.cameras` must include the keys referenced by `server_image_key_map`.
- `--actions_per_chunk 30` matches the default `chunk_size` in `MConfig`. Change both together if you adjust this.

## Camera key mapping

The `server_image_key_map` translates lerobot camera names to the observation keys the external server expects. Default (MolmoAct YAM 3-view):

| Robot camera        | lerobot key | server key   | Role               |
|---------------------|-------------|--------------|--------------------|
| Front / scene view  | `front`     | `top_cam`    | Primary scene view |
| Left wrist          | `left`      | `left_cam`   | Left wrist camera  |
| Right wrist         | `right`     | `right_cam`  | Right wrist camera |

The lerobot-side keys (`front` / `left` / `right`) must match the keys you pass in `--robot.cameras` on the `RobotClient` — if they don't, the wrapper logs a warning and the server receives no images for those slots. The server-side key names are arbitrary labels agreed with your external inference server.

To switch to a **2-view MolmoAct2-style server**, override in `lerobot_config.json`:

```json
"server_image_key_map": {"top": "external_cam", "right": "wrist_cam"},
"primary_image_key": "top",
"wrist_image_keys": ["right"],
"num_images_in_input": 2
```

## Troubleshooting

### "Could not reach external server"
The external inference server isn't running yet, the HTTPS URL is wrong, or the TLS certificate is untrusted. Check your server logs and make sure `server_url` in `lerobot_config.json` matches the public HTTPS endpoint. If you're using the `LEROBOT_M_SERVER_URL` env var, confirm it's set correctly in the environment where `PolicyServer` was launched (`echo $LEROBOT_M_SERVER_URL`) — it takes precedence over `lerobot_config.json`.

### "External M server response missing 'actions' key"
The external server responded with a JSON object that doesn't contain an `actions` field. Verify your server implementation returns `{"actions": <np.ndarray>}`.

### Wrong action dimensions
Make sure `action_dim` and `proprio_dim` in `lerobot_config.json` match what the external server was trained for (default: 14 for bi-arm).
