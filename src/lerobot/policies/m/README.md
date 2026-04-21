# M Policy (External Server Mode)

This policy integrates an external inference server into lerobot's async inference system by delegating inference to a FastAPI-style HTTP server over a simple `POST /act` JSON contract.

The model runs on a GPU VM. Lerobot's `PolicyServer` acts as a thin proxy — it receives observations from the `RobotClient`, forwards them to the external server, and returns the predicted actions. No heavy dependencies (transformers, peft, etc.) are needed on the lerobot side.

## Architecture

```
Robot (cameras + arms)
  │
  │  gRPC
  ▼
PolicyServer  (VM, lerobot — lightweight HTTP proxy)
  │
  │  HTTP POST localhost:8777/act
  ▼
External server (VM — loads model, runs inference on GPU)
  │
  │  JSON response (actions)
  ▼
PolicyServer → gRPC → RobotClient → robot arms
```

The `PolicyServer` and the external server run on the **same VM**. The `RobotClient` runs on the robot machine and connects to the VM over gRPC.

## HTTP contract

The M policy wrapper sends a single `POST {server_url}` request per inference. The body is a raw `json_numpy`-serialized dict with `Content-Type: application/json`:

| Key | Type | Notes |
|---|---|---|
| `instruction` | `str` | Task string from `--task` |
| `state` | `np.ndarray` shape `(proprio_dim,)` | Proprioception (default `(14,)` for bi-arm) |
| `external_cam` | `np.ndarray` shape `(H, W, 3)` uint8 | Primary camera (default: `top`) |
| `wrist_cam` | `np.ndarray` shape `(H, W, 3)` uint8 | Wrist camera (default: `right`, configurable) |
| `timestamp` | `float` | Client-side wall-clock time |

Response:

```json
{"actions": <np.ndarray shape (N, action_dim)>}
```

The wrapper reshapes this into `(1, chunk_size, action_dim)` for the lerobot pipeline.

## Prerequisites

### VM (GPU machine)

- The external inference server (listening on `0.0.0.0:8777/act` by default).
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

## Step 1: Start the external inference server (VM, terminal 1)

Start the server so it listens at `http://localhost:8777/act`. The server is responsible for model loading, image preprocessing, and action (un)normalization. The exact launch command depends on your server implementation.

## Step 2: (Optional) Create a lerobot config directory on the VM

> **You can skip this step entirely if the `MConfig` defaults work for you** (right wrist, `http://localhost:8777/act`, 14-dim state/action, chunk size 50). No model weights, no dataset statistics, nothing at all is loaded locally by the wrapper — the model lives behind the HTTP server.
>
> The `--pretrained_name_or_path` CLI flag is still **required** by lerobot's argument parser (it must be a non-empty string), but the value is only used as an optional lookup path for a `lerobot_config.json` override file. If the path doesn't exist or doesn't contain that file, the defaults are used silently.
>
> **If defaults work, just pass any placeholder string, e.g. `--pretrained_name_or_path m`.**

If you need to override any default (e.g. swap to the left wrist camera, change `server_url`, change dims or chunk size), create a small directory with a `lerobot_config.json`:

```bash
mkdir -p /home/<user>/m-lerobot
```

Then create `/home/<user>/m-lerobot/lerobot_config.json`:

```json
{
    "server_url": "http://localhost:8777/act",
    "server_image_key_map": {
        "top": "external_cam",
        "left": "wrist_cam"
    },
    "primary_image_key": "top",
    "wrist_image_keys": ["left"],
    "num_images_in_input": 2,
    "action_dim": 14,
    "proprio_dim": 14,
    "chunk_size": 50,
    "n_action_steps": 50
}
```

The example above swaps the wrist camera from `right` (default) to `left`. The third robot camera is simply not included in `server_image_key_map` and is dropped on the wrapper side — it's never sent to the server.

### What goes in `--pretrained_name_or_path`?

| Situation | What to pass |
|---|---|
| Defaults work, no config file needed | Any non-empty placeholder, e.g. `m` |
| Custom overrides in a local dir on the VM | Absolute path, e.g. `/home/<user>/m-lerobot` |
| `lerobot_config.json` published to an HF Hub repo | The repo ID, e.g. `<org>/m-config` |

The wrapper never loads or downloads model weights — only an optional `lerobot_config.json`.

## Step 3: Start lerobot PolicyServer (VM, terminal 2)

```bash
python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8000 \
  --fps=30
```

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
  --pretrained_name_or_path m \
  --policy_device cpu \
  --actions_per_chunk 50 \
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
- `--policy_device cpu` — no GPU needed on the policy server side, it's just an HTTP proxy.
- `--pretrained_name_or_path` can be any non-empty placeholder (the example uses `m`) when defaults are fine. Pass an absolute path on the VM only if you need to override defaults via `lerobot_config.json` (see Step 2).
- Camera keys (`top`, `left`, `right`) in `--robot.cameras` must include the keys referenced by `server_image_key_map`.

## Camera key mapping

The `server_image_key_map` translates lerobot camera names to the observation keys the external server expects. Default:

| Robot camera | lerobot key | server key     | Role          |
|--------------|-------------|----------------|---------------|
| Top overhead | `top`       | `external_cam` | Primary image |
| Right wrist  | `right`     | `wrist_cam`    | Wrist camera  |

To use the left wrist camera instead, set `server_image_key_map` in `lerobot_config.json` to `{"top": "external_cam", "left": "wrist_cam"}` and update `wrist_image_keys` to `["left"]`.

## Troubleshooting

### "Could not reach external server"
The external inference server isn't running yet or is on a different port. Check your server logs and make sure `server_url` in `lerobot_config.json` matches.

### "External M server response missing 'actions' key"
The external server responded with a JSON object that doesn't contain an `actions` field. Verify your server implementation returns `{"actions": <np.ndarray>}`.

### Wrong action dimensions
Make sure `action_dim` and `proprio_dim` in `lerobot_config.json` match what the external server was trained for (default: 14 for bi-arm).
