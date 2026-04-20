# OpenVLA-OFT Policy (External Server Mode)

This policy integrates [OpenVLA-OFT](https://github.com/moojink/openvla-oft) into lerobot's async inference system by delegating inference to an external `deploy.py` FastAPI server over HTTP.

The model runs on a GPU VM via `deploy.py`. Lerobot's `PolicyServer` acts as a thin proxy — it receives observations from the `RobotClient`, forwards them to `deploy.py`, and returns the predicted actions. No heavy dependencies (prismatic, transformers, peft) are needed on the lerobot side.

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
deploy.py     (VM, openvla-oft — loads model, runs inference on GPU)
  │
  │  JSON response (actions)
  ▼
PolicyServer → gRPC → RobotClient → robot arms
```

The `PolicyServer` and `deploy.py` run on the **same VM**. The `RobotClient` runs on the robot machine and connects to the VM over gRPC.

## Prerequisites

### VM (GPU machine)

- Clone and install openvla-oft:
  ```bash
  git clone https://github.com/moojink/openvla-oft.git
  cd openvla-oft
  pip install -e .
  pip install json-numpy
  ```

- A fine-tuned checkpoint, either:
  - A local directory (e.g. `/home/suveen/checkpoints/openvla-oft-cubestack/`)
  - Or a HuggingFace Hub model cached locally (use the snapshot path)

### Robot machine

- lerobot installed from this branch:
  ```bash
  pip install -e ".[all]"
  pip install json-numpy
  ```

## Step 1: Fix prismatic constants (VM)

OpenVLA-OFT auto-detects robot platform from `sys.argv` and often defaults to LIBERO (ACTION_DIM=7). You must set the correct values for your robot.

Edit `prismatic/vla/constants.py` in the openvla-oft repo:

```python
NUM_ACTIONS_CHUNK = 25    # must match training (ALOHA default: 25)
ACTION_DIM = 14           # 7 per arm × 2 for bimanual
PROPRIO_DIM = 14          # 7 per arm × 2 for bimanual
ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType.BOUNDS  # ALOHA uses BOUNDS
```

## Step 2: Start deploy.py (VM, terminal 1)

```bash
cd ~/openvla-oft

nohup python vla-scripts/deploy.py \
  --pretrained_checkpoint /path/to/your/checkpoint \
  --unnorm_key your_dataset_key \
  --use_l1_regression true \
  --use_proprio true \
  --use_film false \
  --num_images_in_input 3 \
  --center_crop true \
  --host 0.0.0.0 \
  --port 8777 \
  > deploy.log 2>&1 &
```

Monitor with `tail -f deploy.log`. Wait for the uvicorn startup message.

### Concrete example

```bash
nohup python vla-scripts/deploy.py \
  --pretrained_checkpoint /home/suveen/.cache/huggingface/hub/models--swang23--openvla-oft-cubestack-20260406/snapshots/<HASH>/ \
  --unnorm_key cube_stacking_combined \
  --use_l1_regression true \
  --use_proprio true \
  --use_film false \
  --num_images_in_input 3 \
  --center_crop true \
  --host 0.0.0.0 \
  --port 8777 \
  > deploy.log 2>&1 &
```

### Finding `--pretrained_checkpoint`

If your model was downloaded from HuggingFace Hub, the cache path is:
```
~/.cache/huggingface/hub/models--<org>--<model>/snapshots/<hash>/
```

Run `ls` on the snapshots directory to find the hash.

Alternatively, download to a clean local directory:
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('swang23/openvla-oft-cubestack-20260406',
                  local_dir='/home/suveen/checkpoints/openvla-oft-cubestack')
"
```

### Finding `--unnorm_key`

This is the dataset name used during training. Check `dataset_statistics.json` in the checkpoint:
```bash
python -c "
import json, sys
with open('/path/to/checkpoint/dataset_statistics.json') as f:
    print(list(json.load(f).keys()))
"
```

Pick the key that matches your fine-tuning dataset (not base model datasets like `bridge_dataset`).

## Step 3: Create lerobot config directory (VM)

Create a directory that the lerobot `PolicyServer` will use. It needs `dataset_statistics.json` (for dimension detection) and optionally a `lerobot_config.json` for custom settings.

```bash
mkdir -p /home/suveen/openvla-oft-lerobot

# Copy dataset stats from checkpoint
cp /path/to/checkpoint/dataset_statistics.json /home/suveen/openvla-oft-lerobot/
```

Create `/home/suveen/openvla-oft-lerobot/lerobot_config.json`:

```json
{
    "server_url": "http://localhost:8777/act",
    "server_image_key_map": {
        "top": "full_image",
        "left": "wrist_image_left",
        "right": "wrist_image_right"
    },
    "primary_image_key": "top",
    "wrist_image_keys": ["left", "right"],
    "num_images_in_input": 3,
    "chunk_size": 25,
    "n_action_steps": 25,
    "action_dim": 14,
    "proprio_dim": 14
}
```

> If the defaults in `OpenVLAOFTConfig` already match your setup, this file is optional — the defaults will be used.

## Step 4: Start lerobot PolicyServer (VM, terminal 2)

```bash
python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8000 \
  --fps=30
```

## Step 5: Start RobotClient (robot machine)

```bash
python -m lerobot.async_inference.robot_client \
  --server_address <VM_IP>:8000 \
  --robot.type bi_yam_follower \
  --robot.left_arm_port 1235 \
  --robot.right_arm_port 1234 \
  --robot.cameras '{
top: {"type": "intelrealsense", "serial_number_or_name": "406122071208", "width": 640, "height": 360, "fps": 30},
left: {"type": "intelrealsense", "serial_number_or_name": "335122271899", "width": 640, "height": 360, "fps": 30},
right: {"type": "intelrealsense", "serial_number_or_name": "323622271967", "width": 640, "height": 360, "fps": 30}
}' \
  --task "Stack the cubes." \
  --policy_type openvla_oft \
  --pretrained_name_or_path /home/suveen/openvla-oft-lerobot \
  --policy_device cpu \
  --actions_per_chunk 25 \
  --chunk_size_threshold 0.5 \
  --aggregate_fn_name weighted_average \
  --debug_visualize_queue_size True \
  --dataset.enabled true \
  --dataset.num_episodes 5 \
  --dataset.repo_id cortexairobot/eval_openvla_oft_cubestack \
  --dataset.push_to_hub true \
  --dataset.max_episode_seconds 360 \
  --dataset.reset_time_s 15 \
  --dataset.resume false
```

Notes:
- `--policy_device cpu` — no GPU needed on the policy server side, it's just an HTTP proxy.
- `--pretrained_name_or_path` must point to the config directory on the **VM** (it gets sent to the PolicyServer via gRPC).
- Camera keys (`top`, `left`, `right`) must match the `server_image_key_map` in `lerobot_config.json`.

## Camera key mapping

The `server_image_key_map` translates lerobot camera names to the observation keys that `deploy.py` expects:

| Robot camera | lerobot key | deploy.py key       | Role             |
|-------------|-------------|---------------------|------------------|
| Top/overhead | `top`       | `full_image`        | Primary image    |
| Left wrist   | `left`      | `wrist_image_left`  | Wrist camera 1   |
| Right wrist  | `right`     | `wrist_image_right` | Wrist camera 2   |

The order and mapping must match how the model was trained. Adjust if your training used different camera assignments.

## Troubleshooting

### "Using LIBERO constants" warning from deploy.py
You forgot to edit `prismatic/vla/constants.py`. See Step 1.

### "Unsupported HF Hub pretrained checkpoint found!"
You're passing a HuggingFace Hub ID but it's not in the hardcoded map in `openvla_utils.py`. Use a **local path** instead (see Step 2).

### "Could not reach external server"
`deploy.py` isn't running yet or is on a different port. Check `deploy.log`.

### Wrong action dimensions
Make sure `ACTION_DIM`, `PROPRIO_DIM`, and `NUM_ACTIONS_CHUNK` in `prismatic/vla/constants.py` match your training config, **and** that `action_dim`/`proprio_dim` in `lerobot_config.json` match as well.
