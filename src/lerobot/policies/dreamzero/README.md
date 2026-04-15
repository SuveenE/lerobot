# DreamZero Policy (External Server Mode)

This policy integrates [DreamZero](https://github.com/dreamzero0/dreamzero) into lerobot's async inference system by delegating inference to an external DreamZero distributed WebSocket server.

The model runs on a multi-GPU machine via `socket_test_optimized_AR.py` (requires 2+ GPUs with torch.distributed). Lerobot's `PolicyServer` acts as a thin WebSocket proxy — it receives observations from the `RobotClient`, forwards them to the DreamZero server, and returns the predicted actions. No heavy dependencies (groot, wan, deepspeed, torch.distributed) are needed on the lerobot side.

## Architecture

```
Robot (cameras + arms)
  |
  |  gRPC
  v
PolicyServer  (VM, lerobot -- lightweight WebSocket proxy)
  |
  |  WebSocket ws://localhost:5000 (msgpack)
  v
socket_test_optimized_AR.py  (VM, dreamzero -- multi-GPU distributed inference)
  |
  |  torch.distributed (NCCL)
  v
GPU 0, GPU 1, ...
```

The `PolicyServer` and the DreamZero server run on the **same VM**. The `RobotClient` runs on the robot machine and connects to the VM over gRPC.

## Prerequisites

### VM (GPU machine)

- **Hardware**: 2+ GPUs (tested on H100, GB200). DreamZero is a 14B parameter model and requires multi-GPU inference.
- **CUDA**: 12.9+
- **Python**: 3.11

- Clone and install DreamZero:
  ```bash
  git clone https://github.com/dreamzero0/dreamzero.git
  cd dreamzero
  pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129
  MAX_JOBS=8 pip install --no-build-isolation flash-attn
  ```

- Download the base model weights (Wan2.1-I2V-14B-480P):
  ```bash
  hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
  hf download google/umt5-xxl --local-dir ./checkpoints/umt5-xxl
  ```

- A fine-tuned DreamZero checkpoint, either:
  - The pretrained DROID checkpoint: `hf download GEAR-Dreams/DreamZero-DROID --local-dir ./checkpoints/DreamZero-DROID`
  - A YAM fine-tuned checkpoint (e.g. `swang23/dreamzero_yam_20260413_2000`)
  - Or your own fine-tuned checkpoint

### Robot machine

- lerobot installed from this branch:
  ```bash
  pip install -e ".[all]"
  pip install websockets openpi-client
  ```

## Step 1: Download checkpoint (VM)

### Option A: DROID checkpoint (zero-shot)

```bash
cd ~/dreamzero
hf download GEAR-Dreams/DreamZero-DROID --local-dir ./checkpoints/DreamZero-DROID
```

### Option B: YAM fine-tuned checkpoint

```bash
cd ~/dreamzero
hf download swang23/dreamzero_yam_20260413_2000 --local-dir ./checkpoints/dreamzero_yam
```

For fine-tuning on a new embodiment, see [Adding a New Embodiment to DreamZero](https://github.com/SuveenE/dreamzero/blob/main/docs/DATASET_TO_GEAR_AND_TRAIN.md).

## Step 2: Start the DreamZero server (VM, terminal 1)

The DreamZero server uses `torch.distributed` across multiple GPUs:

```bash
cd ~/dreamzero

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py \
  --port 5000 \
  --enable-dit-cache \
  --model-path ./checkpoints/dreamzero_yam
```

The first few inferences will take a few minutes to warm up (torch.compile). After warmup, inference takes ~0.6s on GB200 and ~3s on H100.

### Command-line arguments

| Argument | Default | Description |
|---|---|---|
| `--port` | `8000` | WebSocket server port |
| `--model-path` | `./checkpoints/dreamzero` | Path to checkpoint directory |
| `--enable-dit-cache` | `false` | Enable DiT caching for faster inference (recommended) |
| `--max-chunk-size` | from config | Override max_chunk_size for inference |
| `--timeout-seconds` | `50000` | Server timeout in seconds |

### Verify the server is running

In a separate terminal:
```bash
python test_client_AR.py --port 5000
```

## Step 3: Create lerobot config directory (VM)

Create a directory with a `lerobot_config.json` that tells lerobot how to communicate with the DreamZero server.

```bash
mkdir -p /home/suveen/dreamzero-lerobot
```

Create `/home/suveen/dreamzero-lerobot/lerobot_config.json`:

```json
{
    "server_url": "ws://localhost:5000",
    "server_image_key_map": {
        "top": "observation/exterior_image_0_left",
        "left": "observation/exterior_image_1_left",
        "right": "observation/wrist_image_left"
    },
    "server_state_key_map": {
        "observation/left_joint_pos": [0, 6],
        "observation/left_gripper_pos": [6, 7],
        "observation/right_joint_pos": [7, 13],
        "observation/right_gripper_pos": [13, 14]
    },
    "chunk_size": 24,
    "n_action_steps": 24,
    "action_dim": 14,
    "state_dim": 14
}
```

> If the defaults in `DreamZeroConfig` already match your setup, this file is optional.

### Config fields explained

| Field | Description |
|---|---|
| `server_url` | WebSocket URL of the DreamZero server |
| `server_image_key_map` | Maps robot camera names (lerobot) to DreamZero observation keys |
| `server_state_key_map` | Maps DreamZero state key names to `[start, end]` index ranges in lerobot's packed `observation.state` vector |
| `chunk_size` | Number of action steps predicted per inference (DreamZero default: 24) |
| `n_action_steps` | Number of action steps to execute before re-querying |
| `action_dim` | Total action dimensions (YAM bimanual: 6+1+6+1 = 14) |
| `state_dim` | Total state dimensions (YAM bimanual: 6+1+6+1 = 14) |

### YAM state/action layout

For the bi_yam_follower robot, the packed state and action vectors have this layout:

| Index range | Key | Description |
|---|---|---|
| `[0, 6)` | `left_joint_pos` | Left arm joint positions (6 DOF) |
| `[6, 7)` | `left_gripper_pos` | Left gripper position (1 DOF) |
| `[7, 13)` | `right_joint_pos` | Right arm joint positions (6 DOF) |
| `[13, 14)` | `right_gripper_pos` | Right gripper position (1 DOF) |

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
  --task "Fold the towel and place it on the side." \
  --policy_type dreamzero \
  --pretrained_name_or_path /home/suveen/dreamzero-lerobot \
  --policy_device cpu \
  --actions_per_chunk 24 \
  --chunk_size_threshold 0.5 \
  --aggregate_fn_name weighted_average \
  --debug_visualize_queue_size True \
  --dataset.enabled true \
  --dataset.num_episodes 5 \
  --dataset.repo_id cortexairobot/eval_dreamzero \
  --dataset.push_to_hub true \
  --dataset.max_episode_seconds 360 \
  --dataset.reset_time_s 15 \
  --dataset.resume false
```

Notes:
- `--policy_device cpu` -- no GPU needed on the policy server side, it's just a WebSocket proxy.
- `--pretrained_name_or_path` must point to the config directory on the **VM** (it gets sent to the PolicyServer via gRPC).
- Camera keys (`top`, `left`, `right`) must match the keys in `server_image_key_map` in `lerobot_config.json`.
- `--actions_per_chunk 24` must match the `chunk_size` / `action_horizon` from DreamZero training.

## Camera key mapping

The `server_image_key_map` translates lerobot camera names to the observation keys that the DreamZero server expects:

| Robot camera | lerobot key | DreamZero server key | Role |
|---|---|---|---|
| Top/overhead | `top` | `observation/exterior_image_0_left` | External camera 1 |
| Left wrist | `left` | `observation/exterior_image_1_left` | External camera 2 |
| Right wrist | `right` | `observation/wrist_image_left` | Wrist camera |

The mapping must match how the DreamZero server wrapper expects observations. For DROID, these are the standard roboarena keys. For a custom embodiment, adjust accordingly.

## State key mapping

The `server_state_key_map` splits the packed `observation.state` vector into individual named state keys for the DreamZero server. Each entry maps a server-side key name to `[start_index, end_index]` in the packed vector.

For DROID (single arm):
```json
{
    "observation/joint_position": [0, 7],
    "observation/gripper_position": [7, 8]
}
```

For YAM (bimanual):
```json
{
    "observation/left_joint_pos": [0, 6],
    "observation/left_gripper_pos": [6, 7],
    "observation/right_joint_pos": [7, 13],
    "observation/right_gripper_pos": [13, 14]
}
```

## Troubleshooting

### "Cannot connect to DreamZero server"
The DreamZero WebSocket server isn't running or is on a different port. Make sure `socket_test_optimized_AR.py` has started and printed its ready message. Check that `server_url` in `lerobot_config.json` matches.

### Slow first inference
This is expected. DreamZero uses `torch.compile` which takes several minutes to warm up on the first few inferences. Subsequent inferences will be much faster (~0.6s on GB200, ~3s on H100).

### Wrong action dimensions
Make sure `action_dim` and `state_dim` in `lerobot_config.json` match your embodiment:
- DROID: `action_dim=8` (7 joint + 1 gripper), `state_dim=8`
- YAM bimanual: `action_dim=14` (6+1+6+1), `state_dim=14`

Also verify that `server_state_key_map` index ranges add up to `state_dim`.

### Connection drops mid-episode
The policy will automatically attempt to reconnect on the next inference call. If the DreamZero server crashed, restart it and the policy will reconnect.

### CUDA out of memory on DreamZero server
DreamZero-14B requires at least 2 GPUs. Try increasing `--nproc_per_node` or using GPUs with more VRAM. For lower VRAM, consider the Wan2.2-TI2V-5B backbone (see DreamZero docs).
