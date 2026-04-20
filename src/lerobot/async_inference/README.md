# Async Inference

Run a policy on a remote GPU and drive a real robot from a separate, low-power machine.

`async_inference` splits the inference loop across two processes connected by gRPC:

- **`PolicyServer`** — loads the policy, runs inference, and returns action chunks. Runs on a GPU VM.
- **`RobotClient`** — talks to the robot hardware, streams observations to the server, and executes the action chunks it receives back. Runs on the robot machine.

This lets you use heavy policies (VLAs, diffusion policies) without co-locating a GPU with the robot, and decouples the policy's inference rate from the robot's control rate.

## Architecture

```
 robot machine (CPU)                 GPU VM
 ┌─────────────────┐               ┌────────────────┐
 │  RobotClient    │   gRPC :8000  │  PolicyServer  │
 │                 │ ◄───────────► │                │
 │  cameras ───────┼──► obs        │                │
 │  arms    ◄──────┼── actions     │   policy.      │
 │                 │               │   predict_     │
 │                 │               │   action_chunk │
 └─────────────────┘               └────────────────┘
```

For policies with heavy dependencies (Cosmos Policy, OpenVLA-OFT), `PolicyServer` further proxies over HTTP to an external `deploy.py` FastAPI server running on the same GPU VM. See [policies/cosmos/README.md](../policies/cosmos/README.md) and [policies/openvla_oft/README.md](../policies/openvla_oft/README.md) for that pattern.

## Supported policies

Currently the async inference system accepts these `--policy_type` values (see [`constants.py`](constants.py)):

| policy_type | Notes |
| --- | --- |
| `act` | |
| `diffusion` | |
| `tdmpc` | |
| `vqbet` | |
| `pi0` | |
| `pi05` | |
| `smolvla` | |
| `xvla` | |
| `openvla_oft` | HTTP proxy to `openvla-oft` `deploy.py` — see its README |
| `cosmos` | HTTP proxy to `cosmos_policy` `deploy.py` — see its README |

## Prerequisites

### GPU VM (policy server)

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[all]"
```

For `cosmos` or `openvla_oft`, you'll also need the upstream package's own runtime and `json-numpy`; follow that policy's README.

### Robot machine (robot client)

```bash
pip install -e ".[all]"
```

The robot hardware (e.g. `bi_yam_follower` arms and cameras) must be addressable from this machine.

### Networking

- Open port **8000** (or your chosen `--port`) on the VM.
- The robot machine must be able to reach the VM's `server_address`. If the VM is behind NAT, use its public IP and make sure the security-group / firewall allows the robot's IP.

## Quickstart

Below is a complete `pi05` example. The two blocks are run on the two different machines, usually in parallel terminals.

### Terminal A — on the GPU VM

```bash
python -m lerobot.async_inference.policy_server \
  --host 0.0.0.0 \
  --port 8000 \
  --fps 30
```

Wait for the `PolicyServer started on 0.0.0.0:8000` line. No policy is loaded yet — that happens when the client sends the first `SendPolicyInstructions` RPC.

### Terminal B — on the robot machine

```bash
python -m lerobot.async_inference.robot_client \
  --server_address <VM_IP>:8000 \
  --robot.type bi_yam_follower \
  --robot.left_arm_port  1235 \
  --robot.right_arm_port 1234 \
  --robot.cameras '{
top:   {"type":"intelrealsense","serial_number_or_name":"406122071208","width":640,"height":360,"fps":30},
left:  {"type":"intelrealsense","serial_number_or_name":"335122271899","width":640,"height":360,"fps":30},
right: {"type":"intelrealsense","serial_number_or_name":"323622271967","width":640,"height":360,"fps":30}
}' \
  --task "Pack snacks into the container" \
  --policy_type pi05 \
  --pretrained_name_or_path cortexairobot/pi05_multi_packing_200k \
  --policy_device cuda \
  --actions_per_chunk 30 \
  --chunk_size_threshold 0.5 \
  --aggregate_fn_name weighted_average \
  --fps 30
```

On first connection the client ships the policy spec to the server; the server then downloads the weights (from HF or a local path) and stands up the policy. Subsequent requests just send observations and receive action chunks.

## PolicyServer options

All fields of [`PolicyServerConfig`](configs.py) are exposed via draccus flags.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` | `localhost` | Interface to bind. Use `0.0.0.0` on a VM. |
| `--port` | `8080` | TCP port. Common choice: `8000`. |
| `--fps` | `30` | Target policy throughput (Hz). Set to match your policy's training rate — e.g. **25 Hz for `cosmos`**. |
| `--inference_latency` | `1/fps` | Upper bound on per-request latency budget (seconds). |
| `--obs_queue_timeout` | `2` | How long to wait for a new observation before giving up (seconds). |

## RobotClient options

All fields of [`RobotClientConfig`](configs.py) are exposed via draccus flags.

**Required:**

| Flag | Meaning |
| --- | --- |
| `--server_address` | `HOST:PORT` of the running `PolicyServer`. |
| `--policy_type` | One of the [supported policies](#supported-policies). |
| `--pretrained_name_or_path` | HF repo id or local dir the server will load from. |
| `--actions_per_chunk` | How many actions from each chunk the client will execute before requesting a new one. |
| `--robot.type` + `--robot.*` | Robot backend (e.g. `bi_yam_follower`) and its options. See the robot's own configuration for the full field list. |

**Commonly used:**

| Flag | Default | Meaning |
| --- | --- | --- |
| `--task` | `""` | Language instruction passed to the policy. VLAs treat this as their prompt. |
| `--policy_device` | `cpu` | Device for any client-side policy work. Most async setups keep this on `cpu` since the heavy compute is on the server. |
| `--chunk_size_threshold` | `0.5` | When the remaining-action queue drops below this fraction of `actions_per_chunk`, request a new chunk. `0.0` = strict serial (wait until empty); `0.5` is a good default. |
| `--fps` | `30` | Control-loop rate (Hz). Should match the server's `--fps`. |
| `--aggregate_fn_name` | `weighted_average` | How to blend overlapping actions from consecutive chunks. Options: `weighted_average` (0.3·old + 0.7·new), `latest_only`, `average`, `conservative` (0.7·old + 0.3·new). |
| `--rename_map` | `{}` | Maps `observation.*` keys from the robot to the keys the policy expects (see [Camera key renames](#camera-key-renames)). |
| `--debug_visualize_queue_size` | `false` | After the run, plot the action-queue size over time to diagnose stalls. |
| `--play_sounds` | `true` | Speak events (recording started, reset, etc.). |

## Dataset recording during inference

Enable `--dataset.enabled true` to save every episode the client executes as a LeRobotDataset v3 while the policy is driving. Useful for A/B evals and for bootstrapping follow-up training data.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dataset.enabled` | `false` | Turn recording on. |
| `--dataset.repo_id` | — | Required if enabled. Format: `user/eval_dataset`. |
| `--dataset.root` | `~/.cache/...` | Local directory for the dataset. |
| `--dataset.push_to_hub` | `false` | Upload to the Hub when the session ends. |
| `--dataset.private` | `true` | Make the Hub repo private. |
| `--dataset.use_videos` | `true` | Encode cameras as video (vs PNG frames). |
| `--dataset.num_episodes` | `None` | Stop after N episodes (`None` = until keyboard stop). |
| `--dataset.max_episode_seconds` | `None` | Per-episode time cap. `None` = keyboard-only. |
| `--dataset.reset_time_s` | `60` | Seconds given to reset the scene between episodes. |
| `--dataset.resume` | `false` | Append to an existing local dataset instead of creating a new one. |
| `--dataset.video_encoding_batch_size` | `1` | Batch multiple episodes before running ffmpeg (speeds up short episodes). |

Minimal example that saves 5 episodes and pushes them:

```bash
  --dataset.enabled true \
  --dataset.num_episodes 5 \
  --dataset.repo_id your-org/eval_cosmos_cube_stack \
  --dataset.push_to_hub true \
  --dataset.max_episode_seconds 120 \
  --dataset.reset_time_s 10
```

### Keyboard controls (only when recording is enabled)

| Key (then Enter) | Effect |
| --- | --- |
| `n` | Save the current episode and start a new one. |
| `s` | Save the current episode and stop recording. |
| `b` | Discard the current episode and re-record it. |

During the reset window, the robot smoothly moves back to an `initial_position` hardcoded in [`robot_client.py`](robot_client.py); edit that constant if your home pose is different.

## Camera key renames

Policies are trained expecting specific observation keys (e.g. `observation.images.image`), but your robot might expose them as `observation.images.top`, `observation.images.right`, etc. Use `--rename_map` to bridge the two:

```bash
  --rename_map '{"observation.images.right":"observation.images.image"}'
```

Multiple remappings in one flag are supported (JSON dict).

For `cosmos` and `openvla_oft`, the policy already does the mapping via its own `server_image_key_map` in `lerobot_config.json`, so you generally leave `--rename_map` empty.

## Aggregate functions

When a new action chunk arrives from the server while the client is still executing the previous one, overlapping timesteps are blended by `aggregate_fn`. The registry lives in [`configs.py`](configs.py):

| Name | Formula |
| --- | --- |
| `weighted_average` (default) | `0.3 · old + 0.7 · new` |
| `latest_only` | `new` |
| `average` | `0.5 · old + 0.5 · new` |
| `conservative` | `0.7 · old + 0.3 · new` |

`weighted_average` usually gives the smoothest trajectories; `latest_only` is useful for debugging because it makes the effect of each new chunk visible.

## Tuning cheat-sheet

| Symptom | Try |
| --- | --- |
| Client logs "queue empty, waiting for server" | Increase `--actions_per_chunk`; lower `--chunk_size_threshold` toward 0; check inference latency. |
| Jittery motion | Use `--aggregate_fn_name weighted_average` or `conservative`; verify client + server `--fps` match; verify policy was trained at that rate. |
| Policy drifts / ignores task | Make sure `--task` is non-empty and phrased as the model expects (especially for VLAs — cosmos requires an exact T5-cache key). |
| High network usage | Lower `--actions_per_chunk` to 20–30; `lerobot`'s observations are already JPEG-compressed by default. |
| Server errors on first request | Policy init runs lazily — check the server log, not the client log, for the real traceback. |

## Per-policy walkthroughs

- **Cosmos Policy (bi_yam_follower, 25 Hz)** → [`policies/cosmos/README.md`](../policies/cosmos/README.md)
- **OpenVLA-OFT** → [`policies/openvla_oft/README.md`](../policies/openvla_oft/README.md)

For the built-in policies (`act`, `pi05`, `smolvla`, etc.) the quickstart above is enough; just swap `--policy_type` and `--pretrained_name_or_path`.
