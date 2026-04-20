# Cosmos Policy (External Server Mode)

This policy integrates [NVIDIA Cosmos Policy](https://github.com/NVlabs/cosmos-policy) — a 2B-parameter diffusion transformer fine-tuned from `Cosmos-Predict2-2B-Video2World` for bimanual robot manipulation — into lerobot's async inference system by delegating inference to an external `deploy.py` FastAPI server over HTTP.

The Cosmos Policy model runs on a GPU VM via `cosmos_policy/experiments/robot/aloha/deploy.py`. lerobot's `PolicyServer` acts as a thin proxy: it receives observations from the `RobotClient`, forwards them to `deploy.py`, and returns the predicted actions. No heavy dependencies (torch.distributed.checkpoint, imaginaire, cosmos_predict2, t5) are needed on the lerobot side.

The reference, known-good schema for stats / embeddings / config is [`nvidia/Cosmos-Policy-ALOHA-Predict2-2B`](https://huggingface.co/nvidia/Cosmos-Policy-ALOHA-Predict2-2B).

## Architecture

```
Robot (3 RealSense cameras + bimanual YAM arms)
  │
  │  gRPC
  ▼
PolicyServer  (VM, lerobot — lightweight HTTP proxy)
  │
  │  HTTP POST localhost:8777/act
  │  json_numpy-encoded {task_description, primary_image, left_wrist_image,
  │                      right_wrist_image, proprio}
  ▼
deploy.py     (VM, Docker, cosmos_policy — loads 2B DiT, runs diffusion on GPU)
  │
  │  JSON response (actions: (50, 14), future images, value)
  ▼
PolicyServer → gRPC → RobotClient → YAM arms @ 25 Hz
```

Both `PolicyServer` and `deploy.py` run on the **same VM** (typically H100). The `RobotClient` runs on the robot machine and connects to the VM over gRPC.

## Running example

All commands below assume this scenario:

- Fine-tuned model: [`swang23/cosmos_policy_20260415`](https://huggingface.co/swang23/cosmos_policy_20260415) (raw `cosmos_policy.scripts.train` output, FSDP distcp shards)
- Training dataset: [`ai2-cortex/02042026-cube-stacking-combined`](https://huggingface.co/datasets/ai2-cortex/02042026-cube-stacking-combined)
- Task string: `"Stack the cubes."`
- Hardware: `bi_yam_follower` with 3 Intel RealSense cameras (`top`, `left`, `right`)
- Artifact root: `$HOME/cosmos_ckpts/yam_cube_stacking_20260415`

## Background

The HF model repo ships only the raw `torch.distributed.checkpoint` output from `cosmos_policy/scripts/train.py` (FSDP-sharded `model/*.distcp`, plus `optim/`, `scheduler/`, `trainer/`). It does **not** include:

1. `dataset_statistics.json` — required by `deploy.py` for proprio normalization and action unnormalization
2. A T5 text-embeddings pickle — required because `deploy.py` looks up `task_description` in a precomputed cache rather than running T5 at inference time
3. A registered inference-only experiment config — consumed by `deploy.py --config <name>`

We synthesise (1) from the training dataset, generate (2) with a cosmos_policy helper script, and reuse (3) from the fork of `NVlabs/cosmos-policy` that was used at train time.

## Prerequisites

### VM (GPU machine, CUDA 12.8, Python 3.10)

```bash
git clone <your-fork-of-NVlabs-cosmos-policy> cosmos-policy
cd cosmos-policy
# Follow the repo's docker setup:
bash docker/build.sh
bash docker/run.sh
# Inside the container:
pip install json-numpy uvicorn fastapi draccus
```

### Robot machine

```bash
pip install -e ".[all]"
pip install json-numpy
```

## Step 0: Set a shared env var on the VM

```bash
export COSMOS_ROOT=$HOME/cosmos_ckpts/yam_cube_stacking_20260415
mkdir -p "$COSMOS_ROOT"
```

> Avoid `/data` on stock cloud VMs — `hf download --local-dir /data/...` fails with `PermissionError: [Errno 13] Permission denied: '/data'` unless you've `sudo chown`'d it. `$HOME/cosmos_ckpts` always works.

## Step 1: Download the fine-tuned checkpoint

```bash
hf download swang23/cosmos_policy_20260415 --local-dir "$COSMOS_ROOT"
```

Produces `$COSMOS_ROOT/model/__{0..7}_0.distcp` + `.metadata` plus `optim/`, `scheduler/`, `trainer/` (the last three are ignored at inference time). The `model/` directory is the one you pass to `--ckpt_path`.

Note: [`nvidia/Cosmos-Policy-ALOHA-Predict2-2B`](https://huggingface.co/nvidia/Cosmos-Policy-ALOHA-Predict2-2B) ships a single consolidated `.pt` instead — `deploy.py` accepts either format.

## Step 2: Generate `yam_dataset_statistics.json`

Cosmos Policy's `deploy.py` requires a **flat** stats JSON with exactly these 10 keys (schema verified against `nvidia/Cosmos-Policy-ALOHA-Predict2-2B/aloha_dataset_statistics.json`):

```
actions_min, actions_max, actions_mean, actions_std, actions_median,
proprio_min, proprio_max, proprio_mean, proprio_std, proprio_median
```

The training dataset `ai2-cortex/02042026-cube-stacking-combined` is a standard lerobot v3 dataset — its `meta/stats.json` has `min, max, mean, std` but no `median`. We ship a converter that maps these cleanly:

```bash
hf download ai2-cortex/02042026-cube-stacking-combined \
  --repo-type dataset \
  --local-dir "$HOME/cosmos_datasets/02042026-cube-stacking-combined"

python -m lerobot.policies.cosmos.scripts.convert_lerobot_stats \
  --lerobot_dataset_dir "$HOME/cosmos_datasets/02042026-cube-stacking-combined" \
  --out_path             "$COSMOS_ROOT/yam_dataset_statistics.json"
```

By default the converter uses `mean` as a stand-in for `median` (these are very close for joint-position data). For exact medians, pass `--compute_median_from_parquet` to scan the dataset shards directly.

Sanity-check:

```bash
python -c "
import json
d = json.load(open('$COSMOS_ROOT/yam_dataset_statistics.json'))
for k, v in d.items():
    print(f'{k}: dim={len(v)}')
"
```

You should see ten lines, each `dim=14`.

## Step 3: Precompute T5 embeddings for every task string you'll send

`deploy.py` looks up `task_description` as an exact key in a pickle; any unknown string raises. Include every phrasing you'll use at inference time:

```bash
uv run --extra cu128 --group aloha --python 3.10 python \
  -m cosmos_policy.datasets.save_aloha_t5_text_embeddings \
  --task_labels "Stack the cubes." \
  --out_path "$COSMOS_ROOT/yam_t5_embeddings.pkl"
```

## Step 4: Point `deploy.py` at your YAM experiment config

`--config <name>` resolves against Python modules under `cosmos_policy/config/experiment/`. If the config for this training run lives in a fork, clone that fork onto the VM so the name is importable. Either bake `model.ckpt_path=$COSMOS_ROOT/model` into an `__inference_only` variant, or pass it inline via `opts`.

## Step 5: Launch `deploy.py` (VM, Docker container, terminal 1)

```bash
uv run -m cosmos_policy.experiments.robot.aloha.deploy \
  --config <yam_cube_stacking_inference_only_config_name> \
  --ckpt_path "$COSMOS_ROOT/model" \
  --dataset_stats_path "$COSMOS_ROOT/yam_dataset_statistics.json" \
  --t5_text_embeddings_path "$COSMOS_ROOT/yam_t5_embeddings.pkl" \
  --use_third_person_image True --use_wrist_image True --num_wrist_images 2 \
  --use_proprio True --normalize_proprio True --unnormalize_actions True \
  --chunk_size 50 --num_open_loop_steps 50 --num_denoising_steps_action 10 \
  --ar_future_prediction False --ar_value_prediction False \
  --host 0.0.0.0 --port 8777
```

Wait for uvicorn's `Application startup complete` line, then verify:

```bash
curl http://localhost:8777/docs
```

## Step 6: Create the lerobot-side pretrained directory (VM)

This is what `--pretrained_name_or_path` on the robot client points to via gRPC. It only has to exist so `PolicyServer` can instantiate the policy class; no weights live here.

```bash
export LEROBOT_COSMOS_CFG=$HOME/cosmos_lerobot/yam_cube_stacking_20260415
mkdir -p "$LEROBOT_COSMOS_CFG"
cat > "$LEROBOT_COSMOS_CFG/lerobot_config.json" <<'EOF'
{
  "server_url": "http://localhost:8777/act",
  "server_image_key_map": {
    "top":   "primary_image",
    "left":  "left_wrist_image",
    "right": "right_wrist_image"
  },
  "primary_image_key": "top",
  "wrist_image_keys":  ["left", "right"],
  "num_images_in_input": 3,
  "image_size": 224,
  "chunk_size": 50,
  "n_action_steps": 50,
  "action_dim": 14,
  "proprio_dim": 14,
  "task_description": "Stack the cubes."
}
EOF
```

## Step 7: Launch lerobot `PolicyServer` at 25 Hz (VM, terminal 2)

**Cosmos Policy must run at 25 Hz**, not 30 or 50. This is a hard requirement from the training regime (verified against the NVIDIA ALOHA reference model card).

```bash
python -m lerobot.async_inference.policy_server \
  --host 0.0.0.0 --port 8000 --fps 25
```

## Step 8: Launch `RobotClient` at 25 Hz (robot machine)

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
  --task "Stack the cubes." \
  --policy_type cosmos \
  --pretrained_name_or_path $HOME/cosmos_lerobot/yam_cube_stacking_20260415 \
  --policy_device cpu \
  --actions_per_chunk 25 \
  --chunk_size_threshold 0.5 \
  --aggregate_fn_name weighted_average \
  --fps 25
```

The cameras can capture at 30 Hz; `RobotClient` subsamples to match `--fps 25`.

## Recording evaluation rollouts while the policy drives

`RobotClient` can save every episode as a LeRobotDataset v3 while Cosmos is executing, which is how you capture eval data for A/B comparisons or bootstrap follow-up training sets. Enable it with `--dataset.enabled true`:

```bash
  --dataset.enabled true \
  --dataset.num_episodes 5 \
  --dataset.repo_id your-org/eval_cosmos_cube_stack \
  --dataset.push_to_hub true \
  --dataset.max_episode_seconds 120 \
  --dataset.reset_time_s 10 \
  --dataset.video_encoding_batch_size 2
```

Common flags (full list in [`async_inference/configs.py`](../../async_inference/configs.py)):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dataset.enabled` | `false` | Turn recording on. |
| `--dataset.repo_id` | — | Required if enabled. Format `user/eval_dataset`. |
| `--dataset.root` | cache dir | Local directory for the dataset. |
| `--dataset.push_to_hub` | `false` | Upload to the Hub when the session ends. |
| `--dataset.private` | `true` | Keep the Hub repo private. |
| `--dataset.use_videos` | `true` | Encode cameras as video (vs PNG frames). |
| `--dataset.num_episodes` | `None` | Stop after N episodes (`None` = keyboard stop). |
| `--dataset.max_episode_seconds` | `None` | Per-episode time cap. `None` = keyboard-only. |
| `--dataset.reset_time_s` | `60` | Seconds given to reset the scene between episodes. |
| `--dataset.resume` | `false` | Append to an existing local dataset. |
| `--dataset.video_encoding_batch_size` | `1` | Batch multiple episodes before ffmpeg runs (speeds up short episodes). |

### Keyboard controls (only when `--dataset.enabled true`)

| Key (then Enter) | Effect |
| --- | --- |
| `n` | Save the current episode and start a new one. |
| `s` | Save the current episode and stop recording. |
| `b` | Discard the current episode and re-record it. |

During the reset window, the robot smoothly moves to an `initial_position` hardcoded in [`async_inference/robot_client.py`](../../async_inference/robot_client.py); edit that constant if your YAM home pose is different.

## Aggregate functions

When a new action chunk arrives while the previous one is still executing, overlapping timesteps are blended by the aggregate function. The registry lives in [`async_inference/configs.py`](../../async_inference/configs.py):

| `--aggregate_fn_name` | Formula |
| --- | --- |
| `weighted_average` (default) | `0.3 · old + 0.7 · new` |
| `latest_only` | `new` |
| `average` | `0.5 · old + 0.5 · new` |
| `conservative` | `0.7 · old + 0.3 · new` |

For Cosmos on YAM, `weighted_average` gives the smoothest motion. Switch to `latest_only` temporarily when debugging — it makes the effect of each new chunk visible in the joint traces.

## Async-inference tuning cheat-sheet

| Symptom | Try |
| --- | --- |
| Client logs "queue empty, waiting for server" | Increase `--actions_per_chunk` (toward 50); lower `--chunk_size_threshold` toward 0; confirm the GPU isn't thrashing. |
| Jittery motion | Stick with `--aggregate_fn_name weighted_average` or try `conservative`; verify client + server `--fps` are both 25; verify `deploy.py` actually loaded the YAM stats file (not a stale default). |
| Policy seems to ignore the scene | Confirm `--task` is non-empty and an exact T5 key; verify all three images appear in `deploy.py` logs. |
| High network usage | Observations already go compressed; lower `--actions_per_chunk` to 15–25. |
| Server errors on first request | Policy initialisation is lazy — the traceback lands in the `PolicyServer` log, not the `RobotClient` log. |

## YAM-specific gotchas

- **Control rate must be 25 Hz.** NVIDIA's ALOHA reference model card explicitly states the policy was trained for 25 Hz control and performs poorly at other rates. Set `--fps 25` on both server and client.
- **`actions_per_chunk 25` of the 50 returned** gives one second of actions @ 25 Hz between queries, with `--chunk_size_threshold 0.5` providing smooth aggregation without stalling.
- **Task string must match T5 cache.** `"Stack the cubes."` must be a verbatim key in `yam_t5_embeddings.pkl`. If `deploy.py` returns the sentinel string `"error"`, the first thing to check is a task typo / missing key.
- **Proprio / action order must match training.** For `bi_yam_follower`, lerobot packs `observation.state` as `left_joint_0..5.pos, left_gripper.pos, right_joint_0..5.pos, right_gripper.pos` (14 dims). If `ai2-cortex/02042026-cube-stacking-combined` was collected via lerobot, the order matches. If it inherited an ALOHA/ViperX frame, you may need to permute; add `proprio_permutation` / `action_permutation` support in a follow-up.
- **Gripper scale.** YAM encoder gripper is `[0, 1]`. If training normalized it differently, you'll see the gripper twitch; check the `actions_min[6]` / `actions_min[13]` and `proprio_min[6]` / `proprio_min[13]` values in `yam_dataset_statistics.json`.
- **Image resolution is 224×224.** Cameras can be any resolution; `deploy.py` resizes internally.
- **Avoid `/data`.** Use `$HOME/cosmos_ckpts` unless you've `sudo chown`'d `/data` to your user.

## Config reference

All fields in `lerobot_config.json` map to attributes of `CosmosConfig`. Defaults (see [`configuration_cosmos.py`](configuration_cosmos.py)):

| Field | Default | Purpose |
| --- | --- | --- |
| `server_url` | `http://0.0.0.0:8777/act` | `deploy.py` endpoint |
| `server_image_key_map` | `{"top":"primary_image","left":"left_wrist_image","right":"right_wrist_image"}` | lerobot camera key → cosmos observation key |
| `primary_image_key` | `"top"` | Primary (third-person) camera key |
| `wrist_image_keys` | `["left","right"]` | Wrist camera keys (order matters) |
| `num_images_in_input` | `3` | Total cameras fed to the model |
| `image_size` | `224` | Per the NVIDIA reference |
| `chunk_size` / `n_action_steps` | `50` / `50` | Action horizon (2s @ 25 Hz) |
| `action_dim` / `proprio_dim` | `14` / `14` | 7 per arm (6 joints + 1 gripper) |
| `task_description` | `""` | Overrides `batch["task"]` when non-empty |
| `request_timeout` | `30.0` | HTTP timeout (diffusion takes <1s on H100) |

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `deploy.py` returns literal string `"error"` | Task string not in T5 pickle, or proprio dim mismatch. Check `deploy.py` logs. |
| `KeyError: primary_image` in deploy.py | lerobot didn't send all 3 images. Verify `server_image_key_map` matches your camera keys. |
| `assert cfg.chunk_size == cosmos_config...chunk_size` | Training used a different chunk size; match it in `--chunk_size` and in `CosmosConfig.chunk_size`. |
| Very slow / laggy policy | Probably running at 30 Hz; drop to `--fps 25` on both server and client. |
| Gripper oscillates | Normalization mismatch; inspect `actions_min[6]/[13]` in `yam_dataset_statistics.json`. |
| `FileNotFoundError: dataset_statistics.json` | You skipped step 2. Run the converter. |
| `PermissionError: /data` | Use `$HOME/cosmos_ckpts` instead of `/data` on stock VMs. |

## Out of scope

- **Training Cosmos Policy through lerobot** — use `NVlabs/cosmos-policy` directly.
- **Best-of-N and planning-model ensembles** — this wrapper consumes only `actions` from the response; `future_image_predictions` and `value_prediction` are ignored.
- **Autoregressive future state / value heads** — set `--ar_future_prediction False --ar_value_prediction False` on `deploy.py`.
