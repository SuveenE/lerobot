# YAMBox GELLO Commands

Use this when collecting with GELLO leader arms driving the bimanual YAM followers.

## One-time setup

```bash
cd ~/lerobot
git submodule update --init --recursive gello_software
source ./lerobot/bin/activate
pip install -e ./gello_software
pip install -e ".[yam]"
```

The `gello_software` submodule tracks:

```text
repo: https://github.com/SuveenE/gello_software.git
branch: weining
```

## Tab 1: reset CAN

```bash
cd ~/lerobot/i2rt/scripts
bash reset_all_can.sh
```

## Tab 2: start follower servers only

```bash
cd ~/lerobot
source ./lerobot/bin/activate
python -m lerobot.scripts.setup_bi_yam_servers --eval
```

This starts only the YAM follower servers:

- left follower: `localhost:1235`
- right follower: `localhost:1234`

Do not start YAM leader-arm servers on `5002` / `5001`; GELLO leaders are read directly over USB.

## Tab 3: optional camera check

```bash
cd ~/lerobot
source ./lerobot/bin/activate

lerobot-find-cameras realsense

# top camera
lerobot-overlay-camera --camera-id 323622270506

# wrist camera
lerobot-overlay-camera --camera-id 323622271967
```

## Tab 4: record with GELLO leaders

```bash
cd ~/lerobot
source ./lerobot/bin/activate

LEROBOT_RERUN_CAMERAS="top" \
LEROBOT_RERUN_MEMORY_LIMIT="5%" \
lerobot-record \
  --robot.type=bi_yam_follower \
  --robot.left_arm_port=1235 \
  --robot.right_arm_port=1234 \
  --robot.cameras='{
    right: {"type": "intelrealsense", "serial_number_or_name": "352122274173", "width": 640, "height": 360, "fps": 30, "use_depth": true},
    left: {"type": "intelrealsense", "serial_number_or_name": "230422270749", "width": 640, "height": 360, "fps": 30, "use_depth": true},
    top: {"type": "intelrealsense", "serial_number_or_name": "352122273311", "width": 640, "height": 360, "fps": 30, "use_depth": true}
  }' \
  --robot.record_torques=true \
  --teleop.type=bi_gello_yam_leader \
  --teleop.left_port=/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAO9WA0-if00-port0 \
  --teleop.right_port=/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAO9WI5-if00-port0 \
  --dataset.repo_id=cortex1/yambox-gello-demo \
  --dataset.num_episodes=10 \
  --dataset.single_task="brush lint off a garment using lint roller" \
  --display_data=false \
  --dataset.episode_time_s=300 \
  --dataset.reset_time_s=20 \
  --dataset.video_encoding_batch_size=10 \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --dataset.vcodec=libsvtav1 \
  --dataset.push_to_hub=false \
  --resume=false
```

Port mapping:

- `left_port`: left GELLO, `FTAO9WA0`
- `right_port`: right GELLO, `FTAO9WI5`

## Notes

`bi_gello_yam_leader` emits the same 14 action keys as `bi_yam_leader`, so the `bi_yam_follower` robot, cameras, and dataset schema stay unchanged.

GELLO leaders are passive and do not soft-align the follower before the first command. Start the GELLO arms near the YAM rest pose before recording.
