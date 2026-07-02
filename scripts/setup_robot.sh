#!/usr/bin/env bash
set -euo pipefail
SESSION=robot

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n cans
tmux new-window -t "$SESSION" -n servers
tmux new-window -t "$SESSION" -n flowbase
tmux new-window -t "$SESSION" -n record
tmux new-window -t "$SESSION" -n cameras

# Drop the stray default session if tmux created one; ignore if absent.
tmux kill-session -t 0 2>/dev/null || true

# Draft (type but DON'T run) a command into each window. No trailing "Enter",
# so each command just sits at the prompt until you press Enter yourself.
# -l = send the text literally instead of interpreting key names.

tmux send-keys -t "$SESSION:cans" -l \
  'cd ~/lerobot/i2rt && bash scripts/reset_all_can.sh'

tmux send-keys -t "$SESSION:servers" -l \
  'cd ~/lerobot && source lerobot/bin/activate && python -m lerobot.scripts.setup_bi_yam_servers --eval'

tmux send-keys -t "$SESSION:flowbase" -l \
  'cd ~/lerobot && source lerobot/bin/activate && python3 i2rt/i2rt/flow_base/flow_base_controller.py --channel can_linearbot --gpio-host 172.16.0.67:8765'

tmux send-keys -t "$SESSION:record" -l \
  'cd ~/lerobot && source lerobot/bin/activate'

tmux send-keys -t "$SESSION:record" -l \
  'cd ~/lerobot && source lerobot/bin/activate && lerobot-find-cameras realsense'

tmux select-window -t "$SESSION:cans"
tmux attach -t "$SESSION"
