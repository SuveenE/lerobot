#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON="${SCRIPT_DIR}/../lerobot/.venv/bin/python"
fi
if [ ! -x "$PYTHON" ]; then
    echo -e "\033[0;31m[FAIL]\033[0m No venv Python found. Create or use the project venv, e.g.:" >&2
    echo "    cd \"$SCRIPT_DIR\" && python3 -m venv .venv && .venv/bin/pip install -e ." >&2
    exit 1
fi
VENV_BIN="$(dirname "$PYTHON")"
export PATH="$VENV_BIN:$PATH"
export HF_HOME="${HOME}/.cache/huggingface"
CACHE_DIR="${HF_HOME}/lerobot/griffinlabs"
MIN_SPACE_GB=8

# Semantic SocketCAN names (assigned by piper_can_activate.sh from USB bus-info)
RIGHT_PIPER_CAN="right_piper_can"
LEFT_PIPER_CAN="left_piper_can"
RIGHT_CAN_USB="1-4.3.2:1.0"
LEFT_CAN_USB="1-4.3.4.1:1.0"

OVERWRITE=false
DATASET_NAME=""
NUM_EPISODES=1
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --overwrite) OVERWRITE=true ;;
        -h|--help)
            echo "Usage: bash record.sh [--overwrite] [<dataset_name>] [<num_episodes>]"
            echo ""
            echo "  dataset_name   Repo id: griffinlabs/<name> (lowercased)"
            echo "  num_episodes   How many episodes to record (default: 1)"
            echo ""
            echo "Examples:"
            echo "  bash record.sh test7 10     # griffinlabs/test7, 10 episodes"
            echo "  record test7 10             # same (shell function in ~/.bashrc)"
            exit 0
            ;;
        -*) ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done

DATASET_NAME="${POSITIONAL[0]:-}"
if [ -n "${POSITIONAL[1]:-}" ]; then
    if ! [[ "${POSITIONAL[1]}" =~ ^[0-9]+$ ]] || [ "${POSITIONAL[1]}" -lt 1 ]; then
        echo -e "${RED}[FAIL]${NC} Second argument must be a positive integer (episode count), got: ${POSITIONAL[1]}"
        echo "  Example: bash record.sh test7 10"
        exit 1
    fi
    NUM_EPISODES="${POSITIONAL[1]}"
fi

if [ -z "$DATASET_NAME" ]; then
    echo -e "  Dataset repo will be ${BOLD}griffinlabs/<name>${NC}"
    read -rp "  Enter dataset name: " DATASET_NAME
    if [ -z "$DATASET_NAME" ]; then
        echo -e "${RED}${BOLD}Dataset name cannot be empty.${NC}"
        exit 1
    fi
fi

DATASET_NAME=$(echo "$DATASET_NAME" | tr '[:upper:]' '[:lower:]')

echo -e "  ${CYAN}[INFO]${NC} Will record ${BOLD}${NUM_EPISODES}${NC} episode(s) to ${BOLD}griffinlabs/${DATASET_NAME}${NC}"

print_step() {
    echo ""
    echo -e "${CYAN}${BOLD}========================================${NC}"
    echo -e "${CYAN}${BOLD}  [Step $1/6] $2${NC}"
    echo -e "${CYAN}${BOLD}========================================${NC}"
    echo ""
}

print_ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; }
print_fail() { echo -e "  ${RED}[FAIL]${NC} $1"; }
print_warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
print_info() { echo -e "  ${CYAN}[INFO]${NC} $1"; }

print_rerun_viewer_urls() {
    local web_port="${LEROBOT_RERUN_WEB_PORT:-9090}"
    local ws_port="${LEROBOT_RERUN_WS_PORT:-9877}"
    local lan_ip

    # Pick the IP on the default-route interface — the one the machine uses to
    # reach the outside, so it matches what another device on the LAN can reach.
    lan_ip=$(ip route get 1.1.1.1 2>/dev/null | awk '/src/{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}')
    if [ -z "$lan_ip" ]; then
        # Fallback: first global IPv4 address
        lan_ip=$(ip -o -4 addr show scope global 2>/dev/null | awk 'NR==1{split($4,a,"/"); print a[1]}')
    fi

    local viewer_url="http://${lan_ip}:${web_port}/?url=ws://${lan_ip}:${ws_port}"
    print_info "Rerun web viewer (no desktop display detected):"
    if [ -n "$lan_ip" ]; then
        echo -e ""
        echo -e "    ${BOLD}${CYAN}${viewer_url}${NC}"
        echo -e ""
    else
        print_warn "Could not detect LAN IP — open http://127.0.0.1:${web_port}/?url=ws://127.0.0.1:${ws_port} on this machine"
    fi
}

# Printed when LEROBOT_RECORD_CONTROL_PORT is set (browser n/b/s — Rerun cannot forward keys).
print_record_control_url() {
    local port="${LEROBOT_RECORD_CONTROL_PORT:-}"
    [ -n "$port" ] && [ "$port" != "0" ] || return 0
    local lan_ip
    lan_ip=$(ip route get 1.1.1.1 2>/dev/null | awk '/src/{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}')
    if [ -z "$lan_ip" ]; then
        lan_ip=$(ip -o -4 addr show scope global 2>/dev/null | awk 'NR==1{split($4,a,"/"); print a[1]}')
    fi
    print_info "Recording control page (n/b/s or buttons; keep this tab focused):"
    if [ -n "$lan_ip" ]; then
        echo -e "    ${BOLD}${CYAN}http://${lan_ip}:${port}/${NC}"
    fi
    echo -e "    ${BOLD}${CYAN}http://127.0.0.1:${port}/${NC}"
}

# Uses LOCAL_PATH / NUM_EPISODES / PYTHON (set before Step 6).
print_bad_episode_help() {
    local info_json="${LOCAL_PATH}/meta/info.json"
    local total=""
    if [ -f "$info_json" ]; then
        total=$("$PYTHON" -c "import json,sys; print(int(json.load(open(sys.argv[1]))['total_episodes']))" "$info_json" 2>/dev/null || true)
    fi
    if [ -n "$total" ] && [ "$total" -ge 1 ] 2>/dev/null; then
        local last=$((total - 1))
        print_info "Valid episode indices to mark bad (0-based): ${BOLD}0 … ${last}${NC} ($total saved in dataset)"
        if [ "$total" -le 30 ]; then
            print_info "Full list: ${BOLD}$(seq 0 "$last" | tr '\n' ' ' | sed 's/[[:space:]]*$//')${NC}"
        fi
    else
        if [ "$NUM_EPISODES" -ge 1 ] 2>/dev/null; then
            local last=$((NUM_EPISODES - 1))
            print_warn "Could not read ${LOCAL_PATH}/meta/info.json (recording may have failed or not saved yet)."
            print_info "If this run completed normally, indices are usually: ${BOLD}0 … ${last}${NC} ($NUM_EPISODES episode(s) requested)."
        else
            print_warn "Could not determine episode count; enter indices you know are bad (0-based)."
        fi
    fi
}

# ──────────────────────────────────────────────
# Step 1: Validate CAN hardware (names are assigned in Step 2)
# ──────────────────────────────────────────────
print_step 1 "Checking CAN interfaces"

mapfile -t CAN_IFACES < <(ip -br link show type can 2>/dev/null | awk '{print $1}')
if [ "${#CAN_IFACES[@]}" -lt 2 ]; then
    echo ""
    print_fail "Need at least two SocketCAN interfaces (found ${#CAN_IFACES[@]}). Plug in both CAN adapters and try again."
    if [ "${#CAN_IFACES[@]}" -gt 0 ]; then
        print_info "Current CAN interfaces: ${CAN_IFACES[*]}"
    fi
    exit 1
fi

print_ok "Two CAN interfaces detected (${#CAN_IFACES[@]}): ${CAN_IFACES[*]}"

# ──────────────────────────────────────────────
# Step 2: Activate CAN Ports
# ──────────────────────────────────────────────
print_step 2 "Activating CAN ports"

print_info "Activating RIGHT arm (${RIGHT_PIPER_CAN}) at USB ${RIGHT_CAN_USB} ..."
if bash "$SCRIPT_DIR/piper_can_activate.sh" "$RIGHT_PIPER_CAN" 1000000 "$RIGHT_CAN_USB"; then
    print_ok "Right arm (${RIGHT_PIPER_CAN}) activated"
else
    print_fail "Failed to activate right arm (${RIGHT_PIPER_CAN})"
    exit 1
fi

echo ""
print_info "Activating LEFT arm (${LEFT_PIPER_CAN}) at USB ${LEFT_CAN_USB} ..."
if bash "$SCRIPT_DIR/piper_can_activate.sh" "$LEFT_PIPER_CAN" 1000000 "$LEFT_CAN_USB"; then
    print_ok "Left arm (${LEFT_PIPER_CAN}) activated"
else
    print_fail "Failed to activate left arm (${LEFT_PIPER_CAN})"
    exit 1
fi

for iface in "$RIGHT_PIPER_CAN" "$LEFT_PIPER_CAN"; do
    if ip link show "$iface" &>/dev/null; then
        if ip link show "$iface" | grep -q "state UP"; then
            print_ok "Interface $iface is up"
        else
            print_warn "Interface $iface exists but is not UP"
        fi
    else
        print_fail "Expected interface $iface missing after activation"
        exit 1
    fi
done

# ──────────────────────────────────────────────
# Step 3: Verify Arm Joint Readings
# ──────────────────────────────────────────────
# print_step 3 "Verifying arm joint readings"

# print_info "Running interactive joint check (right arm first, then left) ..."
# if "$PYTHON" "$SCRIPT_DIR/verify_joints.py" --right-can "$RIGHT_PIPER_CAN" --left-can "$LEFT_PIPER_CAN"; then
#     print_ok "Both arms verified"
# else
#     print_fail "Joint verification failed. Check arm connections and try again."
#     exit 1
# fi

# ──────────────────────────────────────────────
# Step 4: Check Disk Space
# ──────────────────────────────────────────────
print_step 4 "Checking available disk space"

check_space() {
    df --output=avail -BG "$HF_HOME" | tail -1 | tr -d ' G'
}

AVAIL_GB=$(check_space)
print_info "Available disk space: ${AVAIL_GB}GB (minimum required: ${MIN_SPACE_GB}GB)"

if [ "$AVAIL_GB" -lt "$MIN_SPACE_GB" ]; then
    print_warn "Insufficient disk space!"
    print_info "Opening file explorer on cache directory so you can delete old datasets ..."
    mkdir -p "$CACHE_DIR"
    xdg-open "$CACHE_DIR" 2>/dev/null &

    echo ""
    echo -e "  ${YELLOW}Please delete old datasets from:${NC}"
    echo -e "  ${BOLD}$CACHE_DIR${NC}"
    echo ""
    read -rp "  Press Enter after you have freed up space ..."

    AVAIL_GB=$(check_space)
    print_info "Available disk space after cleanup: ${AVAIL_GB}GB"
    if [ "$AVAIL_GB" -lt "$MIN_SPACE_GB" ]; then
        print_fail "Still not enough space (${AVAIL_GB}GB < ${MIN_SPACE_GB}GB). Exiting."
        exit 1
    fi
    print_ok "Disk space is now sufficient"
else
    print_ok "Disk space check passed"
fi

# ──────────────────────────────────────────────
# Step 5: Run LeRobot Recording
# ──────────────────────────────────────────────
print_step 5 "Configuring recording session"

REPO_ID="griffinlabs/${DATASET_NAME}"
print_info "Dataset: ${BOLD}${REPO_ID}${NC}"
LOCAL_PATH="${CACHE_DIR}/${DATASET_NAME}"
RESUME=false

if [ "$OVERWRITE" = true ]; then
    if [ -d "$LOCAL_PATH" ]; then
        echo ""
        print_warn "Overwrite mode: the following directory will be DELETED:"
        echo -e "  ${RED}${BOLD}$LOCAL_PATH${NC}"
        echo ""
        read -rp "  Type 'yes' to confirm deletion: " CONFIRM
        if [ "$CONFIRM" != "yes" ]; then
            print_fail "Overwrite cancelled by user. Exiting."
            exit 1
        fi
        rm -rf "$LOCAL_PATH"
        print_ok "Deleted cached dataset at $LOCAL_PATH"
    else
        print_info "Overwrite flag set but no cached dataset found. Starting fresh."
    fi
    RESUME=false
else
    if [ -d "$LOCAL_PATH" ]; then
        print_info "Existing dataset found at $LOCAL_PATH"
        print_info "Resuming recording (--resume=true)"
        RESUME=true
    else
        print_info "No existing dataset found. Starting new recording (--resume=false)"
        RESUME=false
    fi
fi

CAMERAS='{"top": {"type": "orbbec", "use_depth": false, "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "right": {"type": "intelrealsense", "use_depth": false, "serial_number_or_name": "352122271775", "width": 424, "height": 240, "fps": 30}, "left": {"type": "intelrealsense", "use_depth": false, "serial_number_or_name": "409122272825", "width": 424, "height": 240, "fps": 30}}'

echo ""
print_info "Recording configuration:"
echo -e "    Repo ID        : ${BOLD}${REPO_ID}${NC}"
echo -e "    Episodes       : ${BOLD}${NUM_EPISODES}${NC}"
echo -e "    Task           : ${BOLD}Clean the countertop${NC}"
echo -e "    Episode time   : ${BOLD}400s${NC}"
echo -e "    Reset time     : ${BOLD}20s${NC}"
echo -e "    Resume         : ${BOLD}${RESUME}${NC}"
echo ""
print_info "Launching lerobot recording ..."
echo ""
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ] && [ -z "${WAYLAND_SOCKET:-}" ]; then
    print_warn "No GUI display detected; using Rerun web viewer."
    print_rerun_viewer_urls
    echo ""
    # Rerun UI cannot send n/b/s to Python; optional HTTP panel (disable with LEROBOT_RECORD_CONTROL_PORT=0).
    export LEROBOT_RECORD_CONTROL_PORT="${LEROBOT_RECORD_CONTROL_PORT:-8778}"
    if [ "${LEROBOT_RECORD_CONTROL_PORT}" != "0" ]; then
        print_record_control_url
        echo ""
    fi
fi

set +e
"$PYTHON" -m lerobot.record \
    --robot.type=bi_piper \
    --robot.left_arm_can_port="$LEFT_PIPER_CAN" \
    --robot.right_arm_can_port="$RIGHT_PIPER_CAN" \
    --robot.id=arms \
    --robot.cameras="$CAMERAS" \
    --dataset.repo_id="$REPO_ID" \
    --dataset.num_episodes="$NUM_EPISODES" \
    --dataset.single_task="Clean the countertop" \
    --dataset.episode_time_s=400 \
    --dataset.reset_time_s=20 \
    --display=true \
    --resume="$RESUME"
RECORD_EXIT=$?
set -e

# ──────────────────────────────────────────────
# Step 6: Label Bad Episodes
# ──────────────────────────────────────────────
print_step 6 "Label bad episodes"

print_bad_episode_help
echo ""
echo -e "  Enter the episode numbers that were ${RED}${BOLD}bad${NC} (space-separated, 0-based)."
echo -e "  Press ${BOLD}Enter${NC} to skip if all episodes were good."
echo ""
read -r -p "  Bad episodes: " -a BAD_EPISODES

if [ "${#BAD_EPISODES[@]}" -eq 0 ]; then
    print_ok "No episodes labeled as bad"
else
    BAD_FILE="${LOCAL_PATH}/bad_episodes.txt"
    VALID=true
    for val in "${BAD_EPISODES[@]}"; do
        if ! [[ "$val" =~ ^[0-9]+$ ]]; then
            print_fail "'$val' is not a valid episode number (must be a non-negative integer)"
            VALID=false
        fi
    done

    if [ "$VALID" = true ]; then
        if [ -f "$BAD_FILE" ]; then
            print_info "Merging into existing ${BAD_FILE} (duplicates removed after update)"
        fi
        for val in "${BAD_EPISODES[@]}"; do
            echo "$val" >> "$BAD_FILE"
        done
        TMP_SORT=$(mktemp)
        if sort -n -u "$BAD_FILE" > "$TMP_SORT" && mv "$TMP_SORT" "$BAD_FILE"; then
            :
        else
            rm -f "$TMP_SORT"
            print_fail "Could not deduplicate ${BAD_FILE}"
            exit 1
        fi
        print_ok "Labeled episodes [ ${BAD_EPISODES[*]} ] as bad -> ${BAD_FILE} (sorted, unique)"

        print_info "Uploading bad_episodes.txt to Hugging Face (${REPO_ID}) ..."
        if huggingface-cli upload "$REPO_ID" "$BAD_FILE" bad_episodes.txt --repo-type dataset; then
            print_ok "Uploaded bad_episodes.txt to ${REPO_ID}"
        else
            print_warn "Failed to upload bad_episodes.txt — you can retry manually with:"
            echo -e "    huggingface-cli upload ${REPO_ID} ${BAD_FILE} bad_episodes.txt"
        fi
    else
        print_warn "No episodes were labeled due to invalid input"
    fi
fi

exit "$RECORD_EXIT"
