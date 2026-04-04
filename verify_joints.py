#!/usr/bin/env python
"""
Interactive joint verification for bimanual Piper arms.

Checks each arm sequentially (right first, then left) with a live-updating
terminal display that polls at ~10Hz.  Every joint (1-6 + gripper) must show
movement beyond CHANGE_THRESHOLD.  The display auto-completes when all joints
pass.  If the operator presses Enter before all pass, the arm fails immediately.
"""

import argparse
import select
import sys
import termios
import time
import tty

CHANGE_THRESHOLD = 1.0
POLL_INTERVAL = 0.1

JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper",
]
NUM_DISPLAY_LINES = len(JOINT_NAMES) + 2  # header + joint rows + summary

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
NC = "\033[0m"
CLEAR_LINE = "\033[K"


def print_ok(msg):
    print(f"  {GREEN}[OK]{NC} {msg}")


def print_fail(msg):
    print(f"  {RED}[FAIL]{NC} {msg}")


def print_info(msg):
    print(f"  {CYAN}[INFO]{NC} {msg}")


def parse_joint_values(joint_msgs):
    """Return a dict of {joint_name: float_value} for joints 1-6."""
    msg_str = str(joint_msgs)
    values = {}
    for i in range(1, 7):
        pattern = f"Joint {i}:"
        start_idx = msg_str.find(pattern)
        if start_idx == -1:
            raise ValueError(f"Joint {i} not found in message: {msg_str}")
        value_start = start_idx + len(pattern)
        value_end = msg_str.find("\n", value_start)
        if value_end == -1:
            value_end = len(msg_str)
        values[f"joint_{i}"] = float(msg_str[value_start:value_end].strip())
    return values


def parse_gripper_value(gripper_msgs):
    """Return gripper position in mm."""
    raw = gripper_msgs.gripper_state.grippers_angle
    return float(raw) / 1000.0


def read_arm(arm):
    """Read all joint + gripper values and return as a dict."""
    joints = parse_joint_values(arm.GetArmJointMsgs())
    joints["gripper"] = parse_gripper_value(arm.GetArmGripperMsgs())
    return joints


def draw_table(arm_label, baseline, current, passed, first_draw):
    """Redraw the joint status table in-place using ANSI escape codes."""
    if not first_draw:
        sys.stdout.write(f"\033[{NUM_DISPLAY_LINES}A")

    num_passed = len(passed)
    total = len(JOINT_NAMES)

    sys.stdout.write(
        f"{CLEAR_LINE}  {BOLD}{arm_label} joint verification "
        f"({num_passed}/{total} passed){NC}\n"
    )

    for name in JOINT_NAMES:
        base_val = baseline[name]
        cur_val = current[name]
        delta = abs(cur_val - base_val)
        if name in passed:
            status = f"{GREEN}{BOLD}PASS{NC}"
        else:
            status = f"{YELLOW}    {NC}"
        sys.stdout.write(
            f"{CLEAR_LINE}    {name:>10s} : "
            f"{base_val:>10.3f} -> {cur_val:>10.3f}  "
            f"delta {delta:>7.3f}  [{status}]\n"
        )

    if num_passed == total:
        sys.stdout.write(
            f"{CLEAR_LINE}  {GREEN}{BOLD}All joints verified!{NC}\n"
        )
    else:
        sys.stdout.write(
            f"{CLEAR_LINE}  {YELLOW}Wiggle remaining joints... "
            f"(press Enter to abort){NC}\n"
        )

    sys.stdout.flush()


def verify_arm(can_port, arm_label):
    """
    Connect to one arm, show a live display, and verify every joint moves.

    Polls at ~10Hz and redraws the status table in-place.  Auto-passes when
    all joints exceed CHANGE_THRESHOLD.  Returns False immediately if the
    operator presses Enter before all joints have passed.
    """
    from piper_sdk import C_PiperInterface_V2

    print_info(f"Connecting to {arm_label} arm on {can_port} ...")
    arm = C_PiperInterface_V2(can_port)
    arm.ConnectPort(True)
    time.sleep(0.5)

    try:
        baseline = read_arm(arm)
        passed = set()
        enter_pressed = False

        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        try:
            first_draw = True
            while True:
                current = read_arm(arm)

                for name in JOINT_NAMES:
                    if abs(current[name] - baseline[name]) > CHANGE_THRESHOLD:
                        passed.add(name)

                draw_table(arm_label, baseline, current, passed, first_draw)
                first_draw = False

                if len(passed) == len(JOINT_NAMES):
                    break

                if select.select([sys.stdin], [], [], 0)[0]:
                    ch = sys.stdin.read(1)
                    if ch in ("\n", "\r"):
                        enter_pressed = True
                        break

                time.sleep(POLL_INTERVAL)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        print()
        if len(passed) == len(JOINT_NAMES):
            print_ok(f"{arm_label} arm PASSED — all joints verified")
            return True

        unmoved = sorted(set(JOINT_NAMES) - passed)
        print_fail(
            f"{arm_label} arm FAILED — unmoved: {', '.join(unmoved)}"
        )
        return False

    except Exception as exc:
        print_fail(f"{arm_label} arm ERROR — could not read joints: {exc}")
        return False
    finally:
        try:
            arm.DisconnectPort()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Verify Piper arm joint readings")
    parser.add_argument(
        "--right-can", required=True, help="CAN interface for the right arm (e.g. right_piper_can)"
    )
    parser.add_argument(
        "--left-can", required=True, help="CAN interface for the left arm (e.g. left_piper_can)"
    )
    args = parser.parse_args()

    print(f"\n{CYAN}{BOLD}--- Verifying RIGHT arm ({args.right_can}) ---{NC}")
    if not verify_arm(args.right_can, "RIGHT"):
        sys.exit(1)

    print(f"\n{CYAN}{BOLD}--- Verifying LEFT arm ({args.left_can}) ---{NC}")
    if not verify_arm(args.left_can, "LEFT"):
        sys.exit(1)

    print()
    print_ok("Both arms verified successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
