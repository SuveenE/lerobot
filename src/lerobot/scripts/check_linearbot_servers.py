#!/usr/bin/env python3
"""Check connectivity and read joint positions from all Linear Bot arm servers."""
import argparse
import socket
import time

import numpy as np

LEADER_SERVERS = [
    ("Right leader", 5001),
    ("Left leader",  5002),
]

FOLLOWER_SERVERS = [
    ("Right follower", 1234),
    ("Left follower",  1235),
]


def check_tcp(host: str, port: int, timeout: float = 3.0) -> tuple[bool, float]:
    """Return (reachable, latency_ms) for a TCP port."""
    t0 = time.time()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, (time.time() - t0) * 1000
    except (ConnectionRefusedError, OSError):
        return False, 0.0


def read_arm_state(host: str, port: int) -> dict | None:
    """Connect via portal RPC and read arm state. Returns None on failure."""
    try:
        import portal

        client = portal.Client(f"{host}:{port}")
        dofs = client.num_dofs().result()
        joint_pos = client.get_joint_pos().result()
        obs = client.get_observations().result()
        return {"dofs": dofs, "joint_pos": joint_pos, "observations": obs}
    except Exception as e:
        return {"error": str(e)}


def format_array(arr) -> str:
    if isinstance(arr, np.ndarray):
        return np.array2string(arr, precision=3, suppress_small=True, separator=", ")
    return str(arr)


def main():
    parser = argparse.ArgumentParser(description="Check Linear Bot arm servers and read joint positions.")
    parser.add_argument(
        "--leader_host",
        type=str,
        default="localhost",
        help="IP or hostname of the leader machine (default: localhost).",
    )
    parser.add_argument(
        "--follower_host",
        type=str,
        default=None,
        help="IP or hostname of the follower machine. If omitted, only leader servers are checked.",
    )
    parser.add_argument(
        "--read",
        action="store_true",
        help="Read joint positions from reachable servers (requires portal).",
    )
    args = parser.parse_args()

    servers = [(name, args.leader_host, port) for name, port in LEADER_SERVERS]
    if args.follower_host:
        for name, port in FOLLOWER_SERVERS:
            servers.append((name, args.follower_host, port))

    print("=" * 60)
    print("  Linear Bot — Server Check")
    print("=" * 60)

    reachable = []
    for name, host, port in servers:
        ok, latency = check_tcp(host, port)
        if ok:
            print(f"  ✓ {name:20s}  {host}:{port}  ({latency:.0f} ms)")
            reachable.append((name, host, port))
        else:
            print(f"  ✗ {name:20s}  {host}:{port}  — not reachable")

    print()
    if len(reachable) == len(servers):
        print(f"All {len(servers)} server(s) reachable.")
    else:
        print(f"{len(reachable)}/{len(servers)} server(s) reachable.")

    if not args.follower_host:
        print("\nTip: pass --follower_host <IP> to also check follower arm servers.")

    if args.read and reachable:
        print()
        print("=" * 60)
        print("  Joint Positions")
        print("=" * 60)
        for name, host, port in reachable:
            state = read_arm_state(host, port)
            if state is None or "error" in state:
                err = state.get("error", "unknown") if state else "unknown"
                print(f"\n  {name} ({host}:{port})  — RPC error: {err}")
                continue

            dofs = state["dofs"]
            joint_pos = state["joint_pos"]
            obs = state["observations"]

            print(f"\n  {name} ({host}:{port})")
            print(f"    DOFs:       {dofs}")
            print(f"    joint_pos:  {format_array(joint_pos)}")

            if "gripper_pos" in obs:
                print(f"    gripper:    {format_array(obs['gripper_pos'])}")
            if "joint_vel" in obs:
                print(f"    joint_vel:  {format_array(obs['joint_vel'])}")

    print()


if __name__ == "__main__":
    main()
