#!/usr/bin/env python3
"""Merge selected source LeRobot datasets into 80 final datasets and push them
to the ``ai2-cortex`` Hugging Face org, deleting local copies after each push.

Source lists live in ``temp/selected/{policy}_{id|ood}.txt`` (relative to the
repo root). For each ``(policy, task, split)`` triple we:

  1. ``snapshot_download`` every source repo into ``--workdir/src/<repo_id>``
  2. Call ``aggregate_datasets`` (LeRobot v3) to produce the merged dataset at
     ``--workdir/dst/ai2-cortex/eval_<policy>_<task>_<in-distribution|ood>``
  3. ``upload_folder`` the result to the matching repo on the Hub
  4. Delete the local source copies + aggregated copy
  5. Record outcome in ``--workdir/state.json`` so reruns skip completed groups

Designed to run on a beefy VM. Disk usage at any time is roughly the sources
of the current group + the merged copy. Memory usage is small.

Usage examples
--------------
Preview the 80 groups (no downloads, no uploads, no deletions)::

    python temp/merge_and_push.py --dry-run

Process everything::

    python temp/merge_and_push.py --workdir /scratch/lerobot_merge --all

Process one cell only (good for the first end-to-end test)::

    python temp/merge_and_push.py --workdir /scratch/lerobot_merge \\
        --policy cosmos --task pegboard --split id

Resume after a partial run (the default; existing dest repos are skipped)::

    python temp/merge_and_push.py --workdir /scratch/lerobot_merge --all

Force re-merge a single cell::

    python temp/merge_and_push.py --workdir /scratch/lerobot_merge \\
        --policy cosmos --task pegboard --split id --force
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMP_DIR = Path(__file__).resolve().parent
SELECTED = TEMP_DIR / "selected"

# Make lerobot src importable even if the package isn't pip-installed.
sys.path.insert(0, str(REPO_ROOT / "src"))

DEST_ORG = "ai2-cortex"
SPLIT_LABEL = {"id": "in-distribution", "ood": "ood"}

POLICIES = ["cosmos", "molmoact", "openvla", "pi05", "xvla"]
TASKS = [
    "candy_sorting",
    "cup_stacking",
    "cup_storing",
    "linearbot",
    "pegboard",
    "pipette",
    "test_tube",
    "toy_storing",
]

TASK_RE = {
    "candy_sorting": re.compile(r"candy[_-]?(sorting)?", re.I),
    "cup_stacking":  re.compile(r"cup[_-]?stacking", re.I),
    "cup_storing":   re.compile(r"cup[_-]?storing", re.I),
    "linearbot":     re.compile(r"linear[_-]?bot", re.I),
    "pegboard":      re.compile(r"pegboard", re.I),
    "pipette":       re.compile(r"p+ipette[d]?", re.I),
    "test_tube":     re.compile(r"test[_-]?tube|testtube", re.I),
    "toy_storing":   re.compile(r"toy[_-]?storing", re.I),
}


def task_of(name: str) -> str | None:
    # Hardcoded slug families with no task token in the name (per dataset
    # owner clarification): ``eval_cosmos_03052026-*`` is pegboard/cosmos and
    # ``eval_cosmos_m_03052026-*`` is pegboard/molmoact. The renamed aliases
    # (``..._pegboard_03052026``) are kept for backward compatibility.
    if (
        "eval_cosmos_03052026" in name
        or "eval_cosmos_m_03052026" in name
        or "eval_cosmos_pegboard_03052026" in name
        or "eval_molmoact_pegboard_03052026" in name
    ):
        return "pegboard"
    # Order matters: cup_stacking & cup_storing before generic 'cup',
    # test_tube before generic 'test'.
    for t in [
        "candy_sorting",
        "cup_stacking",
        "cup_storing",
        "test_tube",
        "toy_storing",
        "linearbot",
        "pegboard",
        "pipette",
    ]:
        if TASK_RE[t].search(name):
            return t
    return None


# ---------- Group enumeration ----------------------------------------------

def enumerate_groups() -> dict[tuple[str, str, str], list[str]]:
    """Build {(policy, task, split): [source_repo_ids]} from selected/*.txt."""
    groups: dict[tuple[str, str, str], list[str]] = {}
    unknown: list[str] = []
    for policy in POLICIES:
        for split in ("id", "ood"):
            f = SELECTED / f"{policy}_{split}.txt"
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                t = task_of(line)
                if t is None:
                    unknown.append(line)
                    continue
                groups.setdefault((policy, t, split), []).append(line)
    if unknown:
        raise RuntimeError(f"Unclassified slugs: {unknown}")
    return groups


def dest_repo_id(policy: str, task: str, split: str) -> str:
    return f"{DEST_ORG}/eval_{policy}_{task}_{SPLIT_LABEL[split]}"


# ---------- State persistence ---------------------------------------------

class State:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        if path.exists():
            self.data = json.loads(path.read_text())

    def get(self, dest: str) -> dict | None:
        return self.data.get(dest)

    def set(self, dest: str, **fields) -> None:
        entry = self.data.setdefault(dest, {})
        entry.update(fields)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")


# ---------- HF helpers -----------------------------------------------------

def remote_repo_exists(api, repo_id: str) -> bool:
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        return True
    except Exception:
        return False


def download_source(api, repo_id: str, dest: Path) -> None:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        # Pull everything we need to merge: meta + data + videos. Skip
        # ``images/`` since aggregate uses videos and pushed datasets
        # ignore ``images/`` too.
        ignore_patterns=["images/*", ".gitattributes"],
        max_workers=8,
    )


def upload_aggregated(api, repo_id: str, folder: Path, source_list: list[str]) -> None:
    from lerobot.datasets.utils import create_lerobot_dataset_card, load_info

    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(folder),
        ignore_patterns=["images/*"],
    )

    # Push the same LeRobot dataset card the rest of the cortexairobot
    # datasets use (no custom tags, no description) so the YAML metadata
    # warning goes away and the Hub UI matches the rest of the org.
    info = load_info(folder)
    card = create_lerobot_dataset_card(
        dataset_info=info,
        license="apache-2.0",
    )
    card.push_to_hub(repo_id=repo_id, repo_type="dataset")


# ---------- Core merge step ------------------------------------------------

def merge_one_group(
    policy: str,
    task: str,
    split: str,
    sources: list[str],
    workdir: Path,
    api,
    keep_local: bool,
    log: logging.Logger,
) -> None:
    dest = dest_repo_id(policy, task, split)
    log.info(f"[{dest}] sources={len(sources)}")

    src_dir = workdir / "src"
    src_roots: list[Path] = []
    for s in sources:
        local = src_dir / s.replace("/", "__")
        if not local.exists() or not (local / "meta" / "info.json").exists():
            log.info(f"  download  {s}")
            download_source(api, s, local)
        else:
            log.info(f"  cached    {s}")
        src_roots.append(local)

    dst_root = workdir / "dst" / dest.replace("/", "__")
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"  merge ->  {dst_root}")
    from lerobot.datasets.aggregate import aggregate_datasets  # heavy import

    aggregate_datasets(
        repo_ids=sources,
        aggr_repo_id=dest,
        roots=src_roots,
        aggr_root=dst_root,
    )

    log.info(f"  upload -> {dest}")
    upload_aggregated(api, dest, dst_root, sources)

    if not keep_local:
        log.info("  cleanup local copies")
        for r in src_roots:
            shutil.rmtree(r, ignore_errors=True)
        shutil.rmtree(dst_root, ignore_errors=True)


# ---------- CLI ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", type=Path, default=Path.home() / "lerobot_merge",
                   help="Where source/aggregated dirs are written. Default: ~/lerobot_merge")
    p.add_argument("--all", action="store_true", help="Process every group.")
    p.add_argument("--policy", choices=POLICIES, help="Filter by policy.")
    p.add_argument("--task", choices=TASKS, help="Filter by task.")
    p.add_argument("--split", choices=("id", "ood"), help="Filter by split.")
    p.add_argument("--dry-run", action="store_true", help="Print plan only.")
    p.add_argument("--force", action="store_true",
                   help="Process even if dest repo already exists on the Hub.")
    p.add_argument("--keep-local", action="store_true",
                   help="Don't delete source/aggregated dirs after upload (debug).")
    p.add_argument("--max-groups", type=int, default=None,
                   help="Stop after processing this many groups (sanity check).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("merge")

    if not (args.all or args.policy or args.task or args.split or args.dry_run):
        log.error("Specify --all or at least one of --policy/--task/--split (or --dry-run).")
        return 2

    groups = enumerate_groups()
    selected: list[tuple[tuple[str, str, str], list[str]]] = []
    for key, srcs in sorted(groups.items()):
        p, t, s = key
        if args.policy and p != args.policy:
            continue
        if args.task and t != args.task:
            continue
        if args.split and s != args.split:
            continue
        selected.append((key, srcs))

    log.info(f"Plan: {len(selected)} group(s) (out of {len(groups)} total).")
    for (p, t, s), srcs in selected:
        log.info(f"  {dest_repo_id(p, t, s):<70}  {len(srcs)} sources")

    if args.dry_run:
        return 0

    args.workdir.mkdir(parents=True, exist_ok=True)
    state = State(args.workdir / "state.json")

    from huggingface_hub import HfApi
    api = HfApi()

    processed = 0
    failures: list[tuple[str, str]] = []
    for (p, t, s), srcs in selected:
        if args.max_groups is not None and processed >= args.max_groups:
            log.info(f"Reached --max-groups={args.max_groups}, stopping.")
            break
        dest = dest_repo_id(p, t, s)

        prev = state.get(dest)
        if prev and prev.get("status") == "ok" and not args.force:
            log.info(f"[{dest}] already done in state.json, skip")
            continue
        if not args.force and remote_repo_exists(api, dest):
            log.info(f"[{dest}] exists on Hub, skip (use --force to redo)")
            state.set(dest, status="ok", reason="pre-existing")
            continue

        t0 = time.time()
        try:
            merge_one_group(
                policy=p,
                task=t,
                split=s,
                sources=srcs,
                workdir=args.workdir,
                api=api,
                keep_local=args.keep_local,
                log=log,
            )
            state.set(dest, status="ok", sources=srcs, seconds=round(time.time() - t0, 1))
            log.info(f"[{dest}] done in {time.time() - t0:.1f}s")
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            log.error(f"[{dest}] FAILED: {e}\n{tb}")
            state.set(
                dest,
                status="failed",
                error=str(e),
                seconds=round(time.time() - t0, 1),
            )
            failures.append((dest, str(e)))
        processed += 1

    log.info(f"Processed {processed} group(s); {len(failures)} failed.")
    for d, msg in failures:
        log.info(f"  FAIL {d}: {msg}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
