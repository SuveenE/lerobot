"""Convert a lerobot dataset's ``meta/stats.json`` into Cosmos Policy's flat
``dataset_statistics.json`` schema.

Schema verified against the canonical reference release
[nvidia/Cosmos-Policy-ALOHA-Predict2-2B/aloha_dataset_statistics.json](
https://huggingface.co/nvidia/Cosmos-Policy-ALOHA-Predict2-2B/blob/main/aloha_dataset_statistics.json
):

```json
{
  "actions_min":    [14 floats],
  "actions_max":    [14 floats],
  "actions_mean":   [14 floats],
  "actions_std":    [14 floats],
  "actions_median": [14 floats],
  "proprio_min":    [14 floats],
  "proprio_max":    [14 floats],
  "proprio_mean":   [14 floats],
  "proprio_std":    [14 floats],
  "proprio_median": [14 floats]
}
```

lerobot's ``meta/stats.json`` is nested as
``{feature: {stat: list, ...}, ...}`` with ``min, max, mean, std, count``.  It
does NOT contain ``median`` natively, so we either:

1. fall back to using ``mean`` as a stand-in for ``median`` (default, fast), or
2. compute true per-dim medians from the dataset's parquet shards (opt in with
   ``--compute_median_from_parquet``).

Example:

    python -m lerobot.policies.cosmos.scripts.convert_lerobot_stats \
      --lerobot_dataset_dir $HOME/cosmos_datasets/02042026-cube-stacking-combined \
      --out_path             $HOME/cosmos_ckpts/yam_cube_stacking_20260415/yam_dataset_statistics.json
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ACTION_FEATURE_CANDIDATES = ("action", "actions")
STATE_FEATURE_CANDIDATES = (
    "observation.state",
    "observation.proprio",
    "observation.proprioception",
    "state",
    "proprio",
)
STATS_KEYS = ("min", "max", "mean", "std")


def _find_first(d: dict, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in d:
            return c
    return None


def _as_list(v) -> list[float]:
    return np.asarray(v, dtype=np.float64).reshape(-1).tolist()


def _compute_median_from_parquet(dataset_dir: Path, column: str) -> list[float] | None:
    """Compute per-dim median by concatenating the column across all parquet shards.

    Returns ``None`` if no parquet files are found or the column is absent.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.warning("pyarrow not installed; skipping parquet-based median computation.")
        return None

    parquet_files = sorted(
        glob.glob(str(dataset_dir / "data" / "**" / "*.parquet"), recursive=True)
    )
    if not parquet_files:
        parquet_files = sorted(
            glob.glob(str(dataset_dir / "**" / "*.parquet"), recursive=True)
        )
    if not parquet_files:
        logger.warning("No parquet files found under %s; cannot compute median.", dataset_dir)
        return None

    chunks: list[np.ndarray] = []
    for fp in parquet_files:
        try:
            tbl = pq.read_table(fp, columns=[column])
        except Exception as e:
            logger.debug("Skipping %s (column %r missing or unreadable: %s)", fp, column, e)
            continue
        arr = tbl.column(column).to_pylist()
        if not arr:
            continue
        chunks.append(np.asarray(arr, dtype=np.float64))

    if not chunks:
        logger.warning("Column %r not found in any parquet shard; falling back to mean.", column)
        return None

    stacked = np.concatenate(chunks, axis=0)
    if stacked.ndim == 1:
        stacked = stacked.reshape(-1, 1)
    median = np.median(stacked, axis=0)
    return median.reshape(-1).tolist()


def _load_lerobot_stats(dataset_dir: Path) -> dict:
    stats_path = dataset_dir / "meta" / "stats.json"
    episodes_stats_path = dataset_dir / "meta" / "episodes_stats.jsonl"
    if stats_path.is_file():
        logger.info("Loading %s", stats_path)
        with open(stats_path) as f:
            return json.load(f)

    if episodes_stats_path.is_file():
        logger.info(
            "meta/stats.json missing; aggregating per-episode stats from %s",
            episodes_stats_path,
        )
        per_ep: dict[str, list[dict]] = {}
        with open(episodes_stats_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                ep_stats = ep.get("stats", ep)
                for feature, feat_stats in ep_stats.items():
                    if not isinstance(feat_stats, dict) or "min" not in feat_stats:
                        continue
                    per_ep.setdefault(feature, []).append(feat_stats)
        agg: dict[str, dict] = {}
        for feature, ep_list in per_ep.items():
            mins = np.stack([np.asarray(e["min"], dtype=np.float64) for e in ep_list])
            maxs = np.stack([np.asarray(e["max"], dtype=np.float64) for e in ep_list])
            means = np.stack([np.asarray(e["mean"], dtype=np.float64) for e in ep_list])
            stds = np.stack([np.asarray(e["std"], dtype=np.float64) for e in ep_list])
            counts = np.asarray(
                [np.asarray(e.get("count", [1]), dtype=np.float64).reshape(-1)[0] for e in ep_list]
            )
            total = counts.sum()
            weighted_mean = (means * counts[:, None]).sum(axis=0) / total
            between = ((means - weighted_mean) ** 2 * counts[:, None]).sum(axis=0) / total
            within = ((stds**2) * counts[:, None]).sum(axis=0) / total
            weighted_std = np.sqrt(within + between)
            agg[feature] = {
                "min": mins.min(axis=0).tolist(),
                "max": maxs.max(axis=0).tolist(),
                "mean": weighted_mean.tolist(),
                "std": weighted_std.tolist(),
                "count": [int(total)],
            }
        return agg

    raise FileNotFoundError(
        f"Could not find meta/stats.json or meta/episodes_stats.jsonl under {dataset_dir}"
    )


def convert(
    lerobot_dataset_dir: Path,
    out_path: Path,
    action_feature: str | None = None,
    state_feature: str | None = None,
    compute_median_from_parquet: bool = False,
) -> dict:
    """Convert lerobot stats to cosmos-policy flat schema and write to ``out_path``."""
    lerobot_dataset_dir = Path(lerobot_dataset_dir)
    out_path = Path(out_path)

    stats = _load_lerobot_stats(lerobot_dataset_dir)

    action_key = action_feature or _find_first(stats, ACTION_FEATURE_CANDIDATES)
    state_key = state_feature or _find_first(stats, STATE_FEATURE_CANDIDATES)

    if action_key is None:
        raise KeyError(
            f"No action feature found in lerobot stats.  Expected one of {ACTION_FEATURE_CANDIDATES}; "
            f"got keys={sorted(stats.keys())}.  Use --action_feature to override."
        )
    if state_key is None:
        raise KeyError(
            f"No proprio/state feature found in lerobot stats.  Expected one of "
            f"{STATE_FEATURE_CANDIDATES}; got keys={sorted(stats.keys())}.  "
            f"Use --state_feature to override."
        )

    for key, stats_block in [("action", stats[action_key]), ("proprio", stats[state_key])]:
        missing = [k for k in STATS_KEYS if k not in stats_block]
        if missing:
            raise KeyError(
                f"lerobot stats for {key!r} is missing required fields {missing}.  "
                f"Present fields: {sorted(stats_block.keys())}."
            )

    action_stats = stats[action_key]
    state_stats = stats[state_key]

    logger.info(
        "Mapping lerobot[%r] -> cosmos[actions_*], lerobot[%r] -> cosmos[proprio_*]",
        action_key,
        state_key,
    )

    if compute_median_from_parquet:
        logger.info("Computing medians from parquet shards (this can be slow)...")
        action_median = _compute_median_from_parquet(lerobot_dataset_dir, action_key)
        state_median = _compute_median_from_parquet(lerobot_dataset_dir, state_key)
    else:
        action_median = None
        state_median = None

    if action_median is None:
        logger.info("Using actions_mean as a stand-in for actions_median.")
        action_median = _as_list(action_stats["mean"])
    if state_median is None:
        logger.info("Using proprio_mean as a stand-in for proprio_median.")
        state_median = _as_list(state_stats["mean"])

    action_dim = len(_as_list(action_stats["min"]))
    state_dim = len(_as_list(state_stats["min"]))
    logger.info("Inferred action_dim=%d, proprio_dim=%d", action_dim, state_dim)

    out = {
        "actions_min": _as_list(action_stats["min"]),
        "actions_max": _as_list(action_stats["max"]),
        "actions_mean": _as_list(action_stats["mean"]),
        "actions_std": _as_list(action_stats["std"]),
        "actions_median": action_median,
        "proprio_min": _as_list(state_stats["min"]),
        "proprio_max": _as_list(state_stats["max"]),
        "proprio_mean": _as_list(state_stats["mean"]),
        "proprio_std": _as_list(state_stats["std"]),
        "proprio_median": state_median,
    }

    for k, v in out.items():
        if len(v) not in (action_dim, state_dim):
            raise ValueError(
                f"Converted field {k!r} has length {len(v)}, expected {action_dim} or {state_dim}."
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", out_path)
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert a lerobot dataset's meta/stats.json to Cosmos Policy's "
        "flat dataset_statistics.json schema."
    )
    p.add_argument(
        "--lerobot_dataset_dir",
        type=Path,
        required=True,
        help="Path to the lerobot dataset root (contains a meta/ directory).",
    )
    p.add_argument(
        "--out_path",
        type=Path,
        required=True,
        help="Destination JSON file for the cosmos-schema stats.",
    )
    p.add_argument(
        "--action_feature",
        type=str,
        default=None,
        help=f"Name of the action feature in the lerobot stats.  "
        f"Default: first of {ACTION_FEATURE_CANDIDATES} that exists.",
    )
    p.add_argument(
        "--state_feature",
        type=str,
        default=None,
        help=f"Name of the proprio/state feature in the lerobot stats.  "
        f"Default: first of {STATE_FEATURE_CANDIDATES} that exists.",
    )
    p.add_argument(
        "--compute_median_from_parquet",
        action="store_true",
        help="Compute per-dim medians by scanning the dataset's parquet shards.  "
        "If omitted, mean is used as a stand-in for median.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    convert(
        lerobot_dataset_dir=args.lerobot_dataset_dir,
        out_path=args.out_path,
        action_feature=args.action_feature,
        state_feature=args.state_feature,
        compute_median_from_parquet=args.compute_median_from_parquet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
