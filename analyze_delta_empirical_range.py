#!/usr/bin/env python3
"""
Empirical calibration of the controlled log-domain response-shift magnitudes.

For each selected sample, compute the sample-wise mean log response

    m = 0.5 * [ mean(log(H)) + mean(log(L)) ]

using normalized [0,1] intensities and log clipping at eps. This matches the
interpretation used in the current manuscript: a synchronized additive log
shift delta changes m by approximately delta before output clipping.

Outputs:
  - per_sample_log_response.csv
  - log_response_summary.csv
  - delta_context.csv
  - config.json

The delta_context table reports each requested |delta| in units of the observed
standard deviation and inter-percentile spans of m.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze empirical DE-XRT log-response range")
    p.add_argument("--csv-path", type=Path, default=Path("/root/projects/Hou_swin/split_outputs/copper_xray_all_splits.csv"))
    p.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/data/原始购买的二分类数据集/原始购买的二分类数据集"))
    p.add_argument("--split-column", type=str, default="split")
    p.add_argument("--split-values", type=str, default="train", help="Comma-separated values; use 'all' for entire CSV")
    p.add_argument("--deltas", type=str, default="0.05,0.15,0.25")
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def resolve_path(path_str: str, data_root: Path) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    path_norm = str(path_str).replace("\\", "/")
    filename = Path(path_norm).name

    # Try direct recursive filename lookup only as a last resort because some
    # datasets may contain duplicated filenames.
    parts = [x for x in Path(path_norm).parts if x not in ("/", "\\")]
    for k in range(min(6, len(parts)), 0, -1):
        cand = data_root.joinpath(*parts[-k:])
        if cand.exists():
            return cand

    direct = data_root / filename
    if direct.exists():
        return direct

    hits = list(data_root.rglob(filename))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise RuntimeError(f"Ambiguous filename {filename}: found {len(hits)} candidates")
    raise FileNotFoundError(f"Cannot resolve image: {path_str}")


def load_channel(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    # Current project loader uses /255.0. Preserve that convention here.
    # If future source images are 16-bit, this script will detect it and scale
    # by the observed integer range to avoid values >1.
    maxv = float(np.max(arr)) if arr.size else 0.0
    if maxv <= 255.0:
        arr = arr / 255.0
    elif maxv <= 65535.0:
        arr = arr / 65535.0
    else:
        arr = arr / max(maxv, 1.0)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def mean_log_response(high: np.ndarray, low: np.ndarray, eps: float) -> Dict[str, float]:
    h = np.log(np.clip(high, eps, 1.0))
    l = np.log(np.clip(low, eps, 1.0))
    mh = float(np.mean(h))
    ml = float(np.mean(l))
    return {
        "mean_log_high": mh,
        "mean_log_low": ml,
        "mean_log_response": 0.5 * (mh + ml),
        "log_energy_difference": mh - ml,
    }


def summarize_vector(v: np.ndarray) -> Dict[str, float]:
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    out: Dict[str, float] = {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "sd": float(np.std(v, ddof=1)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
        "p05_p95_span": float(np.percentile(v, 95) - np.percentile(v, 5)),
        "p01_p99_span": float(np.percentile(v, 99) - np.percentile(v, 1)),
    }
    for q in qs:
        out[f"p{q:02d}"] = float(np.percentile(v, q))
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    required = {"sample_id", "label", "high_path", "low_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    split_values = [x.strip() for x in args.split_values.split(",") if x.strip()]
    if split_values != ["all"]:
        if args.split_column not in df.columns:
            raise ValueError(f"CSV missing split column {args.split_column}")
        df = df[df[args.split_column].astype(str).isin(split_values)].copy()

    rows: List[Dict[str, object]] = []
    total = len(df)
    for j, (_, row) in enumerate(df.iterrows(), start=1):
        hp = resolve_path(str(row["high_path"]), args.data_root)
        lp = resolve_path(str(row["low_path"]), args.data_root)
        high = load_channel(hp)
        low = load_channel(lp)
        m = mean_log_response(high, low, args.eps)
        rows.append({
            "sample_id": str(row["sample_id"]),
            "label": int(row["label"]),
            args.split_column: row.get(args.split_column, ""),
            **m,
        })
        if j % 500 == 0 or j == total:
            print(f"[INFO] processed {j}/{total}", flush=True)

    per = pd.DataFrame(rows)
    per.to_csv(args.out_dir / "per_sample_log_response.csv", index=False)

    summary_rows: List[Dict[str, object]] = []
    for name, sdf in [("all_selected", per)] + [(f"label_{lab}", g) for lab, g in per.groupby("label")]:
        for col in ["mean_log_response", "mean_log_high", "mean_log_low", "log_energy_difference"]:
            stats = summarize_vector(sdf[col].to_numpy(dtype=float))
            summary_rows.append({"subset": name, "variable": col, **stats})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "log_response_summary.csv", index=False)

    m = per["mean_log_response"].to_numpy(dtype=float)
    stats = summarize_vector(m)
    sd = stats["sd"]
    iqr = stats["iqr"]
    pspan = stats["p05_p95_span"]
    deltas = [abs(float(x.strip())) for x in args.deltas.split(",") if x.strip()]
    drows = []
    for d in deltas:
        drows.append({
            "abs_delta": d,
            "multiplicative_factor_exp_delta": float(np.exp(d)),
            "delta_in_sd": float(d / sd) if sd > 0 else float("nan"),
            "delta_as_fraction_of_iqr": float(d / iqr) if iqr > 0 else float("nan"),
            "delta_as_fraction_of_p05_p95_span": float(d / pspan) if pspan > 0 else float("nan"),
            "two_sided_span_2delta_in_sd": float((2 * d) / sd) if sd > 0 else float("nan"),
        })
    pd.DataFrame(drows).to_csv(args.out_dir / "delta_context.csv", index=False)

    config = {
        "csv_path": str(args.csv_path),
        "data_root": str(args.data_root),
        "split_column": args.split_column,
        "split_values": split_values,
        "eps": args.eps,
        "deltas": deltas,
        "definition": "m=0.5*(mean(log(H))+mean(log(L)))",
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[RESULT] mean_log_response")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[OK] outputs -> {args.out_dir}")


if __name__ == "__main__":
    main()
