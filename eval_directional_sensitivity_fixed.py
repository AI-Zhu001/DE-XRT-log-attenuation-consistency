#!/usr/bin/env python3
"""
Directional sensitivity analysis for dual-energy X-ray classifiers.

Purpose
-------
This script evaluates already-trained checkpoints. It does NOT retrain models.
It measures local prediction sensitivity along two physically interpretable
log-response directions in the paired high/low-energy input space:

  common:       (delta_H, delta_L) = (+eps, +eps) vs (-eps, -eps)
  differential: (delta_H, delta_L) = (+eps, -eps) vs (-eps, +eps)

For binary classifiers, the script reports both probability sensitivity and
logit-margin sensitivity, plus prediction flip rates. It can evaluate a single
checkpoint or a manifest of checkpoints and aggregate run-level results by
method.

Expected project layout
-----------------------
The script follows the existing project code and imports:
    from datasets.copper_xray_dataset import CopperXRayDataset

Place this file in <project_root>/lunwen1xiugai/. The script resolves the local
datasets/copper_xray_dataset.py module automatically, so no PYTHONPATH change is needed.

Manifest format (optional)
--------------------------
CSV columns:
    method,seed,arch,ckpt_path,img_size

Example:
    Baseline,42,resnet18,/path/to/baseline_seed42.pt,192
    Consistency,42,resnet18,/path/to/consistency_seed42.pt,192
    CI-Consistency,42,resnet18,/path/to/ci_consistency_seed42.pt,192

Supported arch values: resnet18, swin_t
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

# This script is intended to live in <project_root>/lunwen1xiugai/.
# Import the local dataset module by putting <project_root>/datasets first on
# sys.path. This avoids conflicts with the third-party HuggingFace package
# that is also named "datasets".
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOCAL_DATASETS_DIR = PROJECT_ROOT / "datasets"
if not (LOCAL_DATASETS_DIR / "copper_xray_dataset.py").exists():
    raise FileNotFoundError(
        f"Expected local dataset module at: {LOCAL_DATASETS_DIR / 'copper_xray_dataset.py'}"
    )
sys.path.insert(0, str(LOCAL_DATASETS_DIR))
from copper_xray_dataset import CopperXRayDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate local DE-XRT response-direction sensitivity")

    p.add_argument("--csv-path", type=Path, required=True)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--split-column", type=str, default="split")
    p.add_argument("--split-value", type=str, default="test")

    # Multi-checkpoint mode.
    p.add_argument("--manifest", type=Path, default=None,
                   help="Optional CSV with method,seed,arch,ckpt_path,img_size columns")

    # Single-checkpoint mode.
    p.add_argument("--ckpt-path", type=Path, default=None)
    p.add_argument("--method-name", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "swin_t"])
    p.add_argument("--img-size", type=int, default=None,
                   help="Default: 192 for ResNet18, 224 for Swin-T")

    p.add_argument("--eps-list", type=str, default="0.01,0.025,0.05")
    p.add_argument("--directions", type=str, default="common,differential",
                   help="Comma-separated subset of common,differential,high_only,low_only")
    p.add_argument("--log-eps", type=float, default=1e-6)

    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--save-samples", action="store_true", default=False,
                   help="Save sample-level sensitivity values in addition to run summaries")

    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def build_resnet18_2ch(num_classes: int = 2) -> nn.Module:
    # No pretrained download is needed for evaluation because the checkpoint
    # overwrites the parameters.
    m = resnet18(weights=None)
    old = m.conv1
    m.conv1 = nn.Conv2d(
        2,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    )
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_swin_t_2ch(num_classes: int = 2) -> nn.Module:
    try:
        from timm import create_model
    except ImportError as exc:
        raise RuntimeError("Swin-T evaluation requires timm") from exc

    m = create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_classes)
    old = m.patch_embed.proj
    # Match the training/evaluation architecture used in the existing code.
    m.patch_embed.proj = nn.Conv2d(
        2,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
    )
    return m


def build_model(arch: str) -> nn.Module:
    if arch == "resnet18":
        return build_resnet18_2ch()
    if arch == "swin_t":
        return build_swin_t_2ch()
    raise ValueError(f"Unsupported architecture: {arch}")


def _unwrap_state_dict(obj: object) -> Dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint is not a state-dict-like object")

    # Direct state_dict is the format used by the existing training scripts.
    if obj and all(isinstance(k, str) for k in obj.keys()) and any(
        isinstance(v, torch.Tensor) for v in obj.values()
    ):
        state = obj  # type: ignore[assignment]
    else:
        state = None
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = obj.get(key)  # type: ignore[union-attr]
            if isinstance(candidate, dict):
                state = candidate
                break
        if state is None:
            raise KeyError("Could not find state_dict/model_state_dict/model in checkpoint")

    # Be tolerant of DataParallel checkpoints.
    if any(str(k).startswith("module.") for k in state.keys()):
        state = {str(k).removeprefix("module."): v for k, v in state.items()}
    return state


def load_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    obj = torch.load(ckpt_path, map_location=device)
    state = _unwrap_state_dict(obj)
    model.load_state_dict(state, strict=True)


def make_dataset(
    csv_path: Path,
    data_root: str,
    split_column: str,
    split_value: str,
    img_size: int,
) -> CopperXRayDataset:
    return CopperXRayDataset(
        csv_path=csv_path,
        split_column=split_column,
        split_value=split_value,
        data_root=data_root,
        out_size=img_size,
        hflip_p=0.0,
        thickness_aug=False,
        use_brightness_aug=False,
    )


def apply_log_response_shift(
    x: torch.Tensor,
    delta_high: float,
    delta_low: float,
    log_eps: float = 1e-6,
) -> torch.Tensor:
    """Apply deterministic channel-wise additive shifts in log intensity."""
    x_safe = torch.clamp(x, log_eps, 1.0)
    out = x.clone()
    out[:, 0:1] = torch.clamp(
        torch.exp(torch.log(x_safe[:, 0:1]) + float(delta_high)), 0.0, 1.0
    )
    out[:, 1:2] = torch.clamp(
        torch.exp(torch.log(x_safe[:, 1:2]) + float(delta_low)), 0.0, 1.0
    )
    return out


def direction_pair(name: str, eps: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return (+ direction, - direction) channel offsets for a given epsilon."""
    if name == "common":
        return (eps, eps), (-eps, -eps)
    if name == "differential":
        return (eps, -eps), (-eps, eps)
    if name == "high_only":
        return (eps, 0.0), (-eps, 0.0)
    if name == "low_only":
        return (0.0, eps), (0.0, -eps)
    raise ValueError(f"Unknown direction: {name}")


def _stats(v: np.ndarray) -> Dict[str, float]:
    if v.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(v)),
        "sd": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
    }


@torch.no_grad()
def evaluate_one(
    model: nn.Module,
    loader: DataLoader,
    dataset: CopperXRayDataset,
    device: torch.device,
    method: str,
    seed: int,
    arch: str,
    eps_list: Iterable[float],
    directions: Iterable[str],
    log_eps: float,
    amp: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()

    run_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []

    # Sample metadata follows dataset order because loader shuffle=False.
    meta = dataset.samples.reset_index(drop=True)

    for direction in directions:
        for eps in eps_list:
            plus_delta, minus_delta = direction_pair(direction, float(eps))

            all_prob_sens: List[np.ndarray] = []
            all_margin_sens: List[np.ndarray] = []
            all_clean_dev_prob: List[np.ndarray] = []
            all_pm_flip: List[np.ndarray] = []
            all_any_clean_flip: List[np.ndarray] = []
            all_clean_prob: List[np.ndarray] = []
            all_plus_prob: List[np.ndarray] = []
            all_minus_prob: List[np.ndarray] = []
            all_clean_pred: List[np.ndarray] = []
            all_plus_pred: List[np.ndarray] = []
            all_minus_pred: List[np.ndarray] = []
            all_labels: List[np.ndarray] = []

            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                x_plus = apply_log_response_shift(x, plus_delta[0], plus_delta[1], log_eps)
                x_minus = apply_log_response_shift(x, minus_delta[0], minus_delta[1], log_eps)

                use_amp = amp and device.type == "cuda"
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    z0 = model(x)
                    zp = model(x_plus)
                    zm = model(x_minus)

                p0 = torch.softmax(z0.float(), dim=1)[:, 1]
                pp = torch.softmax(zp.float(), dim=1)[:, 1]
                pm = torch.softmax(zm.float(), dim=1)[:, 1]

                m0 = (z0[:, 1] - z0[:, 0]).float()
                mp = (zp[:, 1] - zp[:, 0]).float()
                mm = (zm[:, 1] - zm[:, 0]).float()

                pred0 = z0.argmax(dim=1)
                predp = zp.argmax(dim=1)
                predm = zm.argmax(dim=1)

                # Central finite-difference sensitivity. Because both directions
                # use the same per-channel eps magnitude, common and differential
                # values are directly comparable under this parameterization.
                prob_sens = torch.abs(pp - pm) / (2.0 * float(eps))
                margin_sens = torch.abs(mp - mm) / (2.0 * float(eps))

                # Symmetric deviation from the clean prediction. This is useful
                # when finite-epsilon nonlinearity causes central-difference
                # cancellation.
                clean_dev_prob = (
                    torch.abs(pp - p0) + torch.abs(pm - p0)
                ) / (2.0 * float(eps))

                pm_flip = (predp != predm).float()
                any_clean_flip = ((predp != pred0) | (predm != pred0)).float()

                for src, dst in [
                    (prob_sens, all_prob_sens),
                    (margin_sens, all_margin_sens),
                    (clean_dev_prob, all_clean_dev_prob),
                    (pm_flip, all_pm_flip),
                    (any_clean_flip, all_any_clean_flip),
                    (p0, all_clean_prob),
                    (pp, all_plus_prob),
                    (pm, all_minus_prob),
                    (pred0, all_clean_pred),
                    (predp, all_plus_pred),
                    (predm, all_minus_pred),
                    (y, all_labels),
                ]:
                    dst.append(src.detach().cpu().numpy())

            prob_sens = np.concatenate(all_prob_sens)
            margin_sens = np.concatenate(all_margin_sens)
            clean_dev_prob = np.concatenate(all_clean_dev_prob)
            pm_flip = np.concatenate(all_pm_flip)
            any_clean_flip = np.concatenate(all_any_clean_flip)
            clean_prob = np.concatenate(all_clean_prob)
            plus_prob = np.concatenate(all_plus_prob)
            minus_prob = np.concatenate(all_minus_prob)
            clean_pred = np.concatenate(all_clean_pred)
            plus_pred = np.concatenate(all_plus_pred)
            minus_pred = np.concatenate(all_minus_pred)
            labels = np.concatenate(all_labels)

            s_prob = _stats(prob_sens)
            s_margin = _stats(margin_sens)
            s_dev = _stats(clean_dev_prob)

            run_rows.append({
                "method": method,
                "seed": seed,
                "arch": arch,
                "direction": direction,
                "eps": float(eps),
                "n": int(prob_sens.size),
                "prob_sens_mean": s_prob["mean"],
                "prob_sens_sd": s_prob["sd"],
                "prob_sens_median": s_prob["median"],
                "prob_sens_p95": s_prob["p95"],
                "margin_sens_mean": s_margin["mean"],
                "margin_sens_sd": s_margin["sd"],
                "margin_sens_median": s_margin["median"],
                "margin_sens_p95": s_margin["p95"],
                "clean_dev_prob_mean": s_dev["mean"],
                "clean_dev_prob_median": s_dev["median"],
                "plus_minus_flip_rate": float(np.mean(pm_flip)),
                "any_vs_clean_flip_rate": float(np.mean(any_clean_flip)),
                "clean_acc": float(np.mean(clean_pred == labels)),
                "plus_acc": float(np.mean(plus_pred == labels)),
                "minus_acc": float(np.mean(minus_pred == labels)),
                "delta_high_plus": float(plus_delta[0]),
                "delta_low_plus": float(plus_delta[1]),
                "delta_high_minus": float(minus_delta[0]),
                "delta_low_minus": float(minus_delta[1]),
            })

            # Sample-level output, joined to metadata by dataset order.
            for i in range(prob_sens.size):
                row = {
                    "method": method,
                    "seed": seed,
                    "arch": arch,
                    "direction": direction,
                    "eps": float(eps),
                    "sample_index": i,
                    "label": int(labels[i]),
                    "prob_sensitivity": float(prob_sens[i]),
                    "margin_sensitivity": float(margin_sens[i]),
                    "clean_dev_prob": float(clean_dev_prob[i]),
                    "plus_minus_flip": int(pm_flip[i]),
                    "any_vs_clean_flip": int(any_clean_flip[i]),
                    "clean_prob_class1": float(clean_prob[i]),
                    "plus_prob_class1": float(plus_prob[i]),
                    "minus_prob_class1": float(minus_prob[i]),
                    "clean_pred": int(clean_pred[i]),
                    "plus_pred": int(plus_pred[i]),
                    "minus_pred": int(minus_pred[i]),
                }
                if i < len(meta):
                    for col in ("sample_id", "thickness_group", "mean_gray"):
                        if col in meta.columns:
                            val = meta.iloc[i][col]
                            if pd.notna(val):
                                row[col] = val
                sample_rows.append(row)

    return pd.DataFrame(run_rows), pd.DataFrame(sample_rows)


def load_runs(args: argparse.Namespace) -> pd.DataFrame:
    if args.manifest is not None:
        df = pd.read_csv(args.manifest)
        required = {"method", "seed", "ckpt_path"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        if "arch" not in df.columns:
            df["arch"] = "resnet18"
        if "img_size" not in df.columns:
            df["img_size"] = np.nan
        return df

    if args.ckpt_path is None or args.method_name is None:
        raise ValueError("Use --manifest, or provide both --ckpt-path and --method-name")

    return pd.DataFrame([{
        "method": args.method_name,
        "seed": args.seed,
        "arch": args.arch,
        "ckpt_path": str(args.ckpt_path),
        "img_size": args.img_size,
    }])


def aggregate_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()

    metrics = [
        "prob_sens_mean",
        "margin_sens_mean",
        "clean_dev_prob_mean",
        "plus_minus_flip_rate",
        "any_vs_clean_flip_rate",
        "clean_acc",
        "plus_acc",
        "minus_acc",
    ]
    rows: List[Dict[str, object]] = []
    for keys, g in run_df.groupby(["method", "arch", "direction", "eps"], dropna=False):
        method, arch, direction, eps = keys
        row: Dict[str, object] = {
            "method": method,
            "arch": arch,
            "direction": direction,
            "eps": eps,
            "n_runs": int(g.shape[0]),
        }
        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{m}_run_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            row[f"{m}_run_sd"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0 if vals.size == 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "arch", "direction", "eps"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    eps_list = [float(s.strip()) for s in args.eps_list.split(",") if s.strip()]
    directions = [s.strip() for s in args.directions.split(",") if s.strip()]
    valid_directions = {"common", "differential", "high_only", "low_only"}
    if not directions or any(d not in valid_directions for d in directions):
        raise ValueError(f"directions must be chosen from {sorted(valid_directions)}")
    if any(e <= 0 for e in eps_list):
        raise ValueError("All eps values must be > 0")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    runs = load_runs(args)

    all_run: List[pd.DataFrame] = []
    all_samples: List[pd.DataFrame] = []

    for idx, r in runs.iterrows():
        method = str(r["method"])
        seed = int(r["seed"])
        arch = str(r.get("arch", "resnet18"))
        ckpt_path = Path(str(r["ckpt_path"]))

        raw_size = r.get("img_size", np.nan)
        if pd.notna(raw_size):
            img_size = int(raw_size)
        elif args.img_size is not None:
            img_size = int(args.img_size)
        else:
            img_size = 224 if arch == "swin_t" else 192

        print(f"[INFO] {idx + 1}/{len(runs)} method={method} seed={seed} arch={arch}")
        print(f"       ckpt={ckpt_path}")

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        dataset = make_dataset(
            args.csv_path, args.data_root, args.split_column, args.split_value, img_size
        )
        loader = DataLoader(
            dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        model = build_model(arch).to(device)
        load_checkpoint(model, ckpt_path, device)

        run_df, sample_df = evaluate_one(
            model=model,
            loader=loader,
            dataset=dataset,
            device=device,
            method=method,
            seed=seed,
            arch=arch,
            eps_list=eps_list,
            directions=directions,
            log_eps=args.log_eps,
            amp=args.amp,
        )
        all_run.append(run_df)
        if args.save_samples:
            all_samples.append(sample_df)

        # Free GPU memory between checkpoints.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    run_all = pd.concat(all_run, ignore_index=True) if all_run else pd.DataFrame()
    agg = aggregate_runs(run_all)

    run_path = args.out_dir / "directional_sensitivity_by_run.csv"
    agg_path = args.out_dir / "directional_sensitivity_summary.csv"
    run_all.to_csv(run_path, index=False)
    agg.to_csv(agg_path, index=False)

    if args.save_samples and all_samples:
        sample_path = args.out_dir / "directional_sensitivity_by_sample.csv"
        pd.concat(all_samples, ignore_index=True).to_csv(sample_path, index=False)
        print(f"[OK] sample-level: {sample_path}")

    config_path = args.out_dir / "directional_sensitivity_config.json"
    config = {
        "csv_path": str(args.csv_path),
        "data_root": args.data_root,
        "split_column": args.split_column,
        "split_value": args.split_value,
        "eps_list": eps_list,
        "directions": directions,
        "log_eps": args.log_eps,
        "amp": bool(args.amp),
        "n_checkpoints": int(len(runs)),
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] run-level: {run_path}")
    print(f"[OK] summary:   {agg_path}")
    print(f"[OK] config:    {config_path}")


if __name__ == "__main__":
    main()
