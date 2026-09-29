#!/usr/bin/env python3
"""
2D high-/low-energy response-plane evaluation for trained DE-XRT classifiers.

For each checkpoint and each (delta_H, delta_L) point on a user-defined grid,
apply deterministic channel-wise log-domain shifts and report:
  - Accuracy
  - Balanced accuracy
  - Macro-F1
  - mean absolute logit-margin change vs clean input
  - mean absolute positive-class probability change vs clean input
  - prediction flip rate vs clean input

The script also summarizes response-plane "flatness" for each run and method,
including common-mode diagonal and differential-mode anti-diagonal behavior.

Manifest CSV columns:
    method,seed,arch,ckpt_path,img_size

Supported arch: resnet18, swin_t
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18


def _import_local_dataset():
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "datasets",
        Path(os.environ.get("HOU_PROJECT_ROOT", "/root/projects/Hou_swin")) / "datasets",
    ]
    for d in candidates:
        if (d / "copper_xray_dataset.py").exists():
            sys.path.insert(0, str(d))
            from copper_xray_dataset import CopperXRayDataset  # type: ignore
            return CopperXRayDataset
    raise FileNotFoundError(
        "Cannot locate datasets/copper_xray_dataset.py. "
        "Place this script under <Hou_swin>/lunwen1xiugai/ or set HOU_PROJECT_ROOT."
    )


CopperXRayDataset = _import_local_dataset()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate 2D DE-XRT response plane")
    p.add_argument("--csv-path", type=Path, default=Path("/root/projects/Hou_swin/split_outputs/copper_xray_all_splits.csv"))
    p.add_argument("--data-root", type=str, default="/root/autodl-tmp/data/原始购买的二分类数据集/原始购买的二分类数据集")
    p.add_argument("--split-column", type=str, default="split")
    p.add_argument("--split-value", type=str, default="test")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument(
        "--grid",
        type=str,
        default="-0.15,-0.10,-0.05,0,0.05,0.10,0.15",
        help="Comma-separated delta values used for both H and L axes",
    )
    p.add_argument("--log-eps", type=float, default=1e-6)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def build_resnet18_2ch() -> nn.Module:
    m = resnet18(weights=None)
    old = m.conv1
    m.conv1 = nn.Conv2d(
        2, old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    )
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def build_swin_t_2ch() -> nn.Module:
    try:
        from timm import create_model
    except ImportError as exc:
        raise RuntimeError("Swin-T evaluation requires timm") from exc
    m = create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=2)
    old = m.patch_embed.proj
    m.patch_embed.proj = nn.Conv2d(
        2,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    )
    return m


def build_model(arch: str) -> nn.Module:
    arch = str(arch).strip().lower()
    if arch == "resnet18":
        return build_resnet18_2ch()
    if arch == "swin_t":
        return build_swin_t_2ch()
    raise ValueError(f"Unsupported architecture: {arch}")


def _unwrap_state_dict(obj: object) -> Dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint is not a dict/state_dict")
    if obj and any(isinstance(v, torch.Tensor) for v in obj.values()):
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
    if any(str(k).startswith("module.") for k in state.keys()):
        state = {str(k).removeprefix("module."): v for k, v in state.items()}
    return state


def load_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    obj = torch.load(ckpt_path, map_location=device)
    state = _unwrap_state_dict(obj)
    model.load_state_dict(state, strict=True)


def make_dataset(args: argparse.Namespace, img_size: int):
    return CopperXRayDataset(
        args.csv_path,
        split_column=args.split_column,
        split_value=args.split_value,
        data_root=args.data_root,
        out_size=img_size,
        hflip_p=0.0,
        thickness_aug=False,
        use_brightness_aug=False,
    )


def apply_channel_log_shift(
    x: torch.Tensor,
    delta_high: float,
    delta_low: float,
    log_eps: float,
) -> torch.Tensor:
    x_safe = torch.clamp(x, log_eps, 1.0)
    out = x.clone()
    out[:, 0:1] = torch.clamp(
        torch.exp(torch.log(x_safe[:, 0:1]) + float(delta_high)), 0.0, 1.0
    )
    out[:, 1:2] = torch.clamp(
        torch.exp(torch.log(x_safe[:, 1:2]) + float(delta_low)), 0.0, 1.0
    )
    return out


def binary_margin(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, 1] - logits[:, 0]


def positive_prob(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, 1]


def calc_classification_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(y, pred)),
        "bal_acc": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


@torch.no_grad()
def collect_clean_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ys: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    margins: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    model.eval()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
            z = model(x)
        ys.append(y.numpy())
        preds.append(z.argmax(1).cpu().numpy())
        margins.append(binary_margin(z).float().cpu().numpy())
        probs.append(positive_prob(z).float().cpu().numpy())
    return (
        np.concatenate(ys),
        np.concatenate(preds),
        np.concatenate(margins),
        np.concatenate(probs),
    )


@torch.no_grad()
def collect_shifted_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    delta_high: float,
    delta_low: float,
    log_eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds: List[np.ndarray] = []
    margins: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    model.eval()
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        x_s = apply_channel_log_shift(x, delta_high, delta_low, log_eps)
        with torch.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
            z = model(x_s)
        preds.append(z.argmax(1).cpu().numpy())
        margins.append(binary_margin(z).float().cpu().numpy())
        probs.append(positive_prob(z).float().cpu().numpy())
    return np.concatenate(preds), np.concatenate(margins), np.concatenate(probs)


def summarize_flatness(run_df: pd.DataFrame) -> Dict[str, float]:
    off = run_df[~((run_df.delta_high == 0.0) & (run_df.delta_low == 0.0))].copy()
    common = run_df[np.isclose(run_df.delta_high, run_df.delta_low) & ~np.isclose(run_df.delta_high, 0.0)]
    diff = run_df[np.isclose(run_df.delta_high, -run_df.delta_low) & ~np.isclose(run_df.delta_high, 0.0)]

    def mean_or_nan(df: pd.DataFrame, col: str) -> float:
        return float(df[col].mean()) if len(df) else float("nan")

    return {
        "plane_mean_abs_margin_change": mean_or_nan(off, "mean_abs_margin_change"),
        "plane_max_abs_margin_change": float(off["mean_abs_margin_change"].max()) if len(off) else float("nan"),
        "plane_mean_flip_rate": mean_or_nan(off, "flip_rate_vs_clean"),
        "plane_worst_macro_f1": float(off["macro_f1"].min()) if len(off) else float("nan"),
        "plane_mean_macro_f1": mean_or_nan(off, "macro_f1"),
        "common_diag_mean_abs_margin_change": mean_or_nan(common, "mean_abs_margin_change"),
        "common_diag_mean_macro_f1": mean_or_nan(common, "macro_f1"),
        "differential_diag_mean_abs_margin_change": mean_or_nan(diff, "mean_abs_margin_change"),
        "differential_diag_mean_macro_f1": mean_or_nan(diff, "macro_f1"),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    grid = [float(x.strip()) for x in args.grid.split(",") if x.strip()]
    manifest = pd.read_csv(args.manifest)
    required = {"method", "seed", "arch", "ckpt_path", "img_size"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    all_rows: List[Dict[str, object]] = []
    flat_rows: List[Dict[str, object]] = []

    dataset_cache: Dict[int, object] = {}
    loader_cache: Dict[int, DataLoader] = {}

    for idx, rec in manifest.iterrows():
        method = str(rec["method"])
        seed = int(rec["seed"])
        arch = str(rec["arch"])
        ckpt = Path(str(rec["ckpt_path"]))
        img_size = int(rec["img_size"])

        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        if img_size not in dataset_cache:
            ds = make_dataset(args, img_size)
            loader = DataLoader(
                ds,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            dataset_cache[img_size] = ds
            loader_cache[img_size] = loader
        loader = loader_cache[img_size]

        model = build_model(arch).to(device)
        load_checkpoint(model, ckpt, device)
        model.eval()

        print(f"[RUN {idx+1}/{len(manifest)}] {method} seed={seed} arch={arch}", flush=True)
        y, clean_pred, clean_margin, clean_prob = collect_clean_outputs(model, loader, device, args.amp)
        clean_metrics = calc_classification_metrics(y, clean_pred)

        run_rows: List[Dict[str, object]] = []
        for dh in grid:
            for dl in grid:
                if math.isclose(dh, 0.0, abs_tol=1e-12) and math.isclose(dl, 0.0, abs_tol=1e-12):
                    pred = clean_pred
                    margin = clean_margin
                    prob = clean_prob
                else:
                    pred, margin, prob = collect_shifted_outputs(
                        model, loader, device, args.amp, dh, dl, args.log_eps
                    )
                metrics = calc_classification_metrics(y, pred)
                row: Dict[str, object] = {
                    "method": method,
                    "seed": seed,
                    "arch": arch,
                    "ckpt_path": str(ckpt),
                    "img_size": img_size,
                    "delta_high": float(dh),
                    "delta_low": float(dl),
                    "acc": metrics["acc"],
                    "bal_acc": metrics["bal_acc"],
                    "macro_f1": metrics["macro_f1"],
                    "clean_acc": clean_metrics["acc"],
                    "clean_bal_acc": clean_metrics["bal_acc"],
                    "clean_macro_f1": clean_metrics["macro_f1"],
                    "macro_f1_drop": clean_metrics["macro_f1"] - metrics["macro_f1"],
                    "mean_abs_margin_change": float(np.mean(np.abs(margin - clean_margin))),
                    "mean_abs_prob_change": float(np.mean(np.abs(prob - clean_prob))),
                    "flip_rate_vs_clean": float(np.mean(pred != clean_pred)),
                }
                run_rows.append(row)
                all_rows.append(row)

        run_df = pd.DataFrame(run_rows)
        flat = summarize_flatness(run_df)
        flat_rows.append({
            "method": method,
            "seed": seed,
            "arch": arch,
            "ckpt_path": str(ckpt),
            **flat,
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    by_run = pd.DataFrame(all_rows)
    by_run.to_csv(args.out_dir / "response_plane_by_run.csv", index=False)

    metric_cols = [
        "acc", "bal_acc", "macro_f1", "macro_f1_drop",
        "mean_abs_margin_change", "mean_abs_prob_change", "flip_rate_vs_clean",
    ]
    summary = by_run.groupby(["method", "delta_high", "delta_low"], as_index=False)[metric_cols].agg(["mean", "std", "count"])
    summary.columns = ["_".join([str(x) for x in col if str(x)]) for col in summary.columns.to_flat_index()]
    summary = summary.rename(columns={"method_": "method", "delta_high_": "delta_high", "delta_low_": "delta_low"})
    summary.to_csv(args.out_dir / "response_plane_summary.csv", index=False)

    flat_df = pd.DataFrame(flat_rows)
    flat_df.to_csv(args.out_dir / "response_plane_flatness_by_run.csv", index=False)

    flat_metric_cols = [c for c in flat_df.columns if c not in {"method", "seed", "arch", "ckpt_path"}]
    flat_summary = flat_df.groupby("method", as_index=False)[flat_metric_cols].agg(["mean", "std", "count"])
    flat_summary.columns = ["_".join([str(x) for x in col if str(x)]) for col in flat_summary.columns.to_flat_index()]
    flat_summary = flat_summary.rename(columns={"method_": "method"})
    flat_summary.to_csv(args.out_dir / "response_plane_flatness_summary.csv", index=False)

    config = {
        "grid": grid,
        "split_column": args.split_column,
        "split_value": args.split_value,
        "manifest": str(args.manifest),
        "csv_path": str(args.csv_path),
        "data_root": args.data_root,
        "log_eps": args.log_eps,
    }
    (args.out_dir / "response_plane_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] Saved 2D response-plane outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
