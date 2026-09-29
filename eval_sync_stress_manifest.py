#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate synchronized log-domain stress for all checkpoints in a manifest")
    p.add_argument("--csv-path", type=Path, default=Path("/root/projects/Hou_swin/split_outputs/copper_xray_all_splits.csv"))
    p.add_argument("--data-root", type=str, default="/root/autodl-tmp/data/原始购买的二分类数据集/原始购买的二分类数据集")
    p.add_argument("--dataset-py", type=Path, default=Path("/root/projects/Hou_swin/datasets/copper_xray_dataset.py"))
    p.add_argument("--manifest", type=Path, default=Path("/root/autodl-tmp/lunwen1xiugai_extra/response_plane_manifest_7methods_5seeds.csv"))
    p.add_argument("--delta-list", type=str, default="-0.25,-0.15,0,0.15,0.25")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)
    p.add_argument("--out-dir", type=Path, default=Path("/root/autodl-tmp/lunwen1xiugai_extra/sync_stress_7methods"))
    return p.parse_args()


def load_dataset_class(dataset_py: Path):
    spec = importlib.util.spec_from_file_location("local_copper_xray_dataset", dataset_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load dataset module: {dataset_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CopperXRayDataset


def build_resnet18_2ch(num_classes: int = 2) -> nn.Module:
    # No pretrained download is needed for evaluation; checkpoint overwrites all weights.
    m = resnet18(weights=None)
    old = m.conv1
    m.conv1 = nn.Conv2d(
        2, old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    )
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def load_state(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            obj = obj["state_dict"]
        elif "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            obj = obj["model_state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            obj = obj["model"]
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(obj)}")
    # tolerate DataParallel prefix
    if all(str(k).startswith("module.") for k in obj.keys()):
        obj = {str(k)[7:]: v for k, v in obj.items()}
    model.load_state_dict(obj, strict=True)


def apply_sync_shift(x: torch.Tensor, delta: float, eps: float = 1e-6) -> torch.Tensor:
    if abs(delta) < 1e-15:
        return x
    xs = torch.clamp(x, eps, 1.0)
    y = torch.exp(torch.log(xs) + float(delta))
    return torch.clamp(y, 0.0, 1.0)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, delta: float, amp: bool) -> Dict[str, float]:
    model.eval()
    ys: List[int] = []
    ps: List[int] = []
    margins: List[float] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = apply_sync_shift(x, delta)
        with torch.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
            z = model(x)
        pred = z.argmax(1)
        margin = (z[:, 1] - z[:, 0]).float()
        ys.extend(y.cpu().tolist())
        ps.extend(pred.cpu().tolist())
        margins.extend(margin.cpu().tolist())
    return {
        "acc": float(accuracy_score(ys, ps)),
        "bal_acc": float(balanced_accuracy_score(ys, ps)),
        "macro_f1": float(f1_score(ys, ps, average="macro", zero_division=0)),
        "pred1_rate": float(np.mean(np.asarray(ps) == 1)),
        "mean_abs_logit_margin": float(np.mean(np.abs(margins))),
    }


def paired_stats(by_run: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("Consistency", "DirectAttenAug-Matched"),
        ("Consistency", "Brightness-Consistency"),
        ("CI-Consistency", "CI-DirectAttenAug"),
        ("CI-Consistency", "Consistency"),
        ("CI-Consistency", "Brightness-Consistency"),
    ]
    rows = []
    for delta in sorted(by_run["delta"].unique()):
        d = by_run[by_run["delta"] == delta]
        for a, b in pairs:
            da = d[d.method == a].set_index("seed")
            db = d[d.method == b].set_index("seed")
            seeds = sorted(set(da.index).intersection(db.index))
            if len(seeds) < 2:
                continue
            for metric in ["macro_f1", "bal_acc"]:
                xa = da.loc[seeds, metric].to_numpy(float)
                xb = db.loc[seeds, metric].to_numpy(float)
                diff = xa - xb
                t = stats.ttest_rel(xa, xb)
                try:
                    w = stats.wilcoxon(xa, xb, alternative="two-sided", method="exact")
                    w_stat, w_p = float(w.statistic), float(w.pvalue)
                except Exception:
                    w_stat, w_p = np.nan, np.nan
                rows.append({
                    "delta": float(delta),
                    "metric": metric,
                    "method_a": a,
                    "method_b": b,
                    "n": len(seeds),
                    "mean_a": float(xa.mean()),
                    "mean_b": float(xb.mean()),
                    "mean_diff_a_minus_b": float(diff.mean()),
                    "all_a_gt_b": bool(np.all(diff > 0)),
                    "all_a_lt_b": bool(np.all(diff < 0)),
                    "paired_t_stat": float(t.statistic),
                    "paired_t_p": float(t.pvalue),
                    "wilcoxon_stat": w_stat,
                    "wilcoxon_exact_p": w_p,
                })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    DatasetCls = load_dataset_class(args.dataset_py)
    ds = DatasetCls(
        args.csv_path,
        split_column="split",
        split_value="test",
        data_root=args.data_root,
        out_size=192,
        hflip_p=0.0,
        thickness_aug=False,
        use_brightness_aug=False,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    manifest = pd.read_csv(args.manifest)
    deltas = [float(x.strip()) for x in args.delta_list.split(",") if x.strip()]
    rows = []

    for rec in manifest.itertuples(index=False):
        arch = str(rec.arch).lower()
        if arch != "resnet18":
            print(f"[SKIP] unsupported arch={arch}: {rec.method} seed={rec.seed}")
            continue
        ckpt = Path(str(rec.ckpt_path))
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        model = build_resnet18_2ch().to(device)
        load_state(model, ckpt, device)
        for delta in deltas:
            m = evaluate(model, loader, device, delta, args.amp)
            row = {"method": str(rec.method), "seed": int(rec.seed), "delta": delta, **m}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    by_run = pd.DataFrame(rows)
    by_run.to_csv(args.out_dir / "sync_stress_by_run.csv", index=False)

    metric_cols = ["acc", "bal_acc", "macro_f1", "pred1_rate", "mean_abs_logit_margin"]
    agg = by_run.groupby(["method", "delta"])[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join([str(x) for x in c if str(x)]) if isinstance(c, tuple) else str(c) for c in agg.columns]
    agg.to_csv(args.out_dir / "sync_stress_summary.csv", index=False)

    paired = paired_stats(by_run)
    paired.to_csv(args.out_dir / "sync_stress_paired_stats.csv", index=False)

    cfg = {
        "csv_path": str(args.csv_path),
        "manifest": str(args.manifest),
        "deltas": deltas,
        "note": "Overall test metrics only. WGA intentionally omitted here because grouping is being audited separately.",
    }
    (args.out_dir / "sync_stress_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved to {args.out_dir}")


if __name__ == "__main__":
    main()
