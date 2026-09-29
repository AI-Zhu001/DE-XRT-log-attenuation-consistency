#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


GROUP_CANDIDATES = [
    "thickness_group",
    "mean_gray_group",
    "mean_gray_third",
    "indicator_group",
    "response_group",
    "gray_group",
    "group",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recalculate corrected indicator-group WGA/WGBAcc for DE-XRT checkpoints."
    )
    p.add_argument(
        "--csv-path",
        type=Path,
        default=Path("/root/projects/Hou_swin/split_outputs/copper_xray_all_splits_mean_gray_groups.csv"),
        help="Corrected split CSV containing fixed clean-response groups.",
    )
    p.add_argument(
        "--data-root",
        type=str,
        default="/root/autodl-tmp/data/原始购买的二分类数据集/原始购买的二分类数据集",
    )
    p.add_argument(
        "--dataset-py",
        type=Path,
        default=Path("/root/projects/Hou_swin/datasets/copper_xray_dataset.py"),
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--split-column", type=str, default="split")
    p.add_argument("--split-value", type=str, default="test")
    p.add_argument("--group-column", type=str, default=None)
    p.add_argument("--img-size", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)
    p.add_argument("--skip-missing-checkpoints", action="store_true", default=True)
    p.add_argument(
        "--conditions",
        type=str,
        default="-0.25:-0.25,-0.15:-0.15,0:0,0.15:0.15,0.25:0.25,"
                "0.15:-0.15,-0.15:0.15,0.25:-0.25,-0.25:0.25",
        help="Comma-separated deltaH:deltaL pairs.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/root/autodl-tmp/lunwen1xiugai_extra/wga_corrected"),
    )
    return p.parse_args()


def load_dataset_class(dataset_py: Path):
    if not dataset_py.exists():
        raise FileNotFoundError(f"Dataset module not found: {dataset_py}")
    spec = importlib.util.spec_from_file_location("hou_copper_xray_dataset", str(dataset_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import dataset module: {dataset_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "CopperXRayDataset"):
        raise AttributeError("CopperXRayDataset not found in dataset module")
    return module.CopperXRayDataset


def build_resnet18_2ch(num_classes: int = 2) -> nn.Module:
    # Do not download weights during evaluation; the checkpoint overwrites all model weights.
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


def normalize_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint does not contain a state_dict-like object.")
    clean = {}
    for k, v in obj.items():
        nk = k
        for prefix in ("module.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        clean[nk] = v
    return clean


def load_model(ckpt: Path, device: torch.device) -> nn.Module:
    m = build_resnet18_2ch().to(device)
    raw = torch.load(ckpt, map_location=device)
    sd = normalize_state_dict(raw)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"State-dict mismatch for {ckpt}\nmissing={missing}\nunexpected={unexpected}"
        )
    m.eval()
    return m


def find_group_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(
                f"--group-column={requested!r} not found. Available columns: {list(df.columns)}"
            )
        return requested
    for c in GROUP_CANDIDATES:
        if c in df.columns:
            return c

    # Fallback: search for a column with exactly 3 non-null unique values and a group-ish name.
    for c in df.columns:
        lc = c.lower()
        if any(t in lc for t in ("group", "third", "indicator")):
            vals = df[c].dropna().astype(str).unique()
            if len(vals) == 3:
                return c

    raise ValueError(
        "Could not auto-detect the corrected group column. "
        f"Pass --group-column explicitly. Columns: {list(df.columns)}"
    )


class GroupWrappedDataset(Dataset):
    def __init__(self, base_ds, groups: List[str]):
        if len(base_ds) != len(groups):
            raise ValueError(f"Dataset/group length mismatch: {len(base_ds)} vs {len(groups)}")
        self.base_ds = base_ds
        self.groups = groups

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        return x, y, self.groups[idx]


def apply_log_shift_pair(
    x: torch.Tensor,
    delta_h: float,
    delta_l: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    x_safe = torch.clamp(x, eps, 1.0)
    out = x_safe.clone()
    out[:, 0:1] = torch.clamp(
        torch.exp(torch.log(x_safe[:, 0:1]) + float(delta_h)), 0.0, 1.0
    )
    out[:, 1:2] = torch.clamp(
        torch.exp(torch.log(x_safe[:, 1:2]) + float(delta_l)), 0.0, 1.0
    )
    return out


def safe_group_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"acc": np.nan, "bal_acc": np.nan, "macro_f1": np.nan}
    acc = float(accuracy_score(y_true, y_pred))
    if len(set(y_true)) < 2:
        bal = np.nan
        mf1 = np.nan
    else:
        bal = float(balanced_accuracy_score(y_true, y_pred))
        mf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {"acc": acc, "bal_acc": bal, "macro_f1": mf1}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    delta_h: float,
    delta_l: float,
    amp: bool,
):
    ys: List[int] = []
    ps: List[int] = []
    gs: List[str] = []

    use_amp = amp and device.type == "cuda"
    for x, y, group in loader:
        x = x.to(device, non_blocking=True)
        x = apply_log_shift_pair(x, delta_h, delta_l)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        pred = logits.argmax(1).cpu().numpy().tolist()

        ys.extend(y.cpu().numpy().tolist())
        ps.extend(pred)
        gs.extend([str(g) for g in group])

    overall = {
        "acc": float(accuracy_score(ys, ps)),
        "bal_acc": float(balanced_accuracy_score(ys, ps)),
        "macro_f1": float(f1_score(ys, ps, average="macro", zero_division=0)),
    }

    group_rows = []
    for g in sorted(set(gs)):
        idx = [i for i, gg in enumerate(gs) if gg == g]
        gy = [ys[i] for i in idx]
        gp = [ps[i] for i in idx]
        m = safe_group_metrics(gy, gp)
        counts = Counter(gy)
        group_rows.append(
            {
                "group": g,
                "n": len(idx),
                "label0_n": int(counts.get(0, 0)),
                "label1_n": int(counts.get(1, 0)),
                **m,
            }
        )

    gdf = pd.DataFrame(group_rows)
    wga = float(gdf["acc"].min())
    worst_acc_group = str(gdf.loc[gdf["acc"].idxmin(), "group"])

    valid_bal = gdf.dropna(subset=["bal_acc"])
    wg_bal = float(valid_bal["bal_acc"].min()) if len(valid_bal) else np.nan
    worst_bal_group = (
        str(valid_bal.loc[valid_bal["bal_acc"].idxmin(), "group"])
        if len(valid_bal) else ""
    )

    valid_f1 = gdf.dropna(subset=["macro_f1"])
    wg_f1 = float(valid_f1["macro_f1"].min()) if len(valid_f1) else np.nan
    worst_f1_group = (
        str(valid_f1.loc[valid_f1["macro_f1"].idxmin(), "group"])
        if len(valid_f1) else ""
    )

    return overall, gdf, {
        "wga": wga,
        "worst_acc_group": worst_acc_group,
        "worst_group_bal_acc": wg_bal,
        "worst_bal_acc_group": worst_bal_group,
        "worst_group_macro_f1": wg_f1,
        "worst_macro_f1_group": worst_f1_group,
    }


def parse_conditions(s: str) -> List[Tuple[float, float]]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        a, b = tok.split(":")
        out.append((float(a), float(b)))
    return out


def condition_name(dh: float, dl: float) -> str:
    if abs(dh - dl) < 1e-12:
        return f"sync_{dh:+.2f}"
    return f"H{dh:+.2f}_L{dl:+.2f}"


def summarize(df: pd.DataFrame, keys: List[str], metrics: List[str]) -> pd.DataFrame:
    rows = []
    for vals, g in df.groupby(keys, dropna=False):
        if not isinstance(vals, tuple):
            vals = (vals,)
        row = dict(zip(keys, vals))
        row["runs"] = int(len(g))
        for m in metrics:
            arr = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_mean"] = float(arr.mean())
            row[f"{m}_std"] = float(arr.std(ddof=1)) if arr.notna().sum() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.csv_path.exists():
        raise FileNotFoundError(
            f"Corrected grouping CSV not found: {args.csv_path}\n"
            "Use the corrected mean-gray grouping file, not the old grouping CSV."
        )
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    full_df = pd.read_csv(args.csv_path)
    if args.split_column not in full_df.columns:
        raise ValueError(f"Missing split column: {args.split_column}")
    test_df = full_df[full_df[args.split_column].astype(str) == args.split_value].copy()
    if test_df.empty:
        raise ValueError(f"No rows with {args.split_column}={args.split_value!r}")

    group_col = find_group_column(test_df, args.group_column)
    if test_df[group_col].isna().any():
        raise ValueError(f"Group column {group_col!r} has missing values in the test split.")

    # Audit group composition before any model evaluation.
    audit = (
        test_df.groupby([group_col, "label"]).size()
        .unstack(fill_value=0)
        .rename(columns=lambda x: f"label{x}_n")
        .reset_index()
        .rename(columns={group_col: "group"})
    )
    for col in ("label0_n", "label1_n"):
        if col not in audit.columns:
            audit[col] = 0
    audit["n"] = audit["label0_n"] + audit["label1_n"]
    audit["label1_fraction"] = audit["label1_n"] / audit["n"]
    audit.to_csv(args.out_dir / "group_audit.csv", index=False)

    print(f"[INFO] corrected group column: {group_col}")
    print("[INFO] test-set group composition:")
    print(audit.to_string(index=False))

    if len(audit) != 3:
        print(f"[WARN] Expected 3 groups, found {len(audit)}.")
    if ((audit["label0_n"] == 0) | (audit["label1_n"] == 0)).any():
        print(
            "[WARN] At least one group is single-class. WGA can be class-confounded; "
            "worst-group balanced accuracy will be undefined for that group."
        )

    CopperXRayDataset = load_dataset_class(args.dataset_py)
    base_ds = CopperXRayDataset(
        args.csv_path,
        split_column=args.split_column,
        split_value=args.split_value,
        data_root=args.data_root,
        out_size=args.img_size,
        hflip_p=0.0,
        thickness_aug=False,
        use_brightness_aug=False,
    )

    # base_ds.samples is filtered/reset in the same order as the CSV split.
    if group_col not in base_ds.samples.columns:
        raise ValueError(
            f"Dataset's filtered dataframe does not contain group column {group_col!r}."
        )
    groups = base_ds.samples[group_col].astype(str).tolist()
    ds = GroupWrappedDataset(base_ds, groups)

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    pin = device.type == "cuda"
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    manifest = pd.read_csv(args.manifest)
    req = {"method", "seed", "ckpt_path"}
    if not req.issubset(manifest.columns):
        raise ValueError(f"Manifest must contain {sorted(req)}")

    conditions = parse_conditions(args.conditions)
    run_rows = []
    group_rows = []

    for _, r in manifest.iterrows():
        method = str(r["method"])
        seed = int(r["seed"])
        ckpt = Path(str(r["ckpt_path"]))

        if not ckpt.exists():
            msg = f"[WARN] Missing checkpoint, skipped: {method} seed={seed}: {ckpt}"
            if args.skip_missing_checkpoints:
                print(msg)
                continue
            raise FileNotFoundError(msg)

        print(f"\n[MODEL] {method} seed={seed}\n        {ckpt}")
        model = load_model(ckpt, device)

        for dh, dl in conditions:
            cname = condition_name(dh, dl)
            overall, gdf, worst = evaluate(
                model, loader, device, dh, dl, args.amp
            )
            row = {
                "method": method,
                "seed": seed,
                "condition": cname,
                "delta_h": dh,
                "delta_l": dl,
                **overall,
                **worst,
            }
            run_rows.append(row)

            for _, gr in gdf.iterrows():
                group_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "condition": cname,
                        "delta_h": dh,
                        "delta_l": dl,
                        **gr.to_dict(),
                    }
                )

            print(
                f"  {cname:20s} "
                f"F1={overall['macro_f1']:.4f} "
                f"WGA={worst['wga']:.4f} "
                f"WGBAcc={worst['worst_group_bal_acc']:.4f} "
                f"WGF1={worst['worst_group_macro_f1']:.4f}"
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    run_df = pd.DataFrame(run_rows)
    group_df = pd.DataFrame(group_rows)
    run_df.to_csv(args.out_dir / "wga_by_run.csv", index=False)
    group_df.to_csv(args.out_dir / "group_metrics_by_run.csv", index=False)

    run_metrics = [
        "acc", "bal_acc", "macro_f1",
        "wga", "worst_group_bal_acc", "worst_group_macro_f1",
    ]
    run_summary = summarize(
        run_df,
        ["method", "condition", "delta_h", "delta_l"],
        run_metrics,
    )
    run_summary.to_csv(args.out_dir / "wga_summary.csv", index=False)

    group_summary = summarize(
        group_df,
        ["method", "condition", "delta_h", "delta_l", "group"],
        ["acc", "bal_acc", "macro_f1"],
    )
    group_summary.to_csv(args.out_dir / "group_metrics_summary.csv", index=False)

    # Convenience view for the key paper conditions.
    key_cols = [
        "method", "condition", "runs",
        "macro_f1_mean", "macro_f1_std",
        "wga_mean", "wga_std",
        "worst_group_bal_acc_mean", "worst_group_bal_acc_std",
        "worst_group_macro_f1_mean", "worst_group_macro_f1_std",
    ]
    available = [c for c in key_cols if c in run_summary.columns]
    run_summary[available].to_csv(
        args.out_dir / "paper_wga_key_summary.csv", index=False
    )

    meta = {
        "csv_path": str(args.csv_path),
        "group_column": group_col,
        "split": args.split_value,
        "n_test": int(len(test_df)),
        "conditions": [{"delta_h": a, "delta_l": b} for a, b in conditions],
        "note": (
            "Groups are fixed from the corrected clean-response grouping CSV. "
            "WGA=min group accuracy; WGBAcc=min group balanced accuracy; "
            "WGF1=min group Macro-F1."
        ),
    }
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n[DONE]")
    print(f"Results: {args.out_dir}")
    print(f"  {args.out_dir / 'group_audit.csv'}")
    print(f"  {args.out_dir / 'wga_by_run.csv'}")
    print(f"  {args.out_dir / 'wga_summary.csv'}")
    print(f"  {args.out_dir / 'group_metrics_by_run.csv'}")
    print(f"  {args.out_dir / 'group_metrics_summary.csv'}")
    print(f"  {args.out_dir / 'paper_wga_key_summary.csv'}")


if __name__ == "__main__":
    main()
