#!/usr/bin/env python3
"""
实验三：通道独立衰减响应偏移评估
Evaluate models under channel-independent log-domain attenuation shifts (δ_H ≠ δ_L).

与现有 eval_thickness_shift.py 的区别：
- 原版：δ_H = δ_L = δ（高低能通道施加相同偏移）
- 本版：δ_H 和 δ_L 独立取值，模拟真实场景中高低能通道响应解耦的情况
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet18_Weights, resnet18
from datasets.copper_xray_dataset import CopperXRayDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Channel-independent log-domain attenuation shift evaluation"
    )
    p.add_argument("--csv-path", type=Path, required=True)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--split-column", type=str, default="split")
    p.add_argument("--img-size", type=int, default=192)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-path", type=Path, required=True)
    p.add_argument("--method-name", type=str, required=True)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.set_defaults(pretrained=True)
    p.add_argument(
        "--delta-pairs",
        type=str,
        default="0.0:0.0,-0.15:0.15,0.15:-0.15,-0.25:0.25,0.25:-0.25,"
                "0.0:0.15,0.15:0.0,0.0:-0.15,-0.15:0.0,"
                "0.05:0.15,0.15:0.05,-0.05:-0.15,-0.15:-0.05",
        help="逗号分隔的 δ_H:δ_L 对，如 '0.0:0.15,0.15:0.0'"
    )
    p.add_argument("--apply-p", type=float, default=1.0)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_resnet18_2ch(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = resnet18(weights=weights)
    old = m.conv1
    m.conv1 = nn.Conv2d(
        2, old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    )
    with torch.no_grad():
        if pretrained and old.weight.shape[1] == 3:
            w_mean = old.weight.mean(dim=1, keepdim=True)
            m.conv1.weight.copy_(w_mean.repeat(1, 2, 1, 1))
        else:
            nn.init.kaiming_normal_(m.conv1.weight, mode="fan_out", nonlinearity="relu")
        if m.conv1.bias is not None:
            nn.init.zeros_(m.conv1.bias)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def apply_channel_independent_thickness(
    x: torch.Tensor, delta_high: float, delta_low: float, eps: float = 1e-6
) -> torch.Tensor:
    """
    通道独立对数域衰减扰动。
    x: (B, 2, H, W), 通道0=高能, 通道1=低能
    delta_high: 高能通道偏移量
    delta_low:  低能通道偏移量
    """
    x_out = x.clone()
    x_safe = torch.clamp(x_out, eps, 1.0)
    # 高能通道 (channel 0)
    x_out[:, 0:1] = torch.clamp(torch.exp(torch.log(x_safe[:, 0:1]) + delta_high), 0.0, 1.0)
    # 低能通道 (channel 1)
    x_out[:, 1:2] = torch.clamp(torch.exp(torch.log(x_safe[:, 1:2]) + delta_low), 0.0, 1.0)
    return x_out


@torch.no_grad()
def eval_metrics(
    model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module
) -> Dict[str, float]:
    model.eval()
    all_p: List[int] = []
    all_y: List[int] = []
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        p = logits.argmax(1)
        total_loss += float(loss.item()) * y.numel()
        n += y.numel()
        all_p.extend(p.cpu().tolist())
        all_y.extend(y.cpu().tolist())
    acc = accuracy_score(all_y, all_p)
    bal = balanced_accuracy_score(all_y, all_p)
    mf1 = f1_score(all_y, all_p, average="macro", zero_division=0)
    pre = precision_score(all_y, all_p, average="macro", zero_division=0)
    rec = recall_score(all_y, all_p, average="macro", zero_division=0)
    dist = Counter(all_p)
    return {
        "loss": total_loss / max(n, 1),
        "acc": float(acc),
        "bal_acc": float(bal),
        "macro_f1": float(mf1),
        "precision": float(pre),
        "recall": float(rec),
        "pred0": int(dist.get(0, 0)),
        "pred1": int(dist.get(1, 0)),
    }


def _safe_group_metrics(
    y_true: List[int], y_pred: List[int]
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    out["acc"] = float(accuracy_score(y_true, y_pred))
    uniq = set(y_true)
    if len(uniq) < 2:
        out["bal_acc"] = None
        out["macro_f1"] = None
        out["precision"] = None
        out["recall"] = None
        return out
    out["bal_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    return out


@torch.no_grad()
def group_report_test(
    args: argparse.Namespace,
    model: nn.Module,
    device: torch.device,
    ds: CopperXRayDataset,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = pd.read_csv(args.csv_path)
    required = {"thickness_group", "sample_id", "label", args.split_column}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame(), {}
    test_df = df[df[args.split_column] == "test"].copy()
    if test_df.empty:
        return pd.DataFrame(), {}
    groups = sorted([str(g) for g in test_df["thickness_group"].dropna().unique().tolist()])
    if not groups:
        return pd.DataFrame(), {}
    id_to_idx = {
        str(ds.samples.iloc[i]["sample_id"]).strip(): i
        for i in range(len(ds.samples))
    }
    rows = []
    worst_acc = 1.0
    worst_group = None
    for g in groups:
        gdf = test_df[test_df["thickness_group"].astype(str) == g]
        label_counts = gdf["label"].value_counts().to_dict()
        label0_n = int(label_counts.get(0, 0))
        label1_n = int(label_counts.get(1, 0))
        sids = [str(x).strip() for x in gdf["sample_id"].tolist()]
        idxs = [id_to_idx[sid] for sid in sids if sid in id_to_idx]
        n = len(idxs)
        if n == 0:
            rows.append({
                "thickness_group": g, "n": 0, "label0_n": label0_n, "label1_n": label1_n,
                "acc": None, "bal_acc": None, "macro_f1": None, "precision": None, "recall": None,
                "pred0": 0, "pred1": 0, "note": "empty"
            })
            continue
        subset = Subset(ds, idxs)
        loader = DataLoader(
            subset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers
        )
        y_true: List[int] = []
        y_pred: List[int] = []
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            p = logits.argmax(1).cpu().tolist()
            y_pred.extend(p)
            y_true.extend(y.cpu().tolist())
        dist = Counter(y_pred)
        m = _safe_group_metrics(y_true, y_pred)
        note = "single-class" if (label0_n == 0 or label1_n == 0) else ""
        acc_v = m["acc"] if m["acc"] is not None else None
        if acc_v is not None and acc_v < worst_acc:
            worst_acc = acc_v
            worst_group = g
        rows.append({
            "thickness_group": g,
            "n": n,
            "label0_n": label0_n,
            "label1_n": label1_n,
            "acc": m["acc"],
            "bal_acc": m["bal_acc"],
            "macro_f1": m["macro_f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "pred0": int(dist.get(0, 0)),
            "pred1": int(dist.get(1, 0)),
            "note": note,
        })
    out = pd.DataFrame(rows).sort_values("thickness_group").reset_index(drop=True)
    summary = {}
    if worst_group is not None:
        summary["worst_group_acc"] = float(worst_acc)
        summary["worst_group_name"] = str(worst_group)
    return out, summary


def make_clean_test_ds(args: argparse.Namespace) -> CopperXRayDataset:
    return CopperXRayDataset(
        args.csv_path,
        split_column=args.split_column,
        split_value="test",
        data_root=args.data_root,
        out_size=args.img_size,
        hflip_p=0.0,
        thickness_aug=False,
        use_brightness_aug=False,
    )


def parse_delta_pairs(delta_pairs_str: str) -> List[Tuple[float, float]]:
    pairs = []
    for token in delta_pairs_str.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(f"无效的 δ_H:δ_L 格式: {token}")
        pairs.append((float(parts[0]), float(parts[1])))
    return pairs


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = build_resnet18_2ch(pretrained=args.pretrained).to(device)
    print(f"[INFO] Loading checkpoint: {args.ckpt_path}", flush=True)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    delta_pairs = parse_delta_pairs(args.delta_pairs)
    print(f"[INFO] Evaluating {len(delta_pairs)} (δ_H, δ_L) pairs:", flush=True)
    for dh, dl in delta_pairs:
        print(f"       δ_H={dh:+.2f}, δ_L={dl:+.2f}", flush=True)

    # 加载干净测试集用于扰动
    clean_ds = make_clean_test_ds(args)
    pin = args.device == "cuda" and torch.cuda.is_available()

    summary_rows = []
    for dh, dl in delta_pairs:
        print(f"\n[INFO] Evaluating δ_H={dh:+.2f}, δ_L={dl:+.2f}", flush=True)

        # 直接对 batch 施加通道独立扰动，不走 dataset 的 aug 逻辑
        loader = DataLoader(
            clean_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin,
        )

        all_p: List[int] = []
        all_y: List[int] = []
        total_loss = 0.0
        n = 0
        for x, y in loader:
            x = apply_channel_independent_thickness(
                x.to(device), delta_high=dh, delta_low=dl
            )
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            p = logits.argmax(1)
            total_loss += float(loss.item()) * y.numel()
            n += y.numel()
            all_p.extend(p.cpu().tolist())
            all_y.extend(y.cpu().tolist())

        test = {
            "loss": total_loss / max(n, 1),
            "acc": float(accuracy_score(all_y, all_p)),
            "bal_acc": float(balanced_accuracy_score(all_y, all_p)),
            "macro_f1": float(f1_score(all_y, all_p, average="macro", zero_division=0)),
            "precision": float(precision_score(all_y, all_p, average="macro", zero_division=0)),
            "recall": float(recall_score(all_y, all_p, average="macro", zero_division=0)),
            "pred0": int(Counter(all_p).get(0, 0)),
            "pred1": int(Counter(all_p).get(1, 0)),
        }

        # 用已扰动后的数据集做 group report
        # 构建一个临时的扰动后 subset
        perturbed_ds = make_clean_test_ds(args)
        gr, gr_summary = group_report_test(args, model, device, perturbed_ds)

        # 实际上 group_report 需要的是扰动后的预测，但上面 group_report_test 用的是原始数据。
        # 为正确获取 group-level 指标，我们直接基于 all_p/all_y 和 thickness_group 标签构建。
        # 这里简化处理：用 clean DS 的 group 信息 + 扰动后的预测
        df = pd.read_csv(args.csv_path)
        if "thickness_group" in df.columns and "sample_id" in df.columns:
            test_df = df[df[args.split_column] == "test"].copy()
            ds_samples = clean_ds.samples
            id_to_group = {}
            for i in range(len(ds_samples)):
                sid = str(ds_samples.iloc[i]["sample_id"]).strip()
                match = test_df[test_df["sample_id"].astype(str).str.strip() == sid]
                if not match.empty:
                    id_to_group[i] = str(match.iloc[0]["thickness_group"])

            groups = sorted(set(id_to_group.values()))
            gr_rows = []
            worst_acc = 1.0
            worst_group = None
            for g in groups:
                idxs = [i for i, grp in id_to_group.items() if grp == g]
                if not idxs:
                    continue
                g_y_true = [all_y[i] for i in idxs]
                g_y_pred = [all_p[i] for i in idxs]
                m = _safe_group_metrics(g_y_true, g_y_pred)
                c0 = sum(1 for v in g_y_true if v == 0)
                c1 = sum(1 for v in g_y_true if v == 1)
                note = "single-class" if (c0 == 0 or c1 == 0) else ""
                acc_v = m["acc"]
                if acc_v is not None and acc_v < worst_acc:
                    worst_acc = acc_v
                    worst_group = g
                gr_rows.append({
                    "thickness_group": g, "n": len(idxs),
                    "label0_n": c0, "label1_n": c1,
                    "acc": m["acc"], "bal_acc": m["bal_acc"],
                    "macro_f1": m["macro_f1"],
                    "precision": m["precision"], "recall": m["recall"],
                    "note": note,
                })
            gr = pd.DataFrame(gr_rows)
            gr_summary = {}
            if worst_group is not None:
                gr_summary = {"worst_group_acc": float(worst_acc), "worst_group_name": str(worst_group)}
        else:
            gr = pd.DataFrame()
            gr_summary = {}

        tag = (
            f"dh{'+' if dh>=0 else ''}{dh:.2f}_dl{'+' if dl>=0 else ''}{dl:.2f}"
            .replace("+", "p").replace("-", "m").replace(".", "_")
        )
        payload = {
            "method": args.method_name,
            "delta_high": dh,
            "delta_low": dl,
            "apply_p": args.apply_p,
            "overall_test": test,
            "worst_group_acc": gr_summary.get("worst_group_acc", None),
            "worst_group_name": gr_summary.get("worst_group_name", None),
        }
        js_path = args.out_dir / f"summary_{args.method_name}_{tag}.json"
        js_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        row = {
            "method": args.method_name,
            "delta_high": dh,
            "delta_low": dl,
            "apply_p": args.apply_p,
            "loss": test["loss"],
            "acc": test["acc"],
            "bal_acc": test["bal_acc"],
            "macro_f1": test["macro_f1"],
            "precision": test["precision"],
            "recall": test["recall"],
            "pred0": test["pred0"],
            "pred1": test["pred1"],
            "worst_group_acc": gr_summary.get("worst_group_acc", None),
            "worst_group_name": gr_summary.get("worst_group_name", None),
        }
        if not gr.empty:
            for _, r in gr.iterrows():
                g = str(r["thickness_group"])
                row[f"{g}_acc"] = r["acc"]
                row[f"{g}_bal_acc"] = r["bal_acc"]
                row[f"{g}_macro_f1"] = r["macro_f1"]
        summary_rows.append(row)
        print(f"  acc={test['acc']:.4f}  bal_acc={test['bal_acc']:.4f}  macro_f1={test['macro_f1']:.4f}", flush=True)

    out_csv = args.out_dir / f"channel_independent_summary_{args.method_name}.csv"
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"\n[OK] saved: {out_csv}", flush=True)


if __name__ == "__main__":
    main()