#!/usr/bin/env python3
"""Summarize the directional-only ablation with matched 5-seed statistics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--directional-by-run", type=Path, required=True)
    p.add_argument("--plane-by-run", type=Path, required=True)
    p.add_argument("--plane-flatness-by-run", type=Path, required=True)
    p.add_argument("--eps", type=float, default=0.025)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def paired_stats(df, method_a, method_b, value_col, extra_filter=None):
    d = df.copy()
    if extra_filter is not None:
        d = d.loc[extra_filter(d)].copy()
    a = d[d.method == method_a][["seed", value_col]].rename(columns={value_col: "a"})
    b = d[d.method == method_b][["seed", value_col]].rename(columns={value_col: "b"})
    m = a.merge(b, on="seed", how="inner").sort_values("seed")
    if len(m) < 2:
        return None
    diff = m.a - m.b
    t = ttest_rel(m.a, m.b)
    try:
        w = wilcoxon(m.a, m.b, alternative="two-sided", method="exact")
        wp = float(w.pvalue)
        ws = float(w.statistic)
    except Exception:
        w = wilcoxon(m.a, m.b, alternative="two-sided")
        wp = float(w.pvalue)
        ws = float(w.statistic)
    return {
        "method_a": method_a,
        "method_b": method_b,
        "metric": value_col,
        "n": int(len(m)),
        "a_mean": float(m.a.mean()),
        "a_sd": float(m.a.std(ddof=1)),
        "b_mean": float(m.b.mean()),
        "b_sd": float(m.b.std(ddof=1)),
        "mean_difference_a_minus_b": float(diff.mean()),
        "all_a_lower_than_b": bool((m.a < m.b).all()),
        "all_a_higher_than_b": bool((m.a > m.b).all()),
        "paired_t_stat": float(t.statistic),
        "paired_t_p": float(t.pvalue),
        "wilcoxon_stat": ws,
        "wilcoxon_exact_two_sided_p": wp,
        "seeds": m.seed.astype(int).tolist(),
        "a_values": m.a.astype(float).tolist(),
        "b_values": m.b.astype(float).tolist(),
    }


def mean_sd_table(df, group_cols, metric_cols):
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_runs"] = int(len(g))
        for c in metric_cols:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_sd"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds = pd.read_csv(args.directional_by_run)
    plane = pd.read_csv(args.plane_by_run)
    flat = pd.read_csv(args.plane_flatness_by_run)

    eps_df = ds[np.isclose(ds["eps"], args.eps)].copy()
    directional_summary = mean_sd_table(
        eps_df,
        ["method", "direction"],
        ["margin_sens_mean", "prob_sens_mean", "plus_minus_flip_rate", "clean_acc"],
    )
    directional_summary.to_csv(args.out_dir / "directional_ablation_eps_summary.csv", index=False)

    flat_metrics = [
        "plane_mean_abs_margin_change",
        "plane_mean_flip_rate",
        "plane_worst_macro_f1",
        "plane_mean_macro_f1",
        "common_diag_mean_abs_margin_change",
        "common_diag_mean_macro_f1",
        "differential_diag_mean_abs_margin_change",
        "differential_diag_mean_macro_f1",
    ]
    flat_summary = mean_sd_table(flat, ["method"], flat_metrics)
    flat_summary.to_csv(args.out_dir / "response_plane_ablation_summary.csv", index=False)

    selected_pairs = [(0.15, -0.15), (-0.15, 0.15), (0.15, 0.15), (-0.15, -0.15), (0.0, 0.0)]
    selected = plane[
        plane.apply(lambda r: any(np.isclose(r.delta_high, a) and np.isclose(r.delta_low, b) for a, b in selected_pairs), axis=1)
    ].copy()
    selected_summary = mean_sd_table(
        selected,
        ["method", "delta_high", "delta_low"],
        ["macro_f1", "bal_acc", "mean_abs_margin_change", "flip_rate_vs_clean"],
    )
    selected_summary.to_csv(args.out_dir / "selected_response_pairs_summary.csv", index=False)

    comparisons = []
    for direction in ["common", "differential"]:
        filt = lambda d, direction=direction: (d.direction == direction) & np.isclose(d.eps, args.eps)
        for a, b in [
            ("Differential-Consistency", "Consistency"),
            ("Differential-Consistency", "CI-Consistency"),
            ("CI-Consistency", "Consistency"),
        ]:
            s = paired_stats(ds, a, b, "margin_sens_mean", filt)
            if s:
                s["context"] = f"direction={direction}, eps={args.eps}"
                comparisons.append(s)

    # Key pointwise Macro-F1 comparisons.
    for dh, dl in [(0.15, -0.15), (-0.15, 0.15), (0.15, 0.15), (-0.15, -0.15)]:
        filt = lambda d, dh=dh, dl=dl: np.isclose(d.delta_high, dh) & np.isclose(d.delta_low, dl)
        for a, b in [
            ("Differential-Consistency", "Consistency"),
            ("Differential-Consistency", "CI-Consistency"),
        ]:
            s = paired_stats(plane, a, b, "macro_f1", filt)
            if s:
                s["context"] = f"response_pair=({dh:+.2f},{dl:+.2f})"
                comparisons.append(s)

    pd.DataFrame(comparisons).to_csv(args.out_dir / "paired_comparisons.csv", index=False)
    (args.out_dir / "paired_comparisons.json").write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== Directional sensitivity at eps=%.3f ===" % args.eps)
    print(directional_summary.to_string(index=False))
    print("\n=== Response-plane flatness ===")
    print(flat_summary.to_string(index=False))
    print("\n[OK] Results written to", args.out_dir)
    print("[NOTE] With n=5, an exact two-sided Wilcoxon test cannot be <0.0625 when all five differences share one sign.")


if __name__ == "__main__":
    main()
