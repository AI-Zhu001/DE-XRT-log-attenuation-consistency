#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extended measured-geometry / dual-energy response analysis for the 60-particle subset")
    p.add_argument("--xlsx", type=Path, required=True)
    p.add_argument("--sheet", type=str, default="60样本数据记录")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20260929)
    return p.parse_args()


def load_records(path: Path, sheet: str) -> pd.DataFrame:
    wb = load_workbook(path, data_only=True, read_only=True)
    ws = wb[sheet]
    headers = [ws.cell(2, c).value for c in range(1, ws.max_column + 1)]
    rows = []
    for r in range(3, ws.max_row + 1):
        row = [ws.cell(r, c).value for c in range(1, ws.max_column + 1)]
        if row[0] is None:
            continue
        rows.append(row)
    df = pd.DataFrame(rows, columns=headers)

    ren = {
        "Sample ID": "sample_id",
        "类别": "class_cn",
        "Hmax_1/mm": "hmax1",
        "Hmax_2/mm": "hmax2",
        "Hmax平均/mm": "hmax_mean",
        "Hcenter_1/mm": "hcenter1",
        "Hcenter_2/mm": "hcenter2",
        "Hcenter平均/mm": "hcenter_mean",
        "Hproxy/mm": "hproxy",
        "高能均值 IH": "ih_mean",
        "低能均值 IL": "il_mean",
        "mean log response m": "mean_log_response_pixelwise",
        "mask_area_pixels": "mask_area_pixels",
        "mask_note": "mask_note",
        "高能图文件名": "high_filename",
        "低能图文件名": "low_filename",
    }
    df = df.rename(columns=ren)
    required = ["sample_id", "class_cn", "hmax1", "hmax2", "hcenter1", "hcenter2", "hproxy", "ih_mean", "il_mean", "mean_log_response_pixelwise"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after rename: {missing}")

    num_cols = ["hmax1", "hmax2", "hcenter1", "hcenter2", "hproxy", "ih_mean", "il_mean", "mean_log_response_pixelwise", "mask_area_pixels"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["class_label"] = df["class_cn"].map({"铜矿": "copper", "废石": "waste"}).fillna(df["class_cn"].astype(str))
    df["class_indicator"] = (df["class_label"] == "copper").astype(int)

    # Channel-level log responses using log of the mask-mean normalized intensities.
    # This is distinct from the workbook's pixelwise mean log response m.
    eps = 1e-12
    df["log_ih_mean"] = np.log(np.clip(df["ih_mean"].to_numpy(float), eps, None))
    df["log_il_mean"] = np.log(np.clip(df["il_mean"].to_numpy(float), eps, None))
    df["common_log_mean"] = 0.5 * (df["log_ih_mean"] + df["log_il_mean"])
    df["differential_log_mean"] = df["log_ih_mean"] - df["log_il_mean"]
    df["hmax_abs_diff"] = (df["hmax1"] - df["hmax2"]).abs()
    df["hcenter_abs_diff"] = (df["hcenter1"] - df["hcenter2"]).abs()
    if "mask_area_pixels" in df.columns:
        df["log_mask_area"] = np.log(np.clip(df["mask_area_pixels"].to_numpy(float), 1.0, None))
    return df


def bootstrap_slope_pearson_ci(x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    xb = x[idx]
    yb = y[idx]
    xm = xb.mean(axis=1, keepdims=True)
    ym = yb.mean(axis=1, keepdims=True)
    xc = xb - xm
    yc = yb - ym
    sxx = np.sum(xc * xc, axis=1)
    syy = np.sum(yc * yc, axis=1)
    sxy = np.sum(xc * yc, axis=1)
    valid_slope = sxx > 0
    slopes = sxy[valid_slope] / sxx[valid_slope]
    valid_r = (sxx > 0) & (syy > 0)
    rs = sxy[valid_r] / np.sqrt(sxx[valid_r] * syy[valid_r])
    slope_ci = tuple(np.quantile(slopes, [0.025, 0.975]).tolist())
    pear_ci = tuple(np.quantile(rs, [0.025, 0.975]).tolist())
    return slope_ci, pear_ci


def summarize_xy(df: pd.DataFrame, scope: str, ycol: str, yname: str, n_boot: int, rng: np.random.Generator) -> Dict[str, float | str | int]:
    d = df[["hproxy", ycol]].dropna()
    x = d["hproxy"].to_numpy(float)
    y = d[ycol].to_numpy(float)
    lr = stats.linregress(x, y)
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    slope_ci, pear_ci = bootstrap_slope_pearson_ci(x, y, n_boot, rng)
    return {
        "scope": scope,
        "response": yname,
        "n": len(d),
        "slope_per_mm": float(lr.slope),
        "slope_boot_ci_low": float(slope_ci[0]),
        "slope_boot_ci_high": float(slope_ci[1]),
        "intercept": float(lr.intercept),
        "r2": float(lr.rvalue ** 2),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "pearson_boot_ci_low": float(pear_ci[0]),
        "pearson_boot_ci_high": float(pear_ci[1]),
        "spearman_rho": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def repeatability_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, a, b in [("Hmax", "hmax1", "hmax2"), ("Hcenter", "hcenter1", "hcenter2")]:
        d = df[[a, b]].dropna()
        diff = d[a].to_numpy(float) - d[b].to_numpy(float)
        absdiff = np.abs(diff)
        rows.append({
            "measure": label,
            "n": len(d),
            "mean_difference_mm": float(np.mean(diff)),
            "sd_difference_mm": float(np.std(diff, ddof=1)),
            "median_abs_difference_mm": float(np.median(absdiff)),
            "mean_abs_difference_mm": float(np.mean(absdiff)),
            "max_abs_difference_mm": float(np.max(absdiff)),
            "repeatability_coefficient_1.96sd_mm": float(1.96 * np.std(diff, ddof=1)),
            "pearson_between_repeats": float(stats.pearsonr(d[a], d[b]).statistic),
        })
    return pd.DataFrame(rows)


def adjusted_models(df: pd.DataFrame, outcomes: Iterable[Tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for ycol, yname in outcomes:
        d = df[[ycol, "hproxy", "class_label"]].dropna().copy()
        # waste is the reference level, copper indicator enters via C(class_label)
        model_add = smf.ols(f"{ycol} ~ hproxy + C(class_label)", data=d).fit(cov_type="HC3")
        model_int = smf.ols(f"{ycol} ~ hproxy * C(class_label)", data=d).fit(cov_type="HC3")

        rows.append({
            "response": yname,
            "model": "class-adjusted additive",
            "n": int(model_add.nobs),
            "height_coef": float(model_add.params["hproxy"]),
            "height_se_hc3": float(model_add.bse["hproxy"]),
            "height_p_hc3": float(model_add.pvalues["hproxy"]),
            "height_ci_low": float(model_add.conf_int().loc["hproxy", 0]),
            "height_ci_high": float(model_add.conf_int().loc["hproxy", 1]),
            "interaction_coef": np.nan,
            "interaction_p_hc3": np.nan,
            "r2": float(model_add.rsquared),
            "adj_r2": float(model_add.rsquared_adj),
        })

        interaction_name = next((x for x in model_int.params.index if x.startswith("hproxy:C(class_label)") or x.startswith("C(class_label)") and ":hproxy" in x), None)
        rows.append({
            "response": yname,
            "model": "height-by-class interaction",
            "n": int(model_int.nobs),
            "height_coef": float(model_int.params["hproxy"]),
            "height_se_hc3": float(model_int.bse["hproxy"]),
            "height_p_hc3": float(model_int.pvalues["hproxy"]),
            "height_ci_low": float(model_int.conf_int().loc["hproxy", 0]),
            "height_ci_high": float(model_int.conf_int().loc["hproxy", 1]),
            "interaction_coef": float(model_int.params[interaction_name]) if interaction_name else np.nan,
            "interaction_p_hc3": float(model_int.pvalues[interaction_name]) if interaction_name else np.nan,
            "r2": float(model_int.rsquared),
            "adj_r2": float(model_int.rsquared_adj),
        })
    return pd.DataFrame(rows)


def maskarea_adjusted_models(df: pd.DataFrame, outcomes: Iterable[Tuple[str, str]]) -> pd.DataFrame:
    rows = []
    if "log_mask_area" not in df.columns:
        return pd.DataFrame()
    for ycol, yname in outcomes:
        d = df[[ycol, "hproxy", "class_label", "log_mask_area"]].dropna().copy()
        model = smf.ols(f"{ycol} ~ hproxy + C(class_label) + log_mask_area", data=d).fit(cov_type="HC3")
        ci = model.conf_int().loc["hproxy"]
        rows.append({
            "response": yname,
            "n": int(model.nobs),
            "height_coef": float(model.params["hproxy"]),
            "height_se_hc3": float(model.bse["hproxy"]),
            "height_p_hc3": float(model.pvalues["hproxy"]),
            "height_ci_low": float(ci[0]),
            "height_ci_high": float(ci[1]),
            "log_mask_area_coef": float(model.params["log_mask_area"]),
            "log_mask_area_p_hc3": float(model.pvalues["log_mask_area"]),
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
        })
    return pd.DataFrame(rows)


def make_scatter_plot(df: pd.DataFrame, ycol: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for cls, sub in df.groupby("class_label"):
        ax.scatter(sub["hproxy"], sub[ycol], alpha=0.8, label=cls)
        lr = stats.linregress(sub["hproxy"], sub[ycol])
        xs = np.linspace(sub["hproxy"].min(), sub["hproxy"].max(), 100)
        ax.plot(xs, lr.intercept + lr.slope * xs, linewidth=1.8)
    ax.set_xlabel("Measured geometric-height proxy (mm)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    df = load_records(args.xlsx, args.sheet)

    # Core outcomes. log(IH mean) and log(IL mean) are directly comparable in log units.
    outcomes = [
        ("ih_mean", "IH mean intensity"),
        ("il_mean", "IL mean intensity"),
        ("log_ih_mean", "log(mean IH)"),
        ("log_il_mean", "log(mean IL)"),
        ("common_log_mean", "common log response = [log(mean IH)+log(mean IL)]/2"),
        ("differential_log_mean", "differential log response = log(mean IH)-log(mean IL)"),
        ("mean_log_response_pixelwise", "pixelwise mean log response m"),
    ]

    rows = []
    for scope, sub in [("all", df), ("copper", df[df["class_label"] == "copper"]), ("waste", df[df["class_label"] == "waste"])]:
        for ycol, yname in outcomes:
            rows.append(summarize_xy(sub, scope, ycol, yname, args.bootstrap, rng))
    corr = pd.DataFrame(rows)

    # Multiplicity correction for the seven overall Pearson tests and separately for class-specific tests.
    corr["pearson_p_holm_within_scope"] = np.nan
    corr["spearman_p_holm_within_scope"] = np.nan
    for scope, idx in corr.groupby("scope").groups.items():
        inds = list(idx)
        corr.loc[inds, "pearson_p_holm_within_scope"] = multipletests(corr.loc[inds, "pearson_p"], method="holm")[1]
        corr.loc[inds, "spearman_p_holm_within_scope"] = multipletests(corr.loc[inds, "spearman_p"], method="holm")[1]

    repeat = repeatability_summary(df)
    adjusted = adjusted_models(df, [
        ("log_ih_mean", "log(mean IH)"),
        ("log_il_mean", "log(mean IL)"),
        ("common_log_mean", "common log response"),
        ("differential_log_mean", "differential log response"),
        ("mean_log_response_pixelwise", "pixelwise mean log response m"),
    ])
    maskarea_adj = maskarea_adjusted_models(df, [
        ("log_ih_mean", "log(mean IH)"),
        ("log_il_mean", "log(mean IL)"),
        ("common_log_mean", "common log response"),
        ("differential_log_mean", "differential log response"),
        ("mean_log_response_pixelwise", "pixelwise mean log response m"),
    ])

    # Primary test of unequal channel geometry sensitivities:
    # slope(logIH - logIL vs h) = slope(logIH vs h) - slope(logIL vs h).
    diff_all = corr[(corr.scope == "all") & (corr.response.str.startswith("differential log response"))].iloc[0]
    channel_h = corr[(corr.scope == "all") & (corr.response == "log(mean IH)")].iloc[0]
    channel_l = corr[(corr.scope == "all") & (corr.response == "log(mean IL)")].iloc[0]

    summary = {
        "n": int(len(df)),
        "class_counts": df["class_label"].value_counts().to_dict(),
        "hproxy_min_mm": float(df["hproxy"].min()),
        "hproxy_max_mm": float(df["hproxy"].max()),
        "hproxy_mean_mm": float(df["hproxy"].mean()),
        "hproxy_sd_mm": float(df["hproxy"].std(ddof=1)),
        "mask_note_counts": df["mask_note"].value_counts(dropna=False).astype(int).to_dict() if "mask_note" in df.columns else {},
        "primary_channel_log_slopes": {
            "log_mean_IH_per_mm": float(channel_h["slope_per_mm"]),
            "log_mean_IL_per_mm": float(channel_l["slope_per_mm"]),
            "difference_H_minus_L_per_mm": float(diff_all["slope_per_mm"]),
            "difference_95_boot_ci": [float(diff_all["slope_boot_ci_low"]), float(diff_all["slope_boot_ci_high"])],
            "difference_pearson_p": float(diff_all["pearson_p"]),
            "difference_pearson_p_holm": float(diff_all["pearson_p_holm_within_scope"]),
        },
        "interpretation_boundary": "The 60-particle set was acquired on a different DE-XRT system from the main 7,245-particle dataset. Use it for within-device measured-geometry/response association only; do not use it to calibrate delta to millimeters or as cross-device classifier validation for the main model.",
        "channel_log_definition": "log(mean IH) and log(mean IL) are logarithms of mask-mean normalized intensities from the workbook; they are distinct from the workbook's pixelwise mean log response m.",
    }

    df.to_csv(args.out_dir / "geometry60_derived_per_sample.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(args.out_dir / "geometry60_correlations_slopes.csv", index=False, encoding="utf-8-sig")
    repeat.to_csv(args.out_dir / "geometry60_repeatability.csv", index=False, encoding="utf-8-sig")
    adjusted.to_csv(args.out_dir / "geometry60_class_adjusted_models.csv", index=False, encoding="utf-8-sig")
    maskarea_adj.to_csv(args.out_dir / "geometry60_class_maskarea_adjusted_models.csv", index=False, encoding="utf-8-sig")
    make_scatter_plot(df, "log_ih_mean", "log(mean high-energy intensity)", args.out_dir / "geometry60_height_vs_logIH.png")
    make_scatter_plot(df, "log_il_mean", "log(mean low-energy intensity)", args.out_dir / "geometry60_height_vs_logIL.png")
    make_scatter_plot(df, "differential_log_mean", "log(mean IH) - log(mean IL)", args.out_dir / "geometry60_height_vs_differential.png")

    with open(args.out_dir / "geometry60_key_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Plain text compact report for quick server inspection.
    lines = []
    lines.append(f"N={len(df)}; class counts={summary['class_counts']}")
    lines.append(f"Hproxy range={summary['hproxy_min_mm']:.3f} to {summary['hproxy_max_mm']:.3f} mm")
    lines.append("Repeatability:")
    for _, r in repeat.iterrows():
        lines.append(
            f"  {r['measure']}: median |repeat diff|={r['median_abs_difference_mm']:.3f} mm, "
            f"max={r['max_abs_difference_mm']:.3f} mm, repeatability coefficient={r['repeatability_coefficient_1.96sd_mm']:.3f} mm"
        )
    lines.append("Primary channel log-response slopes:")
    lines.append(f"  log(mean IH) slope={channel_h['slope_per_mm']:.6f}/mm, r={channel_h['pearson_r']:.4f}, p={channel_h['pearson_p']:.3e}")
    lines.append(f"  log(mean IL) slope={channel_l['slope_per_mm']:.6f}/mm, r={channel_l['pearson_r']:.4f}, p={channel_l['pearson_p']:.3e}")
    lines.append(
        f"  differential [log(mean IH)-log(mean IL)] slope={diff_all['slope_per_mm']:.6f}/mm, "
        f"bootstrap95%CI=[{diff_all['slope_boot_ci_low']:.6f}, {diff_all['slope_boot_ci_high']:.6f}], "
        f"r={diff_all['pearson_r']:.4f}, p={diff_all['pearson_p']:.3e}, Holm p={diff_all['pearson_p_holm_within_scope']:.3e}"
    )
    lines.append("Class-adjusted models (HC3 robust SE):")
    for _, r in adjusted[adjusted["model"] == "class-adjusted additive"].iterrows():
        lines.append(
            f"  {r['response']}: height coef={r['height_coef']:.6f}/mm, "
            f"95%CI=[{r['height_ci_low']:.6f},{r['height_ci_high']:.6f}], p={r['height_p_hc3']:.3e}, R2={r['r2']:.3f}"
        )
    if not maskarea_adj.empty:
        r = maskarea_adj[maskarea_adj["response"] == "differential log response"].iloc[0]
        lines.append(
            f"Mask-area adjusted differential response: height coef={r['height_coef']:.6f}/mm, "
            f"95%CI=[{r['height_ci_low']:.6f},{r['height_ci_high']:.6f}], p={r['height_p_hc3']:.3e}; "
            f"log(mask area) p={r['log_mask_area_p_hc3']:.3e}."
        )
    lines.append("Boundary: different acquisition device from the main dataset; use as within-device geometry-response validation, not delta-to-mm calibration or cross-device classifier validation.")
    (args.out_dir / "geometry60_quick_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
