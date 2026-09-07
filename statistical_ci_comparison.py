#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

from scipy.stats import (
    ttest_rel,
    wilcoxon
)


# =====================================================
# paths
# =====================================================

DIRECT_DIR = Path(
    "/root/autodl-tmp/ci_direct_shift_eval"
)

CONS_DIR = Path(
    "/root/autodl-tmp/ci_consistency_shift_eval_v2"
)


OUT_FILE = Path(
    "/root/autodl-tmp/CI_shift_statistical_comparison_with_worstgroup.csv"
)



# =====================================================
# collect
# =====================================================

def collect_csv(folder, method):

    rows = []

    for f in sorted(folder.glob("*.csv")):

        df = pd.read_csv(f)
        df["method"] = method
        rows.append(df)

    if len(rows) == 0:
        raise RuntimeError(
            f"No csv found in {folder}"
        )

    return pd.concat(
        rows,
        ignore_index=True
    )



# =====================================================
# CI
# =====================================================

def mean_ci(values):

    values = np.array(values)

    n = len(values)

    mean = values.mean()

    std = values.std(
        ddof=1
    )

    ci = 1.96 * std / np.sqrt(n)

    return (
        mean,
        std,
        mean-ci,
        mean+ci
    )



# =====================================================
# paired test helper
# =====================================================

def paired_stats(
        consistency,
        direct
):

    t_stat, p_t = ttest_rel(
        consistency,
        direct
    )

    try:

        _, p_w = wilcoxon(
            consistency,
            direct
        )

    except Exception:

        p_w = np.nan


    return p_t, p_w



# =====================================================
# main
# =====================================================

def main():


    direct = collect_csv(
        DIRECT_DIR,
        "CI-DirectAttenAug"
    )


    consistency = collect_csv(
        CONS_DIR,
        "CI-Consistency"
    )


    results = []


    shifts = sorted(
        set(
            zip(
                direct.delta_high,
                direct.delta_low
            )
        )
    )


    for dh, dl in shifts:


        d = direct[
            (direct.delta_high == dh)
            &
            (direct.delta_low == dl)
        ]


        c = consistency[
            (consistency.delta_high == dh)
            &
            (consistency.delta_low == dl)
        ]


        # only matched 5 runs

        if len(d) != 5 or len(c) != 5:
            continue



        # =========================
        # Macro-F1
        # =========================

        d_f1 = d.macro_f1.values

        c_f1 = c.macro_f1.values


        d_f1_mean, d_f1_std, _, _ = mean_ci(
            d_f1
        )

        c_f1_mean, c_f1_std, c_low, c_high = mean_ci(
            c_f1
        )


        f1_p_t, f1_p_w = paired_stats(
            c_f1,
            d_f1
        )



        # =========================
        # Worst-group accuracy
        # =========================

        d_w = d.worst_group_acc.values

        c_w = c.worst_group_acc.values


        d_w_mean, d_w_std, _, _ = mean_ci(
            d_w
        )

        c_w_mean, c_w_std, w_low, w_high = mean_ci(
            c_w
        )


        w_p_t, w_p_w = paired_stats(
            c_w,
            d_w
        )



        results.append({

            "delta_high":
            dh,

            "delta_low":
            dl,


            # Macro-F1

            "direct_mean_f1":
            d_f1_mean,

            "direct_std_f1":
            d_f1_std,


            "consistency_mean_f1":
            c_f1_mean,

            "consistency_std_f1":
            c_f1_std,


            "f1_difference":
            c_f1_mean - d_f1_mean,


            "f1_paired_ttest_p":
            f1_p_t,

            "f1_wilcoxon_p":
            f1_p_w,



            # Worst-group

            "direct_mean_worst_group":
            d_w_mean,

            "direct_std_worst_group":
            d_w_std,


            "consistency_mean_worst_group":
            c_w_mean,

            "consistency_std_worst_group":
            c_w_std,


            "worst_group_difference":
            c_w_mean - d_w_mean,


            "worst_group_paired_ttest_p":
            w_p_t,


            "worst_group_wilcoxon_p":
            w_p_w,


            "consistency_better_seeds_f1":
            int(
                np.sum(
                    c_f1 > d_f1
                )
            ),


            "consistency_better_seeds_worst_group":
            int(
                np.sum(
                    c_w > d_w
                )
            )

        })


    out = pd.DataFrame(
        results
    )


    out = out.sort_values(
        [
            "delta_high",
            "delta_low"
        ]
    )


    out.to_csv(
        OUT_FILE,
        index=False
    )


    print(
        "Saved:",
        OUT_FILE
    )


    print("\nKey shifts:\n")


    print(
        out[
            (
                (out.delta_high == 0.25)
                &
                (out.delta_low == -0.25)
            )
            |
            (
                (out.delta_high == -0.25)
                &
                (out.delta_low == 0.25)
            )
        ]
    )



if __name__ == "__main__":

    main()