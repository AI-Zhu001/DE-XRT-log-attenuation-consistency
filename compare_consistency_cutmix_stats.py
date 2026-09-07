import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ttest_rel, wilcoxon


# =====================================================
# read CI-Consistency
# =====================================================

def read_ci(folder, shift_h, shift_l):

    files = sorted(
        Path(folder).glob(
            "channel_independent_summary_ci_consistency_seed*.csv"
        )
    )

    f1 = []
    wg = []

    print("\nCI files:")

    for f in files:

        df = pd.read_csv(f)

        row = df[
            (df["delta_high"] == shift_h) &
            (df["delta_low"] == shift_l)
        ]

        if len(row) == 0:
            continue

        row = row.iloc[0]

        print(
            f.name,
            row["macro_f1"],
            row["worst_group_acc"]
        )

        f1.append(
            float(row["macro_f1"])
        )

        wg.append(
            float(row["worst_group_acc"])
        )


    return np.array(f1), np.array(wg)



# =====================================================
# read CutMix
# =====================================================

def read_cutmix(shift_h, shift_l):

    files = sorted(
        Path("/root/autodl-tmp").glob(
            "cutmix_shift_eval_seed*/"
            "channel_independent_summary_CutMix_seed*.csv"
        )
    )


    f1 = []
    wg = []

    print("\nCutMix files:")

    for f in files:

        df = pd.read_csv(f)


        row = df[
            (df["delta_high"] == shift_h) &
            (df["delta_low"] == shift_l)
        ]


        if len(row) == 0:
            continue


        row = row.iloc[0]


        print(
            f.name,
            row["macro_f1"],
            row["worst_group_acc"]
        )


        f1.append(
            float(row["macro_f1"])
        )

        wg.append(
            float(row["worst_group_acc"])
        )


    return np.array(f1), np.array(wg)



# =====================================================
# statistics
# =====================================================

def compare(
    shift,
    metric,
    cutmix,
    consistency
):

    diff = consistency - cutmix


    result = {

        "shift":
        shift,

        "metric":
        metric,


        "CutMix_mean":
        cutmix.mean(),

        "CutMix_std":
        cutmix.std(ddof=1),


        "CI-Consistency_mean":
        consistency.mean(),

        "CI-Consistency_std":
        consistency.std(ddof=1),


        "difference(CI-CutMix)":
        diff.mean(),


        "better_seeds":
        f"{int((diff>0).sum())}/{len(diff)}"

    }


    if len(diff) >= 2:

        result["paired_t_p"] = (
            ttest_rel(
                consistency,
                cutmix
            ).pvalue
        )

        try:
            result["wilcoxon_p"] = (
                wilcoxon(
                    consistency,
                    cutmix
                ).pvalue
            )

        except:

            result["wilcoxon_p"] = None


    else:

        result["paired_t_p"] = None
        result["wilcoxon_p"] = None


    return result



# =====================================================
# main
# =====================================================

def main():


    results = []


    # ----------------------------
    # + shift
    # ----------------------------

    ci_f1_pos, ci_wg_pos = read_ci(
        "/root/autodl-tmp/ci_consistency_shift_eval_v2",
        0.25,
        -0.25
    )


    cut_f1_pos, cut_wg_pos = read_cutmix(
        0.25,
        -0.25
    )



    results.append(
        compare(
            "(+0.25,-0.25)",
            "Macro-F1",
            cut_f1_pos,
            ci_f1_pos
        )
    )


    results.append(
        compare(
            "(+0.25,-0.25)",
            "Worst-group accuracy",
            cut_wg_pos,
            ci_wg_pos
        )
    )



    # ----------------------------
    # - shift
    # ----------------------------

    ci_f1_neg, ci_wg_neg = read_ci(
        "/root/autodl-tmp/ci_consistency_shift_eval_v2",
        -0.25,
        0.25
    )


    cut_f1_neg, cut_wg_neg = read_cutmix(
        -0.25,
        0.25
    )



    results.append(
        compare(
            "(-0.25,+0.25)",
            "Macro-F1",
            cut_f1_neg,
            ci_f1_neg
        )
    )


    results.append(
        compare(
            "(-0.25,+0.25)",
            "Worst-group accuracy",
            cut_wg_neg,
            ci_wg_neg
        )
    )



    out = pd.DataFrame(results)


    save = (
        "/root/autodl-tmp/"
        "CI_vs_CutMix_statistics.csv"
    )


    out.to_csv(
        save,
        index=False
    )


    print("\n========== Result ==========")

    print(
        out.to_string(
            index=False
        )
    )


    print(
        "\nSaved:",
        save
    )



if __name__ == "__main__":

    main()