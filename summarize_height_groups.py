#!/usr/bin/env python3

from pathlib import Path
import pandas as pd



# =====================================================
# paths
# =====================================================

INPUT_DIR = Path(
    "/root/projects/Hou_swin/"
    "measured60_eval_outputs/"
    "source60_all_models"
)


OUT_RAW = Path(
    "/root/autodl-tmp/"
    "height_group_summary_raw.csv"
)


OUT_SUMMARY = Path(
    "/root/autodl-tmp/"
    "height_group_summary_mean_std.csv"
)



# =====================================================
# parse filename
# =====================================================

def parse_method_seed(filename):

    """
    Example:

    group_metrics_source60_consistency_seed43

    return:
        consistency, 43
    """

    prefix = "group_metrics_source60_"


    if not filename.startswith(prefix):

        return (
            "unknown",
            -1
        )


    name = filename[len(prefix):]


    if "_seed" not in name:

        return (
            name,
            -1
        )


    method, seed = name.rsplit(
        "_seed",
        1
    )


    try:

        seed = int(seed)

    except:

        seed = -1


    return (
        method,
        seed
    )



# =====================================================
# main
# =====================================================

def main():


    rows = []


    files = sorted(
        INPUT_DIR.rglob(
            "group_metrics_source60_*.csv"
        )
    )


    print(
        "Found files:",
        len(files)
    )


    if len(files) == 0:

        raise RuntimeError(
            "No source60 group csv found"
        )



    for f in files:


        method, seed = parse_method_seed(
            f.stem
        )


        print(
            "Reading:",
            f.name,
            "=>",
            method,
            seed
        )


        df = pd.read_csv(
            f
        )


        if "group_type" not in df.columns:

            continue



        # only height groups

        height_df = df[
            df["group_type"] == "height"
        ]



        for _, r in height_df.iterrows():


            rows.append({

                "method":
                method,


                "seed":
                seed,


                "height_group":
                r["group"],


                "n":
                r["n"],


                "acc":
                r["acc"],


                "bal_acc":
                r["bal_acc"],


                "macro_f1":
                r["macro_f1"],


                "precision":
                r["precision"],


                "recall":
                r["recall"]

            })



    raw = pd.DataFrame(
        rows
    )


    print(
        "\nCollected rows:",
        len(raw)
    )


    if len(raw) == 0:

        raise RuntimeError(
            "No height rows collected"
        )


    print(
        raw.head()
    )



    # save raw

    raw.to_csv(
        OUT_RAW,
        index=False
    )



    # =====================================================
    # mean std summary
    # =====================================================

    summary = (

        raw
        .groupby(
            [
                "method",
                "height_group"
            ]
        )
        .agg(

            runs=(
                "seed",
                "nunique"
            ),


            mean_n=(
                "n",
                "mean"
            ),


            acc_mean=(
                "acc",
                "mean"
            ),

            acc_std=(
                "acc",
                "std"
            ),


            bal_acc_mean=(
                "bal_acc",
                "mean"
            ),

            bal_acc_std=(
                "bal_acc",
                "std"
            ),


            macro_f1_mean=(
                "macro_f1",
                "mean"
            ),

            macro_f1_std=(
                "macro_f1",
                "std"
            )

        )

        .reset_index()

    )



    summary.to_csv(
        OUT_SUMMARY,
        index=False
    )



    print(
        "\nSaved raw:"
    )

    print(
        OUT_RAW
    )


    print(
        "\nSaved summary:"
    )

    print(
        OUT_SUMMARY
    )


    print(
        "\nSummary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )



if __name__ == "__main__":

    main()