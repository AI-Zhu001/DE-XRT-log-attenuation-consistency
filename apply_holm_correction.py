#!/usr/bin/env python3

from pathlib import Path
import argparse

import pandas as pd
from statsmodels.stats.multitest import multipletests



# =====================================================
# args
# =====================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Apply Holm correction to correlation p-values"
    )


    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )


    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )


    return parser.parse_args()



# =====================================================
# Holm function
# =====================================================

def apply_holm(
        df,
        p_col,
        prefix
):

    if p_col not in df.columns:
        return df


    valid = df[p_col].notna()


    if valid.sum() == 0:
        return df



    reject, corrected, _, _ = multipletests(
        df.loc[valid, p_col],
        method="holm"
    )


    df.loc[
        valid,
        f"{prefix}_p_holm"
    ] = corrected


    df.loc[
        valid,
        f"{prefix}_significant_holm"
    ] = reject


    return df



# =====================================================
# main
# =====================================================

def main():

    args = parse_args()


    df = pd.read_csv(
        args.input
    )


    print(
        "Loaded:",
        args.input
    )

    print(
        "Rows:",
        len(df)
    )


    # Pearson

    df = apply_holm(
        df,
        "pearson_p",
        "pearson"
    )


    # Spearman

    df = apply_holm(
        df,
        "spearman_p",
        "spearman"
    )



    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    df.to_csv(
        args.output,
        index=False
    )


    print(
        "Saved:",
        args.output
    )


    print(
        df.head()
    )



if __name__ == "__main__":

    main()