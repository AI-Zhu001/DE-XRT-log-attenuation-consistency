#!/usr/bin/env python3

from pathlib import Path
import argparse

import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader


from eval_thickness_shift import (
    build_resnet18_2ch,
    make_test_ds,
    eval_metrics,
    group_report_test,
)



# ======================================================
# paths
# ======================================================

CKPT_DIR = Path(
    "/root/autodl-tmp/ablation_lambda/checkpoints"
)


OUT_FILE = Path(
    "/root/autodl-tmp/lambda_ablation_worst_group.csv"
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



# ======================================================
# build args compatible with eval_thickness_shift.py
# ======================================================

def build_args():


    args = argparse.Namespace()


    # data

    args.csv_path = (
        "/root/projects/Hou_swin/"
        "split_outputs/"
        "copper_xray_all_splits.csv"
    )


    args.data_root = (
        "/root/autodl-tmp/data/"
        "原始购买的二分类数据集/"
        "原始购买的二分类数据集"
    )


    args.split_column = "split"


    args.img_size = 192


    args.eval_batch_size = 64

    args.batch_size = 64


    args.num_workers = 4



    # model

    args.pretrained = True



    # thickness augmentation arguments

    args.apply_p = 1.0

    args.fixed_delta = 0.0


    # augmentation flags

    args.thickness_aug = False

    args.use_brightness_aug = False

    args.hflip_p = 0.0


    return args



# ======================================================
# main
# ======================================================

def main():


    args = build_args()


    loss_fn = nn.CrossEntropyLoss()



    # clean test dataset

    test_ds = make_test_ds(
        args,
        fixed_delta=0.0
    )


    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )



    rows=[]



    ckpts = sorted(
        CKPT_DIR.glob(
            "best_lambda_*.pth"
        )
    )


    if len(ckpts)==0:

        raise RuntimeError(
            f"No checkpoint found in {CKPT_DIR}"
        )



    for ckpt in ckpts:


        print(
            "\nEvaluating:",
            ckpt.name
        )


        model = build_resnet18_2ch(
            num_classes=2,
            pretrained=args.pretrained
        )


        state = torch.load(
            ckpt,
            map_location=DEVICE
        )


        model.load_state_dict(
            state
        )


        model.to(
            DEVICE
        )


        model.eval()



        # normal metrics

        metrics = eval_metrics(
            model,
            test_loader,
            DEVICE,
            loss_fn
        )



        # worst-group

        group_df, group_summary = group_report_test(
            args,
            model,
            DEVICE,
            test_ds
        )



        rows.append({

            "lambda":
            ckpt.stem.split(
                "lambda_"
            )[1].split(
                "_"
            )[0],


            "checkpoint":
            ckpt.name,


            "acc":
            metrics["acc"],


            "bal_acc":
            metrics["bal_acc"],


            "macro_f1":
            metrics["macro_f1"],


            "worst_group_acc":
            group_summary.get(
                "worst_group_acc",
                None
            ),


            "worst_group_name":
            group_summary.get(
                "worst_group_name",
                None
            )

        })


    df = pd.DataFrame(
        rows
    )


    df = df.sort_values(
        "lambda"
    )


    df.to_csv(
        OUT_FILE,
        index=False
    )


    print("\n========== Result ==========")

    print(df)


    print(
        "\nSaved:",
        OUT_FILE
    )



if __name__ == "__main__":

    main()