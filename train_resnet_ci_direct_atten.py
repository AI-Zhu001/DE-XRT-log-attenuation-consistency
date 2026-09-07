#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)

from datasets.copper_xray_dataset import CopperXRayDataset


try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass


# =========================================================
# arguments
# =========================================================

def parse_args():

    p = argparse.ArgumentParser(
        description=
        "ResNet18 + channel-independent log-domain DirectAttenAug"
    )

    p.add_argument(
        "--csv-path",
        type=Path,
        required=True
    )

    p.add_argument(
        "--data-root",
        type=str,
        required=True
    )

    p.add_argument(
        "--split-column",
        type=str,
        default="split"
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=64
    )

    p.add_argument(
        "--img-size",
        type=int,
        default=192
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=4
    )

    p.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42
    )

    p.add_argument(
        "--amp",
        action="store_true"
    )

    p.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false"
    )

    p.set_defaults(
        amp=True
    )

    p.add_argument(
        "--log-dir",
        type=Path,
        required=True
    )

    p.add_argument(
        "--ckpt-path",
        type=Path,
        required=True
    )

    p.add_argument(
        "--best-metric",
        type=str,
        default="macro_f1",
        choices=[
            "macro_f1",
            "bal_acc",
            "acc"
        ]
    )

    p.add_argument(
        "--hflip-p",
        type=float,
        default=0.5
    )

    p.add_argument(
        "--pretrained",
        action="store_true"
    )

    p.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false"
    )

    p.set_defaults(
        pretrained=True
    )


    # brightness branch
    p.add_argument(
        "--brightness-delta-min",
        type=float,
        default=-0.15
    )

    p.add_argument(
        "--brightness-delta-max",
        type=float,
        default=0.15
    )

    p.add_argument(
        "--brightness-apply-p",
        type=float,
        default=0.5
    )


    # channel independent attenuation
    p.add_argument(
        "--delta-min",
        type=float,
        default=0.0
    )

    p.add_argument(
        "--delta-max",
        type=float,
        default=0.15
    )


    return p.parse_args()



# =========================================================
# seed
# =========================================================

def set_seed(seed: int):

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# =========================================================
# model
# =========================================================

def build_resnet18_2ch(
        pretrained=True
):

    weights = (
        ResNet18_Weights.IMAGENET1K_V1
        if pretrained
        else None
    )

    model = resnet18(
        weights=weights
    )

    old = model.conv1


    model.conv1 = nn.Conv2d(
        2,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None)
    )


    with torch.no_grad():

        if pretrained:

            w_mean = old.weight.mean(
                dim=1,
                keepdim=True
            )

            model.conv1.weight.copy_(
                w_mean.repeat(
                    1,
                    2,
                    1,
                    1
                )
            )


    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    return model



# =========================================================
# dataset
# =========================================================

def make_loader(
        args,
        split
):

    ds = CopperXRayDataset(
        args.csv_path,
        split_column=args.split_column,
        split_value=split,
        data_root=args.data_root,
        out_size=args.img_size,

        hflip_p=(
            args.hflip_p
            if split == "train"
            else 0.0
        ),

        thickness_aug=False,
        use_brightness_aug=False,
    )


    loader = DataLoader(
        ds,

        batch_size=(
            args.batch_size
            if split == "train"
            else args.eval_batch_size
        ),

        shuffle=(
            True
            if split == "train"
            else False
        ),

        num_workers=args.num_workers
    )

    return loader
    # =========================================================
# brightness augmentation
# =========================================================

def apply_brightness_batch(
        x,
        delta_min,
        delta_max,
        apply_p
):

    x_aug = x.clone()

    B = x.shape[0]

    probs = torch.rand(
        B,
        device=x.device
    )

    deltas = torch.empty(
        B,
        device=x.device
    ).uniform_(
        delta_min,
        delta_max
    )


    for i in range(B):

        if probs[i] < apply_p:

            x_aug[i] = torch.clamp(
                x_aug[i] * (1.0 + deltas[i]),
                0.0,
                1.0
            )


    return x_aug



# =========================================================
# channel independent attenuation perturbation
# =========================================================

def apply_ci_attenuation(
        x,
        delta_min,
        delta_max,
        eps=1e-6
):

    """
    Independent log-domain attenuation perturbation.

    channel 0: high-energy
    channel 1: low-energy

    delta_H and delta_L are sampled independently.
    """

    x_safe = torch.clamp(
        x,
        eps,
        1.0
    )

    x_out = x_safe.clone()

    B = x.shape[0]


    delta_h = torch.empty(
        B,
        1,
        1,
        1,
        device=x.device
    ).uniform_(
        delta_min,
        delta_max
    )


    delta_l = torch.empty(
        B,
        1,
        1,
        1,
        device=x.device
    ).uniform_(
        delta_min,
        delta_max
    )


    # high energy channel

    x_out[:, 0:1] = torch.clamp(
        torch.exp(
            torch.log(
                x_safe[:, 0:1]
            )
            +
            delta_h
        ),
        0.0,
        1.0
    )


    # low energy channel

    x_out[:, 1:2] = torch.clamp(
        torch.exp(
            torch.log(
                x_safe[:, 1:2]
            )
            +
            delta_l
        ),
        0.0,
        1.0
    )


    return x_out



# =========================================================
# evaluation
# =========================================================

@torch.no_grad()
def evaluate(
        model,
        loader,
        device
):

    model.eval()


    all_y = []
    all_p = []


    for x, y in loader:

        x = x.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device
        )


        logits = model(x)

        pred = logits.argmax(
            dim=1
        )


        all_y.extend(
            y.cpu().tolist()
        )

        all_p.extend(
            pred.cpu().tolist()
        )


    return {

        "acc":
        float(
            accuracy_score(
                all_y,
                all_p
            )
        ),


        "bal_acc":
        float(
            balanced_accuracy_score(
                all_y,
                all_p
            )
        ),


        "macro_f1":
        float(
            f1_score(
                all_y,
                all_p,
                average="macro",
                zero_division=0
            )
        ),


        "precision":
        float(
            precision_score(
                all_y,
                all_p,
                average="macro",
                zero_division=0
            )
        ),


        "recall":
        float(
            recall_score(
                all_y,
                all_p,
                average="macro",
                zero_division=0
            )
        )
    }



# =========================================================
# training
# =========================================================

def main():

    args = parse_args()


    set_seed(
        args.seed
    )


    device = torch.device(
        "cuda"
        if (
            args.device == "cuda"
            and torch.cuda.is_available()
        )
        else "cpu"
    )


    args.log_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    args.ckpt_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    writer = SummaryWriter(
        str(args.log_dir)
    )


    train_loader = make_loader(
        args,
        "train"
    )

    val_loader = make_loader(
        args,
        "val"
    )

    test_loader = make_loader(
        args,
        "test"
    )


    model = build_resnet18_2ch(
        pretrained=args.pretrained
    )


    model = model.to(
        device
    )


    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )


    scaler = GradScaler(
        enabled=args.amp
    )


    loss_fn = nn.CrossEntropyLoss()


    best = -1.0



    for epoch in range(
        1,
        args.epochs + 1
    ):


        model.train()


        total_loss = 0.0
        n = 0


        for x, y in train_loader:


            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device
            )


            # brightness supervised branch

            x_b = apply_brightness_batch(
                x,
                args.brightness_delta_min,
                args.brightness_delta_max,
                args.brightness_apply_p
            )


            # channel-independent attenuation branch

            x_ci = apply_ci_attenuation(
                x,
                args.delta_min,
                args.delta_max
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            with autocast(
                enabled=args.amp
            ):


                # clean branch

                logits = model(x)


                # brightness CE branch

                logits_b = model(x_b)


                # CI attenuation CE branch

                logits_ci = model(x_ci)


                ce = loss_fn(
                    logits,
                    y
                )


                ce_b = loss_fn(
                    logits_b,
                    y
                )


                ce_ci = loss_fn(
                    logits_ci,
                    y
                )


                # CI-DirectAttenAug objective
                #
                # The only difference from CI-Consistency:
                # supervised CE replaces consistency loss

                loss = (
                    ce
                    +
                    ce_b
                    +
                    ce_ci
                )


            scaler.scale(
                loss
            ).backward()


            scaler.step(
                optimizer
            )


            scaler.update()


            total_loss += (
                float(loss.item())
                *
                y.numel()
            )

            n += y.numel()



        train_loss = total_loss / max(n, 1)


        writer.add_scalar(
            "train/loss",
            train_loss,
            epoch
        )


        val = evaluate(
            model,
            val_loader,
            device
        )


        writer.add_scalar(
            "val/macro_f1",
            val["macro_f1"],
            epoch
        )


        writer.add_scalar(
            "val/bal_acc",
            val["bal_acc"],
            epoch
        )


        score = val[
            args.best_metric
        ]


        if score > best:

            best = score

            torch.save(
                model.state_dict(),
                args.ckpt_path
            )


        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_macro_f1={val['macro_f1']:.4f}",
            flush=True
        )
                # end epoch


    # =====================================================
    # load best checkpoint
    # =====================================================

    model.load_state_dict(
        torch.load(
            args.ckpt_path,
            map_location=device
        )
    )


    test = evaluate(
        model,
        test_loader,
        device
    )


    print(
        "[TEST]",
        json.dumps(
            test,
            indent=2
        ),
        flush=True
    )


    # =====================================================
    # save summary
    # =====================================================

    summary = {

        "method":
        "CI-DirectAttenAug",


        "perturbation_mode":
        "channel_independent",


        "training_objective":
        "supervised_CE_on_independent_attenuation",


        "seed":
        int(args.seed),


        "delta_min":
        float(args.delta_min),


        "delta_max":
        float(args.delta_max),


        "brightness_delta_min":
        float(args.brightness_delta_min),


        "brightness_delta_max":
        float(args.brightness_delta_max),


        "brightness_apply_p":
        float(args.brightness_apply_p),


        "best_metric":
        args.best_metric,


        "best_metric_value":
        float(best),


        "test":
        test
    }



    out_json = (
        args.log_dir /
        "summary.json"
    )


    out_json.write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )


    print(
        f"[SUMMARY] saved to {out_json}",
        flush=True
    )


    writer.close()



if __name__ == "__main__":

    main()