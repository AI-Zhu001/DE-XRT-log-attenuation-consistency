#!/usr/bin/env python3

import time
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

from datasets.copper_xray_dataset import CopperXRayDataset


# =========================
# config
# =========================

CSV_PATH = "/root/projects/Hou_swin/split_outputs/copper_xray_all_splits.csv"

DATA_ROOT = (
    "/root/autodl-tmp/data/"
    "原始购买的二分类数据集/"
    "原始购买的二分类数据集"
)

IMG_SIZE = 192

BATCH_SIZE = 32

NUM_WORKERS = 4

DEVICE = "cuda"

NUM_EPOCHS_ESTIMATE = 20



# =========================
# model
# =========================

def build_resnet18_2ch():

    model = resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1
    )


    old = model.conv1


    model.conv1 = nn.Conv2d(
        2,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False
    )


    with torch.no_grad():

        w = old.weight.mean(
            dim=1,
            keepdim=True
        )

        model.conv1.weight.copy_(
            w.repeat(
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



# =========================
# dataset
# =========================

def get_loader():

    ds = CopperXRayDataset(
        CSV_PATH,
        split_column="split",
        split_value="train",
        data_root=DATA_ROOT,
        out_size=IMG_SIZE,
        hflip_p=0.5,
        thickness_aug=False,
        use_brightness_aug=False
    )


    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    return loader



# =========================
# augmentation
# =========================

def apply_brightness(
        x,
        low=-0.15,
        high=0.15
):

    delta = torch.empty(
        x.shape[0],
        1,
        1,
        1,
        device=x.device
    ).uniform_(
        low,
        high
    )

    return torch.clamp(
        x*(1+delta),
        0,
        1
    )



def apply_ci_attenuation(
        x,
        low=0.0,
        high=0.15
):

    eps=1e-6

    x_safe=torch.clamp(
        x,
        eps,
        1
    )


    out=x_safe.clone()

    B=x.shape[0]


    dh=torch.empty(
        B,1,1,1,
        device=x.device
    ).uniform_(
        low,
        high
    )


    dl=torch.empty(
        B,1,1,1,
        device=x.device
    ).uniform_(
        low,
        high
    )


    out[:,0:1]=torch.clamp(
        torch.exp(
            torch.log(x_safe[:,0:1])+dh
        ),
        0,
        1
    )


    out[:,1:2]=torch.clamp(
        torch.exp(
            torch.log(x_safe[:,1:2])+dl
        ),
        0,
        1
    )


    return out



# =========================
# benchmark baseline
# =========================

def benchmark_baseline():

    print("\nBenchmark Baseline")


    loader=get_loader()


    model=build_resnet18_2ch().to(
        DEVICE
    )


    optimizer=AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )


    loss_fn=nn.CrossEntropyLoss()


    scaler=GradScaler()


    model.train()


    start=time.time()


    for x,y in loader:

        x=x.to(DEVICE)
        y=y.to(DEVICE)


        optimizer.zero_grad()


        with autocast():

            logits=model(x)

            loss=loss_fn(
                logits,
                y
            )


        scaler.scale(
            loss
        ).backward()


        scaler.step(
            optimizer
        )

        scaler.update()


    if DEVICE=="cuda":
        torch.cuda.synchronize()


    elapsed=time.time()-start


    print(
        f"1 epoch: {elapsed:.2f}s"
    )

    print(
        f"20 epochs: {elapsed*20:.2f}s"
    )


    return elapsed



# =========================
# benchmark consistency
# =========================

def benchmark_consistency():

    print("\nBenchmark CI-Consistency")


    loader=get_loader()


    model=build_resnet18_2ch().to(
        DEVICE
    )


    optimizer=AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )


    loss_fn=nn.CrossEntropyLoss()


    scaler=GradScaler()


    model.train()


    start=time.time()


    for x,y in loader:


        x=x.to(DEVICE)
        y=y.to(DEVICE)


        x_b=apply_brightness(
            x
        )


        x_ci=apply_ci_attenuation(
            x
        )


        optimizer.zero_grad()


        with autocast():

            logits=model(x)

            logits_b=model(x_b)

            logits_ci=model(x_ci)


            ce=loss_fn(
                logits,
                y
            )


            ce_b=loss_fn(
                logits_b,
                y
            )


            cons=nn.functional.mse_loss(
                logits_ci,
                logits.detach()
            )


            loss=(
                ce
                +
                ce_b
                +
                0.5*cons
            )


        scaler.scale(
            loss
        ).backward()


        scaler.step(
            optimizer
        )


        scaler.update()



    if DEVICE=="cuda":
        torch.cuda.synchronize()


    elapsed=time.time()-start


    print(
        f"1 epoch: {elapsed:.2f}s"
    )


    print(
        f"20 epochs: {elapsed*20:.2f}s"
    )


    return elapsed



# =========================
# main
# =========================

if __name__=="__main__":


    baseline_time = benchmark_baseline()

    consistency_time = benchmark_consistency()


    print("\n========== Summary ==========")

    print(
        f"Baseline epoch: {baseline_time:.2f}s"
    )

    print(
        f"Consistency epoch: {consistency_time:.2f}s"
    )


    print(
        "Training overhead:"
    )

    print(
        f"{consistency_time/baseline_time:.2f}x"
    )