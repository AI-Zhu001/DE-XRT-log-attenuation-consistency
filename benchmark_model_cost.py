#!/usr/bin/env python3

import time
import torch
from torchvision.models import resnet18, ResNet18_Weights

from thop import profile


# ==========================
# config
# ==========================

DEVICE = "cuda"

IMG_SIZE = 192

NUM_RUNS = 200



# ==========================
# ResNet18 2-channel
# ==========================

def build_resnet18_2ch():

    model = resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1
    )


    old_conv = model.conv1


    model.conv1 = torch.nn.Conv2d(
        2,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )


    with torch.no_grad():

        w = old_conv.weight.mean(
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


    model.fc = torch.nn.Linear(
        model.fc.in_features,
        2
    )


    return model



# ==========================
# benchmark
# ==========================

def benchmark(
        model,
        name
):

    model.eval()

    model.to(DEVICE)


    x = torch.randn(
        1,
        2,
        IMG_SIZE,
        IMG_SIZE
    ).to(DEVICE)



    # FLOPs

    flops, params = profile(
        model,
        inputs=(x,),
        verbose=False
    )


    params_m = params / 1e6

    flops_g = flops / 1e9



    # warmup

    with torch.no_grad():

        for _ in range(20):

            _ = model(x)



    if DEVICE=="cuda":

        torch.cuda.synchronize()



    start=time.time()


    with torch.no_grad():

        for _ in range(NUM_RUNS):

            _ = model(x)



    if DEVICE=="cuda":

        torch.cuda.synchronize()



    end=time.time()


    latency_ms = (
        (end-start)
        /
        NUM_RUNS
        *
        1000
    )


    fps = 1000 / latency_ms



    print("\n====================")

    print(name)

    print("====================")

    print(
        f"Params: {params_m:.3f} M"
    )

    print(
        f"FLOPs: {flops_g:.3f} G"
    )

    print(
        f"Latency: {latency_ms:.3f} ms/sample"
    )

    print(
        f"FPS: {fps:.2f}"
    )



if __name__=="__main__":


    model = build_resnet18_2ch()


    benchmark(
        model,
        "ResNet18-DE-XRT"
    )