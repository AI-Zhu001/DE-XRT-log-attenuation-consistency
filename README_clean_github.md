# DE-XRT Directional Consistency

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the main training, evaluation, and analysis code for the manuscript:

**Directional Response Consistency Regularization for Robust Dual-Energy X-Ray Ore Sorting**

**Authors:** Zhi-yong Zhu, Jian-feng He, Xue-yuan Wang, Fei Xia, Feng-jun Nie, Wen Wang, Yang-hui Zou, Wei-dong Li, Guo-yun Zhong, Zhi-xiang Ye, Fan Diao

> The manuscript is currently under preparation/submission. The title and citation information will be updated after publication.

---

## Overview

This work studies robustness in dual-energy X-ray transmission (DE-XRT) ore classification from the perspective of paired high- and low-energy response sensitivity.

We consider two representative response directions:

- **Common-mode response:** both channels change in the same direction, for example `(delta_H, delta_L) = (delta, delta)`.
- **Differential response:** the two channels change in opposite directions, for example `(delta_H, delta_L) = (delta, -delta)`.

The main method, **Differential-Consistency**, specifically regularizes the differential response direction.

During training:

- `a ~ Uniform(0, 0.15)`
- `s` is randomly sampled from `{-1, +1}`
- `delta_H = s * a`
- `delta_L = -s * a`

The model is trained with:

1. cross-entropy loss on the clean sample;
2. cross-entropy loss on a brightness-augmented sample;
3. consistency loss between the logits of the clean sample and the differential-response-perturbed sample.

The consistency weight is `lambda = 0.5`.

The perturbation is treated as a **controlled DE-XRT channel-response shift**. It is not interpreted as a calibrated physical thickness change in millimeters.

---

## Main Results

All primary matched experiments use the same five random seeds:

```text
42, 52, 62, 72, 82
```

### Differential-Consistency

| Evaluation condition | Macro-F1 |
|---|---:|
| Clean test set | 0.9136 +/- 0.0031 |
| Common shift (-0.15, -0.15) | 0.9020 +/- 0.0063 |
| Common shift (+0.15, +0.15) | 0.8924 +/- 0.0135 |
| Differential shift (+0.15, -0.15) | 0.8341 +/- 0.0424 |
| Differential shift (-0.15, +0.15) | 0.8369 +/- 0.0350 |

### Full 2-D response-plane evaluation

The response plane uses:

```text
delta_H, delta_L in {-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15}
```

This gives `7 x 7 = 49` controlled response conditions.

Differential-Consistency achieved:

- mean Macro-F1: `0.8800 +/- 0.0061`
- worst-point Macro-F1: `0.8128 +/- 0.0328`
- mean prediction-flip rate: `0.0758 +/- 0.0096`

### Directional sensitivity

At local perturbation magnitude `epsilon = 0.025`, differential logit-margin sensitivity was:

| Method | Differential sensitivity |
|---|---:|
| Ordinary synchronized Consistency | 176.70 +/- 16.33 |
| CI-Consistency | 13.08 +/- 7.83 |
| Differential-Consistency | 19.97 +/- 3.96 |

Differential-Consistency reduced differential sensitivity by approximately `88.7%` relative to ordinary synchronized Consistency.

### Corrected worst-group evaluation

The previous grouping implementation was corrected before the final analysis.

Selected Worst-Group Accuracy (WGA) values for Differential-Consistency are:

| Condition | WGA |
|---|---:|
| Clean | 0.9071 |
| (-0.25, -0.25) | 0.8635 |
| (+0.25, +0.25) | 0.8592 |
| (+0.15, -0.15) | 0.7956 |
| (-0.15, +0.15) | 0.8301 |
| (+0.25, -0.25) | 0.5599 |
| (-0.25, +0.25) | 0.7385 |

Worst-group balanced accuracy and worst-group Macro-F1 are also included in the released CSV results.

---

## Independent Measured-Geometry Evidence

A separate set of 60 newly collected physical particles was used to study the relationship between measured particle geometry and paired DE-XRT response.

The set contains:

- 30 copper particles
- 30 waste particles

Each particle includes repeated measurements of maximum height and center height.

Important limitation:

> These 60 particles were acquired using a different DE-XRT system from the main 7,245-particle industrial dataset. They are used only as independent geometry-response evidence and not as cross-device classifier validation.

Key observations include:

- high-energy log-response slope: `-0.02130 / mm`
- low-energy log-response slope: `-0.05645 / mm`
- differential-response slope: `+0.03515 / mm`
- bootstrap 95% CI for the differential slope: `[0.01643, 0.05600]`

After adjustment for copper/waste class:

- height coefficient: `0.02557 / mm`
- p-value: `1.17e-12`

After additional adjustment for projected particle area:

- height coefficient: `0.02539 / mm`
- p-value: `1.64e-8`

These measurements support the existence of a geometry-associated channel-differential response component.

They do not establish a calibration between the perturbation parameter `delta` and physical thickness in millimeters.

---

## Dataset

The main industrial dataset contains 7,245 paired DE-XRT particle samples:

- 2,806 copper ore particles
- 4,439 waste particles

Data split:

- training: 5,071
- validation: 1,087
- test: 1,087

Main DE-XRT acquisition settings:

- tube voltage: 160 kV
- tube current: 550 microampere

The samples were collected onsite at Dexing Copper Mine.

### Data availability

The full industrial dataset cannot be publicly released because of proprietary restrictions.

This repository therefore provides code, example data organization, and lightweight result tables rather than the complete industrial image dataset.

The binary labels used in the study were operational labels provided with the industrial dataset.

Particle-wise assay/XRF measurements were not available.

No unverified particle-level copper cutoff should be inferred from these labels.

---

## Data Format

Each sample consists of paired high- and low-energy images.

Example manifest:

```csv
sample_id,label,high_path,low_path,split,response_group
sample_0001,1,path/to/high/sample_0001.png,path/to/low/sample_0001.png,train,medium_indicator
sample_0002,0,path/to/high/sample_0002.png,path/to/low/sample_0002.png,test,low_indicator
```

Columns:

- `sample_id`: sample identifier
- `label`: binary class label
- `high_path`: path to the high-energy image
- `low_path`: path to the low-energy image
- `split`: `train`, `val`, or `test`
- `response_group`: optional clean-response group used for worst-group evaluation

For the ResNet18 experiments, paired images are resized to `192 x 192`.

---

## Installation

### Requirements

- Python 3.8+
- PyTorch
- torchvision
- NumPy
- pandas
- SciPy
- scikit-learn
- Pillow
- NVIDIA GPU recommended

Install dependencies:

```bash
pip install -r requirements.txt
```

Clone the repository:

```bash
git clone https://github.com/AI-Zhu001/DE-XRT-directional-consistency.git
cd DE-XRT-directional-consistency
```

---

## Reproducibility

All primary five-seed experiments use exactly:

```text
42 52 62 72 82
```

Use these same seeds to reproduce the reported means and standard deviations.

Statistical note:

> With five paired observations, the smallest possible exact two-sided Wilcoxon signed-rank p-value when all five paired differences have the same sign is 0.0625. Therefore, p = 0.0625 is not described as conventionally statistically significant.

---

## Training

### 1. Differential-Consistency

This is the main method.

```bash
python src/train_resnet_differential_consistency.py     --csv-path /path/to/manifest.csv     --data-root /path/to/dataset     --epochs 20     --batch-size 32     --img-size 192     --lr 1e-4     --weight-decay 1e-4     --consistency-weight 0.5     --log-dir logs/differential_consistency_seed42     --ckpt-path checkpoints/differential_consistency_seed42/best.pth     --pretrained     --seed 42
```

Repeat using:

```text
42, 52, 62, 72, 82
```

---

### 2. Brightness-Consistency Control

This control tests whether generic shared-intensity consistency is sufficient to obtain channel-differential robustness.

```bash
python src/train_resnet_brightness_consistency.py     --csv-path /path/to/manifest.csv     --data-root /path/to/dataset     --epochs 20     --batch-size 32     --img-size 192     --lr 1e-4     --weight-decay 1e-4     --consistency-weight 0.5     --log-dir logs/brightness_consistency_seed42     --ckpt-path checkpoints/brightness_consistency_seed42/best.pth     --pretrained     --seed 42
```

---

### 3. DirectAttenAug-Matched Control

This matched control uses response perturbation exposure without the paired-view consistency objective.

```bash
python src/train_resnet_direct_atten_aug_matched.py     --csv-path /path/to/manifest.csv     --data-root /path/to/dataset     --epochs 20     --batch-size 32     --img-size 192     --lr 1e-4     --weight-decay 1e-4     --log-dir logs/direct_atten_matched_seed42     --ckpt-path checkpoints/direct_atten_matched_seed42/best.pth     --pretrained     --seed 42
```

---

## Evaluation

### Directional sensitivity

```bash
python src/eval_directional_sensitivity.py     --csv-path /path/to/manifest.csv     --data-root /path/to/dataset     --manifest /path/to/checkpoint_manifest.csv     --eps-list 0.01,0.025,0.05     --out-dir results/directional_sensitivity
```

### Full 2-D response plane

```bash
python src/eval_response_plane_2d.py     --csv-path /path/to/manifest.csv     --data-root /path/to/dataset     --manifest /path/to/checkpoint_manifest.csv     --grid=-0.15,-0.10,-0.05,0,0.05,0.10,0.15     --out-dir results/response_plane_2d
```

### Synchronized and opposite-direction stress tests

```bash
python src/eval_sync_stress.py     --csv-path /path/to/manifest.csv     --data-root /path/to/dataset     --manifest /path/to/checkpoint_manifest.csv     --out-dir results/sync_stress
```

### Corrected worst-group evaluation

```bash
python src/recalc_wga_corrected.py     --csv-path /path/to/manifest_with_response_groups.csv     --data-root /path/to/dataset     --dataset-py datasets/copper_xray_dataset.py     --manifest /path/to/checkpoint_manifest.csv     --out-dir results/wga
```

Reported group-level metrics include:

- Worst-Group Accuracy
- Worst-Group Balanced Accuracy
- Worst-Group Macro-F1

### Empirical perturbation-scale analysis

```bash
python src/analyze_delta_empirical_range.py     --csv-path /path/to/manifest.csv     --data-root /path/to/dataset     --out-dir results/delta_range
```

The training-set sample-wise mean log-response standard deviation was:

```text
sigma = 0.12585
```

Therefore:

- `delta = 0.05` is approximately `0.40 sigma`
- `delta = 0.15` is approximately `1.19 sigma`
- `delta = 0.25` is approximately `1.99 sigma`

These values provide an empirical response-dispersion reference only.

### Measured-geometry response analysis

```bash
python src/analyze_geometry_response_60.py     --input /path/to/geometry60_measurements.csv     --out-dir results/geometry60
```

This analysis is independent of the main classifier evaluation.

---

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
│
├── datasets/
│   └── copper_xray_dataset.py
│
├── src/
│   ├── train_resnet_differential_consistency.py
│   ├── train_resnet_brightness_consistency.py
│   ├── train_resnet_direct_atten_aug_matched.py
│   ├── eval_directional_sensitivity.py
│   ├── eval_response_plane_2d.py
│   ├── eval_sync_stress.py
│   ├── summarize_directional_ablation.py
│   ├── recalc_wga_corrected.py
│   ├── analyze_delta_empirical_range.py
│   └── analyze_geometry_response_60.py
│
├── scripts/
│   ├── run_differential_ablation_5seed.sh
│   └── run_wga_corrected.sh
│
├── data/
│   └── example_manifest.csv
│
└── results/
    ├── main_ablation/
    ├── controls/
    ├── sync_stress/
    ├── wga/
    ├── geometry60/
    └── delta_range/
```

Only final and reproduction-relevant scripts are included.

Preliminary, superseded, debugging, foreground/background diagnostic, and checkpoint-management scripts are intentionally omitted.

---

## Interpretation Boundaries

The released code and results support the following interpretation:

- DE-XRT classifier robustness can be analyzed in a paired high-/low-energy response space.
- Common-mode and channel-differential response directions produce different classifier sensitivities.
- Generic common-mode consistency does not automatically solve channel-differential sensitivity.
- Differential-Consistency substantially reduces differential sensitivity while maintaining strong clean classification performance.
- Independent measured-geometry data support the existence of a geometry-associated differential high-/low-energy response component.

The following claims are not made:

- `delta` is not calibrated to physical thickness in millimeters.
- Mean grayscale is not treated as a direct physical thickness measurement.
- The controlled response perturbation is not presented as a literal local particle-thickness edit.
- The 60-particle geometry dataset is not used as cross-device classifier validation.
- The binary labels are not claimed to arise from a verified particle-level 0.1% copper cutoff.
- A five-seed exact two-sided Wilcoxon p-value of `0.0625` is not described as statistically significant.

---

## Results Files

The repository includes lightweight CSV and JSON summaries used to generate the reported tables and comparisons.

Large model checkpoints, raw industrial X-ray images, intermediate caches, and exploratory foreground/background experiments are not included.

The released results include:

- five-seed directional sensitivity;
- full 7 x 7 response-plane results;
- selected response-pair comparisons;
- synchronized and opposite-direction stress tests;
- corrected WGA and worst-group balanced metrics;
- empirical response-dispersion statistics;
- independent measured-geometry response statistics.

---

## Citation

If you find this repository useful, please cite the associated manuscript:

```bibtex
@article{Zhu2026DirectionalResponseConsistency,
  title   = {Directional Response Consistency Regularization for Robust Dual-Energy X-Ray Ore Sorting},
  author  = {Zhu, Zhi-yong and He, Jian-feng and Wang, Xue-yuan and Xia, Fei and Nie, Feng-jun and Wang, Wen and Zou, Yang-hui and Li, Wei-dong and Zhong, Guo-yun and Ye, Zhi-xiang and Diao, Fan},
  journal = {Submitted},
  year    = {2026}
}
```

The BibTeX entry will be updated after formal publication.

---

## Contact

For questions or issues, please contact the corresponding author:

**Jian-feng He**  
hjf_10@yeah.net

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
