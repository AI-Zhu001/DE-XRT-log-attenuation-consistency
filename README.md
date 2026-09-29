# Response-Subspace Consistency Regularization for Robust DE-XRT Ore Sorting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the main training, evaluation, and analysis code associated with the manuscript:

**"Directional Response Consistency Regularization for Robust Dual-Energy X-Ray Ore Sorting"**

**Zhi-yong Zhu, Jian-feng He, Xue-yuan Wang, Fei Xia, Feng-jun Nie, Wen Wang, Yang-hui Zou, Wei-dong Li, Guo-yun Zhong, Zhi-xiang Ye, Fan Diao**

> The manuscript is currently under preparation/submission. The title and bibliographic information may be updated after publication.

---

## Overview

This work studies robustness in dual-energy X-ray transmission (DE-XRT) ore classification from the perspective of **paired high-/low-energy response sensitivity**.

Instead of treating the perturbation as a calibrated physical-thickness transformation, we formulate controlled changes directly in the paired log-response space:
We formulate controlled changes directly in the paired log-response space:

$$
\mathbf{u}
=
\begin{bmatrix}
\log I_H \\
\log I_L
\end{bmatrix}.
$$

A two-channel response perturbation can be decomposed into:

$$
\Delta
=
\delta_c
\begin{bmatrix}
1 \\
1
\end{bmatrix}
+
\delta_d
\begin{bmatrix}
1 \\
-1
\end{bmatrix}.
$$

Here,

- $v_{\mathrm{com}}=(1,1)$ denotes the **common-mode response direction**;
- $v_{\mathrm{diff}}=(1,-1)$ denotes the **channel-differential response direction**.

The main method, **Differential-Consistency**, applies paired-view consistency regularization specifically along the differential response direction:

$$
\delta_H = s a,
\qquad
\delta_L = -s a,
$$

where

$$
a \sim U(0,0.15),
\qquad
s \in \{-1,+1\}.
$$

The training objective is:

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{CE,clean}}
+
\mathcal{L}_{\mathrm{CE,brightness}}
+
\lambda
\left\|
z_{\mathrm{diff}}
-
\operatorname{sg}(z_{\mathrm{clean}})
\right\|_2^2,
$$

where $\operatorname{sg}(\cdot)$ denotes the stop-gradient operation and $\lambda=0.5$.

The perturbation is used as a **controlled channel-level DE-XRT response shift**. It is not interpreted as a calibrated millimeter-scale thickness transformation.

---

## Main Findings

Using five matched random seeds

```text
42, 52, 62, 72, 82
```

the final Differential-Consistency model achieved:

| Evaluation | Macro-F1 |
|---|---:|
| Clean test set | **0.9136 ± 0.0031** |
| Common shift \((-0.15,-0.15)\) | **0.9020 ± 0.0063** |
| Common shift \((+0.15,+0.15)\) | **0.8924 ± 0.0135** |
| Differential shift \((+0.15,-0.15)\) | **0.8341 ± 0.0424** |
| Differential shift \((-0.15,+0.15)\) | **0.8369 ± 0.0350** |

Across the full controlled \(7\times7\) response plane,

$$
\delta_H,\delta_L
\in
\{-0.15,-0.10,-0.05,0,0.05,0.10,0.15\},
$$

Differential-Consistency obtained:

- mean Macro-F1: **0.8800 ± 0.0061**
- worst-point Macro-F1: **0.8128 ± 0.0328**
- mean prediction-flip rate: **0.0758 ± 0.0096**

At local perturbation magnitude \(\epsilon=0.025\), the differential logit-margin sensitivity was:

- ordinary synchronized Consistency: **176.70 ± 16.33**
- CI-Consistency: **13.08 ± 7.83**
- Differential-Consistency: **19.97 ± 3.96**

Thus, targeted differential consistency reduced differential sensitivity by approximately **88.7%** relative to ordinary synchronized Consistency.

### Corrected worst-group evaluation

Worst-group metrics were recomputed using corrected clean-response grouping.

For Differential-Consistency, corrected Worst-Group Accuracy (WGA) included:

| Condition | WGA |
|---|---:|
| Clean | **0.9071** |
| \((-0.25,-0.25)\) | **0.8635** |
| \((+0.25,+0.25)\) | **0.8592** |
| \((+0.15,-0.15)\) | **0.7956** |
| \((-0.15,+0.15)\) | **0.8301** |
| \((+0.25,-0.25)\) | **0.5599** |
| \((-0.25,+0.25)\) | **0.7385** |

Worst-group balanced accuracy and worst-group Macro-F1 are also included in the released result tables.

---

## Independent Measured-Geometry Evidence

An additional set of **60 newly collected physical particles** was used to examine whether measured particle geometry is associated with both common and differential DE-XRT response variation.

The set contains:

- 30 copper particles
- 30 waste particles

Each particle was measured repeatedly for maximum and center height.

Important limitation:

> These 60 particles were acquired on a different DE-XRT system from the main 7,245-particle industrial dataset. They are therefore used only as independent **geometry-response evidence**, not as cross-device classifier validation.

For the differential response

$$
d=
\log(\bar I_H)-\log(\bar I_L),
$$

the observed slope with respect to the measured height proxy was

$$
\beta_d = 0.03515/\mathrm{mm},
$$

with bootstrap 95% CI

$$
[0.01643,\,0.05600].
$$

After adjustment for copper/waste class,

$$
\beta_h=0.02557/\mathrm{mm},
$$

with

$$
p=1.17\times10^{-12}.
$$

After further adjustment for projected particle area,

$$
\beta_h=0.02539/\mathrm{mm},
$$

with

$$
p=1.64\times10^{-8}.
$$

These measurements support the existence of a geometry-associated **channel-differential response component**. They do **not** establish a calibration between the synthetic perturbation parameter \(\delta\) and physical thickness in millimeters.

---

## Dataset

The main industrial dataset contains **7,245 paired DE-XRT particle samples**:

- 2,806 copper ore particles
- 4,439 waste particles

The split used in the experiments is:

- training: 5,071
- validation: 1,087
- test: 1,087

The main DE-XRT system was operated at:

- tube voltage: 160 kV
- tube current: 550 μA

The samples were collected onsite at Dexing Copper Mine.

### Data availability

The complete industrial dataset cannot be publicly released because of proprietary restrictions.

This repository therefore provides code and an example manifest format rather than the complete dataset.

The binary labels used in the study were operational labels provided with the industrial dataset. Particle-wise assay/XRF measurements were not available.

No unverified particle-level Cu cutoff should be inferred from these labels.

---

## Data Format

Each sample consists of paired high- and low-energy images.

A recommended manifest format is:

```csv
sample_id,label,high_path,low_path,split,response_group
sample_0001,1,path/to/high/sample_0001.png,path/to/low/sample_0001.png,train,medium_indicator
sample_0002,0,path/to/high/sample_0002.png,path/to/low/sample_0002.png,test,low_indicator
```

where

- `label`: binary class label;
- `high_path`: path to the high-energy image;
- `low_path`: path to the low-energy image;
- `split`: `train`, `val`, or `test`;
- `response_group`: optional clean-response grouping used for worst-group analysis.

For the ResNet18 experiments, paired images are resized to \(192\times192\).

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

Install dependencies with:

```bash
pip install -r requirements.txt
```

Clone the repository:

```bash
git clone https://github.com/AI-Zhu001/DE-XRT-thickness-regularization.git
cd DE-XRT-thickness-regularization
```

> The repository name may be renamed to reflect the final response-subspace formulation.

---

## Reproducibility

All primary matched experiments use exactly five random seeds:

```text
42 52 62 72 82
```

Do not replace these with arbitrary seeds when reproducing the reported means and standard deviations.

A statistical note for the five-seed paired comparisons:

> With \(n=5\), the smallest possible exact two-sided Wilcoxon signed-rank p-value when all paired differences have the same sign is 0.0625. Therefore \(p=0.0625\) is not described as conventionally statistically significant.

---

## Training

### 1. Differential-Consistency — main method

```bash
python src/train_resnet_differential_consistency.py \
    --csv-path /path/to/manifest.csv \
    --data-root /path/to/dataset \
    --epochs 20 \
    --batch-size 32 \
    --img-size 192 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --brightness-delta-min -0.15 \
    --brightness-delta-max 0.15 \
    --brightness-apply-p 0.5 \
    --diff-delta-max 0.15 \
    --consistency-weight 0.5 \
    --log-dir logs/differential_consistency_seed42 \
    --ckpt-path checkpoints/differential_consistency_seed42/best.pth \
    --pretrained \
    --seed 42
```

Repeat the run using:

```text
42, 52, 62, 72, 82
```

or use the provided five-seed batch script.

---

### 2. Brightness-Consistency control

This control tests whether generic shared-intensity consistency is sufficient to obtain differential robustness.

```bash
python src/train_resnet_brightness_consistency.py \
    --csv-path /path/to/manifest.csv \
    --data-root /path/to/dataset \
    --epochs 20 \
    --batch-size 32 \
    --img-size 192 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --consistency-weight 0.5 \
    --log-dir logs/brightness_consistency_seed42 \
    --ckpt-path checkpoints/brightness_consistency_seed42/best.pth \
    --pretrained \
    --seed 42
```

This control achieves strong common-mode robustness but retains substantially larger channel-differential sensitivity.

---

### 3. DirectAttenAug-Matched control

This matched control uses the same response perturbation exposure without the paired-view consistency objective.

```bash
python src/train_resnet_direct_atten_aug_matched.py \
    --csv-path /path/to/manifest.csv \
    --data-root /path/to/dataset \
    --epochs 20 \
    --batch-size 32 \
    --img-size 192 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --log-dir logs/direct_atten_matched_seed42 \
    --ckpt-path checkpoints/direct_atten_matched_seed42/best.pth \
    --pretrained \
    --seed 42
```

This experiment is used to distinguish the effect of perturbation exposure from the effect of consistency regularization.

---

## Evaluation

### 1. Directional sensitivity

Directional sensitivity is evaluated from the logit margin

$$
m=z_1-z_0
$$

using

$$
S_m=
\frac{|m(x^+)-m(x^-)|}{2\epsilon}.
$$

The primary analysis uses

$$
\epsilon=0.025.
$$

Run:

```bash
python src/eval_directional_sensitivity.py \
    --csv-path /path/to/manifest.csv \
    --data-root /path/to/dataset \
    --manifest /path/to/checkpoint_manifest.csv \
    --eps-list 0.01,0.025,0.05 \
    --out-dir results/directional_sensitivity
```

---

### 2. Full 2-D response-plane evaluation

Evaluate the paired high-/low-energy response space:

```bash
python src/eval_response_plane_2d.py \
    --csv-path /path/to/manifest.csv \
    --data-root /path/to/dataset \
    --manifest /path/to/checkpoint_manifest.csv \
    --grid=-0.15,-0.10,-0.05,0,0.05,0.10,0.15 \
    --out-dir results/response_plane_2d
```

This evaluates all

$$
7\times7=49
$$

pairs of \((\delta_H,\delta_L)\).

---

### 3. Synchronized and opposite-direction stress tests

```bash
python src/eval_sync_stress.py \
    --csv-path /path/to/manifest.csv \
    --data-root /path/to/dataset \
    --manifest /path/to/checkpoint_manifest.csv \
    --out-dir results/sync_stress
```

The released results include both synchronized response shifts and opposite-channel stress conditions.

---

### 4. Corrected worst-group evaluation

```bash
python src/recalc_wga_corrected.py \
    --csv-path /path/to/manifest_with_response_groups.csv \
    --data-root /path/to/dataset \
    --dataset-py datasets/copper_xray_dataset.py \
    --manifest /path/to/checkpoint_manifest.csv \
    --out-dir results/wga
```

Reported group-level metrics include:

- Worst-Group Accuracy (WGA)
- Worst-Group Balanced Accuracy
- Worst-Group Macro-F1

The corrected test-set grouping contains both classes in every group.

---

### 5. Empirical perturbation-scale analysis

```bash
python src/analyze_delta_empirical_range.py \
    --csv-path /path/to/manifest.csv \
    --data-root /path/to/dataset \
    --out-dir results/delta_range
```

The training-set sample-wise mean log-response standard deviation was:

$$
\sigma = 0.12585.
$$

Therefore:

- \(\delta=0.05\approx0.40\sigma\)
- \(\delta=0.15\approx1.19\sigma\)
- \(\delta=0.25\approx1.99\sigma\)

These values provide an empirical response-dispersion reference only; they are not physical thickness calibration.

---

### 6. Measured-geometry response analysis

```bash
python src/analyze_geometry_response_60.py \
    --input /path/to/geometry60_measurements.csv \
    --out-dir results/geometry60
```

This analysis is independent of the main classifier evaluation and is used only to assess measured geometry-response associations.

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

Only final and reproduction-relevant scripts are included in the public repository. Preliminary, superseded, debugging, foreground/background diagnostic, and checkpoint-management scripts are intentionally omitted.

---

## Interpretation Boundaries

The released code and manuscript support the following interpretation:

- DE-XRT classifier robustness can be analyzed in a paired high-/low-energy response space.
- Common-mode and channel-differential response directions produce different classifier sensitivities.
- Generic/common-mode consistency does not automatically solve channel-differential sensitivity.
- Targeted Differential-Consistency substantially reduces differential sensitivity while maintaining strong clean classification performance.
- Independent measured-geometry data support the existence of a geometry-associated differential H/L response component.

The following claims are **not** made:

- \(\delta\) is not calibrated to a physical thickness in millimeters.
- Mean grayscale is not treated as a direct physical thickness measurement.
- The controlled response perturbation is not presented as a literal local particle-thickness edit.
- The 60-particle geometry dataset is not used as cross-device classifier validation.
- The binary labels are not claimed to arise from a verified particle-level 0.1% Cu cutoff.
- Five-seed Wilcoxon \(p=0.0625\) is not described as statistically significant.

---

## Results Files

The repository includes lightweight CSV/JSON summaries used to generate the reported tables and comparisons.

Large model checkpoints, raw industrial X-ray images, intermediate caches, and exploratory foreground/background experiments are not included.

The released result tables include:

- five-seed directional sensitivity;
- full \(7\times7\) response-plane results;
- selected response-pair comparisons;
- synchronized and opposite-direction stress tests;
- corrected WGA/WGBAcc/Worst-group Macro-F1;
- empirical response-dispersion statistics;
- independent measured-geometry response statistics.

---

## Citation

If you find this repository useful, please cite the associated manuscript.

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
