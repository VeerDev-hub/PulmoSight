<div align="center">

# PulmoSight

**Deep learning that reads chest X-rays for signs of tuberculosis.**

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-EfficientNet--B0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/vision/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.46%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Jupyter](https://img.shields.io/badge/Notebooks-11%20executed-F37626?logo=jupyter&logoColor=white)](notebooks/)

[![Test accuracy](https://img.shields.io/badge/Test%20accuracy-86.7%25-2EA44F)](#results)
[![AUROC](https://img.shields.io/badge/AUROC-0.937-2EA44F)](#results)
[![Models](https://img.shields.io/badge/Models-3%20CNNs%20%2B%20ensemble-087F78)](#model-architectures)
[![Status](https://img.shields.io/badge/Status-research%20prototype-F59E0B)](#limitations)
[![Medical use](https://img.shields.io/badge/Medical%20use-not%20a%20diagnostic%20device-B91C1C)](#limitations)

</div>

PulmoSight classifies a frontal chest X-ray as **TB** or **Normal** and returns a calibrated probability.
It compares three convolutional neural networks (a basic CNN, a custom CNN built for this project, and an
ImageNet transfer-learning model) and combines them into an ensemble that beats each of them on held-out data.
A Streamlit app lets anyone test the models on their own X-ray.

> ⚠️ Research and portfolio project. **Not a medical device and not a diagnosis.**

**Trailer:** [`assets/trailer/PulmoSight_trailer.mp4`](assets/trailer/PulmoSight_trailer.mp4) (46 s, with sound). It also plays on the app's home page.

| Test an X-ray | Model comparison |
|---|---|
| ![Test page](assets/trailer/shots/test_sample3.png) | ![Comparison page](assets/trailer/shots/comparison.png) |

**Contents:** [Results](#results) · [System design](#system-design) · [The app](#the-app) · [Notebooks](#notebooks) ·
[Findings](#findings-worth-knowing) · [Limitations](#limitations) · [Project structure](#project-structure)

---

## Results

All models were trained on the same **13,445** X-rays and scored once on **1,681** test X-rays they never saw.
Calibration, thresholds and ensemble weights were fixed on the **1,659**-image validation set before the test set was opened.

| Rank | Model | Test accuracy [95% CI] | AUROC | Sensitivity | Specificity | Params |
|:-:|---|:-:|:-:|:-:|:-:|:-:|
| 🥇 | **Ensemble** (weighted vote of all three) | **86.7%** [85.0–88.4] | **0.937** | 80.6% | 92.6% | 8.5 M |
| 🥈 | **EfficientNet-B0**: ImageNet transfer learning | 85.4% [83.6–87.1] | 0.933 | 82.8% | 87.8% | 4.0 M |
| 🥉 | **PulmoNet**: custom residual + squeeze-excite CNN | 83.5% [81.7–85.4] | 0.906 | 74.2% | 92.5% | 4.1 M |
| 🥉 | **Simple CNN**: 5 plain conv → pool blocks | 83.5% [81.7–85.2] | 0.916 | 77.7% | 89.0% | 0.39 M |

**What the experiments show**

1. **A basic CNN gets surprisingly far.** The Simple CNN ties PulmoNet (McNemar p = 1.0, DeLong p = 0.08) with 10× fewer parameters.
2. **Transfer learning is the strongest single model.** EfficientNet-B0 beats both from-scratch networks significantly
   (DeLong p = 0.001 vs the Simple CNN and 5×10⁻⁷ vs PulmoNet; McNemar p = 0.035 for both), in the fewest epochs.
3. **Together they are better still.** The ensemble (EfficientNet 50%, Simple CNN 30%, PulmoNet 20%) beats EfficientNet-B0
   alone: +1.4 accuracy points (McNemar p = 0.007) and +0.005 AUROC (DeLong p = 0.03). It mainly cuts false alarms
   (64 vs 105) at the cost of slightly lower sensitivity.

---

## System design

PulmoSight has two halves that share nothing at runtime except a small set of **versioned artifacts**: an offline
research pipeline (Jupyter notebooks on a GPU) and an online app (Streamlit on a CPU).

### 1. System overview

```mermaid
flowchart LR
    subgraph OFF["Offline · research and training (notebooks, GPU)"]
        direction TB
        RAW[("Raw X-rays<br/>data/raw")] --> CACHE["scripts/prepare_cache.py<br/>grayscale 512×512"]
        CACHE --> SPLIT["01 · audit + frozen split<br/>train · validation · test"]
        SPLIT --> TRAIN["02–04 · train<br/>Simple CNN · PulmoNet · EfficientNet-B0"]
        TRAIN --> EVAL["05 · calibrate and set thresholds on validation<br/>score test once"]
        EVAL --> ENS["11 · choose ensemble on validation<br/>score test once"]
    end

    subgraph ART["Versioned artifacts (in git)"]
        direction TB
        CK[("models/*/best.pth<br/>3 checkpoints · 34 MB")]
        J1[("comparison_3models.json<br/>temperatures · thresholds · test metrics")]
        J2[("ensemble.json<br/>weights · threshold · test metrics")]
    end

    subgraph ON["Online · Streamlit app (CPU)"]
        direction TB
        UI["streamlit_app.py<br/>Introduction · Model comparison · Test an X-ray"] --> SC["pulmosight.inference.Screener<br/>loaded once, cached"]
        SC --> NETS["pulmosight.models<br/>3 CNNs in eval mode"]
    end

    TRAIN --> CK
    EVAL --> J1
    ENS --> J2
    CK --> SC
    J1 --> SC
    J2 --> SC
    J1 --> UI
    J2 --> UI
    USER(("User")) <--> UI
```

### 2. Evaluation protocol (no test-set leakage)

Every choice is made on validation. The test set is opened once, after the system is frozen.

```mermaid
flowchart LR
    D["16,785 chest X-rays<br/>duplicates removed"] --> TR["Train · 13,445<br/>learn the weights"]
    D --> VA["Validation · 1,659<br/>make every decision"]
    D --> TE["Test · 1,681<br/>scored once"]
    TR --> M["Trained networks"]
    VA --> C1["Best epoch<br/>early stopping on AUROC"]
    VA --> C2["Temperature scaling"]
    VA --> C3["Decision thresholds"]
    VA --> C4["Ensemble weights"]
    M --> F["Frozen system"]
    C1 --> F
    C2 --> F
    C3 --> F
    C4 --> F
    F --> TE
    TE --> R["Reported metrics<br/>95% bootstrap CIs · DeLong · McNemar"]
```

### 3. Inference request

```mermaid
sequenceDiagram
    actor U as User
    participant UI as Streamlit page
    participant S as Screener (cached)
    participant N as CNNs
    U->>UI: upload an X-ray and choose a model
    UI->>S: predict(image, model)
    S->>S: load grayscale, resize to 512×512, quality checks
    S->>N: forward pass, no gradients
    N-->>S: two logits per network
    S->>S: temperature scaling, weighted vote, threshold
    S-->>UI: verdict, P(TB), certainty, votes, warnings
    UI-->>U: result card, what it means, next steps
```

### 4. Decision pipeline (ensemble)

```mermaid
flowchart LR
    IN["Uploaded X-ray<br/>PNG · JPG · BMP · TIF"] --> PRE["Preprocess<br/>8-bit grayscale · bicubic 512×512<br/>normalise (x − 0.449) / 0.226"]
    PRE -.-> QW["Quality warnings<br/>resolution · colour · aspect · contrast"]
    PRE --> A["Simple CNN"]
    PRE --> B["PulmoNet"]
    PRE --> C["EfficientNet-B0"]
    A --> TA["sigmoid(Δlogit / 1.19)"]
    B --> TPN["sigmoid(Δlogit / 1.09)"]
    C --> TC["sigmoid(Δlogit / 1.23)"]
    TA -- "× 0.3" --> W["Weighted vote<br/>P(TB)"]
    TPN -- "× 0.2" --> W
    TC -- "× 0.5" --> W
    W --> TH{"P(TB) ≥ 0.535?"}
    TH -- yes --> POS["TB suspected"]
    TH -- no --> NEG["No TB detected"]
```

Each network's probability is temperature-calibrated before the vote, so the three are on the same scale. Choosing a
single model in the app skips the vote and uses that network's own calibrated probability and threshold.

### 5. Model architectures

```mermaid
flowchart TB
    IN["Chest X-ray · 1×512×512"]
    subgraph S["Simple CNN · 0.39 M"]
        direction TB
        s1["From scratch<br/>5 × Conv 3×3 → BN → ReLU → MaxPool<br/>16 → 32 → 64 → 128 → 256 channels"]
        s2["Features 256×16×16"]
        s3["Global avg pool → Dropout → Linear"]
        s1 --> s2 --> s3
    end
    subgraph P["PulmoNet · 4.09 M"]
        direction TB
        p1["From scratch<br/>Stem: Conv 5×5 s2 · Conv 3×3 · Conv 3×3 s2"]
        p2["4 stages of residual SE blocks<br/>depths 2·2·3·2 · widths 48·96·160·256"]
        p3["Conv 1×1 → 512×16×16"]
        p4["Global avg pool → Dropout → Linear"]
        p1 --> p2 --> p3 --> p4
    end
    subgraph E["EfficientNet-B0 · 4.01 M"]
        direction TB
        e1["ImageNet weights<br/>grayscale stem: RGB filters summed"]
        e2["16 MBConv blocks<br/>depthwise 3×3 / 5×5 + squeeze-excite"]
        e3["Conv 1×1 → 1280×16×16"]
        e4["Global avg pool → Dropout → Linear"]
        e1 --> e2 --> e3 --> e4
    end
    OUT["2 logits per network<br/>Normal · TB"]
    IN --> s1
    IN --> p1
    IN --> e1
    s3 --> OUT
    p4 --> OUT
    e4 --> OUT
```

Layer-by-layer tables, training recipes and the reasoning behind each design are in
[`docs/CNN_Explanation.md`](docs/CNN_Explanation.md).

### 6. Artifact contract

The app never trains or recomputes anything. It only reads these files, which the notebooks produce.

| Artifact | Produced by | Read by | Contents |
|---|---|---|---|
| `splits/splits.csv` | notebook 01 | notebooks 02–11 | frozen train / validation / test assignment of every image |
| `models/simple_cnn/best.pth` | notebook 02 | app, notebooks 05 and 11 | weights + preprocessing constants |
| `models/pulmonet/best.pth` | notebook 03 | app, notebooks 05, 06, 10, 11 | weights + preprocessing constants |
| `models/efficientnet_b0/best.pth` | notebook 04 | app, notebooks 05, 06, 10, 11 | weights + preprocessing constants |
| `results/comparison/comparison_3models.json` | notebook 05 | app | per-network temperature, thresholds, test metrics with CIs, ROC curve, confusion matrix, cost, significance tests |
| `results/comparison/ensemble.json` | notebook 11 | app | ensemble weights, thresholds, test metrics with CIs, ROC curve, comparison with EfficientNet-B0 |

### 7. Design decisions

| Decision | Why |
|---|---|
| **Notebooks for research, a small package for serving** | The notebooks keep the full experimental record with outputs. The app imports only `pulmosight/` (~300 lines), so serving stays simple and testable. |
| **Artifacts, not retraining, at deploy time** | The deployed app is reproducible and starts in seconds. Every number it shows comes from a file written by a notebook. |
| **Calibrate each network before voting** | Raw network probabilities are overconfident on different scales. Temperature scaling makes them comparable and meaningful ("80%" ≈ 80% of such X-rays had TB). |
| **All choices on validation, test scored once** | Keeps the reported 86.7% an honest estimate rather than a tuned one. |
| **Models loaded once with `st.cache_resource`** | The three networks (~34 MB) load at startup; each prediction is then a single forward pass, under a second on a CPU. |
| **CPU-only PyTorch for deployment** | Free hosting has no GPU, and the CPU build is a fraction of the size of the CUDA build. |
| **Uploads processed in memory** | No patient images are written to disk or kept between requests. |
| **No heatmaps in the app** | Grad-CAM revealed the models also use non-lung cues. Showing it as an explanation of TB would mislead, so it is documented as a limitation instead. |
| **Image-quality warnings, not hard rejections** | The app flags low resolution, colour photos, odd aspect ratios and low contrast, but still lets the user see the result. |

---

## The app

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py          # → http://localhost:8501
```

| Page | What it does |
|---|---|
| **Introduction** | What PulmoSight does, the trailer, how it works, the models, and what the system can and cannot do. |
| **Model comparison** | Accuracy and AUROC with 95% CIs, ROC curves, errors per model, cost, significance tests and conclusions. |
| **Test an X-ray** | Pick one model (the ensemble is recommended), then upload an X-ray or try a test-set sample. You get the verdict, a calibrated TB probability against the model's threshold, how certain the call is, how each network voted, what the model's sensitivity and specificity mean for this result, image-quality warnings and next steps. |

**CLI:** `python -m pulmosight.inference path/to/xray.png --model ensemble` (or `efficientnet_b0`, `pulmonet`, `simple_cnn`).

**New to CNNs?** [`docs/CNN_Explanation.md`](docs/CNN_Explanation.md) walks through all three networks layer by layer,
how they were trained, and how the ensemble decides.

### Deploy on Streamlit Community Cloud

1. Push this repository to GitHub. The `.gitignore` already keeps the datasets out and lets through the three model
   checkpoints (~34 MB) and the two result files the app needs.
2. On [share.streamlit.io](https://share.streamlit.io) create a new app, choose this repo and set the main file to `streamlit_app.py`.
3. Under **Advanced settings**, choose **Python 3.12**. `requirements.txt` installs the CPU build of PyTorch.

---

## Notebooks

Executed notebooks with outputs, in `notebooks/`. Run them in order.

| # | Notebook | Content |
|:-:|---|---|
| 01 | `01_Data_Exploration_and_Splits` | dataset audit, duplicate check, frozen train / validation / test split |
| 02 | `02_Simple_CNN_Baseline` | basic CNN trained from scratch (100-epoch budget) |
| 03 | `03_PulmoNet_Custom_CNN` | custom residual + squeeze-excite CNN trained from scratch |
| 04 | `04_EfficientNetB0_Transfer_Learning` | ImageNet EfficientNet-B0: head warm-up, then fine-tuning |
| 05 | `05_Model_Comparison` | all three on the test set: calibration, thresholds, CIs, significance, Grad-CAM, export for the app |
| 06 | `06_Diagnostics_Overfitting_Robustness` | overfitting curves, robustness to corruptions, shortcut (hidden-lungs) test |
| 07 | `07_Lung_Segmentation_and_Crop` | polarity audit, lung segmentation (TorchXRayVision), lung-cropped dataset |
| 08–10 | `08`–`10` v2 experiment | retraining on lung crops (PulmoNet, EfficientNet × 3 seeds) and its evaluation |
| 11 | `11_Ensemble` | ensembles chosen on validation, tested once against EfficientNet-B0, export for the app |

`scripts/run_all_notebooks.sh` re-executes 02–05 and `scripts/run_v2_notebooks.sh` re-executes 08–10.

### Reproducing from scratch

1. Place the dataset as `data/raw/train/{normal,tb}/` and `data/raw/val/{normal,tb}/`.
2. Build the 512×512 grayscale cache: `python scripts/prepare_cache.py --src data/raw --dst data/cache512`.
3. Run the notebooks in order. Notebook 01 pools the images and writes the frozen split to `splits/splits.csv`.
   Notebook 07 writes the lung-cropped images and masks to `data/lungcrop512/`, `data/lungmask512/` and `data/lungcrop_mask512/`.

Training used an RTX 4060 Laptop GPU (8 GB). Training the three networks took about 2.2 hours in total.

---

## Findings worth knowing

* **Image polarity.** About 93% of the dataset is stored inverted (lungs bright). The minority in standard polarity is mostly
  Normal, a shortcut the models could exploit. Converting uploads to one polarity *lowered* test accuracy, because the models
  learned both, so the app feeds images as uploaded.
* **Shortcut learning.** With the lungs masked out, the models still reach AUROC 0.75–0.86, so part of their signal comes from
  outside the lungs.
* **Robustness trade-off.** Retraining on polarity-standardised, lung-cropped images (notebooks 07–10) made the models far more
  robust (EfficientNet AUROC under moderate noise 0.50 → 0.83) but cost 2–4 accuracy points here. The app uses the more
  accurate original models.

## Limitations

* **One data distribution.** Accuracy on X-rays from other hospitals, scanners or phone photos is unknown and likely lower.
* **Image quality.** Strong noise, blur or very low resolution reduce accuracy.
* **Labels are image-level only.** There is no lesion localisation or severity, and probability is not severity.
* **Patient-level independence** of the splits cannot be verified from the available metadata.

A positive result calls for clinical assessment and confirmatory testing (sputum smear, culture, or a molecular test such as
Xpert MTB/RIF) under local TB guidance. A negative result does not rule out TB.

---

## Project structure

```
PulmoSight/
├── streamlit_app.py          Streamlit app: introduction, model comparison, X-ray testing
├── requirements.txt
├── .streamlit/config.toml    app theme
├── pulmosight/               Python package used by the app
│   ├── models.py             Simple CNN, PulmoNet and EfficientNet-B0 definitions
│   └── inference.py          Screener: preprocessing, calibrated prediction, ensemble (also a CLI)
├── notebooks/                01–11 executed notebooks (the whole training and evaluation story)
├── docs/
│   └── CNN_Explanation.md    how the three CNNs and the ensemble work
├── scripts/
│   ├── prepare_cache.py      builds the 512×512 grayscale image cache
│   ├── run_all_notebooks.sh  re-executes notebooks 02–05
│   ├── run_v2_notebooks.sh   re-executes notebooks 08–10
│   ├── make_soundtrack.py    synthesises the trailer soundtrack
│   └── render_trailer.py     renders assets/trailer/trailer.html to MP4
├── assets/
│   ├── samples/              six test-set X-rays offered as samples in the app
│   └── trailer/              trailer MP4, its HTML source, app screenshots, feature maps
├── models/                   trained checkpoints (only the three used by the app are in git)
├── results/comparison/       test-set results, figures, and the two JSON files the app reads
├── splits/                   frozen train / validation / test split (+ lung-crop metadata)
└── data/                     datasets, git-ignored: raw/{train,val}, cache512, lungcrop512, lung masks, external_samples
```

## Acknowledgements

* Chest X-ray dataset: *add the dataset name, source and licence here.*
* Lung segmentation: [TorchXRayVision](https://github.com/mlmed/torchxrayvision) (PSPNet trained on ChestX-Det).
* EfficientNet-B0 ImageNet weights: [torchvision](https://pytorch.org/vision/).
* TB statistics in the trailer: WHO Global Tuberculosis Report 2024.
