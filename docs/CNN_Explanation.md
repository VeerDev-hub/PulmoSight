# How PulmoSight's CNNs Work

This document explains the three convolutional neural networks (CNNs) in PulmoSight, layer by layer, how they were
trained, and how their outputs are combined into a final TB / Normal decision. All shapes and parameter counts are taken
from the trained models in this repository (`pulmosight/models.py`).

**Contents**
1. [CNNs in one page](#1-cnns-in-one-page)
2. [What the networks receive](#2-what-the-networks-receive)
3. [Model 1: Simple CNN (the baseline)](#3-model-1-simple-cnn-the-baseline)
4. [Model 2: PulmoNet (our custom CNN)](#4-model-2-pulmonet-our-custom-cnn)
5. [Model 3: EfficientNet-B0 (transfer learning)](#5-model-3-efficientnet-b0-transfer-learning)
6. [How the networks were trained](#6-how-the-networks-were-trained)
7. [From network output to decision](#7-from-network-output-to-decision)
8. [The ensemble](#8-the-ensemble)
9. [Results and what they teach](#9-results-and-what-they-teach)
10. [Why the app shows no heatmaps](#10-why-the-app-shows-no-heatmaps)
11. [Glossary](#11-glossary)

---

## 1. CNNs in one page

A CNN turns an image into a prediction by passing it through a stack of layers. Each layer turns its input into a
set of **feature maps**: grids of numbers that say "how strongly does this pattern appear here?".

| Building block | What it does | Why it matters for X-rays |
|---|---|---|
| **Convolution** | Slides small learned filters (3×3, 5×5) over the image and computes a weighted sum at every position. One filter → one feature map. | Early filters learn edges and gradients (rib borders, the heart outline). Later filters combine them into textures and shapes (opacities, nodules, cavities). |
| **Activation** (ReLU / SiLU) | Applies a non-linearity: keeps strong positive responses and suppresses the rest. | Without it, stacking layers would still be a single linear filter. |
| **Batch normalisation** | Re-centres and re-scales each feature map during training. | Makes deep networks train quickly and stably from random weights. |
| **Downsampling** (max-pool or stride 2) | Halves the width and height of the feature maps. | Each later layer sees a larger area of the X-ray (its **receptive field**) at lower cost. |
| **Global average pooling** | Averages each final feature map to a single number. | Gives a fixed-length summary of the whole image, whatever was detected where. |
| **Linear classifier** | Maps that summary to two scores (**logits**): Normal and TB. | The final decision layer. |

As data flows through the network, **resolution falls and abstraction rises**. PulmoSight's 512×512 X-ray becomes
16×16 feature maps that no longer look like an X-ray but encode which patterns are present.

---

## 2. What the networks receive

Every model sees exactly the same input:

1. Load the image as **8-bit grayscale**. 16-bit X-rays are min–max scaled to 0–255, never clipped.
2. Resize to **512 × 512** (bicubic), the same code that built the training cache.
3. Normalise: `(pixel / 255 − 0.449) / 0.226`.

Input tensor: **1 × 512 × 512**. One channel, because X-rays have no colour.

---

## 3. Model 1: Simple CNN (the baseline)

*The kind of network taught in an introductory deep-learning course.* It answers the question: how far does a plain CNN get?

Five identical blocks of **Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2**, doubling the channels each time.

| Block | Layers | Output shape | Parameters | Receptive field |
|---|---|---|---:|---:|
| 1 | Conv 3×3 (1→16) · BN · ReLU · MaxPool | 16 × 256 × 256 | 176 | 4 px |
| 2 | Conv 3×3 (16→32) · BN · ReLU · MaxPool | 32 × 128 × 128 | 4,672 | 10 px |
| 3 | Conv 3×3 (32→64) · BN · ReLU · MaxPool | 64 × 64 × 64 | 18,560 | 22 px |
| 4 | Conv 3×3 (64→128) · BN · ReLU · MaxPool | 128 × 32 × 32 | 73,984 | 46 px |
| 5 | Conv 3×3 (128→256) · BN · ReLU · MaxPool | 256 × 16 × 16 | 295,424 | 94 px |
| Head | Global average pool → Dropout 0.3 → Linear (256→2) | 2 logits | 514 | — |
| **Total** | | | **393,330** | |

**Design notes**
* **BatchNorm** is the one concession to modern practice; without it a from-scratch CNN of this depth trains very slowly.
* **Global average pooling instead of Flatten → Dense.** Flattening 16×16×256 features would need a 65,536-input dense
  layer (~8 M weights in one layer) that overfits immediately.
* **Limitation by design:** each final feature only "sees" a 94×94-pixel patch (18% of the image width), and there is a
  single convolution per stage. There are no skip connections, no attention and no pre-training.

Cost: 1.25 GFLOPs per image, about 8 ms on a CPU.

---

## 4. Model 2: PulmoNet (our custom CNN)

*A network designed for this project and trained from random weights.* It uses three ideas from modern architectures:

* **Residual connections:** each block learns a correction that is *added* to its input (`out = act(body(x) + skip(x))`).
  Gradients flow straight through the skip path, so deeper networks train reliably.
* **Squeeze-and-excitation (SE) attention:** each block averages every feature map to one number, passes those numbers
  through a tiny two-layer network, and uses the result (0–1) to scale the maps. The network learns *which* features matter
  for *this* image.
* **Stochastic depth:** during training, residual branches are randomly skipped, with the probability rising to 0.1 in the
  deepest block. This is a regulariser: the network cannot rely on any single block.

Activations are **SiLU** (`x · sigmoid(x)`), a smooth version of ReLU.

| Part | Structure | Output shape | Parameters |
|---|---|---|---:|
| Stem | Conv 5×5 s2 (1→32) · Conv 3×3 (32→32) · Conv 3×3 s2 (32→48), each + BN + SiLU | 48 × 128 × 128 | 24,064 |
| Stage 1 | 2 residual SE blocks | 48 × 128 × 128 | 84,588 |
| Stage 2 | 2 residual SE blocks (first with stride 2) | 96 × 64 × 64 | 300,696 |
| Stage 3 | 3 residual SE blocks (first with stride 2) | 160 × 32 × 32 | 1,327,580 |
| Stage 4 | 2 residual SE blocks (first with stride 2) | 256 × 16 × 16 | 2,214,976 |
| Head conv | Conv 1×1 (256→512) + BN + SiLU | 512 × 16 × 16 | 132,096 |
| Classifier | Global average pool → Dropout 0.3 → Linear (512→2) | 2 logits | 1,026 |
| **Total** | | | **4,085,026** |

**One residual SE block:**

```
x ─┬─▶ Conv 3×3 (stride s) ─▶ BN ─▶ SiLU ─▶ Conv 3×3 ─▶ BN ─▶ SE attention ─▶ (stochastic depth) ─┐
   │                                                                                             (+) ─▶ SiLU ─▶ out
   └─▶ skip: identity, or AvgPool(s) + Conv 1×1 + BN when the shape changes ──────────────────────┘
```

The aggressive stem (4× downsampling) keeps compute manageable at 512×512 input. Cost: 5.38 GFLOPs per image, about 29 ms on a CPU.

---

## 5. Model 3: EfficientNet-B0 (transfer learning)

*A published architecture whose weights were first trained on ImageNet (1.2 million everyday photos), then fine-tuned
on chest X-rays.* The network starts with edge, texture and shape detectors that already work, instead of learning
them from scratch.

EfficientNet-B0 is built from **MBConv** blocks (mobile inverted bottlenecks):

```
x ─┬─▶ Conv 1×1 expand (×6 channels) ─▶ depthwise Conv k×k ─▶ SE attention ─▶ Conv 1×1 project ─┐
   └──────────────────────────────── skip (when shapes match) ─────────────────────────────────(+)─▶ out
```

A **depthwise** convolution filters each channel separately, which is much cheaper than a normal convolution. The 1×1
convolutions mix channels. That is how EfficientNet-B0 gets strong accuracy from only 4 M parameters and 2 GFLOPs.

| Stage | Block (kernel) × repeats | Output shape | Parameters |
|---|---|---|---:|
| Stem | Conv 3×3 s2 (**1**→32) + BN + SiLU | 32 × 256 × 256 | 352 |
| 1 | MBConv1 (3×3) × 1 | 16 × 256 × 256 | 1,448 |
| 2 | MBConv6 (3×3) × 2 | 24 × 128 × 128 | 16,714 |
| 3 | MBConv6 (5×5) × 2 | 40 × 64 × 64 | 46,640 |
| 4 | MBConv6 (3×3) × 3 | 80 × 32 × 32 | 242,930 |
| 5 | MBConv6 (5×5) × 3 | 112 × 32 × 32 | 543,148 |
| 6 | MBConv6 (5×5) × 4 | 192 × 16 × 16 | 2,026,348 |
| 7 | MBConv6 (3×3) × 1 | 320 × 16 × 16 | 717,232 |
| Head | Conv 1×1 (320→1280) + BN + SiLU | 1280 × 16 × 16 | 412,160 |
| Classifier | Global average pool → Dropout 0.3 → Linear (1280→2) | 2 logits | 2,562 |
| **Total** | | | **4,009,534** |

**Adapting an RGB network to grayscale X-rays.** ImageNet models expect 3 colour channels. The pretrained stem filters
have shape 32 × **3** × 3 × 3. PulmoSight replaces the stem with a 1-channel convolution and initialises it by **summing
the three RGB filters**:

```python
gray = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
gray.weight.copy_(stem.weight.sum(dim=1, keepdim=True))     # 32×3×3×3 → 32×1×3×3
```

A grayscale image is what an RGB image looks like when R = G = B. Summing the colour filters gives exactly the response
the pretrained filter would have had on that image, so all the learned edge and texture detectors survive the conversion.
The 1000-class ImageNet head is replaced by a 2-class head (Normal, TB).

Cost: 1.97 GFLOPs per image, about 37 ms on a CPU (depthwise convolutions are cheap in FLOPs but less CPU-friendly).

---

## 6. How the networks were trained

Same data, same augmentation and the same model-selection rule for all three, so the comparison is fair.

| | Simple CNN | PulmoNet | EfficientNet-B0 |
|---|---|---|---|
| Initial weights | random | random | ImageNet |
| Optimiser | AdamW, weight decay 0.05 | AdamW, weight decay 0.05 | AdamW, weight decay 0.01 |
| Learning rate | One-Cycle, peak 2e-3 | One-Cycle, peak 2e-3 | stage 1: head only, 1e-3 (2 epochs); stage 2: everything, backbone 3e-4 |
| Batch size | 32 | 32 | 16 |
| Epoch budget / early-stopping patience | 100 / 15 | 40 / 8 | 2 + 20 / 6 |
| Best epoch (epochs run) | 63 (78) | 38 (40) | 9 (15) |
| Training time | 53 min | 48 min | 29 min |

**Shared settings**
* **Loss:** cross-entropy with label smoothing 0.05, so targets are 0.975 / 0.025 instead of 1 / 0. This discourages overconfidence.
* **Augmentation (on the GPU):** random rotation ±7°, shift ±4%, zoom 0.93–1.07, brightness and contrast ±15%.
  Every epoch sees slightly different versions of each X-ray.
* **Mixed precision:** bfloat16 autocast with channels-last memory, plus gradient clipping at 2.0.
* **Model selection:** keep the epoch with the best **validation AUROC**. The test set is never used for any choice.
* **Hardware:** RTX 4060 Laptop GPU (8 GB).

**Why EfficientNet is trained in two stages.** A freshly added random head produces large, noisy gradients. If the whole
network trained from step one, those gradients would scramble the pretrained features. Warming up the head first, then
fine-tuning everything with a lower learning rate for the backbone, keeps the ImageNet knowledge intact.

---

## 7. From network output to decision

1. **Logits → probability.** With two classes, softmax reduces to `P(TB) = sigmoid(logit_TB − logit_Normal)`.
2. **Temperature scaling.** Raw network probabilities are usually overconfident. Each network gets one number *T*,
   fitted on the validation set, and uses `P(TB) = sigmoid((logit_TB − logit_Normal) / T)`. This never changes which
   image ranks above which (AUROC is unchanged), but it makes "80%" mean roughly "80% of such X-rays had TB".
3. **Threshold.** The *balanced* threshold maximises sensitivity + specificity on validation. P(TB) at or above it means **TB**.
   A stricter *high-specificity* threshold (≥ 95% validation specificity) is reported alongside it.

Temperatures and thresholds were fixed on validation *before* the test set was opened.

---

## 8. The ensemble

The ensemble averages the three calibrated probabilities with weights chosen on validation:

```
P(TB) = 0.5 · P_EfficientNet + 0.3 · P_SimpleCNN + 0.2 · P_PulmoNet        → TB if P(TB) ≥ 0.535
```

**Why it helps:** the networks make *different* mistakes. Their architectures and starting points differ, so an image that
confuses one is often clear to another, and averaging cancels part of the error. The Simple CNN earns a bigger weight than
PulmoNet even though they score the same alone, most likely because its mistakes complement EfficientNet's better.

**How it was chosen without cheating:** all pairs, the equal-weight triple and a grid of weighted triples were scored on
**validation**. The best one was picked there, along with its threshold, and only then scored once on test (notebook 11).

---

## 9. Results and what they teach

Test set: 1,681 X-rays that no model saw during training or any decision.

| Model | Test accuracy | AUROC | Sensitivity | Specificity | Params |
|---|:-:|:-:|:-:|:-:|:-:|
| **Ensemble** | **86.7%** | **0.937** | 80.6% | 92.6% | 8.5 M |
| EfficientNet-B0 | 85.4% | 0.933 | 82.8% | 87.8% | 4.0 M |
| PulmoNet | 83.5% | 0.906 | 74.2% | 92.5% | 4.1 M |
| Simple CNN | 83.5% | 0.916 | 77.7% | 89.0% | 0.39 M |

1. **A basic CNN is a strong baseline.** It ties PulmoNet (McNemar p = 1.0) with 10× fewer parameters. Residual blocks and
   attention did not, on their own, buy a measurable gain on this dataset.
2. **Pre-training is the biggest single lever.** EfficientNet-B0 beats both from-scratch networks significantly
   (DeLong p = 0.001 and 5×10⁻⁷) and converged in 9 epochs instead of 38–63.
3. **Diversity beats size.** The ensemble beats the best single model (McNemar p = 0.007), mostly by cutting false alarms
   (64 vs 105), at a small cost in sensitivity.

---

## 10. Why the app shows no heatmaps

**Grad-CAM** colours the regions whose features pushed the TB score up. It was computed for all models (notebooks 05 and
06). It often highlighted **bones, film borders and the burned-in "R" side marker**, not just the lungs. A follow-up
test that masked the lungs out still let the models reach AUROC 0.75–0.86.

So the heatmaps were telling the truth: part of what the networks learned comes from outside the lungs (**shortcut
learning**, often linked to differences between the hospitals or scanners in a dataset). Presenting those maps as
"where the TB is" would mislead users, so the app does not show them. The limitation is documented instead, and
notebooks 07–10 explore a fix (training on lung-cropped images): far more robust, at a cost of 2–4 accuracy points.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **AUROC** | Area under the ROC curve. The probability that a random TB X-ray scores higher than a random Normal one. 0.5 = chance, 1.0 = perfect. Independent of the threshold. |
| **Sensitivity** | Share of TB X-rays the model flags (recall). Low sensitivity = missed TB. |
| **Specificity** | Share of Normal X-rays the model correctly clears. Low specificity = false alarms. |
| **Feature map** | The output grid of one filter: where, and how strongly, its pattern occurs. |
| **Receptive field** | The area of the input image that one value in a feature map depends on. |
| **Logit** | The raw score a network outputs before it is turned into a probability. |
| **Calibration** | How well predicted probabilities match observed frequencies. |
| **Transfer learning** | Starting from weights learned on another task (here ImageNet) instead of random weights. |
| **DeLong / McNemar tests** | Paired significance tests: DeLong compares two AUROCs; McNemar compares which images two models get wrong. |
| **Shortcut learning** | When a model relies on cues that happen to correlate with the label (markers, borders, scanner type) rather than the disease itself. |
