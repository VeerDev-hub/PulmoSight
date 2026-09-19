"""Validation evaluation and presentation report for the TB classifier.

This script evaluates existing checkpoints only. It does not retrain and it does
not create an independent test set from the validation data.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from Inference import build_model, load_gray

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_ROOT = Path(__file__).resolve().parent
MEAN, STD = 0.449, 0.226
sns.set_theme(style="whitegrid", context="talk")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved TB checkpoints on validation data.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", choices=("val", "test"), default="val",
                        help="Dataset split to evaluate; test requires an existing independent split.")
    parser.add_argument("--models-path", type=Path, default=DEFAULT_ROOT / "models")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_ROOT / "results")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Evaluate one checkpoint, e.g. tb_best_model.pth.")
    parser.add_argument("--locked-threshold", type=float, default=None,
                        help="Required for --split test; threshold selected on validation data only.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--min-specificity", type=float, default=0.95,
                        help="Validation specificity target for the low-FP threshold.")
    return parser.parse_args()


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    return model, checkpoint


def collect_predictions(model, data_loader, positive_index):
    probabilities, labels, paths = [], [], []
    with torch.inference_mode():
        for images, batch_labels in data_loader:
            images = images.to(DEVICE, non_blocking=True)
            logits = model(images).float()
            probabilities.append(logits.softmax(dim=1)[:, positive_index].cpu().numpy())
            labels.append((batch_labels.numpy() == positive_index).astype(np.int64))
    probabilities = np.concatenate(probabilities)
    labels = np.concatenate(labels)
    paths = np.array([path for path, _ in data_loader.dataset.samples])
    return probabilities, labels, paths


def metric_row(labels, probabilities, threshold):
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "npv": float(tn / max(tn + fn, 1)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def threshold_table(labels, probabilities):
    thresholds = np.round(np.arange(0.10, 0.951, 0.01), 2)
    return pd.DataFrame([metric_row(labels, probabilities, threshold) for threshold in thresholds])


def select_threshold(table, min_specificity):
    eligible = table[table["specificity"] >= min_specificity]
    if eligible.empty:
        selected = table.sort_values(
            ["specificity", "sensitivity", "f1"], ascending=False
        ).iloc[0].to_dict()
        selected["specificity_target_met"] = False
        selected["selection_note"] = "No evaluated threshold met the specificity target."
        return selected
    selected = eligible.sort_values(
        ["sensitivity", "f1", "specificity"], ascending=False
    ).iloc[0].to_dict()
    selected["specificity_target_met"] = True
    selected["selection_note"] = "Highest sensitivity among thresholds meeting the specificity target."
    return selected


def save_confusion_matrix(row, output_path, title):
    matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=int)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Normal", "TB"], yticklabels=["Normal", "TB"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_curves(labels, probabilities, table, output_dir, title_prefix):
    fpr, tpr, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    auroc = roc_auc_score(labels, probabilities)
    auprc = average_precision_score(labels, probabilities)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}", linewidth=2.5)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set(xlabel="False-positive rate", ylabel="Sensitivity", title=f"{title_prefix}: ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall, precision, label=f"AUPRC = {auprc:.3f}", linewidth=2.5)
    ax.set(xlabel="Sensitivity / Recall", ylabel="Precision", title=f"{title_prefix}: Precision-recall curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_dir / "pr_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for name in ["sensitivity", "specificity", "precision", "f1", "accuracy"]:
        ax.plot(table["threshold"], table[name], label=name.capitalize())
    ax.set(xlabel="TB probability threshold", ylabel="Metric", ylim=(0, 1.02),
           title=f"{title_prefix}: threshold trade-offs")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_metrics.png", dpi=180)
    plt.close(fig)
    return auroc, auprc


def save_calibration(labels, probabilities, output_dir, title_prefix):
    fraction_positive, mean_probability = calibration_curve(labels, probabilities, n_bins=10, strategy="quantile")
    brier = float(np.mean((probabilities - labels) ** 2))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(mean_probability, fraction_positive, "o-", label=f"Model (Brier={brier:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axes[0].set(xlabel="Mean predicted TB probability", ylabel="Observed TB frequency",
                 title=f"{title_prefix}: calibration")
    axes[0].legend()
    axes[1].hist(probabilities[labels == 0], bins=20, alpha=0.65, label="Normal", density=True)
    axes[1].hist(probabilities[labels == 1], bins=20, alpha=0.65, label="TB", density=True)
    axes[1].set(xlabel="Predicted TB probability", ylabel="Density", title="Probability distributions")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "calibration.png", dpi=180)
    plt.close(fig)
    return brier


def save_error_lists(labels, probabilities, paths, threshold, output_dir):
    predictions = probabilities >= threshold
    frame = pd.DataFrame({
        "path": paths,
        "ground_truth": np.where(labels == 1, "TB", "Normal"),
        "prediction": np.where(predictions, "TB", "Normal"),
        "tb_probability": probabilities,
    })
    frame[frame.ground_truth != frame.prediction].to_csv(output_dir / "errors.csv", index=False)
    frame[(frame.ground_truth == "Normal") & (frame.prediction == "TB")].to_csv(
        output_dir / "false_positives.csv", index=False)
    frame[(frame.ground_truth == "TB") & (frame.prediction == "Normal")].to_csv(
        output_dir / "false_negatives.csv", index=False)


def evaluate_checkpoint(checkpoint_path, validation_loader, classes, positive_index, output_root,
                        min_specificity, split, locked_threshold=None):
    model, checkpoint = load_checkpoint(checkpoint_path)
    probabilities, labels, paths = collect_predictions(model, validation_loader, positive_index)
    checkpoint_name = checkpoint_path.stem
    output_dir = output_root / checkpoint_name
    output_dir.mkdir(parents=True, exist_ok=True)
    table = threshold_table(labels, probabilities)
    selected = (metric_row(labels, probabilities, locked_threshold)
                if locked_threshold is not None else select_threshold(table, min_specificity))
    if locked_threshold is not None:
        selected["specificity_target_met"] = None
        selected["selection_note"] = "Threshold locked before this evaluation."
    default = metric_row(labels, probabilities, 0.5)
    auroc, auprc = save_curves(labels, probabilities, table, output_dir, checkpoint_name)
    brier = save_calibration(labels, probabilities, output_dir, checkpoint_name)
    save_confusion_matrix(selected, output_dir / "confusion_matrix.png",
                          f"{checkpoint_name}: threshold {selected['threshold']:.2f}")
    table.to_csv(output_dir / "threshold_metrics.csv", index=False)
    save_error_lists(labels, probabilities, paths, selected["threshold"], output_dir)
    summary = {
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "classes": classes,
        "positive_class": classes[positive_index],
        "split": split,
        "validation_only": split == "val",
        "threshold_selection": ("locked before held-out evaluation" if locked_threshold is not None
                     else f"highest sensitivity with specificity >= {min_specificity:.2f}"),
        "selected_threshold_metrics": selected,
        "threshold_0.5_metrics": default,
        "auroc": auroc,
        "auprc": auprc,
        "brier_score": brier,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    args = parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)
    normalize = transforms.Normalize([MEAN], [STD])
    validation_transform = transforms.Compose([
        transforms.Resize((512, 512)), transforms.ToTensor(), normalize,
    ])
    if args.split == "test" and args.locked_threshold is None:
        raise ValueError("--locked-threshold is required when evaluating an independent test split.")
    split_path = args.data_path / args.split
    if not split_path.is_dir():
        raise FileNotFoundError(
            f"Missing {args.split!r} split at {split_path}. Do not create a random test split; "
            "provide an independent patient-level test set instead.")
    validation_data = datasets.ImageFolder(split_path, validation_transform, loader=load_gray)
    if validation_data.class_to_idx != {"normal": 0, "tb": 1}:
        raise ValueError(f"Unexpected class mapping: {validation_data.class_to_idx}")
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=DEVICE.type == "cuda")
    print(f"Device: {DEVICE}")
    print(f"{args.split.title()} images: {len(validation_data)} | Mapping: {validation_data.class_to_idx}")
    summaries = []
    checkpoint_names = ([args.checkpoint] if args.checkpoint else
                        ["tb_best_model.pth", "tb_last.pth"])
    for checkpoint_name in checkpoint_names:
        checkpoint_path = args.models_path / checkpoint_name
        if checkpoint_path.exists():
            summary = evaluate_checkpoint(
                checkpoint_path, validation_loader, validation_data.classes, 1,
                args.output_path / args.split, args.min_specificity, args.split,
                args.locked_threshold)
            summaries.append(summary)
            selected = summary["selected_threshold_metrics"]
            print(f"{checkpoint_name}: AUROC={summary['auroc']:.4f} AUPRC={summary['auprc']:.4f} "
                  f"threshold={selected['threshold']:.2f} sensitivity={selected['sensitivity']:.3f} "
                  f"specificity={selected['specificity']:.3f}")
    (args.output_path / "evaluation_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"Saved validation report to {args.output_path}")


if __name__ == "__main__":
    main()
