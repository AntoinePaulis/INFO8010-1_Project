#CLAUDE GENERATED FILE TO PLOT ERRORS

"""
Loads a saved predictions .pt file from inference.py and produces:
  1. A positioning-error histogram (PNG) for the poster
  2. Printed test metrics summary

Run from src/ball_tracking/ on Alan:
    python plot_errors.py --pt /path/to/predictions_xxx.pt --out /path/to/output.png
"""

import argparse
import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

W = 640  # frame width used during inference

DETECT_THRESH = 2.55  # same as inference.py
TP_THRESH     = 5     # pixels — same as compute_ball_metrics


def load_and_analyse(pt_path):
    data = torch.load(pt_path, map_location="cpu")
    preds  = data["predictions"]    # (N, H, W)  int class 0-255
    gts    = data["ground_truths"]  # (N, 1, H, W) float [0,1]
    metrics = data.get("metrics", {})
    return preds, gts, metrics


def compute_errors(preds, gts):
    """Return pixel distances for every (visible, detected) sample."""
    N = preds.shape[0]
    distances = []
    tp = fp = fn = tn = 0

    for i in range(N):
        gt_map   = gts[i, 0]           # (H, W)
        pred_map = preds[i].float()    # (H, W)

        ball_visible = gt_map.max().item() > 0.0
        ball_detected = pred_map.max().item() > DETECT_THRESH

        if not ball_visible:
            if not ball_detected:
                tn += 1
            else:
                fp += 1
            continue

        # ball is visible
        if not ball_detected:
            fn += 1
            continue

        gt_idx   = torch.argmax(gt_map).item()
        pred_idx = torch.argmax(pred_map).item()

        gt_y,   gt_x   = divmod(gt_idx,   W)
        pred_y, pred_x = divmod(pred_idx, W)

        dist = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        distances.append(dist)

        if dist < TP_THRESH:
            tp += 1
        else:
            fp += 1

    return distances, tp, fp, fn, tn


def plot_histogram(distances, out_path, cap_px=10):
    distances = np.array(distances)

    fig, ax = plt.subplots(figsize=(4.5, 1.6))

    bins = np.arange(0, cap_px + 2, 1)
    counts, edges = np.histogram(distances, bins=bins)
    pct = 100.0 * counts / len(distances)

    bar_colors = ["#1565C0" if e < TP_THRESH else "#90A4AE" for e in edges[:-1]]
    ax.bar(edges[:-1], pct, width=0.85, color=bar_colors, linewidth=0, align="edge")

    ax.axvline(TP_THRESH, color="#E65100", linewidth=1.2, linestyle="--",
               label=f"TP threshold ({TP_THRESH} px)")

    within_tp = 100.0 * (distances < TP_THRESH).sum() / len(distances)
    clipped    = 100.0 * (distances > cap_px).sum() / len(distances)
    ax.text(TP_THRESH + 0.25, max(pct[:TP_THRESH]) * 0.92,
            f"{within_tp:.1f}% ≤ {TP_THRESH} px",
            fontsize=6.5, color="#E65100", va="top")

    ax.set_xlim(0, cap_px + 1)
    ax.set_xlabel("Positioning error (px)", fontsize=7.5)
    ax.set_ylabel("% of detections", fontsize=7.5)
    ax.set_title("Ball positioning error distribution", fontsize=8, weight="medium")
    ax.legend(fontsize=6, framealpha=0.7)
    ax.tick_params(labelsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)

    if clipped > 0.05:
        ax.text(0.98, 0.95, f"{clipped:.1f}% of detections\nnot shown (>{cap_px} px)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=5.5, color="gray", style="italic")

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved histogram → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt",  required=True, help="Path to predictions .pt file")
    parser.add_argument("--out", default="ball_error_dist.png", help="Output PNG path")
    args = parser.parse_args()

    print(f"Loading {args.pt} …")
    preds, gts, saved_metrics = load_and_analyse(args.pt)
    print(f"  Samples: {preds.shape[0]}")

    distances, tp, fp, fn, tn = compute_errors(preds, gts)

    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy   = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n── Test metrics ──────────────────────")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Accuracy    = {accuracy:.4f}")
    print(f"  Precision   = {precision:.4f}")
    print(f"  Recall      = {recall:.4f}")
    print(f"  Specificity = {specificity:.4f}")
    print(f"  F1          = {f1:.4f}")
    if saved_metrics:
        print("\n── Metrics from saved file ───────────")
        for k, v in saved_metrics.items():
            print(f"  {k} = {v}")

    if distances:
        dists = np.array(distances)
        print(f"\n── Error distribution ({len(dists)} detections on visible ball) ──")
        print(f"  Mean  = {dists.mean():.2f} px")
        print(f"  Median= {np.median(dists):.2f} px")
        print(f"  ≤5 px = {100*(dists < 5).mean():.1f}%")
        print(f"  ≤10px = {100*(dists < 10).mean():.1f}%")
        plot_histogram(dists, args.out)
    else:
        print("\nNo detections on visible frames — cannot plot.")


if __name__ == "__main__":
    main()
