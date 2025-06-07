import argparse
import re

import numpy as np
from scipy.stats import ttest_rel


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05):
    """
    Compute a (1−alpha) bootstrap confidence interval on the mean of diffs.
    """
    if len(diffs) < 2:
        return 0.0, 0.0
    boots = np.random.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
    return lo, hi


def parse_image_level_iou(results_path):
    """
    Parse a DualSight evaluation file to extract paired image‑level IOUs,
    always treating "(Primary)" as the baseline tag (even in YOLO files).
    Returns two lists of floats: (baseline_ious, ds_ious), in the order encountered.
    """
    baseline_list = []
    ds_list = []

    # Always match "(Primary)" as the baseline detector
    img_primary_pattern = re.compile(r"^(.+)\s+\(Primary\):\s+IoU=([\d\.]+),")
    img_ds_pattern      = re.compile(r"^(.+)\s+\(DualSight\):\s+IoU=([\d\.]+),")

    with open(results_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m_primary = img_primary_pattern.match(line)
        if m_primary:
            baseline_iou = float(m_primary.group(2))
            # Next line should be DualSight
            i += 1
            if i < len(lines):
                ds_line = lines[i].strip()
                m_ds = img_ds_pattern.match(ds_line)
                if m_ds:
                    ds_iou = float(m_ds.group(2))
                    baseline_list.append(baseline_iou)
                    ds_list.append(ds_iou)
                else:
                    # If the following line isn't DualSight, roll back
                    i -= 1
        i += 1

    return baseline_list, ds_list


def analyze_improvement(results_path, label):
    """
    For a single DualSight‐evaluation file (e.g., Nano, X‑Large, or Mask R‑CNN),
    compute paired t‐test between "Primary" vs. "DualSight" image‐level IOUs.
    Returns a dict with: model, t, p, mean_unrefined, mean_refined, std_diff, ci_lo, ci_hi.
    """
    base_ious, ds_ious = parse_image_level_iou(results_path)
    base_arr = np.array(base_ious)
    ds_arr = np.array(ds_ious)
    diffs = ds_arr - base_arr

    n = len(base_arr)
    if n < 2:
        # Insufficient data for t‐test
        return {
            "model": label,
            "t": 0.0,
            "p": 1.0,
            "mean_unrefined": 0.0,
            "mean_refined": 0.0,
            "std_diff": 0.0,
            "ci_lo": 0.0,
            "ci_hi": 0.0,
        }

    t_stat, p_val = ttest_rel(ds_arr, base_arr)
    mean_unrefined = base_arr.mean()
    mean_refined = ds_arr.mean()
    std_diff = diffs.std(ddof=1)
    ci_lo, ci_hi = bootstrap_ci(diffs)

    return {
        "model": label,
        "t": t_stat,
        "p": p_val,
        "mean_unrefined": mean_unrefined,
        "mean_refined": mean_refined,
        "std_diff": std_diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute paired t‑tests on DualSight IoU improvement over Primary for all three models."
    )
    parser.add_argument(
        "--nano-results",
        required=True,
        help="Path to dualSight_nano_evaluation.txt (expects lines “(Primary): IoU=…”)"
    )
    parser.add_argument(
        "--xlarge-results",
        required=True,
        help="Path to dualSight_xlarge_evaluation.txt (expects lines “(Primary): IoU=…”)"
    )
    parser.add_argument(
        "--maskrcnn-results",
        required=True,
        help="Path to dualSight_maskrcnn_evaluation.txt (expects lines “(Primary): IoU=…”)"
    )
    args = parser.parse_args()

    # Analyze YOLOv8 Nano (but now matching "(Primary)" instead of "(YOLO)")
    nano_stats = analyze_improvement(
        args.nano_results,
        label="YOLOv8 Nano"
    )

    # Analyze YOLOv8 X‑Large (same: match "(Primary)")
    xlarge_stats = analyze_improvement(
        args.xlarge_results,
        label="YOLOv8 X‑Large"
    )

    # Analyze Mask R‑CNN (match "(Primary)")
    maskrcnn_stats = analyze_improvement(
        args.maskrcnn_results,
        label="Mask R‑CNN"
    )

    # Print LaTeX table
    print(r"\begin{table}[!h]")
    print(r"\centering")
    print(r"\caption{Paired $t$‑Tests for DualSight IP Improvement over Primary}")
    print(r"\label{tab:dualsight_improvement_primary}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"Model & $t$ & $p$ & Mean Unrefined & Mean Refined & Std $\Delta$IoU & 95\% CI \\")
    print(r"\midrule")

    for stats in (nano_stats, xlarge_stats, maskrcnn_stats):
        print(
            f"{stats['model']} & "
            f"{stats['t']:.2f} & {stats['p']:.4f} & "
            f"{stats['mean_unrefined']:.3f} & {stats['mean_refined']:.3f} & "
            f"{stats['std_diff']:.3f} & {stats['ci_lo']:.3f}--{stats['ci_hi']:.3f}\\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()

    """
    Example usage:
      python _outputEval.py \
        --nano-results   _results4/dualSight_nano_evaluation.txt \
        --xlarge-results _results4/dualSight_xlarge_evaluation.txt \
        --maskrcnn-results _results4/dualSight_maskrcnn_evaluation.txt
    """
