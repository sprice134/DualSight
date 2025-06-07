#!/usr/bin/env python3
import argparse
import re
import ast
import numpy as np
from sklearn.metrics import r2_score

def parse_per_instance_metrics(results_path, baseline_tag="Primary"):
    """
    Parse a “*_evaluation.txt” (DualSight) file and return a list of dicts,
    one per instance, each containing:
        - 'filename' (string)
        - 'GT Morph'         (dict of nine regionprops)
        - 'Primary Morph'    (dict of nine regionprops)
        - 'DS Morph'         (dict of nine regionprops)
    """
    re_primary_header = re.compile(
        rf"^(\S+)\s+\({baseline_tag}\):\s*"
        r"IoU=[\d\.]+,\s*Prec=[\d\.]+,\s*Rec=[\d\.]+,\s*Morph=(\{.*\})"
    )
    re_ds_header = re.compile(
        r"^(\S+)\s+\((?:MASKRCNN\+SAM|YOLO\+SAM|DualSight)\):\s*"
        r"IoU=[\d\.]+,\s*Prec=[\d\.]+,\s*Rec=[\d\.]+,\s*Morph=(\{.*\})"
    )
    re_per_instance_start = re.compile(r"^(\S+)\s+Per-instance details:")
    re_instance_label = re.compile(r"^\s*Instance\s+(\d+):\s*$")
    re_primary_instance = re.compile(
        r"^\s*Primary IoU=[\d\.]+,\s*Primary Prec=[\d\.]+,\s*Primary Rec=[\d\.]+\s*$"
    )
    re_ds_instance = re.compile(
        r"^\s*DualSight IoU=[\d\.]+,\s*DS Prec=[\d\.]+,\s*DS Rec=[\d\.]+\s*$"
    )
    re_gt_morph      = re.compile(r"^\s*GT Morph=(\{.*\})\s*$")
    re_primary_morph = re.compile(r"^\s*Primary Morph=(\{.*\})\s*$")
    re_ds_morph      = re.compile(r"^\s*DS Morph=(\{.*\})\s*$")

    instances = []
    lines = [L.rstrip("\n") for L in open(results_path, 'r')]
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        m_primary = re_primary_header.match(line)
        if not m_primary:
            i += 1
            continue

        filename = m_primary.group(1)

        # 1) Advance to next line, which must be the DS header for the same filename
        i += 1
        if i >= n:
            break
        next_line = lines[i].strip()
        m_ds = re_ds_header.match(next_line)
        if not m_ds or m_ds.group(1) != filename:
            raise ValueError(
                f"Expected “{filename} (MASKRCNN+SAM|YOLO+SAM|DualSight)” at line {i+1}, got: {lines[i]}"
            )

        # 2) Advance to “Per-instance details:”
        i += 1
        if i >= n:
            break
        per_inst_line = lines[i].strip()
        m_inst_start = re_per_instance_start.match(per_inst_line)
        if not m_inst_start or m_inst_start.group(1) != filename:
            # No per-instance block for this image → skip ahead
            i += 1
            continue

        # 3) Inside the per-instance block:
        i += 1  # move to first “Instance X:” or blank
        while i < n:
            if not re_instance_label.match(lines[i]):
                # End of this image’s per-instance block
                break

            # We have “Instance X:”
            # 3a) Next line must be “Primary IoU=…, Primary Prec=…, Primary Rec=…”
            i += 1
            if i >= n:
                raise ValueError(f"Unexpected EOF before Primary IoU line for {filename}")
            if not re_primary_instance.match(lines[i]):
                raise ValueError(f"Expected “Primary IoU=…” at line {i+1}, got: {lines[i]}")

            # 3b) Next line must be “DualSight IoU=…, DS Prec=…, DS Rec=…”
            i += 1
            if i >= n:
                raise ValueError(f"Unexpected EOF before DS IoU line for {filename}")
            if not re_ds_instance.match(lines[i]):
                raise ValueError(f"Expected “DualSight IoU=…” at line {i+1}, got: {lines[i]}")

            # 3c) Next line: “GT Morph={…}”
            i += 1
            if i >= n:
                raise ValueError(f"Unexpected EOF before GT Morph for {filename}")
            m_gtm = re_gt_morph.match(lines[i])
            if not m_gtm:
                raise ValueError(f"Expected “GT Morph={{…}}” at line {i+1}, got: {lines[i]}")
            try:
                gt_dict = ast.literal_eval(m_gtm.group(1))
            except Exception as e:
                raise ValueError(f"Failed to parse GT Morph on line {i+1}: {e}")

            # 3d) Next line: “Primary Morph={…}”
            i += 1
            if i >= n:
                raise ValueError(f"Unexpected EOF before Primary Morph for {filename}")
            m_prm = re_primary_morph.match(lines[i])
            if not m_prm:
                raise ValueError(f"Expected “Primary Morph={{…}}” at line {i+1}, got: {lines[i]}")
            try:
                pr_dict = ast.literal_eval(m_prm.group(1))
            except Exception as e:
                raise ValueError(f"Failed to parse Primary Morph on line {i+1}: {e}")

            # 3e) Next line: “DS Morph={…}”
            i += 1
            if i >= n:
                raise ValueError(f"Unexpected EOF before DS Morph for {filename}")
            m_dsm = re_ds_morph.match(lines[i])
            if not m_dsm:
                raise ValueError(f"Expected “DS Morph={{…}}” at line {i+1}, got: {lines[i]}")
            try:
                ds_dict = ast.literal_eval(m_dsm.group(1))
            except Exception as e:
                raise ValueError(f"Failed to parse DS Morph on line {i+1}: {e}")

            # Store a single record for one instance
            instances.append({
                'filename': filename,
                'GT Morph': gt_dict,
                'Primary Morph': pr_dict,
                'DS Morph': ds_dict
            })

            # Move to the next line (blank or next “Instance X:”)
            i += 1

        # End of this image’s per-instance block → continue scanning
        continue

    return instances


def compute_mape_summary_across_instances(inst_list):
    """
    Given a list of instance-dicts (each with 'GT Morph', 'Primary Morph', 'DS Morph'),
    compute for each of the nine metrics using only the middle 95% of YOLO relative errors:
      - YOLO_rel_errors = (Primary - GT) / GT  (for GT != 0)
      - Keep only those instances whose YOLO_rel_error is between the 2.5th and 97.5th percentile.
      - On that filtered subset, compute:
          * MAPE_primary = mean(|(Primary - GT) / GT|) * 100
          * MAPE_ds      = mean(|(DS - GT) / GT|) * 100
          * R2_primary   = r2_score(GT, Primary)
          * R2_ds        = r2_score(GT, DS)

    Returns two dicts keyed by metric_name:
      summary_primary[metric] = {'MAPE': float, 'R2': float}
      summary_ds[metric]      = {'MAPE': float, 'R2': float}
    """
    metric_keys = [
        'area',
        'area_convex',
        'major_axis_length',
        'minor_axis_length',
        'eccentricity',
        'equivalent_diameter',
        'feret_diameter_max',
        'perimeter',
        'solidity',
    ]

    # Gather arrays for each metric
    gt_vals = {k: [] for k in metric_keys}
    pr_vals = {k: [] for k in metric_keys}
    ds_vals = {k: [] for k in metric_keys}

    for inst in inst_list:
        gt_dict = inst['GT Morph']
        pr_dict = inst['Primary Morph']
        ds_dict = inst['DS Morph']
        for k in metric_keys:
            gt_vals[k].append(float(gt_dict.get(k, 0.0)))
            pr_vals[k].append(float(pr_dict.get(k, 0.0)))
            ds_vals[k].append(float(ds_dict.get(k, 0.0)))

    summary_primary = {}
    summary_ds = {}

    for k in metric_keys:
        gt_arr = np.array(gt_vals[k])
        pr_arr = np.array(pr_vals[k])
        ds_arr = np.array(ds_vals[k])

        # 1) Compute YOLO relative errors for GT != 0
        nonzero_mask = gt_arr != 0
        if np.any(nonzero_mask):
            yolo_rel = (pr_arr[nonzero_mask] - gt_arr[nonzero_mask]) / gt_arr[nonzero_mask]
            lower, upper = np.percentile(yolo_rel, [2.5, 97.5])
            # 2) Build a mask of instances to keep:
            rel_values = (pr_arr - gt_arr) / np.where(gt_arr == 0, 1, gt_arr)
            mask_within = (rel_values >= lower) & (rel_values <= upper) & nonzero_mask
            keep_mask = mask_within
        else:
            keep_mask = np.zeros_like(gt_arr, dtype=bool)

        # 3) From the filtered subset, compute MAPE
        if np.any(keep_mask):
            gt_filt = gt_arr[keep_mask]
            pr_filt = pr_arr[keep_mask]
            ds_filt = ds_arr[keep_mask]

            mape_pr = float(np.mean(np.abs((pr_filt - gt_filt) / gt_filt)) * 100)
            mape_ds = float(np.mean(np.abs((ds_filt - gt_filt) / gt_filt)) * 100)
        else:
            mape_pr = 0.0
            mape_ds = 0.0

        # 4) Compute R² on the same filtered subset (if any)
        if np.any(keep_mask) and len(gt_filt) > 1:
            try:
                r2_pr = float(r2_score(gt_filt, pr_filt))
            except ValueError:
                r2_pr = 0.0
            try:
                r2_ds = float(r2_score(gt_filt, ds_filt))
            except ValueError:
                r2_ds = 0.0
        else:
            r2_pr = 1.0
            r2_ds = 1.0

        summary_primary[k] = {'MAPE': mape_pr, 'R2': r2_pr}
        summary_ds[k]      = {'MAPE': mape_ds, 'R2': r2_ds}

    return summary_primary, summary_ds


def format_combined_latex_table(nano_baseline, nano_ds,
                                xlarge_baseline, xlarge_ds,
                                mask_baseline, mask_ds):
    """
    Produce a single LaTeX table in this style:

    \begin{table}[h!]
        \renewcommand{\arraystretch}{1.2} % Increase row height for padding
        \centering
        % Use adjustbox to fit the table to page width
        \begin{adjustbox}{width=\textwidth}
        \begin{tabular}{|l|c|c|c|c|c|c|}
            \hline
            \textbf{Metric} & \multicolumn{2}{c|}{\textbf{YOLOv8 Nano}} & 
                              \multicolumn{2}{c|}{\textbf{YOLOv8 X-Large}} & 
                              \multicolumn{2}{c|}{\textbf{Mask R-CNN}} \\ \hline
             & \makecell[l]{\textbf{Unrefined}\\\textbf{MAPE (\%)}} & 
               \makecell[l]{\textbf{Refined}\\\textbf{MAPE (\%)}} & 
               \makecell[l]{\textbf{Unrefined}\\\textbf{MAPE (\%)}} & 
               \makecell[l]{\textbf{Refined}\\\textbf{MAPE (\%)}} & 
               \makecell[l]{\textbf{Unrefined}\\\textbf{MAPE (\%)}} & 
               \makecell[l]{\textbf{Refined}\\\textbf{MAPE (\%)}} \\ \hline
            Area              & … & … & … & … & … & … \\ \hline
            Convex Area       & … & … & … & … & … & … \\ \hline
            Major Axis Length & … & … & … & … & … & … \\ \hline
            Minor Axis Length & … & … & … & … & … & … \\ \hline
            Eccentricity      & … & … & … & … & … & … \\ \hline
            Equivalent Diam.  & … & … & … & … & … & … \\ \hline
            Feret Diameter    & … & … & … & … & … & … \\ \hline
            Perimeter         & … & … & … & … & … & … \\ \hline
            Solidity          & … & … & … & … & … & … \\ \hline
        \end{tabular}
        \end{adjustbox}
        \caption{Comparison of Unrefined MAPE vs. Refined MAPE across three models}
        \label{table:combined_mape_metrics}
    \end{table}

    In each data row, replace “…” with the numeric values, bolding the smaller
    of unrefined vs. refined in each pair.
    """
    metrics = [
        ('area', 'Area'),
        ('area_convex', 'Convex Area'),
        ('major_axis_length', 'Major Axis Length'),
        ('minor_axis_length', 'Minor Axis Length'),
        ('eccentricity', 'Eccentricity'),
        ('equivalent_diameter', 'Equivalent Diameter'),
        ('feret_diameter_max', 'Feret Diameter'),
        ('perimeter', 'Perimeter'),
        ('solidity', 'Solidity'),
    ]

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"    \renewcommand{\arraystretch}{1.2} % Increase row height for padding")
    lines.append(r"    \centering")
    lines.append(r"    % Use adjustbox to fit the table to page width")
    lines.append(r"    \begin{adjustbox}{width=\textwidth}")
    lines.append(r"    \begin{tabular}{|l|c|c|c|c|c|c|}")
    lines.append(r"        \hline")

    # First header row
    lines.append(
        r"        \textbf{Metric} & "
        r"\multicolumn{2}{c|}{\textbf{YOLOv8 Nano}} & "
        r"\multicolumn{2}{c|}{\textbf{YOLOv8 X-Large}} & "
        r"\multicolumn{2}{c|}{\textbf{Mask R-CNN}} \\ \hline"
    )

    # Second header row, using \makecell for each two-line column
    lines.append(
        r"         & \makecell[l]{\textbf{Unrefined}\\\textbf{MAPE (\%)}} "
        r"& \makecell[l]{\textbf{Refined}\\\textbf{MAPE (\%)}} "
        r"& \makecell[l]{\textbf{Unrefined}\\\textbf{MAPE (\%)}} "
        r"& \makecell[l]{\textbf{Refined}\\\textbf{MAPE (\%)}} "
        r"& \makecell[l]{\textbf{Unrefined}\\\textbf{MAPE (\%)}} "
        r"& \makecell[l]{\textbf{Refined}\\\textbf{MAPE (\%)}} \\ \hline"
    )

    for key, display_name in metrics:
        # YOLOv8 Nano
        nano_unref = nano_baseline.get(key, {}).get('MAPE', 0.0)
        nano_refm  = nano_ds.get(key, {}).get('MAPE', 0.0)

        # YOLOv8 X-Large
        xl_unref = xlarge_baseline.get(key, {}).get('MAPE', 0.0)
        xl_refm  = xlarge_ds.get(key, {}).get('MAPE', 0.0)

        # Mask R-CNN
        mrc_unref = mask_baseline.get(key, {}).get('MAPE', 0.0)
        mrc_refm  = mask_ds.get(key, {}).get('MAPE', 0.0)

        # Bold the smaller value in each pair
        if nano_unref <= nano_refm:
            nano_unref_str = fr"\textbf{{{nano_unref:.3f}}}"
            nano_refm_str  = f"{nano_refm:.3f}"
        else:
            nano_unref_str = f"{nano_unref:.3f}"
            nano_refm_str  = fr"\textbf{{{nano_refm:.3f}}}"

        if xl_unref <= xl_refm:
            xl_unref_str = fr"\textbf{{{xl_unref:.3f}}}"
            xl_refm_str  = f"{xl_refm:.3f}"
        else:
            xl_unref_str = f"{xl_unref:.3f}"
            xl_refm_str  = fr"\textbf{{{xl_refm:.3f}}}"

        if mrc_unref <= mrc_refm:
            mrc_unref_str = fr"\textbf{{{mrc_unref:.3f}}}"
            mrc_refm_str  = f"{mrc_refm:.3f}"
        else:
            mrc_unref_str = f"{mrc_unref:.3f}"
            mrc_refm_str  = fr"\textbf{{{mrc_refm:.3f}}}"

        lines.append(
            f"        {display_name} & "
            f"{nano_unref_str} & {nano_refm_str} & "
            f"{xl_unref_str} & {xl_refm_str} & "
            f"{mrc_unref_str} & {mrc_refm_str} \\\\ \\hline"
        )

    lines.append(r"    \end{tabular}")
    lines.append(r"    \end{adjustbox}")
    lines.append(r"    \caption{Comparison of Unrefined MAPE vs. Refined MAPE across three models}")
    lines.append(r"    \label{table:combined_mape_metrics}")
    lines.append(r"\end{table}")
    lines.append("")  # blank line

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-instance–averaged MAPE (middle 95% YOLO errors) and generate a single combined LaTeX table."
    )
    parser.add_argument(
        "--nano-results",
        required=True,
        help="Path to dualSight_nano_evaluation.txt (Baseline=YOLOv8 Nano)"
    )
    parser.add_argument(
        "--xlarge-results",
        required=True,
        help="Path to dualSight_xlarge_evaluation.txt (Baseline=YOLOv8 X-Large)"
    )
    parser.add_argument(
        "--maskrcnn-results",
        required=True,
        help="Path to dualSight_maskrcnn_evaluation.txt (Baseline=Mask R-CNN)"
    )
    args = parser.parse_args()

    # 1) Parse each file into a list of per-instance “GT/Primary/DS Morph” dicts
    nano_instances     = parse_per_instance_metrics(args.nano_results,     baseline_tag="Primary")
    xlarge_instances   = parse_per_instance_metrics(args.xlarge_results,   baseline_tag="Primary")
    maskrcnn_instances = parse_per_instance_metrics(args.maskrcnn_results, baseline_tag="Primary")

    # 2) Compute summary (MAPE & R²) across ALL instances for each model,
    #    but only keep the middle 95% of YOLO relative errors per metric
    nano_baseline,     nano_ds     = compute_mape_summary_across_instances(nano_instances)
    xlarge_baseline,   xlarge_ds   = compute_mape_summary_across_instances(xlarge_instances)
    maskrcnn_baseline, maskrcnn_ds = compute_mape_summary_across_instances(maskrcnn_instances)

    # 3) Print the LaTeX table
    combined_table = format_combined_latex_table(
        nano_baseline, nano_ds,
        xlarge_baseline, xlarge_ds,
        maskrcnn_baseline, maskrcnn_ds
    )
    print(combined_table)


if __name__ == "__main__":
    main()





    '''
    python _outputMorphEval4.py \
    --nano-results   _results4/dualSight_nano_evaluation.txt \
    --xlarge-results _results4/dualSight_xlarge_evaluation.txt \
    --maskrcnn-results _results4/dualSight_maskrcnn_evaluation.txt 
    '''