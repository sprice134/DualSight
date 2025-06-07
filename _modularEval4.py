import os
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from ultralytics import YOLO

import sys
sys.path.append('utils/')
from samDemo import (
    show_mask,
    show_points,
    show_box,
    mask_to_polygon,
    generate_random_points_within_polygon,
    point_to_polygon_distance,
    find_optimal_points,
    polygon_to_binary_mask,
    expand_bbox_within_border,
    fractal_dimension,
    apply_mask_to_image,
)
from segment_anything import sam_model_registry, SamPredictor
from shapely.geometry import Polygon
from skimage.draw import polygon
from skimage.measure import regionprops
from sklearn.metrics import r2_score

from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as FT

############################################################################
#   Mask R‑CNN loader and predictor
############################################################################

def load_maskrcnn_model(model_path, num_classes=2, backbone="resnet50", device="cuda"):
    """
    Load and return a trained Mask R‑CNN model.
    """
    if backbone != "resnet50":
        raise ValueError(f"Unsupported backbone: {backbone}")
    model = maskrcnn_resnet50_fpn(pretrained=False)
    # Replace the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels=in_features_mask,
        dim_reduced=hidden_layer,
        num_classes=num_classes,
    )
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_maskrcnn_predictions(model, image_path, device="cuda"):
    """
    Generate Mask R‑CNN predictions on an image path.
    Returns (listOfPolygons, listOfBoxes, listOfMasks).
    Each mask is a 0/255 uint8 array.
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = FT.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image_tensor)[0]

    listOfPolygons = []
    listOfBoxes = []
    listOfMasks = []
    score_threshold = 0.5

    for i, score in enumerate(predictions['scores']):
        if score >= score_threshold:
            mask = predictions['masks'][i, 0].cpu().numpy()
            box = predictions['boxes'][i].cpu().numpy()
            if mask.sum() > 200:
                binary_mask = (mask > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    polygon_points = largest_contour.squeeze(axis=1).tolist()
                    if len(polygon_points) > 2:
                        listOfPolygons.append(polygon_points)
                        listOfBoxes.append(box)

                        # Create a 0/255 mask image
                        mask_image = Image.new('L', image.size, 0)
                        draw = ImageDraw.Draw(mask_image)
                        polygon_points_int = [(int(x), int(y)) for x, y in polygon_points]
                        draw.polygon(polygon_points_int, outline=1, fill=1)
                        mask_array = np.array(mask_image) * 255
                        listOfMasks.append(mask_array.astype(np.uint8))

    return listOfPolygons, listOfBoxes, listOfMasks

############################################################################
#   Utility functions
############################################################################

def get_jpg_files(directory):
    """Return a list of all .jpg file paths in the given directory."""
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".jpg")
    ]

def get_image_dimensions(image_path):
    """Return (width, height) of the image at image_path."""
    with Image.open(image_path) as img:
        return img.size  # (width, height)

def label_to_mask(label_line, image_shape):
    """
    Convert a YOLOv8 segmentation label line to a binary mask.
    """
    parts = list(map(float, label_line.split()))
    coords = parts[1:]
    x = np.array(coords[0::2]) * image_shape[1]
    y = np.array(coords[1::2]) * image_shape[0]
    x = np.clip(x, 0, image_shape[1] - 1)
    y = np.clip(y, 0, image_shape[0] - 1)
    rr, cc = polygon(y, x)
    mask = np.zeros(image_shape, dtype=np.uint8)
    mask[rr, cc] = 1
    return mask

def convert_to_binary_mask(mask):
    """Convert a mask with 0s and 255s to 0s and 1s."""
    return np.where(mask == 255, 1, 0)

def binary_mask_to_regionprops_dict(binary_mask):
    """
    Given a 2D binary mask, compute regionprops and return a list of dicts
    containing only the selected properties.
    """
    props = regionprops(binary_mask)
    selected_props = [
        "area",
        "area_convex",
        "major_axis_length",
        "minor_axis_length",
        "eccentricity",
        "equivalent_diameter",
        "euler_number",
        "extent",
        "feret_diameter_max",
        "perimeter",
        "solidity",
    ]
    props_list = []
    for region in props:
        region_dict = {}
        for p in selected_props:
            region_dict[p] = getattr(region, p, None)
        props_list.append(region_dict)
    return props_list

def mean_iou_precision_recall(gt_masks, pred_masks, sam=False):
    """
    Compute per-mask best IoU (and its precision, recall), then average.
    If sam=False, pred_masks should be 0/255 uint8 arrays; this function
    converts them to 0/1 internally via convert_to_binary_mask.
    Returns: (mean_iou, mean_precision, mean_recall)
    """
    iou_scores = []
    precision_scores = []
    recall_scores = []

    for pm in pred_masks:
        if not sam:
            pm = convert_to_binary_mask(pm)

        best_iou = 0
        best_prec = 0
        best_rec = 0

        for gm in gt_masks:
            if pm.shape != gm.shape:
                continue
            intersection = np.logical_and(gm, pm).sum()
            union = np.logical_or(gm, pm).sum()
            if union != 0 and intersection != 0:
                iou_val = intersection / union
                prec_val = intersection / pm.sum() if pm.sum() != 0 else 0
                rec_val = intersection / gm.sum() if gm.sum() != 0 else 0
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_prec = prec_val
                    best_rec = rec_val

        iou_scores.append(best_iou)
        precision_scores.append(best_prec)
        recall_scores.append(best_rec)

    return (
        np.mean(iou_scores) if iou_scores else 0,
        np.mean(precision_scores) if precision_scores else 0,
        np.mean(recall_scores) if recall_scores else 0,
    )

def calculate_morphological_metric_summary(gt_list, pred_list):
    """
    Given two lists of dicts (gt metrics, pred metrics) with the same keys,
    compute RMSE, means, and R² for each key across the list.
    """
    if len(gt_list) != len(pred_list):
        raise ValueError("gt_list and pred_list must be same length.")
    keys = list(gt_list[0].keys()) if gt_list else []
    squared_errors = {k: [] for k in keys}
    gt_vals = {k: [] for k in keys}
    pred_vals = {k: [] for k in keys}

    for gt, pr in zip(gt_list, pred_list):
        for k in keys:
            err = (gt[k] - pr[k]) ** 2
            squared_errors[k].append(err)
            gt_vals[k].append(gt[k])
            pred_vals[k].append(pr[k])

    results = {}
    for k in keys:
        mse = np.mean(squared_errors[k])
        rmse = np.sqrt(mse)
        mean_pred = np.mean(pred_vals[k])
        mean_gt = np.mean(gt_vals[k])
        r2 = r2_score(gt_vals[k], pred_vals[k]) if len(gt_vals[k]) > 1 else None
        results[k] = {
            "RMSE": rmse,
            "Mean of Predictions": mean_pred,
            "Mean of Ground Truth": mean_gt,
            "R2": r2,
        }
    return results

def calculate_morphological_summary_list(summary_list):
    """
    Given a list of per-image morphological-summary dicts,
    average each numeric entry across images.
    """
    if not summary_list:
        return {}
    keys = summary_list[0].keys()
    averaged = {}
    for k in keys:
        rmse_vals = [d[k]["RMSE"] for d in summary_list if k in d]
        pred_means = [d[k]["Mean of Predictions"] for d in summary_list if k in d]
        gt_means = [d[k]["Mean of Ground Truth"] for d in summary_list if k in d]
        r2_vals = [d[k]["R2"] for d in summary_list if k in d and d[k]["R2"] is not None]

        averaged[k] = {
            "RMSE": np.mean(rmse_vals) if rmse_vals else None,
            "Mean of Predictions": np.mean(pred_means) if pred_means else None,
            "Mean of Ground Truth": np.mean(gt_means) if gt_means else None,
            "R2": np.mean(r2_vals) if r2_vals else None,
        }
    return averaged

def get_mean_regionprops(gt_masks, pred_masks, sam=False):
    """
    For each predicted mask, find the ground-truth mask with highest IoU,
    then collect their regionprops (first component only) and run a summary.
    """
    gt_metrics = []
    pred_metrics = []

    for pm in pred_masks:
        if not sam:
            pm = convert_to_binary_mask(pm)

        best_iou = 0
        best_gt_props = None
        best_pred_props = None

        for gm in gt_masks:
            if pm.shape != gm.shape:
                continue
            intersection = np.logical_and(gm, pm).sum()
            union = np.logical_or(gm, pm).sum()
            if union != 0 and intersection != 0:
                iou_val = intersection / union
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_props = binary_mask_to_regionprops_dict(gm)[0]
                    best_pred_props = binary_mask_to_regionprops_dict(pm)[0]

        if best_gt_props is None:
            if gt_metrics:
                zero_dict = {k: 0 for k in gt_metrics[0].keys()}
            else:
                zero_dict = {}
            gt_metrics.append(zero_dict)
            pred_metrics.append(zero_dict)
        else:
            gt_metrics.append(best_gt_props)
            pred_metrics.append(best_pred_props)

    return calculate_morphological_metric_summary(gt_metrics, pred_metrics)

def get_ground_truth_masks(image_path, labels_dir):
    """
    Given an image_path and a labels directory (relative or absolute),
    open the corresponding .txt file and return a list of binary masks.
    """
    img_name = os.path.basename(image_path)
    base = os.path.splitext(img_name)[0]
    w, h = get_image_dimensions(image_path)
    label_file = os.path.join(labels_dir, f"{base}.txt")
    masks = []
    with open(label_file, "r") as f:
        lines = [L.strip() for L in f.readlines()]
        for line in lines:
            masks.append(label_to_mask(line, (h, w)))
    return masks

def get_yolo_prediction_masks(image_path, yolo_model):
    """
    Run YOLOv8 model on an image and convert each detected polygon to a binary mask.
    """
    img = Image.open(image_path)
    results = yolo_model(img)
    pred_masks = []
    for inst in results[0]:
        poly_pts = inst.masks.xy[0]
        if len(poly_pts) > 2:
            mask = polygon_to_binary_mask(poly_pts, img.height, img.width)
            pred_masks.append(mask)
    return pred_masks

def get_dual_sight_masks(image_path, primary_model, primary_type, sam_predictor, device):
    """
    For a given image, run the primary model → polygon & bbox → sample optimal points → SAM → collect
    the first SAM mask for each primary polygon. Returns a list of binary masks.
    """
    img_pil = Image.open(image_path)
    frame = cv2.imread(image_path)

    # Choose which model to run for polygons + bboxes
    if primary_type.lower() == "yolo":
        results = primary_model(img_pil)
        all_polygons = []
        all_bboxes = []
        for inst in results[0]:
            pts = inst.masks.xy[0]
            if len(pts) > 2:
                all_polygons.append(pts)
                all_bboxes.append(inst.boxes.xyxy[0].cpu().numpy())
    else:  # maskrcnn
        polygons, boxes, masks_255 = get_maskrcnn_predictions(primary_model, image_path, device=device)
        all_polygons = polygons
        all_bboxes = boxes

    masks_list = []

    for poly, bbox in zip(all_polygons, all_bboxes):
        # Convert polygon to a binary mask for IoU
        pm = polygon_to_binary_mask(poly, img_pil.height, img_pil.width)

        x1, y1, x2, y2 = bbox
        x1e, y1e, x2e, y2e = expand_bbox_within_border(
            float(x1), float(y1), float(x2), float(y2),
            img_pil.width,
            img_pil.height,
            expansion_rate=0.1,
        )
        bbox_exp = np.array([x1e, y1e, x2e, y2e])

        concave_poly = Polygon(poly)
        sampled_pts = generate_random_points_within_polygon(concave_poly, 50)
        optimal_pts = find_optimal_points(
            sampled_pts, concave_poly,
            num_result_points=3,
            border_weight=2
        )
        inp_pts = np.array([[pt.x, pt.y] for pt in optimal_pts])
        inp_lbls = np.array([1] * len(optimal_pts))

        sam_predictor.set_image(frame)
        masks, scores, logits = sam_predictor.predict(
            point_coords=inp_pts,
            point_labels=inp_lbls,
            box=bbox_exp[None, :],
            multimask_output=True,
        )
        if masks.shape[0] > 0:
            masks_list.append((masks[0] == True).astype(np.uint8))
        else:
            masks_list.append(np.zeros_like(pm))

    return masks_list

############################################################################
#   Evaluation functions
############################################################################

def run_yolo_evaluation(
    directory_path,
    labels_dir,
    primary_model,
    primary_type,
    device,
    output_txt="primary_results.txt"
):
    """
    For each JPG in directory_path/images, load GT masks from labels_dir, run either YOLO or Mask R‑CNN,
    compute IoU/precision/recall + morphological summary (per-image), then write both per-image and
    averaged‐across‐images results to output_txt.  Also prints number of detections per image.
    """
    images_dir = os.path.join(directory_path, "images")
    jpg_files = get_jpg_files(images_dir)

    per_image_morph = []
    all_iou = []
    all_prec = []
    all_rec = []

    prefix = "(YOLO)" if primary_type.lower() == "yolo" else "(MASKRCNN)"

    with open(output_txt, "w") as out_f:
        for img_path in jpg_files:
            base = os.path.splitext(os.path.basename(img_path))[0]
            gt_masks = get_ground_truth_masks(img_path, labels_dir)

            # Run the chosen primary model & count detections
            if primary_type.lower() == "yolo":
                pred_masks = get_yolo_prediction_masks(img_path, primary_model)
                num_dets = len(pred_masks)
            else:  # maskrcnn
                polygons, boxes, masks_255 = get_maskrcnn_predictions(
                    primary_model, img_path, device
                )
                num_dets = len(polygons)
                # Keep masks as 0/255 arrays, do not convert to 0/1 here
                pred_masks = masks_255

            # Compute CV metrics for the chosen primary model
            iou_primary, prec_primary, rec_primary = mean_iou_precision_recall(
                gt_masks, pred_masks
            )

            # Compute morphological summary (per-image)
            morph_primary = {}
            try:
                morph_primary = get_mean_regionprops(gt_masks, pred_masks, sam=False)
                per_image_morph.append(morph_primary)
            except ValueError:
                pass

            all_iou.append(iou_primary)
            all_prec.append(prec_primary)
            all_rec.append(rec_primary)

            # Write line with detections count included
            out_f.write(
                f"{base} {prefix}: Detections={num_dets}, "
                f"IoU={iou_primary:.4f}, Prec={prec_primary:.4f}, Rec={rec_primary:.4f}, Morph={morph_primary}\n"
            )

        # Compute overall averages
        mean_iou_all = np.mean(all_iou) if all_iou else 0
        mean_prec_all = np.mean(all_prec) if all_prec else 0
        mean_rec_all = np.mean(all_rec) if all_rec else 0
        overall_morph = calculate_morphological_summary_list(per_image_morph)

        out_f.write("\n")
        out_f.write(
            f"{prefix} Overall: Mean IoU={mean_iou_all:.4f}, "
            f"Mean Prec={mean_prec_all:.4f}, Mean Rec={mean_rec_all:.4f}, "
            f"Overall Morph={overall_morph}\n"
        )

    print(f"{prefix} evaluation complete. Results saved to {output_txt}")

def run_dual_sight_evaluation(
    directory_path,
    labels_dir,
    primary_model,
    primary_type,
    sam_checkpoint,
    sam_model_type="vit_l",
    device="cuda",
    output_txt="dualsight_results.txt",
):
    """
    For each JPG in directory_path/images, load GT masks, run primary_model → SAM pipeline,
    compute for each image:
      - Primary-model CV metrics + morphological summary
      - DualSight CV metrics + morphological summary
      - Per-instance improvement scores in IoU/precision/recall + per-instance regionprops
    Then write both per-image results and overall averages to output_txt.
    """
    # Load SAM
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    images_dir = os.path.join(directory_path, "images")
    jpg_files = get_jpg_files(images_dir)

    per_image_morph_primary = []
    per_image_iou_primary = []
    per_image_prec_primary = []
    per_image_rec_primary = []

    per_image_morph_ds = []
    per_image_iou_ds = []
    per_image_prec_ds = []
    per_image_rec_ds = []

    prefix = "(YOLO+SAM)" if primary_type.lower() == "yolo" else "(MASKRCNN+SAM)"

    with open(output_txt, "w") as out_f:
        for img_path in jpg_files:
            base = os.path.splitext(os.path.basename(img_path))[0]
            gt_masks = get_ground_truth_masks(img_path, labels_dir)

            # 1) Primary-only
            if primary_type.lower() == "yolo":
                pred_primary = get_yolo_prediction_masks(img_path, primary_model)
            else:
                polygons_p, boxes_p, masks_p255 = get_maskrcnn_predictions(
                    primary_model, img_path, device
                )
                # Keep masks as 0/255 arrays
                pred_primary = masks_p255

            iou_pr, prec_pr, rec_pr = mean_iou_precision_recall(gt_masks, pred_primary)
            try:
                morph_pr = get_mean_regionprops(gt_masks, pred_primary, sam=False)
                per_image_morph_primary.append(morph_pr)
            except ValueError:
                morph_pr = {}
            per_image_iou_primary.append(iou_pr)
            per_image_prec_primary.append(prec_pr)
            per_image_rec_primary.append(rec_pr)

            # 2) DualSight
            # Compute instance‐level improvements and collect refined masks
            img_bgr = cv2.imread(img_path)
            per_instance_info = []
            pred_masks_ds = []

            # First, extract polygons + boxes from primary model
            if primary_type.lower() == "yolo":
                results = primary_model(Image.open(img_path))
                all_polygons = []
                all_bboxes = []
                for inst in results[0]:
                    pts = inst.masks.xy[0]
                    if len(pts) > 2:
                        all_polygons.append(pts)
                        all_bboxes.append(inst.boxes.xyxy[0].cpu().numpy())
            else:
                all_polygons, all_bboxes, masks_p255 = get_maskrcnn_predictions(
                    primary_model, img_path, device
                )

            for poly, bbox in zip(all_polygons, all_bboxes):
                # Build primary mask from polygon
                pm = polygon_to_binary_mask(poly, Image.open(img_path).height, Image.open(img_path).width)

                # Find best matching GT mask
                best_gm = None
                best_iou_pm = 0
                best_prec_pm = 0
                best_rec_pm = 0
                for gm in gt_masks:
                    if pm.shape != gm.shape:
                        continue
                    intersection = np.logical_and(gm, pm).sum()
                    union = np.logical_or(gm, pm).sum()
                    if union != 0 and intersection != 0:
                        iou_val = intersection / union
                        prec_val = intersection / pm.sum() if pm.sum() != 0 else 0
                        rec_val = intersection / gm.sum() if gm.sum() != 0 else 0
                        if iou_val > best_iou_pm:
                            best_iou_pm = iou_val
                            best_prec_pm = prec_val
                            best_rec_pm = rec_val
                            best_gm = gm

                # Compute regionprops for GT and primary (if match found)
                if best_gm is not None:
                    gt_props = binary_mask_to_regionprops_dict(best_gm)[0]
                    pred_primary_props = binary_mask_to_regionprops_dict(pm)[0]
                else:
                    gt_props = {k: 0 for k in binary_mask_to_regionprops_dict(pm)[0].keys()}
                    pred_primary_props = {k: 0 for k in binary_mask_to_regionprops_dict(pm)[0].keys()}

                # Expand bbox for SAM
                x1, y1, x2, y2 = bbox
                x1e, y1e, x2e, y2e = expand_bbox_within_border(
                    float(x1), float(y1), float(x2), float(y2),
                    Image.open(img_path).width,
                    Image.open(img_path).height,
                    expansion_rate=0.1,
                )
                bbox_exp = np.array([x1e, y1e, x2e, y2e])

                concave_poly = Polygon(poly)
                sampled_pts = generate_random_points_within_polygon(concave_poly, 50)
                optimal_pts = find_optimal_points(
                    sampled_pts, concave_poly,
                    num_result_points=3,
                    border_weight=2
                )
                inp_pts = np.array([[pt.x, pt.y] for pt in optimal_pts])
                inp_lbls = np.array([1] * len(optimal_pts))

                predictor.set_image(img_bgr)
                masks_sam, scores, logits = predictor.predict(
                    point_coords=inp_pts,
                    point_labels=inp_lbls,
                    box=bbox_exp[None, :],
                    multimask_output=True,
                )
                if masks_sam.shape[0] > 0:
                    rm = (masks_sam[0] == True).astype(np.uint8)
                else:
                    rm = np.zeros_like(pm)

                # Compute DS metrics against best_gm
                if best_gm is not None:
                    intersection = np.logical_and(best_gm, rm).sum()
                    union = np.logical_or(best_gm, rm).sum()
                    if union != 0 and intersection != 0:
                        best_iou_rm = intersection / union
                        best_prec_rm = intersection / rm.sum() if rm.sum() != 0 else 0
                        best_rec_rm = intersection / best_gm.sum() if best_gm.sum() != 0 else 0
                    else:
                        best_iou_rm, best_prec_rm, best_rec_rm = 0, 0, 0
                    pred_ds_props = binary_mask_to_regionprops_dict(rm)[0]
                else:
                    best_iou_rm, best_prec_rm, best_rec_rm = 0, 0, 0
                    pred_ds_props = {k: 0 for k in binary_mask_to_regionprops_dict(pm)[0].keys()}

                pred_masks_ds.append(rm)

                per_instance_info.append({
                    "Primary IoU": best_iou_pm,
                    "Primary Precision": best_prec_pm,
                    "Primary Recall": best_rec_pm,
                    "DualSight IoU": best_iou_rm,
                    "DualSight Precision": best_prec_rm,
                    "DualSight Recall": best_rec_rm,
                    "GT Morph": gt_props,
                    "Primary Morph": pred_primary_props,
                    "DS Morph": pred_ds_props
                })

            iou_ds, prec_ds, rec_ds = mean_iou_precision_recall(
                gt_masks, pred_masks_ds, sam=True
            )
            try:
                morph_ds = get_mean_regionprops(gt_masks, pred_masks_ds, sam=True)
                per_image_morph_ds.append(morph_ds)
            except ValueError:
                morph_ds = {}
            per_image_iou_ds.append(iou_ds)
            per_image_prec_ds.append(prec_ds)
            per_image_rec_ds.append(rec_ds)

            # Write per-image summary
            out_f.write(
                f"{base} (Primary): IoU={iou_pr:.4f}, Prec={prec_pr:.4f}, Rec={rec_pr:.4f}, Morph={morph_pr}\n"
            )
            out_f.write(
                f"{base} {prefix}: IoU={iou_ds:.4f}, Prec={prec_ds:.4f}, Rec={rec_ds:.4f}, Morph={morph_ds}\n"
            )
            out_f.write(f"{base} Per-instance details:\n")
            for idx, inst in enumerate(per_instance_info, 1):
                out_f.write(
                    f"  Instance {idx}:\n"
                    f"    Primary IoU={inst['Primary IoU']:.4f}, Primary Prec={inst['Primary Precision']:.4f}, Primary Rec={inst['Primary Recall']:.4f}\n"
                    f"    DualSight IoU={inst['DualSight IoU']:.4f}, DS Prec={inst['DualSight Precision']:.4f}, DS Rec={inst['DualSight Recall']:.4f}\n"
                    f"    GT Morph={inst['GT Morph']}\n"
                    f"    Primary Morph={inst['Primary Morph']}\n"
                    f"    DS Morph={inst['DS Morph']}\n"
                )
            out_f.write("\n")

        # Overall Primary summary
        mean_iou_pr_all = np.mean(per_image_iou_primary) if per_image_iou_primary else 0
        mean_prec_pr_all = np.mean(per_image_prec_primary) if per_image_prec_primary else 0
        mean_rec_pr_all = np.mean(per_image_rec_primary) if per_image_rec_primary else 0
        overall_morph_pr = calculate_morphological_summary_list(per_image_morph_primary)

        # Overall DualSight summary
        mean_iou_ds_all = np.mean(per_image_iou_ds) if per_image_iou_ds else 0
        mean_prec_ds_all = np.mean(per_image_prec_ds) if per_image_prec_ds else 0
        mean_rec_ds_all = np.mean(per_image_rec_ds) if per_image_rec_ds else 0
        overall_morph_ds = calculate_morphological_summary_list(per_image_morph_ds)

        out_f.write("===== Overall Summary Across All Images =====\n")
        out_f.write(
            f"Primary Overall: Mean IoU={mean_iou_pr_all:.4f}, "
            f"Mean Prec={mean_prec_pr_all:.4f}, Mean Rec={mean_rec_pr_all:.4f}, "
            f"Overall Morph={overall_morph_pr}\n"
        )
        out_f.write(
            f"{prefix} Overall: Mean IoU={mean_iou_ds_all:.4f}, "
            f"Mean Prec={mean_prec_ds_all:.4f}, Mean Rec={mean_rec_ds_all:.4f}, "
            f"Overall Morph={overall_morph_ds}\n"
        )

    print(f"{prefix} evaluation complete. Results saved to {output_txt}")

############################################################################
#   Main
############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Run Primary-model (YOLO or Mask R‑CNN) and DualSight evaluations on a directory of images."
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Path to the base directory containing an 'images/' subfolder and a labels folder.",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Path to the labels directory containing YOLO .txt files (one per image).",
    )
    parser.add_argument(
        "--primary-model",
        required=True,
        choices=["yolo", "maskrcnn"],
        help="Which primary model to use: 'yolo' or 'maskrcnn'.",
    )
    parser.add_argument(
        "--yolo-weights",
        default=None,
        help="Path to the YOLOv8 .pt weights file (e.g. best.pt). Required if --primary-model yolo.",
    )
    parser.add_argument(
        "--maskrcnn-weights",
        default=None,
        help="Path to the Mask R‑CNN .pth weights file. Required if --primary-model maskrcnn.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        required=True,
        help="Path to the SAM checkpoint .pth file (e.g. sam_vit_l_0b3195.pth).",
    )
    parser.add_argument(
        "--sam-type",
        default="vit_l",
        help="SAM model type (default: vit_l). Must match one of sam_model_registry keys.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for SAM and Mask R‑CNN (default: cuda).",
    )
    parser.add_argument(
        "--primary-output",
        default="primary_results.txt",
        help="Filename to save Primary-model results.",
    )
    parser.add_argument(
        "--dualsight-output",
        default="dualsight_results.txt",
        help="Filename to save DualSight results.",
    )
    args = parser.parse_args()

    primary_type = args.primary_model.lower()
    if primary_type == "yolo":
        if not args.yolo_weights:
            parser.error("--yolo-weights is required when --primary-model is 'yolo'.")
        primary_model = YOLO(args.yolo_weights)
        print("[INFO] YOLOv8 model loaded successfully.")
    else:  # maskrcnn
        if not args.maskrcnn_weights:
            parser.error("--maskrcnn-weights is required when --primary-model is 'maskrcnn'.")
        primary_model = load_maskrcnn_model(
            model_path=args.maskrcnn_weights,
            num_classes=2,
            device=args.device
        )
        print("[INFO] Mask R‑CNN model loaded successfully.")

    # 1) Run Primary-model evaluation
    run_yolo_evaluation(
        directory_path=args.dir,
        labels_dir=args.labels,
        primary_model=primary_model,
        primary_type=primary_type,
        device=args.device,
        output_txt=args.primary_output,
    )

    # 2) Run DualSight evaluation
    run_dual_sight_evaluation(
        directory_path=args.dir,
        labels_dir=args.labels,
        primary_model=primary_model,
        primary_type=primary_type,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_type,
        device=args.device,
        output_txt=args.dualsight_output,
    )

if __name__ == "__main__":
    main()

'''
Usage example (runs the maskrcnn branch):

python _modularEval4.py \
    --dir /home/sprice/DualSight/powder/test \
    --labels /home/sprice/DualSight/powder/test/labels \
    --primary-model maskrcnn \
    --maskrcnn-weights /home/sprice/DualSight/runs/train/maskrcnn.pth \
    --sam-checkpoint /home/sprice/DualSight/modelPerformance/sam_vit_l_0b3195.pth \
    --sam-type vit_l \
    --device cuda \
    --primary-output _results4/maskrcnn_evaluation.txt \
    --dualsight-output _results4/dualSight_maskrcnn_evaluation.txt

python _modularEval4.py \
  --dir /home/sprice/DualSight/powder/test \
  --labels /home/sprice/DualSight/powder/test/labels \
  --primary-model yolo \
  --yolo-weights runs/train/yolov8n-seg-train/weights/best.pt \
  --sam-checkpoint /home/sprice/DualSight/modelPerformance/sam_vit_l_0b3195.pth \
  --sam-type vit_l \
  --device cuda \
  --primary-output _results4/yolo_nano_evaluation.txt \
  --dualsight-output _results4/dualSight_nano_evaluation.txt

python _modularEval4.py \
  --dir /home/sprice/DualSight/powder/test \
  --labels /home/sprice/DualSight/powder/test/labels \
  --primary-model yolo \
  --yolo-weights runs/train/yolov8x-seg-train/weights/best.pt \
  --sam-checkpoint /home/sprice/DualSight/modelPerformance/sam_vit_l_0b3195.pth \
  --sam-type vit_l \
  --device cuda \
  --primary-output _results4/yolo_xlarge_evaluation.txt \
  --dualsight-output _results4/dualSight_xlarge_evaluation.txt








'''
