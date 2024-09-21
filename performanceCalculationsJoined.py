import matplotlib.pyplot as plt  # Import matplotlib for plotting
from PIL import Image, ImageDraw  # For loading images
import numpy as np
from ultralytics import YOLO
from samDemo import show_mask, show_points, show_box, mask_to_polygon, generate_random_points_within_polygon, point_to_polygon_distance, find_optimal_points, polygon_to_binary_mask, expand_bbox_within_border, fractal_dimension, apply_mask_to_image
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch
import cv2
from shapely.geometry import Polygon
from skimage.draw import polygon
from skimage.measure import regionprops
import os
from shapely.geometry import Polygon
from skimage.measure import regionprops
from sklearn.metrics import r2_score


print(torch.cuda.is_available())

def get_jpg_files(directory):
    jpg_files = []
    for file in os.listdir(directory):
        if file.endswith(".jpg"):
            jpg_files.append(os.path.join(directory, file))
    return jpg_files

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def label_to_mask(label, image_shape):
    """
    Convert a YOLOv8 segmentation label line to a binary mask.

    Args:
    label (str): The label line with class index and normalized coordinates, e.g.,
                 '0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2'
    image_shape (tuple): The shape of the image (height, width) on which the mask will be applied.

    Returns:
    np.array: A binary mask of the same size as the image, where pixels inside the polygon are 1, others are 0.
    """
    # Split the label string by spaces and convert to float
    parts = list(map(float, label.split()))
    
    # The first part is the class index, ignore it for mask creation
    coordinates = parts[1:]
    
    # Separate x and y coordinates, scaling them to the image dimensions
    x = np.array(coordinates[0::2]) * image_shape[1]
    y = np.array(coordinates[1::2]) * image_shape[0]
    
    # Clamping coordinates to the maximum allowable index
    x = np.clip(x, 0, image_shape[1] - 1)
    y = np.clip(y, 0, image_shape[0] - 1)
    
    # Generate polygon mask
    rr, cc = polygon(y, x)
    
    # Create an empty mask and fill in the polygon
    mask = np.zeros(image_shape, dtype=np.uint8)
    mask[rr, cc] = 1
    
    return mask

def convert_to_binary_mask(mask):
    """
    Convert a mask with 0s and 255s to a binary mask with 0s and 1s.
    
    Parameters:
    mask (np.ndarray): 2D numpy array with 0s and 255s
    
    Returns:
    np.ndarray: Binary mask with 0s and 1s
    """
    binary_mask = np.where(mask == 255, 1, 0)
    return binary_mask


def binary_mask_to_regionprops_dict(binary_mask):
    
    # Calculate region properties
    props = regionprops(binary_mask)
    
    # List of selected properties
    selected_props = [
        'area', 'area_convex', 'major_axis_length', 'minor_axis_length', 'eccentricity',
        'equivalent_diameter', 'euler_number', 'extent', 'feret_diameter_max', 'perimeter', 'solidity'
    ]
    
    # Convert the selected properties to a dictionary
    props_dicts = []
    for region in props:
        region_dict = {}
        for prop in selected_props:
            try:
                region_dict[prop] = getattr(region, prop)
            except AttributeError:
                region_dict[prop] = None
        props_dicts.append(region_dict)
    
    return props_dicts





def mean_iou(gt_masks, pred_masks):
    """
    Calculate the mean Intersection over Union (IoU) between ground truth and predicted masks.

    Args:
    gt_masks (list of np.array): List of ground truth binary masks.
    pred_masks (list of np.array): List of predicted binary masks.

    Returns:
    float: Mean IoU score across all mask pairs.
    """
    iou_scores = []

    for pred_mask in pred_masks:
        pred_iou_scores = []
        for gt_mask in gt_masks:
            # Ensure the masks are the same size
            if pred_mask.shape != gt_mask.shape:
                raise ValueError("Mismatched dimensions between ground truth and predicted masks.")
            
            # Calculate intersection and union
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
    

            # Calculate IoU and handle case where division by zero might occur
            if union != 0 and intersection != 0:
                pred_iou_scores.append(intersection / union)


        # If no valid IoU scores, append 0
        if len(pred_iou_scores) == 0:
            iou_scores.append(0)
        else:
            # Take the mean of remaining IoU scores
            mean_iou = np.mean(pred_iou_scores)
            iou_scores.append(mean_iou)
    # Calculate mean IoU across all predicted masks
    mean_iou_value = np.mean(iou_scores)
    return mean_iou_value



import numpy as np

def mean_iou_precision_recall(gt_masks, pred_masks, sam=False):
    """
    Calculate the mean Intersection over Union (IoU), mean precision, and mean recall 
    between a composite ground truth mask and a composite predicted mask.

    Args:
    gt_masks (list of np.array): List of ground truth binary masks.
    pred_masks (list of np.array): List of predicted binary masks.
    sam (bool): Flag indicating if SAM model is being used, to skip binary conversion.

    Returns:
    tuple: Mean IoU score, mean precision, mean recall.
    """
    
    # Create composite masks for both gt_masks and pred_masks
    composite_gt_mask = np.zeros_like(gt_masks[0], dtype=bool)
    composite_pred_mask = np.zeros_like(pred_masks[0], dtype=bool)
    
    # Combine all gt_masks into a single composite mask
    for gt_mask in gt_masks:
        composite_gt_mask = np.logical_or(composite_gt_mask, gt_mask)
    
    # Combine all pred_masks into a single composite mask
    for pred_mask in pred_masks:
        if not sam:
            pred_mask = convert_to_binary_mask(pred_mask)
        composite_pred_mask = np.logical_or(composite_pred_mask, pred_mask)
    
    # Calculate intersection and union for the composite masks
    intersection = np.logical_and(composite_gt_mask, composite_pred_mask).sum()
    union = np.logical_or(composite_gt_mask, composite_pred_mask).sum()
    
    # Calculate IoU, precision, and recall
    if union == 0:
        iou = 0.0
    else:
        iou = intersection / union

    precision = intersection / composite_pred_mask.sum() if composite_pred_mask.sum() > 0 else 0.0
    recall = intersection / composite_gt_mask.sum() if composite_gt_mask.sum() > 0 else 0.0
    
    return iou, precision, recall



def calculate_morphologicalMetricSummary(gt_list, pred_list):
    if len(gt_list) != len(pred_list):
        raise ValueError("The ground truth list and prediction list must have the same number of instances.")
    
    # Initialize dictionaries to store squared errors, ground truth, and predictions
    squared_errors = {key: [] for key in gt_list[0].keys()}
    gt_values = {key: [] for key in gt_list[0].keys()}
    pred_values = {key: [] for key in gt_list[0].keys()}
    
    # Iterate through each instance in the lists
    for gt, pred in zip(gt_list, pred_list):
        for key in gt.keys():
            squared_error = (gt[key] - pred[key]) ** 2
            squared_errors[key].append(squared_error)
            gt_values[key].append(gt[key])
            pred_values[key].append(pred[key])
    

    # Calculate RMSE, mean of predictions, mean of ground truth, and R² for each metric
    results = {}
    for key in squared_errors.keys():
        mean_squared_error = np.mean(squared_errors[key])
        rmse = np.sqrt(mean_squared_error)
        
        mean_pred = np.mean(pred_values[key])
        mean_gt = np.mean(gt_values[key])
        
        r2 = r2_score(gt_values[key], pred_values[key])
        
        results[key] = {
            'RMSE': rmse, 
            'Mean of Predictions': mean_pred, 
            'Mean of Ground Truth': mean_gt, 
            'R2': r2
        }

    return results



def get_mean_regionprops(gt_masks, pred_masks, sam=False):
    """
    Calculate the mean Intersection over Union (IoU), mean precision, and mean recall between ground truth and predicted masks.

    Args:
    gt_masks (list of np.array): List of ground truth binary masks.
    pred_masks (list of np.array): List of predicted binary masks.

    Returns:
    tuple: Mean IoU score, mean precision, mean recall.
    """
    iou_scores = []
    pred_metrics = []
    gt_metrics = []
    for pred_mask in pred_masks:
        temp_iou_scores = []
        temp_pred_metrics = []
        temp_gt_metrics = []
        if sam == False:
            pred_mask = convert_to_binary_mask(pred_mask)
        
        for gt_mask in gt_masks:
            # Ensure the masks are the same size
            if pred_mask.shape != gt_mask.shape:
                raise ValueError("Mismatched dimensions between ground truth and predicted masks.")

            # Calculate intersection and union
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            

            # Calculate IoU and handle case where division by zero might occur
            if union != 0 and intersection != 0:
                temp_iou_scores.append(intersection / union)
                temp_pred_metrics.append(binary_mask_to_regionprops_dict(pred_mask))
                temp_gt_metrics.append(binary_mask_to_regionprops_dict(gt_mask))

                


        # If no valid IoU scores, append 0
        if len(temp_iou_scores) == 0:
            iou_scores.append(0)
            pred_metrics.append({})
            gt_metrics.append({})
        

        #Change this to calculate the index with the highest IOU score
        #and then return the IOU, precision, and recall at that index
        else:
            bestIOU = max(temp_iou_scores)
            index = temp_iou_scores.index(bestIOU)
            
            iou_scores.append(temp_iou_scores[index])
            pred_metrics.append(temp_pred_metrics[index][0])
            gt_metrics.append(temp_gt_metrics[index][0])

    # Calculate mean IoU, mean precision, and mean recall across all predicted masks
    
    return calculate_morphologicalMetricSummary(gt_metrics, pred_metrics)


def create_composite_mask(masks):
    """
    Create a composite mask from a list of binary masks.

    Args:
    masks (list of np.array): List of binary masks (2D numpy arrays).

    Returns:
    np.array: A composite binary mask where any pixel that is 1 in any of the input masks is 1 in the composite mask.
    """
    # Initialize a composite mask with zeros, of the same size as the first mask in the list
    if not masks:
        raise ValueError("The list of masks is empty.")
    
    composite_mask = np.zeros_like(masks[0])

    # Iterate over all masks and apply a logical OR to combine them
    for mask in masks:
        composite_mask = np.logical_or(composite_mask, mask)

    return composite_mask.astype(int)  # Convert boolean array to int (0s and 1s)

def get_ground_truth(fileName):
    image_name = fileName.split('/')[-1][:-4]
    image_width, image_height = get_image_dimensions(fileName)
    print(image_width, image_height)
    label_path = 'labels/' + image_name + ".txt"
    with open(directory_path + label_path, 'r') as file:
        lines = file.readlines()
        stripped_lines = [line.strip() for line in lines]
        return [label_to_mask(line, (image_height, image_width)) for line in stripped_lines]
    

def get_prediction_masks(fileName, model):
    predictedMasks = []
    image = Image.open(fileName)
    results = model(image)
    for i in results[0]:
        polygon_points = i.masks.xy[0]  # This is the array of points
        if len(polygon_points) > 2:
            mask = polygon_to_binary_mask(polygon_points, image.height, image.width)
            predictedMasks.append(mask)
    return predictedMasks



def get_dualSight_masks(fileName, yoloModel, samPredictor):
    listOfPolygons = []
    listOfBoxes = []
    image = Image.open(fileName)
    results = yoloModel(image)
    for i in results[0]:
        polygon_points = i.masks.xy[0]  # This is the array of points
        if len(polygon_points) > 2:
            listOfPolygons.append(polygon_points)
            listOfBoxes.append(i.boxes.xyxy[0])
    
    masksList = []

    for INDEX in range(len(listOfPolygons)):
        poly = listOfPolygons[INDEX]
        box = listOfBoxes[INDEX]
        box = box.cpu().numpy()
        # print(image)
        box = np.array(expand_bbox_within_border(box[0], box[1], box[2], box[3], image.width, image.height, expansion_rate = 0.1))
        mask = polygon_to_binary_mask(poly, image.height, image.width)
        concave_polygon = Polygon(poly)
        sampled_points = generate_random_points_within_polygon(concave_polygon, 50)
        optimal_points = find_optimal_points(sampled_points, concave_polygon, num_result_points=3, border_weight=2)
        optimal_points_xy = [[point.x, point.y] for point in optimal_points]
        loop_image = cv2.imread(image_path)


        predictor.set_image(loop_image)
        input_point = np.array(optimal_points_xy)
        input_label = np.array([1, 1, 1])


        masks, scores, logits = samPredictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=box[None, :],
            multimask_output=True,
        )

        for i, (mask, score) in enumerate(zip(masks, scores)):
            if i == 0:
                masksList.append(mask)
    return replace_true_with_one(masksList)
     

def summarize_mask(mask):
    summary = {
        'mean': np.mean(mask),
        'median': np.median(mask),
        'maximum': np.max(mask)
    }
    return summary

def replace_true_with_one(arrays):
    """
    Replace all True values with 1 in a list of 2D arrays.

    Parameters:
    arrays (list of 2D arrays): The input list of 2D arrays.

    Returns:
    list of 2D arrays: A list of 2D arrays with True values replaced by 1.
    """
    modified_arrays = []
    for array in arrays:
        modified_array = np.where(array == True, 1, array)
        modified_arrays.append(modified_array)
    return modified_arrays

# image_path = '/home/sprice/satellite_v2/aim2024/demo.v7i.yolov8/test/images/Cu-Ni-Powder_250x_10_SE_png.rf.cd93ec4589ad8f4e412cb1ec0e805016.jpg'
# gt_masks = get_ground_truth(image_path)
# pred_masks = get_prediction_masks(image_path, model)
# scores = mean_iou_precision_recall(gt_masks, pred_masks)
# print(scores)


# composite_mask = create_composite_mask(gt_masks)
# plt.imshow(composite_mask, cmap='gray')
# plt.title('Binary Mask from YOLOv8 Label')
# plt.axis('off')
# plt.savefig('demo.png')

# composite_mask = create_composite_mask(pred_masks)
# plt.imshow(composite_mask, cmap='gray')
# plt.title('Binary Mask from YOLOv8 Label')
# plt.axis('off')
# plt.savefig('demoYolo.png')


print('Testing All')
model = YOLO('/home/sprice/satellite_v2/aim2024/modelPerformance/models_n/train/weights/best.pt') 

sam_checkpoint = "/home/sprice/satellite_v2/particleTest/segmentAnythingTest/model/sam_vit_l_0b3195.pth"
model_type = "vit_l"
device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)


directory_path = "/home/sprice/satellite_v2/aim2024/demo.v7i.yolov8/test/"
jpg_files = get_jpg_files(directory_path + 'images')
print("JPG Files in the directory:")
iou, prec, rec = [], [], []
for image_path in jpg_files:
    print(f'scoreComparisonMasks/{image_path.split("/")[-1].split("_png")[0]}_gt.png')
    gt_masks = get_ground_truth(image_path)
    
    '''YOLO'''
    # pred_masks = get_prediction_masks(image_path, model)    
    # metrics = get_mean_regionprops(gt_masks, pred_masks)
    # with open('rmse_outputs/yolo_nano_rmse.txt', 'a') as file:
    #     # file.write(f'{image_path.split("/")[-1].split("_png")[0]}_dualSight: {scores[0]}, {scores[1]}, {scores[2]}' + '\n')
    #     file.write(f'{image_path.split("/")[-1].split("_png")[0]}: {metrics}' + '\n')
    '''
    pred_masks = get_prediction_masks(image_path, model)
    scores = mean_iou_precision_recall(gt_masks, pred_masks)

    # composite_mask = create_composite_mask(pred_masks)
    # plt.imshow(composite_mask, cmap='gray')
    # plt.title('Binary Mask from YOLOv8 Label')
    # plt.axis('off')
    # plt.savefig(f'scoreComparisonMasks/{image_path.split("/")[-1].split("_png")[0]}_yolo.png')
    print(scores)
    iou.append(scores[0])
    prec.append(scores[1])
    rec.append(scores[2])
    with open('yolo_nanoOverlapping.txt', 'a') as file:
        # file.write(f'{image_path.split("/")[-1].split("_png")[0]}_dualSight: {scores[0]}, {scores[1]}, {scores[2]}' + '\n')
        file.write(f'{image_path.split("/")[-1].split("_png")[0]}: {scores[0]}, {scores[1]}, {scores[2]}' + '\n')
    '''
    '''DUalSIght'''
    
    # pred_masks = get_dualSight_masks(image_path, model, predictor)
    # metrics = get_mean_regionprops(gt_masks, pred_masks, sam=True)
    # with open('rmse_outputs/dualSight_rmse.txt', 'a') as file:
    #     # file.write(f'{image_path.split("/")[-1].split("_png")[0]}_dualSight: {scores[0]}, {scores[1]}, {scores[2]}' + '\n')
    #     file.write(f'{image_path.split("/")[-1].split("_png")[0]}: {metrics}' + '\n')
    
    pred_masks = get_dualSight_masks(image_path, model, predictor)
    scores = mean_iou_precision_recall(gt_masks, pred_masks, sam=True)

    # composite_mask = create_composite_mask(gt_masks)
    # plt.imshow(composite_mask, cmap='gray')
    # plt.title('Binary Mask from YOLOv8 Label')
    # plt.axis('off')
    # plt.savefig(f'scoreComparisonMasks/{image_path.split("/")[-1].split("_png")[0]}_gt.png')

    # composite_mask = create_composite_mask(pred_masks)
    # plt.imshow(composite_mask, cmap='gray')
    # plt.title('Binary Mask from YOLOv8 Label')
    # plt.axis('off')
    # plt.savefig(f'scoreComparisonMasks/{image_path.split("/")[-1].split("_png")[0]}_dualSight.png')
    
    print(scores)
    iou.append(scores[0])
    prec.append(scores[1])
    rec.append(scores[2])
    with open('ds_nanoOverlapping_v2.txt', 'a') as file:
        # file.write(f'{image_path.split("/")[-1].split("_png")[0]}_dualSight: {scores[0]}, {scores[1]}, {scores[2]}' + '\n')
        file.write(f'{image_path.split("/")[-1].split("_png")[0]}: {scores[0]}, {scores[1]}, {scores[2]}' + '\n')
    
# print(np.mean(iou), np.mean(prec), np.mean(rec))
# with open('yolo_nanoOverlapping.txt', 'a') as file:
#     file.write(f'Final scores: {np.mean(iou)}, {np.mean(prec)}, {np.mean(rec)}' + '\n')

print(np.mean(iou), np.mean(prec), np.mean(rec))
with open('ds_nanoOverlapping_v2.txt', 'a') as file:
    file.write(f'Final scores: {np.mean(iou)}, {np.mean(prec)}, {np.mean(rec)}' + '\n')