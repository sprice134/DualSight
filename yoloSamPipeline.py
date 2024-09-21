import matplotlib.pyplot as plt  # Import matplotlib for plotting
from PIL import Image, ImageDraw  # For loading images
import numpy as np
from ultralytics import YOLO
from samDemo import show_mask, show_points, show_box, mask_to_polygon, generate_random_points_within_polygon, point_to_polygon_distance, find_optimal_points, polygon_to_binary_mask, expand_bbox_within_border, fractal_dimension, apply_mask_to_image
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch
import cv2
from shapely.geometry import Polygon, Point
from itertools import combinations
import random
from skimage.draw import polygon
from skimage.measure import regionprops
from skimage.morphology import convex_hull_image
import math

'''NOTE TO SELF:
When there are detected particles that one of them contains a majority of the other one, 
say, 90% of one particle is contained within another one, delete the small one'''


# Load the trained model (replace 'yolov8n-seg.pt' with your model's weight file)
# model = YOLO('models/yolov8s_trained_weights.pt')  # Use the path to your trained weights
model = YOLO('/home/sprice/satellite_v2/aim2024/modelPerformance/models_s/train/weights/best.pt') 

# Copper
# image_path = '/home/sprice/satellite_v2/particleTest/demo.v5i.yolov8/test/images/Cu-Ni-Powder_250x_2_SE_V1_png.rf.eec5f31cbe6f51d8aa6a574a01f1883c.jpg'
# Big Ones
image_path = '/home/sprice/satellite_v2/particleTest/demo.v5i.yolov8/test/images/S02_03_SE1_1000X24_png.rf.61ceee7fe0a4f4ccabd61c1e71524baf.jpg'
# Demo Sample
# image_path = '/home/sprice/satellite_v2/particleTest/demo.v5i.yolov8/test/images/S05_02_SE1_300X59_png.rf.234bd1c35d0f3a635fd6164b651601f9.jpg'
# Real Big
# image_path = '/home/sprice/satellite_v2/particleTest/demo.v5i.yolov8/test/images/RHA_00-45_500X07_png.rf.2e24ff0e093484de86e43a21ef7e62cb.jpg'
# Agglomerations
# image_path = '/home/sprice/satellite_v2/particleTest/demo.v5i.yolov8/valid/images/RHA_00-45_500X11_png.rf.a1d468233106b607347416e301a98df1.jpg'
# Full test
# image_path = '/home/sprice/satellite_v2/aim2024/demo.v7i.yolov8/train/images/RHA_00-45_500X05_png.rf.559a7282fdc9edcc3c5ea9142294f4f9.jpg'
# image_path = '/home/sprice/satellite_v2/aim2024/demo.v7i.yolov8/test/images/OMG_Cu_1250_07_png.rf.ff417e92b36bba2bb91175ce144ff6ab.jpg'
# image_path = '/home/sprice/satellite_v2/particleTest/demo.v5i.yolov8/test/images/Rec-Cu-Ni-Powder_250x_10_SE_png.rf.2e0d707cd1875e646bed373fcca80ec2.jpg'


# image_path = '/home/sprice/satellite_v2/aim2024/demo.v7i.yolov8/test/images/S05_03_SE1_1000X65_png.rf.b1b275b60d329ab4fae1100a32dcfd4e.jpg'
image = Image.open(image_path)


sam_checkpoint = "/home/sprice/satellite_v2/particleTest/segmentAnythingTest/model/sam_vit_l_0b3195.pth"
model_type = "vit_l"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)


listOfPolygons = []
listOfBoxes = []
results = model(image)
count = 0
image = image.convert("RGBA")
for i in results[0]:
    polygon_points = i.masks.xy[0]  # This is the array of points
    if len(polygon_points) > 2:
        listOfPolygons.append(polygon_points)
        listOfBoxes.append(i.boxes.xyxy[0])
        '''To Save, uncommnet:'''
        # tmp = Image.new('RGBA', image.size)
        # draw = ImageDraw.Draw(tmp)
        # draw.polygon(polygon_points, fill=(30, 144, 255, 180))  # Semi-transparent fill
        # # Composite the original image with the temporary image using alpha blending
        # image_with_polygon = Image.alpha_composite(image, tmp)
        # # Save the result to a file
        # image_with_polygon.convert('RGB').save(f'outputImages/yoloMask/{count}.png')
        # count += 1

polygons = []
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
    op_x, op_y = zip(*optimal_points_xy)
    loop_image = cv2.imread(image_path)
    plt.figure(figsize=(10,8))
    plt.imshow(loop_image)
    # show_mask(mask, plt.gca())
    plt.axis('off')
    plt.savefig(f'outputImages/yoloPipeline/everyMask/{INDEX}_yoloMask.png')
    plt.plot(op_x, op_y, 'ro', markersize=5)
    plt.savefig(f'outputImages/yoloPipeline/everyMask/{INDEX}_yoloCentralPointTest.png')
    plt.close()


    loop_image = cv2.imread(image_path)
    # cv2.imwrite(f'outputImages/yoloPipeline/everyMask/{INDEX}_prePrediction.png', loop_image)


    predictor.set_image(loop_image)
    input_point = np.array(optimal_points_xy)
    input_label = np.array([1, 1, 1])

    # input_point = np.array([[600, 500], [500, 400], [600, 400]])
    # input_label = np.array([1, 1, 1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=box[None, :],
        multimask_output=True,
    )

    for i, (mask, score) in enumerate(zip(masks, scores)):
        if i == 0:
            print('Points found')
            plt.figure(figsize=(10,8))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            # show_box(box, plt.gca())
            # show_points(input_point, input_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.savefig(f'outputImages/yoloPipeline/everyMask/{INDEX}_output_mask_{i+1}_centralPoints.png')
            plt.close()
            masksList.append(mask)


            convertedPolygon = mask_to_polygon(mask)
            x_coords = [point[0] for point in convertedPolygon[0]]
            y_coords = [point[1] for point in convertedPolygon[0]]

            polygon_mask = np.zeros((int(max(y_coords)) + 1, int(max(x_coords)) + 1), dtype=np.uint8)
            rr, cc = polygon(y_coords, x_coords)
            polygon_mask[rr, cc] = 1
            props = regionprops(polygon_mask)
            polygons.append([[x, y] for x, y in zip(x_coords, y_coords)])


            # for prop in props:
            #     print("Area:", prop.area)
            #     print("Perimeter:", prop.perimeter)
            #     print("Eccentricity:", prop.eccentricity)
            #     print('Roundness:', (4 * prop.area) / (np.pi * (prop.major_axis_length ** 2)))
            #     print("Equivalent Diameter:", prop.equivalent_diameter)
            #     print('Feret Diameter:', prop.feret_diameter_max)
            #     print("Centroid Location:", prop.centroid)
            #     print("Major Axis:", prop.axis_major_length)
            #     print("Minor Axis:", prop.axis_minor_length)
            #     print('Elongation:', prop.minor_axis_length / prop.major_axis_length)
            #     print('Crofton Perimeter:', prop.perimeter_crofton)
            #     print('Solidity:', prop.solidity)
            #     print('Convex Area:',prop.area_convex)
            #     print('Extent:', prop.extent)
            #     # print('Convexity:', prop.perimeter / np.sum(np.sqrt(convex_hull_image(prop.coords))))
            #     print('Aspect Ratio:', prop.major_axis_length / prop.minor_axis_length)
            #     print('Fractal Dimension:', fractal_dimension(prop.image))
            #     print('Form Factor:', 4 * math.pi * prop.area / (prop.perimeter ** 2))
            #     print('Rectangularity:', prop.area / prop.area_bbox)
            #     print('Compactness:', (prop.perimeter ** 2) / (4 * math.pi * prop.area))
            #     print('Shape Factor:', math.sqrt(prop.area) / prop.perimeter)
            #     print("Convex Area Ratio:", prop.area / prop.convex_area)
            #     x_center, y_center = (prop.bbox[0] + prop.bbox[2]) / 2.0, (prop.bbox[1] + prop.bbox[3]) / 2.0
            #     centroidToCenter = math.sqrt((prop.centroid[0] - x_center) ** 2 + (prop.centroid[1] - y_center) ** 2)
            #     print("Centroid To Center:", centroidToCenter)
            #     print('-'*30)






composite_mask = np.zeros_like(masksList[0])
# Perform a logical OR operation over all masks
for mask in masksList:
    composite_mask = np.logical_or(composite_mask, mask)

# Convert the boolean array to an integer array (optional, for image saving)
composite_mask_for_image = composite_mask.astype(np.uint8) * 255
composite_image = Image.fromarray(composite_mask_for_image)
composite_image.save("outputImages/yoloPipeline/everyMask/samMask.png")
color_with_alpha = np.array([255, 0, 0, 0.5])#Transparent Red
result_image = apply_mask_to_image(image_path, composite_mask, color_with_alpha)
result_image.save("outputImages/yoloPipeline/everyMask/samImage.png")




image2 = Image.open(image_path).convert('RGBA')

yoloSegs = 0
for i in results[0]:
    polygon_points = i.masks.xy[0]
    if len(polygon_points) != 0:
        flat_polygon_points = [coord for pair in polygon_points for coord in pair]
        mask_img = Image.new('L', image2.size, 0)
        mask_draw = ImageDraw.Draw(mask_img)
        mask_draw.polygon(flat_polygon_points, outline=1, fill=255)
        mask_rgba = Image.new('RGBA', mask_img.size)
        red_mask = Image.new('RGBA', mask_img.size, (0, 0, 255, 128))
        mask_rgba.paste(red_mask, (0, 0), mask_img)
        image2 = Image.alpha_composite(image2, mask_rgba)
        yoloSegs += 1
final_image = image2.convert('RGB')
final_image.save('outputImages/yoloPipeline/everyMask/yoloImage.png')