import torch
from PIL import Image
import cv2
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize

# YOLOv5 setup
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
target_classes = ['person', 'couch']

# ZoeDepth setup
conf = get_config("zoedepth", "infer")
model_zoe_n = build_model(conf)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

def run_yolo_classification(image_cv2, img_pil):
    results = model(img_pil)
    detected_objects = results.xyxy[0].numpy()
    
    # Annotate image with bounding boxes and labels
    for obj in detected_objects:
        x1, y1, x2, y2, conf, class_id = map(int, obj[:6])
        label = results.names[class_id]
        
        if label in target_classes:
            confidence = obj[4]
            cv2.rectangle(image_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f'{label} {confidence:.2f}'
            cv2.putText(image_cv2, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return detected_objects, image_cv2, results.names

# Function to run depth estimation
def run_depth_estimation(image_pil):
    depth = zoe.infer_pil(image_pil)
    colored_depth = colorize(depth)
    return colored_depth, depth

# Combined function to process both depth and object detection
def process_image(image_path):
    # Load the image
    image_pil = Image.open(image_path).convert("RGB")
    image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # Run object detection
    detected_objects, detection_result, class_names = run_yolo_classification(image_cv2.copy(), image_pil)
    
    # Run depth estimation
    depth_color, depth_map = run_depth_estimation(image_pil)

    # Extract depth for each corner of each bounding box
    for obj in detected_objects:
        x1, y1, x2, y2, _, class_id = map(int, obj[:6])
        label = class_names[class_id]

        if label in target_classes:
            
            # Ensure the coordinates are within the image bounds
            h, w = depth_map.shape[:2]
            x1, y1 = min(x1, w - 1), min(y1, h - 1)
            x2, y2 = min(x2, w - 1), min(y2, h - 1)
            
            # Get depth values at the corners
            top_left_depth = depth_map[y1, x1]
            top_right_depth = depth_map[y1, x2]
            bottom_left_depth = depth_map[y2, x1]
            bottom_right_depth = depth_map[y2, x2]
            print(f"Label: {label}")
            print(f"Bounding Box: ({x1}, {y1}), ({x2}, {y2})")
            print(f"Depth at top-left: {top_left_depth}")
            print(f"Depth at top-right: {top_right_depth}")
            print(f"Depth at bottom-left: {bottom_left_depth}")
            print(f"Depth at bottom-right: {bottom_right_depth}")
    
    # Display results
    cv2.imshow("YOLOv5 Object Detection", detection_result)  # OpenCV display for object detection
    cv2.imshow("Depth Estimation", cv2.cvtColor(np.array(Image.fromarray(depth_color)), cv2.COLOR_RGB2BGR))  # OpenCV display for depth
    
    # Wait for a key press to close display windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save outputs if needed
    cv2.imwrite("output_with_bboxes.png", detection_result)  # Save object detection result
    Image.fromarray(depth_result).save("output_depth.png")    # Save depth estimation result

# Path to the image file
image_path = 'assets/rgb.png'
process_image(image_path)
