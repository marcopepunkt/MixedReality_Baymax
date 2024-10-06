import torch
from PIL import Image
import cv2
import numpy as np

# Load YOLOv5 model (pre-trained on COCO)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Define the classes you want to detect (based on YOLOv5 COCO dataset)
# For example, 'person' and 'car'
target_classes = ['person', 'couch']

# Function to run object detection
def run_yolo_classification(image_path):
    # Load image
    img = Image.open(image_path)
    
    # Run YOLO model on image
    results = model(img)
    
    # Results: bounding boxes, class labels, and confidence scores
    detected_objects = results.xyxy[0].numpy()  # Extract bounding boxes and info as NumPy array
    
    # Convert image for displaying results with OpenCV
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Annotate image with bounding boxes and labels
    for obj in detected_objects:
        x1, y1, x2, y2, conf, class_id = map(int, obj[:6])
        label = results.names[class_id]
        
        # Only consider objects in the target_classes list
        if label in target_classes:
            confidence = obj[4]
            
            # Draw bounding box
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label and confidence score
            label_text = f'{label} {confidence:.2f}'
            cv2.putText(img_cv2, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show the image with detections
    cv2.imshow("YOLOv5 Object Detection", img_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the image file
image_path = 'rgb.png'

# Run YOLO object classification with filtered classes
run_yolo_classification(image_path)
