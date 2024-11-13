import collections
import time
from pathlib import Path
import json

import cv2
import numpy as np
from IPython import display
import openvino as ov
from flask import jsonify
from openvino.tools import mo

from typing import List, Tuple

import threading

import os

classes = [
    "background",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "street sign",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "hat",
    "backpack",
    "umbrella",
    "shoe",
    "eye glasses",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "plate",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "mirror",
    "dining table",
    "window",
    "desk",
    "toilet",
    "door",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "blender",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "hair brush",
]

classes_with_priority = {
    "bus": 1,
    "car": 1,
    "motorcycle": 1,
    "pedestrian": 1,
    "stop_sign": 2,
    "traffic_light": 2,
    "fire_hydrant": 3,
    "person": 1,
    "train": 1,
    "truck": 1,
}

for cls in classes:
    # If the class is not already in the mapping, set it to 3
    classes_with_priority.setdefault(cls, 3)


class Object:
    def __init__(self, label, p_center: Tuple[int, int], depth: float, box: Tuple[int, int, float, float]):
        self.label = label
        self.center = p_center  # (x, y)
        self.depth = depth
        self.box = box  # (x, y, w, h)

        self.world_pose = None
    
    def get_box_corners(self) -> List[Tuple[float, float]]:
        x, y, w, h = self.box
        
        # Calculate corners
        bottom_left = (x , y)
        bottom_right = (x + w, y )
        top_left = (x, y + h)
        top_right = (x + w, y + h)
        
        return [bottom_left, bottom_right, top_right, top_left]

# def objects_to_json(objects: List[Object]) -> str:
#     data = []
#     for obj in objects:
#         obj_data = {
#             "label": int(obj.label),
#             "center": obj.center,
#             "depth": obj.depth,
#             "world_pose": obj.world_pose,
#         }
#         data.append(obj_data)
#     return json.dumps(data, indent=4)

def objects_to_json(objects: List[Object], image_description):
    if len(objects) == 0: # no objects detected, just send description to unity
        if image_description is None:
            return json.dumps([])

        data = [{
            'class_name': "none",
            'priority': -1,
            'x': -1,
            'y': -1,
            'z': -1,
            'depth': -1,
            'description': image_description,
        }]
        return jsonify(data)

    data = []
    for obj in objects:
        obj_data = {
            'class_name': classes[obj.label],
            'priority': int(classes_with_priority.get(classes[obj.label])),
            'x': float(obj.world_pose[0]),
            'y': float(obj.world_pose[1]),
            'z': float(obj.world_pose[2]),
            'depth': float(obj.depth),
            'description': image_description,
        }
        data.append(obj_data)
    data.sort(key=lambda t: (t['priority'], t['depth']))
    return jsonify(data)


def get_centers(objects: List[Object]) -> List[Tuple[int, int]]:
    """
    Returns a list of center pixels for a given list of Object instances.

    Parameters:
        objects (List[Object]): List of Object instances.

    Returns:
        List[Tuple[int, int]]: List of center pixel coordinates (x, y).
    """
    return [obj.center for obj in objects]


def remap_bounding_boxes(boxes,frame, target_map):

    mapped_boxes = []
    for label, score, box in boxes:
        x,y,w,h = box

        mask = np.zeros((frame.shape[0],frame.shape[1]))
        mask[y:y+h,x:x+w] = 1.0
        remapped_mask = cv2.remap(mask,target_map[...,0],target_map[...,1], cv2.INTER_LINEAR)

        y_indices, x_indices = np.where(remapped_mask > 0)

        if len(x_indices) > 0 and len(y_indices) > 0:
            xmin = np.min(x_indices)
            ymin = np.min(y_indices)
            w = np.max(x_indices) - xmin
            h = np.max(y_indices) - ymin

            mapped_boxes.append((label,score,(xmin,ymin,w,h)))
    
    return mapped_boxes

def estimate_box_poses(metric_depth, boxes, bin_width=1., stride=2 , cutoff_num=3):
    poses = []

    for label, _, box in boxes:
        x, y, w, h = box
        cx = x + w//2
        cy = y + h//2

        # Sample region of interest from depth map with downsampling, flatten, and filter out NaN/zero
        roi_depths = metric_depth[y:y + h:stride, x:x + w:stride].flatten()
        roi_depths = roi_depths[(~np.isnan(roi_depths)) & (roi_depths > 0)]
        
        if len(roi_depths) > 0:
            # Compute histogram and find top bins
            histogram, bin_edges = np.histogram(roi_depths, bins=np.arange(roi_depths.min(), roi_depths.max() + bin_width, bin_width))
            if len(histogram) == 0:
                estimated_depth = np.nan
                continue
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            top_bins_indices = np.argsort(histogram)[-cutoff_num:]
            
            # Find closest bin among the top bins
            closest_top_bin_index = top_bins_indices[np.argmin(bin_centers[top_bins_indices])]
            bin_start = bin_edges[closest_top_bin_index]
            bin_end = bin_edges[closest_top_bin_index + 1]

            # Calculate median depth within the selected bin range
            in_bin_depth_values = roi_depths[(roi_depths >= bin_start) & (roi_depths < bin_end)]
            estimated_depth = np.median(in_bin_depth_values) if len(in_bin_depth_values) > 0 else np.nan
        else:
            estimated_depth = np.nan
        
        poses.append(Object(label=label,p_center=(cx,cy),depth=estimated_depth,box=box))

    return poses


def process_results(frame, results, thresh=0.6):
    # The size of the original frame.
    h, w = frame.shape[:2]
    # The 'results' variable is a [1, 1, 100, 7] tensor.
    results = results.squeeze()
    boxes = []
    labels = []
    scores = []
    for _, label, score, xmin, ymin, xmax, ymax in results:
        # Create a box with pixels coordinates from the box with normalized coordinates [0,1].
        boxes.append(tuple(map(int, (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h))))
        labels.append(int(label))
        scores.append(float(score))

    # Apply non-maximum suppression to get rid of many overlapping entities.
    # See https://paperswithcode.com/method/non-maximum-suppression
    # This algorithm returns indices of objects to keep.
    indices = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores, score_threshold=thresh, nms_threshold=0.6)

    # If there are no boxes.
    if len(indices) == 0:
        return []

    # Filter detected objects.
    return [(labels[idx], scores[idx], boxes[idx]) for idx in indices.flatten()]
