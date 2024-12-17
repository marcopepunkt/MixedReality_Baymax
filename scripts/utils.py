import json

import cv2
import numpy as np
from flask import jsonify

from typing import List, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from sklearn.cluster import DBSCAN
import open3d as o3d

# Obstacle Buffer settings
MAX_RADIUS = 5
MAX_OBJECTS = 10
MAX_CANDIDATE_OBJECTS = 30
MAX_SIMILARITY_DISTANCE = 0.25
MIN_COUNT = 1
MAX_TIME = 2  # seconds

# Heading buffer settings
MAX_HEADING_COUNT = 5


# Most of these are useless
classes = [
    "background",
    "person",
    # Vehicles
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    # Signs and Street Objects
    "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench",
    # Animals
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    # Accessories
    "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase",
    # Sports and Recreational
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket",
    # Kitchen Items
    "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    # Food
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake",
    # Furniture and Household
    "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk",
    "toilet", "door",
    # Electronics
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "blender",
    # Miscellaneous
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"
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
        self.timestamp = 0.0
        self.radius = 0.0
    
    def get_box_corners(self) -> List[Tuple[float, float]]:
        x, y, w, h = self.box
        
        # Calculate corners
        bottom_left = (x , y)
        bottom_right = (x + w, y )
        top_left = (x, y + h)
        top_right = (x + w, y + h)
        
        return [bottom_left, bottom_right, top_right, top_left]
    
    def dist(self,pos):

        if self.world_pose is not None and pos is not None:
            return np.sqrt((self.world_pose[0]-pos[0])**2+(self.world_pose[2]-pos[2])**2) 
            #return np.sqrt((self.world_pose[0]-pos[0])**2+(self.world_pose[1]-pos[1])**2+(self.world_pose[2]-pos[2])**2)
        else:
            return None
        
class Object_Buffer:

    class Candidate:
        def __init__(self,obj):
            self.object = obj
            self.count = 1
 
    def __init__(self):
        self.buffer: List[Object] = []
        self.candidates: List[Object_Buffer.Candidate] = []

    def update(self, objects, timestamp, head_pos):

        # Process incoming objects
        new_candidates = []
        for obj_new in objects:
            matched = False
            for candidate in self.candidates:
                # Calculate distance to existing candidate
                d = candidate.object.dist(obj_new.world_pose)
                if d is not None and d < MAX_SIMILARITY_DISTANCE:
                    # Update the candidate's position and timestamp
                    candidate.object.world_pose = (
                        candidate.object.world_pose * candidate.count + obj_new.world_pose
                    ) / (candidate.count + 1)
                    candidate.count += 1
                    candidate.object.timestamp = obj_new.timestamp
                    candidate.object.depth = np.linalg.norm(candidate.object.world_pose - head_pos)
                    matched = True
                    break
            if not matched:
                # Add new object as a candidate
                new_candidates.append(self.Candidate(obj_new))
        
        # Add new candidates to the list
        self.candidates.extend(new_candidates)
        
        # Update candidates
        candidates_ = []
        filtered_candidates = []
        for candidate in self.candidates:
            if timestamp - candidate.object.timestamp < MAX_TIME:
                dist = candidate.object.depth
                candidates_.append((dist,candidate))
                if dist is not None and dist < MAX_RADIUS:
                    filtered_candidates.append((dist, candidate))

        # Sort by distance
        candidates_.sort(key=lambda x: x[0])
        filtered_candidates = [candidate for (dist,candidate) in candidates_ if dist < MAX_RADIUS]
        max_candidtates = min(len(filtered_candidates),MAX_CANDIDATE_OBJECTS)
        self.candidates = filtered_candidates[:max_candidtates]
        
        # Update buffer with the MAX_OBJECTS closest candidates
        max_objects = min(len(filtered_candidates),MAX_OBJECTS)
        self.buffer = [candidate.object for candidate in filtered_candidates[:max_objects] if candidate.count > MIN_COUNT]

class Heading_Buffer:
    def __init__(self):
        self.point_buffer = []
        self.angle_buffer = []
        self.heading = None
        self.angle = None
    def update(self, new_heading, new_angle):

        self.point_buffer.append(new_heading)
        self.point_buffer = self.point_buffer[-MAX_HEADING_COUNT:]

        self.angle_buffer.append(new_angle)
        self.angle_buffer = self.angle_buffer[-MAX_HEADING_COUNT:]

        self.heading = np.mean(self.point_buffer, axis=0)
        self.angle = np.mean(self.angle_buffer, axis=0)

def objects_to_json(objects: List[Object]):
    if len(objects) == 0: # no objects detected, just send description to unity
        return json.dumps([])

    data = []
    for obj in objects:
        obj_data = {
            'class_name': classes[obj.label],
            'priority': int(classes_with_priority.get(classes[obj.label])),
            'x': float(obj.world_pose[0]),
            'y': float(obj.world_pose[1]),
            'z': float(obj.world_pose[2]),
            'depth': float(obj.depth)
        }
        data.append(obj_data)
    data.sort(key=lambda t: (t['priority'], t['depth']))
    return jsonify(data)

def objects_to_json_collisions(objects: List[Object]):
    if len(objects) == 0: # no objects detected, just send description to unity
        return {} #TODO: Problem here

    data = []
    for obj in objects:
        obj_data = {
            'class_name': obj.label,
            'priority': 1,
            'x': float(obj.world_pose[0]),
            'y': float(obj.world_pose[1]),
            'z': float(obj.world_pose[2]),
            'depth': float(obj.depth)
        }
        data.append(obj_data)

    return data

def get_centers(objects: List[Object]) -> List[Tuple[int, int]]:
    """
    Returns a list of center pixels for a given list of Object instances.
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

def keep_rotations_xz(rotation_matrix):
    """
    Extract and keep only rotations around the X- and Z-axes from a 3D rotation matrix.
    """
    # Decompose the rotation matrix into Euler angles
    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    
    # Check for gimbal lock
    singular = sy < 1e-6

    if not singular:
        x_angle = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # Rotation around X
        z_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # Rotation around Z
        #print("X rotation: ", x_angle, "Z_rotation: ", z_angle)
    else:
        x_angle = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])  # Rotation around X
        z_angle = 0  # Rotation around Z is undefined in this case

    # Reconstruct rotation matrix with only X and Z rotations
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x_angle), -np.sin(x_angle)],
        [0, np.sin(x_angle), np.cos(x_angle)]
    ])

    Rz = np.array([
        [np.cos(z_angle), -np.sin(z_angle), 0],
        [np.sin(z_angle), np.cos(z_angle), 0],
        [0, 0, 1]
    ])

    # Combine the X and Z rotations
    R_xz = Rz @ Rx

    return R_xz

def downsample_point_cloud(point_cloud, voxel_size):
    """
    Downsample the input point cloud using a voxel grid.
    """
    downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size)
    return downsampled_point_cloud

def find_plane_ransac_o3d(point_cloud, head_height, max_iterations=100, distance_threshold=0.1, min_inliers=250, angle_threshold=30):
    """
    RANSAC algorithm to find the floor plane in a point cloud.
    """
    # Convert Open3D point cloud to NumPy array
    #original_points = np.asarray(point_cloud.points)
    #points = transform_points(original_points, pose)
    points = np.asarray(point_cloud.points)

    up_vector = np.array([0,1,0])

    # Extract the floor normal vector from the IMU pose matrix

    # Initialize variables
    best_inliers = set()
    num_points = points.shape[0]

    if num_points < min_inliers:
        return list()

    def calculate_inliers(params):
        """
        Calculate inliers for a given plane defined by three random points.
        """

        max_retries = 100  # Maximum retries to find a suitable plane
        retries = 0
        plane_height = 0

        while retries < max_retries:
            # Randomly sample 3 points
            if len(points) > 3:
                sample_indices = random.sample(range(num_points), 3)
            p1, p2, p3 = points[sample_indices]

            # Define the plane equation ax + by + cz + d = 0
            vec1, vec2 = p2 - p1, p3 - p1
            plane_normal = np.cross(vec1, vec2)

            # Check for degeneracy
            norm = np.linalg.norm(plane_normal)
            if norm == 0:
                retries += 1
                continue
            plane_normal = plane_normal / norm  # Normalize

            # Check angle with IMU normal
            angle = np.degrees(np.arccos(np.clip(np.dot(up_vector, plane_normal), -1.0, 1.0)))
            if angle <= angle_threshold:  # Plane is within desired angle
                break
            retries += 1
        else:
            return set(), 0  # Return empty set if no valid plane is found within retries

        # Plane equation coefficients
        a, b, c = plane_normal
        d = -(a * p1[0] + b * p1[1] + c * p1[2])

        # Calculate inliers
        plane_norm = np.sqrt(a**2 + b**2 + c**2)
        inliers = set(
            i for i, point in enumerate(points)
            if abs(a * point[0] + b * point[1] + c * point[2] + d) / plane_norm < distance_threshold
        )
        if len(inliers) > 0:
            plane_height = np.mean(points[np.array(list(inliers)), 1])

        return inliers, plane_height

    with ThreadPoolExecutor() as executor:
        # Submit tasks for each RANSAC iteration
        height = 0
        futures = [executor.submit(calculate_inliers, _) for _ in range(max_iterations)]
        for future in futures:
            inliers, plane_height = future.result()
            if len(inliers) > len(best_inliers) and len(inliers) > min_inliers:
                best_inliers = inliers
                height = plane_height

    if height > head_height + 0.25:
        print("Found plane but too high: ", height," with head height: ", head_height)
        return list()

    return list(best_inliers)

def dbscan_clustering(point_cloud, colors, non_floor_mask, eps=0.1, min_samples=10):
    """
    Apply DBSCAN clustering to a point cloud.
    """
    # Convert Open3D point cloud to NumPy array
    points = np.asarray(point_cloud.points)
    filtered_points = points[non_floor_mask]
    filtered_colors = colors[non_floor_mask]

    # Check for edge case: no non-floor points
    if len(filtered_points) == 0:
        print("No non-floor points found for clustering.")
        return np.array([]), filtered_colors

    # Perform DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(filtered_points)
    labels = db.labels_

    # Assign random colors to clusters
    unique_labels = np.unique(labels)
    cluster_colors = np.random.rand(len(unique_labels), 3)  # Generate random colors for all unique labels
    cluster_colors[unique_labels == -1] = [0, 1, 0]  # Green for noise

    # Map labels to their respective colors
    label_to_color_map = cluster_colors[labels - labels.min()]
    filtered_colors = label_to_color_map

    return labels, filtered_colors

def fit_bounding_boxes_with_threshold_and_order(point_cloud, labels, non_floor_mask, min_points=50, reference_point=(0, 0)):
    """
    Fit bounding boxes around clusters in the point cloud, filter by minimum number of points,
    and order by proximity to a reference point in the XZ plane.
    """
    points = np.asarray(point_cloud.points)
    non_floor_points = points[non_floor_mask]
    bounding_boxes = []
    centers_radii = []

    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:
            # Skip noise
            continue

        # Extract points belonging to the current cluster
        cluster_points = non_floor_points[labels == label]

        # Skip clusters with fewer points than the threshold
        if len(cluster_points) < min_points:
            continue

        # Create an Open3D point cloud for the cluster
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)

        # Compute the axis-aligned bounding box (AABB)
        aabb = cluster_pcd.get_axis_aligned_bounding_box()
        bounding_boxes.append(aabb)

        # Compute the center and radius for the XZ plane
        bbox_points = np.asarray(aabb.get_box_points())
        xz_points = bbox_points[:, [0, 2]]  # Project onto XZ plane
        center_xz = xz_points.mean(axis=0)
        min_xz = xz_points.min(axis=0)
        max_xz = xz_points.max(axis=0)
        radius = np.linalg.norm(max_xz - min_xz) / 2

        # if center_xz[1] > 0 :
        #     continue
        
        centers_radii.append((center_xz[0], center_xz[1], radius))

    # Compute distances from the reference point
    distances = [
        np.linalg.norm([cx - reference_point[0], cz - reference_point[1]])
        for cx, cz, _ in centers_radii
    ]

    # Sort bounding boxes and centers_radii by distance
    sorted_indices = np.argsort(distances)
    bounding_boxes = [bounding_boxes[i] for i in sorted_indices]
    centers_radii = [centers_radii[i] for i in sorted_indices]

    return bounding_boxes, centers_radii

def fit_quadratic_and_tangent(points):
    # Extract first and last points
    x1, y1 = points[0]
    xn, yn = points[-1]

    # Create design matrix and solve for coefficients
    A = np.array([[x1**2, x1, 1], [xn**2, xn, 1]])
    b = np.array([y1, yn])
    coeffs = np.linalg.lstsq(A, b, rcond=None)[0]  # [a, b, c]

    a, b, c = coeffs

    # Compute slope at the first point
    slope = 2 * a * x1 + b

    # Determine the sign of the x-component based on the direction of the curve
    x_component = 1 if (xn > x1) else -1

    # Construct tangent vector
    tangent_vector = np.array([x_component, x_component * slope])  # [dx, dy]
    tangent_vector /= np.linalg.norm(tangent_vector)  # Normalize for consistency if needed

    return tangent_vector


def process_bounding_boxes(object_buffer:Object_Buffer,floor_detected,global_pcd,labels, non_floor_mask, min_points=50, global_pose=np.array([0, 0, 0]), timestamp=0.0):
   
    global_points = np.asarray(global_pcd.points)
    global_non_floor_points = global_points[non_floor_mask]

    # Data holders for bounding boxes and centers
    global_centers = []
    bounding_boxes = []

    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:
            # Skip noise
            continue

        # Extract points belonging to the current cluster
        global_cluster_points = global_non_floor_points[labels == label]

        # Skip clusters with fewer points than the threshold
        if len(global_cluster_points) < min_points:
            continue

        # Create an Open3D point cloud for the cluster
        global_cluster_pcd = o3d.geometry.PointCloud()
        global_cluster_pcd.points = o3d.utility.Vector3dVector(global_cluster_points)

        # Compute the axis-aligned bounding box (AABB)
        aabb = global_cluster_pcd.get_axis_aligned_bounding_box()
        bounding_boxes.append(aabb)

        # Compute the center and depth (z-coordinate from local frame)
        bbox_points = np.asarray(aabb.get_box_points())
        center_global = bbox_points.mean(axis=0)
        global_centers.append(center_global)

    # Sort by distance to the local reference point
    distances = [np.linalg.norm([cx - global_pose[0], cy - global_pose[1], cz - global_pose[2]]) for cx, cy, cz in global_centers]
    sorted_indices = np.argsort(distances)

    # Create Object instances based on sorted order
    objects = []
    for idx in sorted_indices:
        center_global = global_centers[idx]
        aabb = bounding_boxes[idx]
        depth = distances[idx]  # TODO:Not actually depth, it is distance, change name
        box_min = aabb.min_bound
        box_max = aabb.max_bound
        box = (box_min[0], box_min[2], box_max[0] - box_min[0], box_max[2] - box_min[2])  # (x, z, w, h)

        obj = Object(
            label="obstacle",
            p_center= None,
            depth=depth,  # Use z from the local frame
            box=box,
        )

        #obj.world_pose = np.array([center_global[0], center_global[1] , center_global[2]])
        obj.world_pose = np.array([center_global[0], global_pose[1]-1.6 , center_global[2]])
        obj.radius = np.sqrt(box[2]**2+box[3]**2)/2
        obj.timestamp = timestamp
        #obj.world_pose = (0, 0 , 0)
        objects.append(obj)

    if floor_detected:
        if objects:
            object_buffer.update(objects,timestamp,global_pose)
        return object_buffer.buffer
    else:
        if objects and depth < 2.0:
            return [objects[0]] # Only give imediate collision risk
        else:
            return []

# def find_heading(object_buffer, head_transform, heading_radius, safety_radius, num_samples, num_stages):
    
#     num_samples = int(num_samples)
#     r = float(heading_radius)

#     origin = np.array([head_transform[0, 3], head_transform[2, 3]])
#     vec_dir = np.array([head_transform[0, 2], head_transform[2, 2]])
#     alpha = np.arctan2(vec_dir[1], vec_dir[0]) # Angle with respect to X-Axis in range [-pi,pi]
#     step = heading_radius / num_stages

    # def find_intermediate_heading(direction, origin, radius, num_samples, alpha, stage_index):

    #     best_point = origin + radius * np.array([np.cos(alpha), np.sin(alpha)])
    #     min_overlap = float('inf')
    #     best_alpha = alpha
    #     best_sample = 0

    #     for i in range(num_samples):

    #         alpha_prime = alpha + np.pi * i / num_samples * direction
    #         point_prime = origin + radius * np.array([np.cos(alpha_prime), np.sin(alpha_prime)])
    #         collision_free = True

    #         overlap = 0
    #         for obj in object_buffer.buffer:

    #             box_x, box_z, box_w, box_h = obj.box
    #             closest_x = max(box_x, min(point_prime[0], box_x + box_w))
    #             closest_z = max(box_z, min(point_prime[1], box_z + box_h))
    #             closest_point = np.array([closest_x, closest_z])
    #             dist_to_obj = np.linalg.norm(closest_point - point_prime)

    #             # Check collisions
    #             overlap += max(0, safety_radius - dist_to_obj)  # Calculate overlap based on circle radius
    #             if dist_to_obj < safety_radius:
    #                 collision_free = False

    #         if overlap < min_overlap:
    #             min_overlap = overlap
    #             best_point = point_prime
    #             best_alpha = alpha_prime if alpha_prime*direction < np.pi else alpha_prime - np.pi*direction
    #             best_sample = i

    #         # If collision-free, return immediately
    #         if collision_free:
    #             #print(f"Stage: {stage_index}, angle: {best_alpha}, sample num: {i}")
    #             return (stage_index, best_point,best_sample)
            
    #     # If no collision-free heading is found, return the best_alpha (least overlap)
    #     return (stage_index, best_point, best_sample)

    # # Use ThreadPoolExecutor for parallel execution
    # best_points = [None] * num_stages * 2
    # with ThreadPoolExecutor() as executor:
    #     # Submit tasks for all stages
    #     futures = [
    #         executor.submit(find_intermediate_heading, direction, origin, step * (i + 1), num_samples, alpha, 2*i+int((1-direction)/2))
    #         for i in range(num_stages)
    #         for direction in [1., -1.]
    #     ]
    #     # Collect results as they complete (plus: even, minus: odd)
    #     for future in as_completed(futures):
    #         stage_index, best_point, best_sample = future.result()
    #         best_points[stage_index] = (best_point , best_sample)

    # # Fit a quadratic function to the best points - direction must be maintained
    # direction = 1.0
    # tolerance = 1e-6
    # diff = 0.0
    # i = 0
    # while diff < tolerance:
    #     diff = best_points[i][1] - best_points[i+1][1]
    #     i +=1
    #     if i == num_stages: break
    # if abs(diff) > tolerance: 
    #     direction = diff / abs(diff) 

    # direction_points = [best_points[2 * i + int((1 - direction) / 2)][0] for i in range(num_stages)]
    # direction_points = [origin] + direction_points

    # vec_dir = fit_quadratic_and_tangent(direction_points)

    # alpha_new = np.arctan2(vec_dir[1], vec_dir[0])
    # heading_point_2d = origin + r * np.array([np.cos(alpha_new), np.sin(alpha_new)])
    # #print("New Heading: ",alpha_new, "Forward:",alpha)
    # best_heading = ((alpha_new - alpha + np.pi) % (2 * np.pi)) - np.pi

    # heading_obj = Object(
    #         label="heading",
    #         p_center= None,
    #         depth=r,  # Use z from the local frame
    #         box=None,
    #     )
    # heading_obj.world_pose = np.array([heading_point_2d[0], head_transform[1, 3] -1.6 , heading_point_2d[1]])

    # return best_heading, [heading_obj]

def find_heading(object_buffer: Object_Buffer, heading_buffer: Heading_Buffer, head_transform, heading_radius, safety_radius, num_samples, num_stages, sound_distance_multiplier):
    
    num_samples = int(num_samples)
    origin = np.array([head_transform[0, 3], head_transform[2, 3]])
    vec_dir = np.array([head_transform[0, 2], head_transform[2, 2]])
    alpha = np.arctan2(vec_dir[1], vec_dir[0]) # Angle with respect to X-Axis in range [-pi,pi]
    step = heading_radius / num_stages

    def find_intermediate_heading(direction, origin, radius, num_samples, alpha, stage_index):

        best_point = origin + radius * np.array([np.cos(alpha), np.sin(alpha)])
        min_overlap = float('inf')
        best_alpha = alpha
        best_sample = 0

        for i in range(num_samples):

            alpha_prime = alpha + np.pi * i / num_samples * direction
            point_prime = origin + radius * np.array([np.cos(alpha_prime), np.sin(alpha_prime)])
            collision_free = True

            overlap = 0
            for obj in object_buffer.buffer:

                box_x, box_z, box_w, box_h = obj.box
                closest_x = max(box_x, min(point_prime[0], box_x + box_w))
                closest_z = max(box_z, min(point_prime[1], box_z + box_h))
                closest_point = np.array([closest_x, closest_z])
                dist_to_obj = np.linalg.norm(closest_point - point_prime)

                # Check collisions
                overlap += max(0, safety_radius - dist_to_obj)  # Calculate overlap based on circle radius
                if dist_to_obj < safety_radius:
                    collision_free = False

            if overlap < min_overlap:
                min_overlap = overlap
                best_point = point_prime
                best_alpha = alpha_prime
                best_sample = i

            # If collision-free, return immediately
            if collision_free:
                #print(f"Stage: {stage_index}, angle: {best_alpha}, sample num: {i}")
                return (stage_index, best_point,best_sample, best_alpha)
            
        # If no collision-free heading is found, return the best_alpha (least overlap)
        return (stage_index, best_point, best_sample, best_alpha)

    # Use ThreadPoolExecutor for parallel execution
    best_points = [None] * num_stages * 2
    with ThreadPoolExecutor() as executor:
        # Submit tasks for all stages
        futures = [
            executor.submit(find_intermediate_heading, direction, origin, step * (i + 1), num_samples, alpha, 2*i+int((1-direction)/2))
            for i in range(num_stages)
            for direction in [1., -1.]
        ]
        # Collect results as they complete (plus: even, minus: odd)
        for future in as_completed(futures):
            stage_index, best_point, best_sample, best_angle = future.result()
            best_points[stage_index] = (best_point , best_sample, best_angle)
    
    
    heading_point_2d = origin + sound_distance_multiplier * heading_radius * np.array([np.cos(alpha), np.sin(alpha)])
    #heading_angle = np.pi
    heading_angle = 0.0
    # Get max angle
    i = 0
    while i < num_stages:
        _ , sample_plus , alpha_plus = best_points[i]
        _ , sample_minus , alpha_minus = best_points[i+1]

        if sample_plus <= sample_minus:
            if heading_angle < abs(alpha_plus - alpha):
                heading_point_2d = origin + sound_distance_multiplier * heading_radius * np.array([np.cos(alpha_plus), np.sin(alpha_plus)])
                heading_angle = abs(alpha_plus - alpha)
        else:
            if heading_angle < abs(alpha_minus - alpha):
                heading_point_2d = origin + sound_distance_multiplier * heading_radius * np.array([np.cos(alpha_minus), np.sin(alpha_minus)])
                heading_angle = abs(alpha_minus - alpha)
        i += 1

    # for (point, _ , alpha_prime) in best_points:

    #     if heading_angle > abs(alpha_prime-alpha):
    #         heading_point_2d = origin + 2 * r * np.array([np.cos(alpha_prime), np.sin(alpha_prime)])
    #         heading_angle = alpha_prime-alpha

    heading_world_pose = np.array([heading_point_2d[0], head_transform[1, 3] - 1.6 , heading_point_2d[1]])
    heading_angle = ((heading_angle + np.pi) % (2 * np.pi)) - np.pi
    heading_buffer.update(heading_world_pose,heading_angle)

    heading_obj = Object(
            label="heading",
            p_center= None,
            depth=heading_radius,  # Use z from the local frame
            box=None,
        )
    heading_obj.world_pose = heading_buffer.heading

    return heading_buffer.angle, [heading_obj]

