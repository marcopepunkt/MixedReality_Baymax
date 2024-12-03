import sys
import os
import time
from typing import List, Tuple
import numpy as np
import cv2
import torch
import winsound
import multiprocessing as mp
from datetime import datetime
import requests
import json

import pyttsx3

# Import HoloLens libraries
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv

# Import detection utilities
import openvino as ov
import utils

# Import collision utilities
import open3d as o3d

# for Azure Computer Vision (object detection) model:
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

#for Azure OpenAI
import openai

# image to byte
from PIL import Image
import io

# Settings
CALIBRATION_PATH = "./calibration"
PV_WIDTH = 640
PV_HEIGHT = 360
PV_FRAMERATE = 15
BUFFER_LENGTH = 5
MAX_SENSOR_DEPTH = 10
VOICE_COMMANDS = ['detect']

# Sound settings
DETECTION_SOUND_FREQ = 1000
DETECTION_SOUND_DURATION = 200
ERROR_SOUND_FREQ = 500

# RANSAC settings
MAX_ITERATIONS = 100
DISTANCE_THRESH = 0.1
ANGLE_THRESH = 30.0
MIN_INLIERS = 250

# Clustering settings
MIN_POINTS = 50

# Colors for visualization
COLORS = np.random.randint(0, 255, size=(len(utils.classes), 3), dtype=np.uint8)
TEXT_COLOR = (255, 255, 255)
TEXT_BACKGROUND = (0, 0, 0)

# Global variables
enable = True
calibration_lt = None
last_detection_time = 0
DETECTION_COOLDOWN = 2

# for Azure Computer Vision resource, gets image description:
# insert key (do not commit them)
region = "switzerlandnorth"
endpoint = "https://baymaxcv.cognitiveservices.azure.com/"
key = "key"
credentials = CognitiveServicesCredentials(key)
client = ComputerVisionClient(
    endpoint=endpoint,
    credentials=credentials
)


class HoloLensDetection:
    def __init__(self, IP_ADDRESS):
        self.HOST = IP_ADDRESS

        self.producer = None
        self.consumer = None
        self.sink_pv = None
        self.sink_lt = None
        self.voice_client = None
        self.latest_frame = None
        self.latest_depth = None
        self.detection_active = False
        self.detection_results = []
        self.last_detection_time = time.time()

        # Initialize OpenVINO
        self.core = ov.Core()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        detection_model_path = os.path.join(current_dir, 'model', 'ssdlite_mobilenet_v2_fp16.xml')
        
        # Create calibration directory if it doesn't exist
        os.makedirs(CALIBRATION_PATH, exist_ok=True)
        
        # Load detection model
        detection_model = self.core.read_model(model=detection_model_path)
        self.compiled_model = self.core.compile_model(model=detection_model, device_name="CPU")
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        self.height, self.width = list(self.input_layer.shape)[1:3]

        # Camera Parameters
        global calibration_lt, calibration_ht, lt_focal_length
        calibration_lt = hl2ss_3dcv.get_calibration_rm(self.HOST,
                                                        hl2ss.StreamPort.RM_DEPTH_LONGTHROW, 
                                                        CALIBRATION_PATH) 
        self.xy1, self.scale_lt = hl2ss_3dcv.rm_depth_compute_rays(calibration_lt.uv2xy, calibration_lt.scale)
        
        self.pv_intrinsics = hl2ss.create_pv_intrinsics_placeholder()
        self.pv_extrinsics = np.eye(4, 4, dtype=np.float32)

        # Open3d Depth Camera model
        self.o3d_lt_intrinsics = o3d.camera.PinholeCameraIntrinsic(hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH,
                                                                hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT,
                                                                calibration_lt.intrinsics[0, 0],calibration_lt.intrinsics[1, 1],
                                                                calibration_lt.intrinsics[2, 0],calibration_lt.intrinsics[2, 1])


    def init_streams(self):
        """Initialize HoloLens streams"""
        try:
            hl2ss_lnm.start_subsystem_pv(self.HOST, hl2ss.StreamPort.PERSONAL_VIDEO, shared=True)
            print("PV subsystem started")

            self.producer = hl2ss_mp.producer()
            self.producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO, 
                                  hl2ss_lnm.rx_pv(self.HOST,
                                                 hl2ss.StreamPort.PERSONAL_VIDEO, 
                                                 width=PV_WIDTH, 
                                                 height=PV_HEIGHT, 
                                                 framerate=PV_FRAMERATE))
            self.producer.configure(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, 
                                  hl2ss_lnm.rx_rm_depth_longthrow(self.HOST,
                                                                 hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
            self.producer.configure(hl2ss.StreamPort.SPATIAL_INPUT, hl2ss_lnm.rx_si(self.HOST, hl2ss.StreamPort.SPATIAL_INPUT))
            
            self.producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, 
                                   BUFFER_LENGTH * PV_FRAMERATE)
            self.producer.initialize(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, 
                                   BUFFER_LENGTH * hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS)

            self.producer.initialize(hl2ss.StreamPort.SPATIAL_INPUT, 
                                     BUFFER_LENGTH * hl2ss.Parameters_SI.SAMPLE_RATE)
            
            self.producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
            self.producer.start(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)
            self.producer.start(hl2ss.StreamPort.SPATIAL_INPUT)    

            manager = mp.Manager()
            self.consumer = hl2ss_mp.consumer()
            self.sink_pv = self.consumer.create_sink(self.producer, 
                                                    hl2ss.StreamPort.PERSONAL_VIDEO, 
                                                    manager, None)
            self.sink_lt = self.consumer.create_sink(self.producer, 
                                                       hl2ss.StreamPort.RM_DEPTH_LONGTHROW, 
                                                       manager, None)
            self.sink_si = self.consumer.create_sink(self.producer, hl2ss.StreamPort.SPATIAL_INPUT, manager, None)
            
            self.sink_pv.get_attach_response()
            self.sink_lt.get_attach_response()
            self.sink_si.get_attach_response()
            print("Streams initialized")

        except Exception as e:
            print(f"Error initializing streams: {str(e)}")
            raise

 

    def draw_detection_results(self, frame, boxes, objects):
        """Draw detection results on the frame"""
        vis_frame = frame.copy()
        overlay = vis_frame.copy()
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(vis_frame, f"Detection Time: {timestamp}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

        for (label, score, box), pose in zip(boxes, objects):
            color = tuple(map(int, COLORS[label]))
            x, y, w, h = box
            
            # Draw box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)

            # Create label
            class_name = utils.classes[label]
            confidence = f"{score:.2f}"
            depth_text = f"{pose.depth:.2f}m" if pose is not None and not np.isnan(pose.depth) else "unknown"
            label_text = f"{class_name} ({confidence}) @ {depth_text}"

            # Draw label background
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_frame, 
                         (x, y - 20), 
                         (x + text_size[0], y),
                         color, -1)

            # Draw label text
            cv2.putText(vis_frame, label_text,
                       (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, TEXT_COLOR, 2)

        # Blend overlay
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0, vis_frame)

        # Add status
        status_text = "Detection Active" if self.detection_active else "Say 'detect' to start"
        cv2.putText(vis_frame, status_text,
                   (10, vis_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 0) if self.detection_active else (0, 0, 255),
                   2)

        return vis_frame
    
    def post_process_objects(self, depth, objects: List[utils.Object], xy1, depth_to_world):

        """ Transform objects into global frame"""

        # Transform to world CF.
        points = hl2ss_3dcv.rm_depth_to_points(depth, xy1)
        points = hl2ss_3dcv.transform(points, depth_to_world)

        for object in objects:
            cx, cy = object.center
            object.world_pose = hl2ss_3dcv.block_to_list(points[cy,cx])[0]
 
    def get_uv_map(self, data_pv, data_depth, depth):

        # Update PV intrinsics ------------------------------------------------
        # PV intrinsics may change between frames due to autofocus
        self.pv_intrinsics = hl2ss.update_pv_intrinsics(self.pv_intrinsics, data_pv.payload.focal_length, data_pv.payload.principal_point)
        color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(self.pv_intrinsics, self.pv_extrinsics)
    
        # Build pointcloud ----------------------------------------------------
        points = hl2ss_3dcv.rm_depth_to_points(self.xy1,depth)
        world_to_pv_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(color_extrinsics) @ hl2ss_3dcv.camera_to_image(color_intrinsics)
        depth_to_world = hl2ss_3dcv.camera_to_rignode(calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(data_depth.pose)
        world_points = hl2ss_3dcv.transform(points, depth_to_world)
        uv_map = hl2ss_3dcv.project(world_points, world_to_pv_image)

        return uv_map

    def process_detection(self, frame, depth, data_pv, depth_data):
        """Process detection on current frame"""
        try:
            if frame is None:
                return None
        
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
            input_img = cv2.resize(src=frame, dsize=(self.width, self.height), interpolation=cv2.INTER_AREA)
            input_img = input_img[np.newaxis, ...]

            results = self.compiled_model([input_img])[self.output_layer]
            boxes = utils.process_results(frame=frame, results=results, thresh=0.5)

            sensor_depth = hl2ss_3dcv.rm_depth_normalize(depth, self.scale_lt)
            sensor_depth[sensor_depth > MAX_SENSOR_DEPTH] = 0

            uv_map = self.get_uv_map(data_pv,depth_data,sensor_depth)

            mapped_boxes = utils.remap_bounding_boxes(boxes,frame,uv_map)
            objects = utils.estimate_box_poses(sensor_depth,mapped_boxes)
            # vis_frame = self.draw_detection_results(frame, boxes, objects)

            if len(objects) > 0:
                depth_to_world = hl2ss_3dcv.camera_to_rignode(calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(depth_data.pose)
                self.post_process_objects(sensor_depth,objects,self.xy1,depth_to_world)

            print("\nDetected objects:")
            print(f"boxes: {boxes}")
            print(f"objects: {objects}")
            if len(objects) == 0:
                print("No objects detected :(")
                return None
            else:
                for (label, score, box), object in zip(boxes, objects):
                    if object is not None and not np.isnan(object.depth):
                        print(f"- {utils.classes[label]} (confidence: {score:.2f}) at {object.depth:.2f} meters")
                    else:
                        print(f"- {utils.classes[label]} (confidence: {score:.2f}) at unknown distance")

            return objects

        except Exception as e:
            print(f"Detection error: {str(e)}")
            return None

    def get_image_description_from_azureCV(self, frame):
        # Convert the NumPy array to a PIL Image
        frame = Image.fromarray(frame.astype('uint8'), 'RGB')
        # Save the image to an in-memory byte stream
        image_stream = io.BytesIO()
        frame.save(image_stream, format="JPEG")  # Save as JPEG or PNG
        image_stream.seek(0)  # Move the cursor to the beginning of the stream

        # Call the Azure Computer Vision API
        description_result = client.describe_image_in_stream(image_stream)

        # Process and print the description
        if description_result.captions:
            description = description_result.captions[0].text
            confidence = description_result.captions[0].confidence
            print(f"Description: {description}")
            print(f"Confidence: {confidence}")
            return description
        else:
            print("No description available from azureCV for this frame.")
            return None

    def start(self):
        try:
            print("Initializing...")
            self.init_streams()
            # cv2.namedWindow("HoloLens Detection", cv2.WINDOW_NORMAL)
            print("Ready!")

        except Exception as e:
            print(f"Runtime error: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_pv_frame(self, data_si):
        # Get depth frame
        _, data_depth = self.sink_lt.get_most_recent_frame()
        if data_depth is None or not hl2ss.is_valid_pose(data_depth.pose):
            print("No valid depth frame")
        # Get PV frame
        _, data_pv = self.sink_pv.get_nearest(data_depth.timestamp)
        if data_pv is None or not hl2ss.is_valid_pose(data_pv.pose):
            print("No valid PV frame")
            return None
        # Get frame and check if it's valid
        frame = data_pv.payload.image
        self.latest_frame = frame
        if frame is None:
            # print("Invalid frame")
            return None
        return frame

    def get_si_pose(self, si):

        head_pose = si.get_head_pose()
        up = head_pose.up
        forward = np.array(head_pose.forward)

        x_flip_rot = np.eye(3)
        x_flip_rot[1,1] = -1
        x_flip_rot[2,2] = -1

        right = np.cross(up, -forward)

        full_rotation = np.column_stack((right, up, -forward))
        roll_yaw_rot = utils.keep_rotations_xz(full_rotation)

        #rotation =  rotation_x * x_flip_rot
        #rotation = np.eye(3) * x_flip_rot
        global_pose = np.eye(4)
        rectified_local_pose = np.eye(4)
        global_pose[:3, :3] = np.matmul(full_rotation, x_flip_rot)
        rectified_local_pose[:3, :3] = roll_yaw_rot @ x_flip_rot

        global_pose[:3, 3] = head_pose.position
        rectified_local_pose[:3, 3] = [0,1.70,0]
        
        # up => y, forward => -z, right => x
        return global_pose, rectified_local_pose

    def run_detection_cycle(self):
        print("Starting object detection!")

        objects = []

        current_time = time.time()

        # Get depth frame
        _, data_depth = self.sink_lt.get_most_recent_frame()
        if data_depth is None or not hl2ss.is_valid_pose(data_depth.pose):
            print("No valid depth frame")
            return []

        # Get PV frame
        _, data_pv = self.sink_pv.get_nearest(data_depth.timestamp)
        if data_pv is None or not hl2ss.is_valid_pose(data_pv.pose):
            print("No valid PV frame")
            return []

        # Get frame and check if it's valid
        frame = data_pv.payload.image
        self.latest_frame = frame
        if frame is None:
            #print("Invalid frame")
            return []

        # vis_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR

        # optional: can get a description of the frame from Azure CV object detection model
        # image_description = self.get_image_description_from_azureCV(frame)

        depth = data_depth.payload.depth
        if depth is None:
            print("Invalid depth")
            return []
        
        try:
            if current_time - self.last_detection_time >= DETECTION_COOLDOWN:
                self.detection_active = True
                print("Starting process_detection")
                objects = self.process_detection(frame, depth, data_pv, data_depth)
                if objects is not None:
                    self.last_detection_time = current_time
                else:
                    return []

            else:
                print("\nPlease wait before next detection")
            
            # Show frame
            # if vis_frame is not None:
            #     vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR
                # cv2.imshow("HoloLens Detection", vis_frame)
                # cv2.waitKey(1)

        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            return []

        return objects

    def run_collision_cycle(self):

        current_time = time.time()

        # Get depth frame
        _, data_lt = self.sink_lt.get_most_recent_frame()
        if data_lt is None:
            print("No valid depth frame")
            return []

        # Get Spatial Input frame
        _, data_si = self.sink_si.get_nearest(data_lt.timestamp)
        if data_si is None:
            print("No valid SI frame")
            return []
        try:
            si = hl2ss.unpack_si(data_si.payload)
            if not si.is_valid_head_pose():
                print("No valid SI frame")
                return []
        except:
            print("No valid SI frame")
            return []
        global_pose, _ = self.get_si_pose(si)

        depth = hl2ss_3dcv.rm_depth_undistort(data_lt.payload.depth, calibration_lt.undistort_map)
        depth = hl2ss_3dcv.rm_depth_normalize(depth, self.scale_lt)

        depth_image = o3d.geometry.Image(depth)
        tmp_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, intrinsic=self.o3d_lt_intrinsics, depth_scale=1)
        ds_pcd = utils.downsample_point_cloud(tmp_pcd,voxel_size=0.05)
        global_pcd = ds_pcd.__copy__()
        global_pcd.transform(global_pose)

        # For debug
        colors = np.full((len(global_pcd.points), 3), [0, 0, 0], dtype=float)  # Base color: black

        # Find floor inliers using RANSAC
        floor_inliers = utils.find_plane_ransac_o3d(global_pcd,
                                                    head_height=1.70,
                                                    max_iterations=MAX_ITERATIONS,distance_threshold=DISTANCE_THRESH,
                                                    angle_threshold=ANGLE_THRESH,min_inliers=MIN_INLIERS)
        bounding_boxes = []
        floor_detected = False
        non_floor_mask = np.ones(len(ds_pcd.points), dtype=bool)
        if len(floor_inliers) > 0:
            floor_detected = True

        colors[floor_inliers] = [1, 0, 0]  # Highlight floor inliers in red
        non_floor_mask[floor_inliers] = False

        # Apply DBSCAN clustering to non-floor points
        cluster_labels, filtered_colors = utils.dbscan_clustering(global_pcd, colors, non_floor_mask)

        if cluster_labels.size > 0:
            colors[non_floor_mask,:] = filtered_colors  # Update non-floor colors with clustering results
            # Bounding boxes in global frame
            obstacles = utils.process_bounding_boxes(global_pcd,ds_pcd,cluster_labels,non_floor_mask,min_points=MIN_POINTS,global_pose=global_pose[:3,3])

        return obstacles
                
        

    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        try:
            # cv2.destroyAllWindows()

            if self.sink_pv:
                self.sink_pv.detach()
            if self.sink_lt:
                self.sink_lt.detach()
            if self.producer:
                self.producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
                self.producer.stop(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

            
            hl2ss_lnm.stop_subsystem_pv(self.HOST, hl2ss.StreamPort.PERSONAL_VIDEO)
                
            print("Cleanup completed")
        except Exception as e:
            print(f"Cleanup error: {str(e)}")

if __name__ == "__main__":
    detector = HoloLensDetection()
    detector.start()
    detector.run()