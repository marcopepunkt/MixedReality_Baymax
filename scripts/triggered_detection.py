import sys
import os
import time
from typing import List, Tuple
import threading
import numpy as np
import cv2
import torch
import winsound
import multiprocessing as mp
from datetime import datetime
import requests
import json

import pyttsx3
import threading

# Import HoloLens libraries
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv

# Import detection utilities
import openvino as ov
import utils

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

# Colors for visualization
COLORS = np.random.randint(0, 255, size=(len(utils.classes), 3), dtype=np.uint8)
TEXT_COLOR = (255, 255, 255)
TEXT_BACKGROUND = (0, 0, 0)

# Global variables
enable = True
calibration_lt = None
last_detection_time = 0
DETECTION_COOLDOWN = 2

# Azure Computer Vision:
region = "switzerlandnorth"
endpoint = "https://baymaxcv.cognitiveservices.azure.com/"
key = "RDUsQ9sjNNm8iq64g7ys2fUT63jalOJYxByykhAYjA2TTsDWIyNFJQQJ99AKACI8hq2XJ3w3AAAFACOG07X7"
credentials = CognitiveServicesCredentials(key)
client = ComputerVisionClient(
    endpoint=endpoint,
    credentials=credentials
)

# Azure OpenAI
openai.api_type = "azure"
openai.api_base = "https://baymaxopenai.openai.azure.com/"
openai.api_version = "2021-04-30"
openai.api_key = "E3XDVx6dBQ3vPgjQkYYbTKMkjdRcel0eJOAdSeHZBLmObDgXL0duJQQJ99AKACI8hq2XJ3w3AAABACOGdk09"

# openAI (chatgpt) key:
openai_key = "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"


class HoloLensDetection:
    def __init__(self, IP_ADDRESS):
        self.HOST = IP_ADDRESS

        self.producer = None
        self.consumer = None
        self.sink_pv = None
        self.sink_depth = None
        self.voice_client = None
        self.latest_frame = None
        self.latest_depth = None
        self.xy1 = None
        self.scale = None
        self.pv_extrinsics = None
        self.pv_extrinsics = None
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


    def init_streams(self):
        """Initialize HoloLens streams"""
        try:
            hl2ss_lnm.start_subsystem_pv(self.HOST, hl2ss.StreamPort.PERSONAL_VIDEO, shared=True)
            print("PV subsystem started")

            global calibration_lt
            calibration_lt = hl2ss_3dcv.get_calibration_rm(self.HOST,
                                                          hl2ss.StreamPort.RM_DEPTH_LONGTHROW, 
                                                          CALIBRATION_PATH)
            
            self.xy1, self.scale = hl2ss_3dcv.rm_depth_compute_rays(calibration_lt.uv2xy, calibration_lt.scale)
            
            self.pv_intrinsics = hl2ss.create_pv_intrinsics_placeholder()
            self.pv_extrinsics = np.eye(4, 4, dtype=np.float32)

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
            
            self.producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, 
                                   BUFFER_LENGTH * PV_FRAMERATE)
            self.producer.initialize(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, 
                                   BUFFER_LENGTH * hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS)
            
            self.producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
            self.producer.start(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

            manager = mp.Manager()
            self.consumer = hl2ss_mp.consumer()
            self.sink_pv = self.consumer.create_sink(self.producer, 
                                                    hl2ss.StreamPort.PERSONAL_VIDEO, 
                                                    manager, None)
            self.sink_depth = self.consumer.create_sink(self.producer, 
                                                       hl2ss.StreamPort.RM_DEPTH_LONGTHROW, 
                                                       manager, None)
            
            self.sink_pv.get_attach_response()
            self.sink_depth.get_attach_response()
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

            sensor_depth = hl2ss_3dcv.rm_depth_normalize(depth, self.scale)
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
            print(f"type description: {type(description)}")
            print(f"Description: {description}")
            print(f"Confidence: {confidence}")
            return description
        else:
            print("No description available from azureCV for this frame.")
            return None

    def get_friendly_text_from_openAI(self, image_description):
        message_text = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information."
            },
            {"role": "user",
             "content": f"Construct full sentences with this text, that are more user-friendly: {image_description}"}
        ]

        completion = openai.ChatCompletion.create(engine="model-gpt-35-turbo-16k",
                                                  messages=message_text,
                                                  temperature=0.7,
                                                  max_tokens=800,
                                                  top_p=0.95,
                                                  frequency_penalty=0,
                                                  presence_penalty=0,
                                                  stop=None)
        if completion is not None:
            print(f"chatgpt response: {completion}")
            print(completion["message"]['content'])
            return completion["message"]['content']
        else:
            print("Could not get response from chatgpt.")
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

    def run(self):
        print("Starting object detection!")

        objects = []

        current_time = time.time()

        # Get depth frame
        _, data_depth = self.sink_depth.get_most_recent_frame()
        if data_depth is None or not hl2ss.is_valid_pose(data_depth.pose):
            print("No valid depth frame")
            return [], None

        # Get PV frame
        _, data_pv = self.sink_pv.get_nearest(data_depth.timestamp)
        if data_pv is None or not hl2ss.is_valid_pose(data_pv.pose):
            print("No valid PV frame")
            return [], None

        # Get frame and check if it's valid
        frame = data_pv.payload.image
        if frame is None:
            #print("Invalid frame")
            return [], None

        # vis_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR
        image_description = self.get_image_description_from_azureCV(frame)
        # if image_description is not None:
        #     openAI_image_description = self.get_friendly_text_from_openAI(image_description)


        depth = data_depth.payload.depth
        if depth is None:
            print("Invalid depth")
            return [], None
        
        try:
            if current_time - self.last_detection_time >= DETECTION_COOLDOWN:
                self.detection_active = True
                print("Starting process_detection")
                objects = self.process_detection(frame, depth, data_pv, data_depth)
                if objects is not None:
                    self.last_detection_time = current_time
                else:
                    return [], None

            else:
                print("\nPlease wait before next detection")
            
            # Show frame
            # if vis_frame is not None:
            #     vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR
                # cv2.imshow("HoloLens Detection", vis_frame)
                # cv2.waitKey(1)

        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            return [], None

        return objects, image_description

    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        try:
            # cv2.destroyAllWindows()

            if self.sink_pv:
                self.sink_pv.detach()
            if self.sink_depth:
                self.sink_depth.detach()
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