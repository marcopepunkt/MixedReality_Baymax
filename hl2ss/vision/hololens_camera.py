import sys
import os
import time
from typing import List, Tuple, Optional
import threading
from pynput import keyboard
import multiprocessing as mp
import numpy as np
import cv2
import open3d as o3d
from datetime import datetime
import openvino as ov
from pathlib import Path

# Import HoloLens libraries
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv

# Import detection utilities
import real_time.utils as utils
from real_time import detect

# Global variables
enable = True
calibration_lt = None

# Settings
HOST = '169.254.174.24'  # Adjust to your HoloLens IP
CALIBRATION_PATH = "./calibration"  # Make sure this directory exists
RECORDING_PATH = "./recordings"  # Directory for saving videos
PV_WIDTH = 640
PV_HEIGHT = 360
PV_FRAMERATE = 15
BUFFER_LENGTH = 5
MAX_SENSOR_DEPTH = 10
SCORE_THRESHOLD = 0.3
DEVICE = 'cpu'  # or 'cuda:0' if using GPU

# Initialize OpenVINO models
core = ov.Core()
device = "CPU"  # You can change this to "GPU" if available

# Get model paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
detection_model_path = os.path.join(parent_dir, 'model', 'ssdlite_mobilenet_v2_fp16.xml')
depth_model_path = os.path.join(parent_dir, 'models_ov', 'depth_anything_v2_metric_vkitti_vits.xml')

# Load and compile models
detection_model = core.read_model(model=detection_model_path)
compiled_detection_model = core.compile_model(model=detection_model, device_name=device)

# Get input/output layers
input_layer = compiled_detection_model.input(0)
output_layer = compiled_detection_model.output(0)
height, width = list(input_layer.shape)[1:3]

class HoloLensDetection:
    def __init__(self):
        self.vis = None
        self.main_pcd = None
        self.first_geometry = True
        self.producer = None
        self.consumer = None
        self.sink_pv = None
        self.sink_depth = None
        self.listener = None
        self.pv_video_writer = None
        self.depth_video_writer = None
        self.recording_start_time = None
        
        # Create necessary directories
        os.makedirs(CALIBRATION_PATH, exist_ok=True)
        os.makedirs(RECORDING_PATH, exist_ok=True)

    def init_video_writers(self):
        """Initialize video writers for PV and depth streams"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Changed file extension to .avi for better compatibility
        pv_filename = os.path.join(RECORDING_PATH, f'pv_stream_{timestamp}.avi')
        depth_filename = os.path.join(RECORDING_PATH, f'depth_stream_{timestamp}.avi')
        
        # Use XVID codec instead of MJPG
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        # Ensure frame size matches the actual frame size
        self.pv_video_writer = cv2.VideoWriter(
            pv_filename, 
            fourcc, 
            PV_FRAMERATE, 
            (PV_WIDTH, PV_HEIGHT),
            isColor=True  # PV stream is color
        )
        self.depth_video_writer = cv2.VideoWriter(
            depth_filename, 
            fourcc, 
            PV_FRAMERATE, 
            (PV_WIDTH, PV_HEIGHT),
            isColor=True  # Depth visualization is also color (due to colormap)
        )
        
        # Verify writers were initialized properly
        if not self.pv_video_writer.isOpened():
            raise RuntimeError(f"Failed to initialize PV video writer: {pv_filename}")
        if not self.depth_video_writer.isOpened():
            raise RuntimeError(f"Failed to initialize depth video writer: {depth_filename}")
            
        self.recording_start_time = time.time()
        print(f"Recording started. Saving to:\n{pv_filename}\n{depth_filename}")


    def init_keyboard_listener(self):
        """Initialize keyboard listener for graceful exit"""
        def on_press(key):
            global enable
            enable = key != keyboard.Key.space
            return enable

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()

    def init_visualizers(self):
        """Initialize OpenCV and Open3D visualizers"""
        cv2.namedWindow('HoloLens Detection', cv2.WINDOW_NORMAL)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.main_pcd = o3d.geometry.PointCloud()

    def process_frame_with_detection(self, data_pv, data_depth, scale, xy1):
        """Process a single frame with object detection"""
        try:
            # Get frame from PV stream
            frame = data_pv.payload.image
            if frame is None:
                return None, None, None

            # Process depth data
            sensor_depth = hl2ss_3dcv.rm_depth_normalize(data_depth.payload.depth, scale)
            sensor_depth[sensor_depth > MAX_SENSOR_DEPTH] = 0

            # Prepare frame for detection
            input_img = cv2.resize(src=frame, dsize=(width, height), interpolation=cv2.INTER_AREA)
            input_img = input_img[np.newaxis, ...]

            # Run detection
            results = compiled_detection_model([input_img])[output_layer]
            boxes = detect.process_results(frame=frame, results=results, thresh=0.5)

            # Estimate poses using depth
            poses = detect.estimate_box_poses(sensor_depth, boxes)
            
            # Draw detection boxes
            frame_with_boxes = frame.copy()
            frame_with_boxes = detect.draw_depth_boxes(frame_with_boxes, boxes, poses)

            # Create point cloud
            points = hl2ss_3dcv.rm_depth_to_points(sensor_depth, xy1)
            depth_to_world = hl2ss_3dcv.camera_to_rignode(calibration_lt.extrinsics) @ \
                            hl2ss_3dcv.reference_to_world(data_depth.pose)
            points = hl2ss_3dcv.transform(points, depth_to_world)

            # Process points for visualization
            points = hl2ss_3dcv.block_to_list(points)
            select = sensor_depth.reshape((-1,)) > 0
            points = points[select, :]

            # Create colored depth visualization
            depth_colored = cv2.normalize(sensor_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_INFERNO)

            return frame_with_boxes, depth_colored, points
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            return None, None, None

    def update_visualization(self, frame: np.ndarray, depth_frame: np.ndarray, points: np.ndarray):
        """Update visualization windows and save frames"""
        try:
            # Show frames
            cv2.imshow('HoloLens Detection', frame)
            
            # Ensure frames are in the correct format before saving
            if frame is not None and self.pv_video_writer is not None:
                # Ensure frame is the correct size
                frame_resized = cv2.resize(frame, (PV_WIDTH, PV_HEIGHT))
                self.pv_video_writer.write(frame_resized)
                
            if depth_frame is not None and self.depth_video_writer is not None:
                # Ensure depth frame is the correct size and format
                depth_resized = cv2.resize(depth_frame, (PV_WIDTH, PV_HEIGHT))
                self.depth_video_writer.write(depth_resized)
            
            # Update recording duration display
            if self.recording_start_time is not None:
                duration = time.time() - self.recording_start_time
                mins, secs = divmod(int(duration), 60)
                print(f"\rRecording duration: {mins:02d}:{secs:02d}", end='', flush=True)
            
            # Update point cloud
            self.main_pcd.points = o3d.utility.Vector3dVector(points)
            
            if self.first_geometry:
                self.vis.add_geometry(self.main_pcd)
                self.first_geometry = False
            else:
                self.vis.update_geometry(self.main_pcd)
            
            self.vis.poll_events()
            self.vis.update_renderer()
            
            cv2.waitKey(1)

        except Exception as e:
            print(f"\nVisualization error: {str(e)}")

    def init_streams(self):
        """Initialize HoloLens streams"""
        try:
            hl2ss_lnm.start_subsystem_pv(HOST, hl2ss.StreamPort.PERSONAL_VIDEO, shared=True)
            print("PV subsystem started successfully")

            global calibration_lt
            calibration_lt = hl2ss_3dcv.get_calibration_rm(HOST, 
                                                          hl2ss.StreamPort.RM_DEPTH_LONGTHROW, 
                                                          CALIBRATION_PATH)
            print("Calibration loaded successfully")

            self.producer = hl2ss_mp.producer()
            self.producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO, 
                                  hl2ss_lnm.rx_pv(HOST, 
                                                 hl2ss.StreamPort.PERSONAL_VIDEO, 
                                                 width=PV_WIDTH, 
                                                 height=PV_HEIGHT, 
                                                 framerate=PV_FRAMERATE))
            self.producer.configure(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, 
                                  hl2ss_lnm.rx_rm_depth_longthrow(HOST, 
                                                                 hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
            
            self.producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, 
                                   BUFFER_LENGTH * PV_FRAMERATE)
            self.producer.initialize(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, 
                                   BUFFER_LENGTH * hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS)
            
            self.producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
            self.producer.start(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)
            print("Streams started successfully")

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
            print("Consumer initialized successfully")

        except Exception as e:
            print(f"Error initializing streams: {str(e)}")
            self.cleanup()
            sys.exit(1)

    def run(self):
        """Main run loop"""
        try:
            print("Initializing HoloLens camera with detection...")
            self.init_keyboard_listener()
            self.init_visualizers()
            self.init_streams()
            self.init_video_writers()
            
            uv2xy = calibration_lt.uv2xy
            xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)
            
            print("Starting main loop. Press SPACE to exit.")
            
            while enable:
                # Get depth frame
                _, data_depth = self.sink_depth.get_most_recent_frame()
                if data_depth is None or not hl2ss.is_valid_pose(data_depth.pose):
                    continue

                # Get nearest PV frame
                _, data_pv = self.sink_pv.get_nearest(data_depth.timestamp)
                if data_pv is None or not hl2ss.is_valid_pose(data_pv.pose):
                    continue

                # Process frame with detection
                frame, depth_frame, points = self.process_frame_with_detection(
                    data_pv, data_depth, scale, xy1
                )
                if frame is not None and points is not None:
                    self.update_visualization(frame, depth_frame, points)

        except Exception as e:
            print(f"Runtime error: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        try:
            # Make sure to properly release video writers
            if self.pv_video_writer is not None:
                self.pv_video_writer.release()
                print("PV video writer released")
            if self.depth_video_writer is not None:
                self.depth_video_writer.release()
                print("Depth video writer released")
                
            if self.sink_pv:
                self.sink_pv.detach()
            if self.sink_depth:
                self.sink_depth.detach()
            if self.producer:
                self.producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
                self.producer.stop(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)
            
            hl2ss_lnm.stop_subsystem_pv(HOST, hl2ss.StreamPort.PERSONAL_VIDEO)
            
            if self.listener:
                self.listener.join()
            
            cv2.destroyAllWindows()
            if self.vis:
                self.vis.destroy_window()
                
            if self.recording_start_time is not None:
                duration = time.time() - self.recording_start_time
                mins, secs = divmod(int(duration), 60)
                print(f"\nRecording completed. Duration: {mins:02d}:{secs:02d}")
                
            print("Cleanup completed")
        except Exception as e:
            print(f"Cleanup error: {str(e)}")

if __name__ == "__main__":
    detector = HoloLensDetection()
    detector.run()