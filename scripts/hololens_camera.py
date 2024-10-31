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
        """Initialize video writers for both processed and raw streams"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Files for processed videos (with detection boxes)
        processed_pv_filename = os.path.join(RECORDING_PATH, f'processed_pv_{timestamp}.avi')
        processed_depth_filename = os.path.join(RECORDING_PATH, f'processed_depth_{timestamp}.avi')
        
        # Files for raw videos - using mp4 format
        raw_pv_filename = os.path.join(RECORDING_PATH, f'raw_pv_{timestamp}.mp4')
        raw_depth_filename = os.path.join(RECORDING_PATH, f'raw_depth_{timestamp}.mp4')
        
        # XVID for processed videos
        fourcc_avi = cv2.VideoWriter_fourcc(*'XVID')
        # H264 for raw videos
        fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Writers for processed videos (for visualization)
        self.pv_video_writer = cv2.VideoWriter(
            processed_pv_filename, 
            fourcc_avi, 
            PV_FRAMERATE, 
            (PV_WIDTH, PV_HEIGHT),
            isColor=True
        )
        self.depth_video_writer = cv2.VideoWriter(
            processed_depth_filename, 
            fourcc_avi, 
            PV_FRAMERATE, 
            (PV_WIDTH, PV_HEIGHT),
            isColor=True
        )
        
        # Get the first frame to determine exact sizes
        _, data_pv = self.sink_pv.get_most_recent_frame()
        _, data_depth = self.sink_depth.get_most_recent_frame()
        
        if data_pv is not None and data_pv.payload.image is not None:
            pv_height, pv_width = data_pv.payload.image.shape[:2]
            print(f"Raw PV size: {pv_width}x{pv_height}")
            
            self.raw_pv_writer = cv2.VideoWriter(
                raw_pv_filename, 
                fourcc_mp4, 
                PV_FRAMERATE, 
                (pv_width, pv_height),
                isColor=True
            )
        else:
            print("Warning: Could not determine PV frame size")
            self.raw_pv_writer = None

        if data_depth is not None and data_depth.payload.depth is not None:
            depth_height, depth_width = data_depth.payload.depth.shape[:2]
            print(f"Raw depth size: {depth_width}x{depth_height}")
            
            self.raw_depth_writer = cv2.VideoWriter(
                raw_depth_filename, 
                fourcc_mp4, 
                PV_FRAMERATE, 
                (depth_width, depth_height),
                isColor=True  # We'll convert depth to BGR for compatibility
            )
        else:
            print("Warning: Could not determine depth frame size")
            self.raw_depth_writer = None
        
        self.recording_start_time = time.time()
        print(f"Recording started. Saving to:\n"
            f"Processed videos:\n{processed_pv_filename}\n{processed_depth_filename}\n"
            f"Raw videos:\n{raw_pv_filename}\n{raw_depth_filename}")

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
        """Process a single frame with object detection and save raw frames"""
        try:
            # Get frame from PV stream
            frame = data_pv.payload.image
            if frame is None:
                return None, None, None

            # Save raw PV frame
            if self.raw_pv_writer is not None:
                try:
                    # Ensure frame is in BGR format
                    if len(frame.shape) == 2:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.shape[2] == 4:  # BGRA format
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    else:
                        frame_bgr = frame.copy()
                    
                    # Write the frame
                    self.raw_pv_writer.write(frame_bgr)
                except Exception as e:
                    print(f"Error saving raw PV frame: {e}")

            # Save raw depth frame
            if self.raw_depth_writer is not None and data_depth.payload.depth is not None:
                try:
                    # Get raw depth data
                    raw_depth = data_depth.payload.depth
                    
                    # Create a colored visualization of the depth data
                    depth_min = np.min(raw_depth)
                    depth_max = np.max(raw_depth)
                    
                    if depth_max > depth_min:
                        # Normalize to 0-255
                        depth_normalized = ((raw_depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                        # Apply colormap for better visualization
                        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                    else:
                        depth_colored = np.zeros((*raw_depth.shape, 3), dtype=np.uint8)
                    
                    # Write the colored depth frame
                    self.raw_depth_writer.write(depth_colored)
                except Exception as e:
                    print(f"Error saving raw depth frame: {e}")

            # Continue with the rest of your processing for visualization
            # [Rest of your code remains the same]
            
            sensor_depth = hl2ss_3dcv.rm_depth_normalize(data_depth.payload.depth, scale)
            sensor_depth[sensor_depth > MAX_SENSOR_DEPTH] = 0

            input_img = cv2.resize(src=frame, dsize=(width, height), interpolation=cv2.INTER_AREA)
            input_img = input_img[np.newaxis, ...]

            results = compiled_detection_model([input_img])[output_layer]
            boxes = detect.process_results(frame=frame, results=results, thresh=0.5)
            poses = detect.estimate_box_poses(sensor_depth, boxes)
            
            frame_with_boxes = frame.copy()
            frame_with_boxes = detect.draw_depth_boxes(frame_with_boxes, boxes, poses)

            points = hl2ss_3dcv.rm_depth_to_points(sensor_depth, xy1)
            depth_to_world = hl2ss_3dcv.camera_to_rignode(calibration_lt.extrinsics) @ \
                            hl2ss_3dcv.reference_to_world(data_depth.pose)
            points = hl2ss_3dcv.transform(points, depth_to_world)

            points = hl2ss_3dcv.block_to_list(points)
            select = sensor_depth.reshape((-1,)) > 0
            points = points[select, :]

            depth_colored_vis = cv2.normalize(sensor_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_colored_vis = cv2.applyColorMap(depth_colored_vis, cv2.COLORMAP_INFERNO)

            return frame_with_boxes, depth_colored_vis, points

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
            # Release all video writers
            writers = [
                (self.pv_video_writer, "Processed PV"),
                (self.depth_video_writer, "Processed Depth"),
                (self.raw_pv_writer, "Raw PV"),
                (self.raw_depth_writer, "Raw Depth")
            ]
            
            for writer, name in writers:
                if writer is not None:
                    writer.release()
                    print(f"{name} video writer released")
                    
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