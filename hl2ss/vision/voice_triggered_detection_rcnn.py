import sys
import os
import time
from typing import List, Tuple
import threading
from pynput import keyboard
import numpy as np
import cv2
import torch
import winsound
import multiprocessing as mp
from datetime import datetime

import pyttsx3
import threading

# Import HoloLens libraries
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv

# Import detection utilities
from real_time import detect_for_rcnn
import openvino as ov

from detecto.core import Model
from detecto import utils, visualize
import ssl

# Settings
HOST = '169.254.174.24'
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
COLORS = np.random.randint(0, 255, size=(len(detect_for_rcnn.classes), 3), dtype=np.uint8)
TEXT_COLOR = (255, 255, 255)
TEXT_BACKGROUND = (0, 0, 0)

# Global variables
enable = True
calibration_lt = None
last_detection_time = 0
DETECTION_COOLDOWN = 2

class HoloLensVoiceDetection:
    def __init__(self):
        self.producer = None
        self.consumer = None
        self.sink_pv = None
        self.sink_depth = None
        self.keyboard_listener = None
        self.voice_client = None
        self.latest_frame = None
        self.latest_depth = None
        self.detection_active = False
        self.detection_results = []
        self.last_detection_time = time.time()

        # Initialize TTS engine for voice notifications
        self.tts_engine = None
        self.init_tts()

        # Initialize OpenVINO
        self.core = ov.Core()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        # detection_model_path = os.path.join(parent_dir, 'real_time\model', 'ssdlite_mobilenet_v2_fp16.xml')
        
        # Create calibration directory if it doesn't exist
        os.makedirs(CALIBRATION_PATH, exist_ok=True)
        
        # Load detection model
        # detection_model = self.core.read_model(model=detection_model_path)
        self.compiled_model = Model(device=torch.device('cpu'), pretrained=True)
        # self.input_layer = self.compiled_model.input(0)
        # self.output_layer = self.compiled_model.output(0)
        # self.height, self.width = list(self.input_layer.shape)[1:3]

    def init_keyboard(self):
        """Initialize keyboard listener"""
        def on_press(key):
            global enable
            enable = key != keyboard.Key.esc
            return enable

        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()
        print("Keyboard listener initialized")

    def init_voice(self):
        """Initialize voice recognition"""
        try:
            self.voice_client = hl2ss_lnm.ipc_vi(HOST, hl2ss.IPCPort.VOICE_INPUT)
            self.voice_client.open()
            self.voice_client.create_recognizer()
            
            if not self.voice_client.register_commands(True, VOICE_COMMANDS):
                print("Failed to register voice commands")
                raise RuntimeError("Voice command registration failed")
                
            print("Voice recognition initialized")
            self.voice_client.start()
            
        except Exception as e:
            print(f"Error initializing voice recognition: {str(e)}")
            raise

    def init_streams(self):
        """Initialize HoloLens streams"""
        try:
            hl2ss_lnm.start_subsystem_pv(HOST, hl2ss.StreamPort.PERSONAL_VIDEO, shared=True)
            print("PV subsystem started")

            global calibration_lt
            calibration_lt = hl2ss_3dcv.get_calibration_rm(HOST, 
                                                          hl2ss.StreamPort.RM_DEPTH_LONGTHROW, 
                                                          CALIBRATION_PATH)

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

    def init_tts(self):
        """Initialize text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            # Configure voice properties
            self.tts_engine.setProperty('rate', 150)    # Speed of speech
            self.tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
            
            # Get available voices and set to english
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if "english" in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
                    
            print("Text-to-speech initialized")
        except Exception as e:
            print(f"Text-to-speech initialization error: {str(e)}")
            self.tts_engine = None

    def speak_text(self, text):
        """Speak text in a separate thread"""
        if self.tts_engine is not None:
            # Create a thread for speech to avoid blocking
            thread = threading.Thread(target=self.tts_engine.say, args=(text,))
            thread.start()
            # Run the engine in the thread
            self.tts_engine.runAndWait()
            
    def announce_detections(self, boxes, poses):
        """Announce detected objects and their distances"""
        if not boxes:
            self.speak_text("No objects detected")
            return
            
        # Create announcement text
        announcements = []
        for (label, score, _), pose in zip(boxes, poses):
            class_name = detect_for_rcnn.classes[label]
            # class_name = label
            if pose is not None and not np.isnan(pose.depth):
                distance = f"{pose.depth:.1f} meters"
                announcement = f"{class_name} at {distance}"
            else:
                announcement = f"{class_name} detected"
            announcements.append(announcement)
        
        # Combine announcements
        if announcements:
            text = "I see: " + ". ".join(announcements)
            self.speak_text(text)

    def play_sound(self, frequency, duration):
        """Play a sound notification"""
        try:
            winsound.Beep(frequency, duration)
        except:
            pass

    def draw_detection_results(self, frame, boxes, poses):
        """Draw detection results on the frame"""
        vis_frame = frame.copy()
        overlay = vis_frame.copy()
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(vis_frame, f"Detection Time: {timestamp}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

        for (label, score, box), pose in zip(boxes, poses):
            color = tuple(map(int, COLORS[label]))
            x, y, w, h = box
            
            # Draw box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)

            # Create label
            class_name = detect_for_rcnn.classes[label]
            # class_name = label
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

    def process_detection(self, frame, depth_data, scale):
        """Process detection on current frame"""
        try:
            if frame is None:
                return None, None, None
        
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
            # input_img = cv2.resize(src=frame, dsize=(self.width, self.height), interpolation=cv2.INTER_AREA)
            # input_img = input_img[np.newaxis, ...]
            input_img = frame

            labels, boxes, scores = self.compiled_model.predict(input_img)
            boxes = detect_for_rcnn.process_results(frame, labels, boxes, scores, thresh=0.5)

            sensor_depth = hl2ss_3dcv.rm_depth_normalize(depth_data, scale)
            sensor_depth[sensor_depth > MAX_SENSOR_DEPTH] = 0

            poses = detect_for_rcnn.estimate_box_poses(sensor_depth, boxes)
            vis_frame = self.draw_detection_results(frame, boxes, poses)

            self.announce_detections(boxes, poses)

            print("\nDetected objects:")
            for (label, score, box), pose in zip(boxes, poses):
                if pose is not None and not np.isnan(pose.depth):
                    print(f"- {detect_for_rcnn.classes[label]} (confidence: {score:.2f}) at {pose.depth:.2f} meters")
                else:
                    print(f"- {detect_for_rcnn.classes[label]} (confidence: {score:.2f}) at unknown distance")

            return boxes, poses, vis_frame

        except Exception as e:
            print(f"Detection error: {str(e)}")
            return None, None, frame

    def run(self):
        """Main run loop"""
        try:
            print("Initializing...")
            self.init_keyboard()
            self.init_voice()
            self.init_streams()
            
            cv2.namedWindow("HoloLens Detection", cv2.WINDOW_NORMAL)
            
            uv2xy = calibration_lt.uv2xy
            xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)
            
            print("Ready! Say 'detect' to perform object detection.")
            self.play_sound(1500, 200)
            
            while enable:
                try:
                    # Get depth frame
                    _, data_depth = self.sink_depth.get_most_recent_frame()
                    if data_depth is None or not hl2ss.is_valid_pose(data_depth.pose):
                        print("No valid depth frame")
                        continue

                    # Get PV frame
                    _, data_pv = self.sink_pv.get_nearest(data_depth.timestamp)
                    if data_pv is None or not hl2ss.is_valid_pose(data_pv.pose):
                        print("No valid PV frame")
                        continue

                    # Get frame and check if it's valid
                    frame = data_pv.payload.image
                    if frame is None:
                        #print("Invalid frame")
                        continue
                    
                    depth = data_depth.payload.depth
                    if depth is None:
                        print("Invalid depth")
                        continue

                    # Create visualization frame
                    vis_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR

                    # Check voice commands
                    events = self.voice_client.pop()
                    for event in events:
                        event.unpack()
                        if event.index == 0:  # "detect" command
                            current_time = time.time()
                            if current_time - self.last_detection_time >= DETECTION_COOLDOWN:
                                print("\nDetection triggered by voice command")
                                self.play_sound(DETECTION_SOUND_FREQ, DETECTION_SOUND_DURATION)
                                self.detection_active = True
                                boxes, poses, vis_frame = self.process_detection(frame, depth, scale)
                                if boxes is not None:
                                    self.last_detection_time = current_time
                                else:
                                    print("Detection failed")
                            else:
                                self.play_sound(ERROR_SOUND_FREQ, DETECTION_SOUND_DURATION)
                                print("\nPlease wait before next detection")

                    # Show frame
                    if vis_frame is not None:
                        cv2.imshow("HoloLens Detection", vis_frame)
                        cv2.waitKey(1)

                except Exception as e:
                    print(f"Frame processing error: {str(e)}")
                    continue

        except Exception as e:
            print(f"Runtime error: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        try:
            cv2.destroyAllWindows()
            
            if self.voice_client is not None:
                self.voice_client.stop()
                self.voice_client.clear()
                self.voice_client.close()
                
            if self.sink_pv:
                self.sink_pv.detach()
            if self.sink_depth:
                self.sink_depth.detach()
            if self.producer:
                self.producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
                self.producer.stop(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)
            if self.tts_engine is not None:
                self.tts_engine.stop()
            
            hl2ss_lnm.stop_subsystem_pv(HOST, hl2ss.StreamPort.PERSONAL_VIDEO)
            
            if self.keyboard_listener:
                self.keyboard_listener.join()
                
            print("Cleanup completed")
        except Exception as e:
            print(f"Cleanup error: {str(e)}")

if __name__ == "__main__":
    detector = HoloLensVoiceDetection()
    detector.run()