import time
import threading

from google.api_core import retry
from google.generativeai.types import RequestOptions
from pynput import keyboard
import numpy as np
import cv2
import multiprocessing as mp
# import pygame  # For cross-platform sound
import pyttsx3
import threading

# Import HoloLens libraries
import hl2ss
import hl2ss_lnm
import hl2ss_mp

# Import Gemini dependencies
import google.generativeai as genai
from dotenv import load_dotenv

import pygame

# Settings
HOST = '10.1.0.143'
PV_WIDTH = 640
PV_HEIGHT = 360
PV_FRAMERATE = 15
BUFFER_LENGTH = 5
VOICE_COMMANDS = ['describe']

# Sound settings
DESCRIPTION_SOUND_FREQ = 1000
DESCRIPTION_SOUND_DURATION = 200
ERROR_SOUND_FREQ = 500

# Global variables
enable = True
last_description_time = 0
DESCRIPTION_COOLDOWN = 3

class AudioFeedback:
    def __init__(self):
        """Initialize audio feedback system"""
        pygame.mixer.init(44100, -16, 2, 1024)
        
        # Create beep sounds of different frequencies
        self.create_beep_sounds()
        
    def create_beep_sounds(self):
        """Create different beep sounds using pygame"""
        sample_rate = 44100
        duration = 0.2  # 200ms
        samples = int(duration * sample_rate)
        
        # Create success beep (1000 Hz)
        t = np.linspace(0, duration, samples, False)
        success_note = np.sin(2 * np.pi * 1000 * t)
        # Ensure array is C-contiguous and properly formatted for stereo
        success_sound = np.empty((samples, 2), dtype=np.int16)
        success_sound[:, 0] = success_sound[:, 1] = np.int16(success_note * 32767)
        success_sound = np.ascontiguousarray(success_sound)
        self.success_sound = pygame.sndarray.make_sound(success_sound)
        
        # Create error beep (500 Hz)
        error_note = np.sin(2 * np.pi * 500 * t)
        # Ensure array is C-contiguous and properly formatted for stereo
        error_sound = np.empty((samples, 2), dtype=np.int16)
        error_sound[:, 0] = error_sound[:, 1] = np.int16(error_note * 32767)
        error_sound = np.ascontiguousarray(error_sound)
        self.error_sound = pygame.sndarray.make_sound(error_sound)
        
    def play_success(self):
        """Play success sound"""
        try:
            self.success_sound.play()
            pygame.time.wait(200)  # Wait for sound to finish
        except Exception as e:
            print(f"Error playing success sound: {e}")
        
    def play_error(self):
        """Play error sound"""
        try:
            self.error_sound.play()
            pygame.time.wait(200)  # Wait for sound to finish
        except Exception as e:
            print(f"Error playing error sound: {e}")
        
    def cleanup(self):
        """Cleanup pygame mixer"""
        try:
            pygame.mixer.quit()
        except:
            pass

class GeminiClient:
    def __init__(self,args_api_key):
        """Initialize Gemini client"""
        load_dotenv()
        genai.configure(api_key=args_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
    
    def analyze_image(self, image_array, prompt: str) -> str:
        """Analyze an image with the given prompt"""
        try:
            # Convert numpy array to bytes
            success, buffer = cv2.imencode('.jpg', image_array)
            if not success:
                raise ValueError("Failed to encode image")
                
            # Create image part directly from bytes
            image_bytes = buffer.tobytes()
            image_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": image_bytes
                }
            ]
            
            # Add prompt to generation config
            prompt_parts = [prompt]
            
            # Generate description
            response = self.model.generate_content(prompt_parts + image_parts, request_options=RequestOptions(retry=retry.Retry(initial=10, multiplier=2, maximum=60, timeout=300)))
            
            # Check if response was blocked
            if response.prompt_feedback.block_reason:
                raise ValueError(f"Response blocked: {response.prompt_feedback.block_reason}")
                
            return response.text

        except Exception as e:
            print(f"Gemini analysis error: {str(e)}")
            return None

class HoloLensSceneDescription:
    def __init__(self):
        self.producer = None
        self.consumer = None
        self.sink_pv = None
        self.keyboard_listener = None
        self.voice_client = None
        self.latest_frame = None
        self.description_active = False
        
        # Initialize audio feedback
        self.audio = AudioFeedback()
        
        # Initialize TTS engine
        self.tts_engine = None
        self.init_tts()
        
        # Initialize Gemini client
        self.gemini_client = GeminiClient()
        
        # Default prompt for scene description
        self.default_prompt = "Describe this photo to a blind person to help them understand where they are and for navigation purposes."

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
            print("\nInitializing voice recognition...")
            self.voice_client = hl2ss_lnm.ipc_vi(HOST, hl2ss.IPCPort.VOICE_INPUT)
            print("Voice client created")
            
            self.voice_client.open()
            print("Voice client opened")
            
            self.voice_client.create_recognizer()
            print("Voice recognizer created")
            
            print(f"Registering voice commands: {VOICE_COMMANDS}")
            success = self.voice_client.register_commands(True, VOICE_COMMANDS)
            if not success:
                print("Failed to register voice commands")
                raise RuntimeError("Voice command registration failed")
            print("Voice commands registered successfully")
            
            self.voice_client.start()
            print("Voice recognition started and listening")
            
        except Exception as e:
            print(f"Error initializing voice recognition: {str(e)}")
            raise

    def check_voice_commands(self):
        """Check and process voice commands"""
        try:
            events = self.voice_client.pop()
            if events:  # Only print if there are events
                print(f"\nReceived {len(events)} voice events")
                
            for event in events:
                print(f"Processing voice event...")
                event.unpack()
                print(f"Event index: {event.index}")
                
                if event.index == 0:  # "describe" command
                    current_time = time.time()
                    if current_time - self.last_description_time >= DESCRIPTION_COOLDOWN:
                        print("\nDescription triggered by voice command")
                        self.audio.play_success()
                        self.description_active = True
                        
                        print("Capturing frame for description...")
                        frame = self.latest_frame
                        if frame is None:
                            print("No valid frame available for description")
                            return
                            
                        print("Processing frame through Gemini...")
                        description = self.process_description(frame)
                        
                        if description is not None:
                            print("Description generated successfully")
                            self.last_description_time = current_time
                        else:
                            print("Description generation failed")
                    else:
                        self.audio.play_error()
                        print(f"\nPlease wait before next description. Cooldown: {DESCRIPTION_COOLDOWN - (current_time - self.last_description_time):.1f}s")
                        
        except Exception as e:
            print(f"Error processing voice commands: {e}")

    def init_streams(self):
        """Initialize HoloLens streams"""
        try:
            hl2ss_lnm.start_subsystem_pv(HOST, hl2ss.StreamPort.PERSONAL_VIDEO, shared=True)
            print("PV subsystem started")

            self.producer = hl2ss_mp.producer()
            self.producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO, 
                                  hl2ss_lnm.rx_pv(HOST, 
                                                 hl2ss.StreamPort.PERSONAL_VIDEO, 
                                                 width=PV_WIDTH, 
                                                 height=PV_HEIGHT, 
                                                 framerate=PV_FRAMERATE))
            
            self.producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, 
                                   BUFFER_LENGTH * PV_FRAMERATE)
            
            self.producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)

            manager = mp.Manager()
            self.consumer = hl2ss_mp.consumer()
            self.sink_pv = self.consumer.create_sink(self.producer, 
                                                    hl2ss.StreamPort.PERSONAL_VIDEO, 
                                                    manager, None)
            
            self.sink_pv.get_attach_response()
            print("Streams initialized")

        except Exception as e:
            print(f"Error initializing streams: {str(e)}")
            raise

    def init_tts(self):
        """Initialize text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
            
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
            thread = threading.Thread(target=self.tts_engine.say, args=(text,))
            thread.start()
            self.tts_engine.runAndWait()

    def process_description(self, frame):
        """Process scene description using Gemini"""
        try:
            if frame is None:
                return None
            
            # Create a window to show original BGRA frame
            cv2.namedWindow("Original BGRA Frame", cv2.WINDOW_NORMAL)
            cv2.imshow("Original BGRA Frame", frame)

            # Convert BGRA to BGR if necessary
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            

            # Show the BGR frame that will be sent to Gemini
            cv2.namedWindow("Frame Sent to Gemini", cv2.WINDOW_NORMAL)
            cv2.imshow("Frame Sent to Gemini", frame)
            cv2.waitKey(1)  # Update the display

            # Get scene description from Gemini
            description = self.gemini_client.analyze_image(frame, self.default_prompt)
            
            # Speak the description
            self.speak_text(description)
            
            # Print the description
            print("\nScene Description:")
            print(description)
            
            return description

        except Exception as e:
            print(f"Description error: {str(e)}")
            return None
        
    def create_window(self):
        """Create OpenCV window safely"""
        try:
            print("Creating window...")
            # Try to destroy any existing windows first
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            
            # Create new window with a shorter timeout
            cv2.namedWindow("HoloLens View", cv2.WINDOW_NORMAL)
            cv2.waitKey(1)  # Process any pending events
            
            # Set window size
            cv2.resizeWindow("HoloLens View", PV_WIDTH, PV_HEIGHT)
            cv2.waitKey(1)
            
            self.window_created = True
            print("Window created successfully")
        except Exception as e:
            print(f"Warning: Failed to create window: {e}")
            self.window_created = False

    def run(self):
        """Main run loop"""
        try:
            print("Initializing...")
            self.init_keyboard()
            self.init_voice()
            self.init_streams()
            
            # Initialize tkinter window
            import tkinter as tk
            from PIL import Image, ImageTk
            
            root = tk.Tk()
            root.title("HoloLens View")
            label = tk.Label(root)
            label.pack()
            
            def update_image():
                try:
                    if hasattr(self, 'latest_frame') and self.latest_frame is not None:
                        # Convert BGRA to RGB
                        rgb_frame = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGRA2RGB)
                        # Convert to PIL Image
                        image = Image.fromarray(rgb_frame)
                        # Convert to PhotoImage
                        photo = ImageTk.PhotoImage(image=image)
                        # Update label
                        label.configure(image=photo)
                        label.image = photo  # Keep a reference
                    root.after(33, update_image)  # Update every ~30ms
                except Exception as e:
                    print(f"Display error: {e}")
                    root.after(33, update_image)
            
            update_image()
            
            print("Ready! Say 'describe' to get a scene description.")
            
            def process_frames():
                try:
                    # Get PV frame
                    _, data_pv = self.sink_pv.get_most_recent_frame()
                    if data_pv is not None and hl2ss.is_valid_pose(data_pv.pose):
                        frame = data_pv.payload.image
                        if frame is not None:
                            self.latest_frame = frame
                    root.after(1, process_frames)
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    root.after(1, process_frames)
            
            process_frames()
            root.mainloop()

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
            cv2.waitKey(1)  # Give time for windows to close
            
            if self.voice_client is not None:
                self.voice_client.stop()
                self.voice_client.clear()
                self.voice_client.close()
                
            if self.sink_pv:
                self.sink_pv.detach()
            if self.producer:
                self.producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
            if self.tts_engine is not None:
                self.tts_engine.stop()
            
            # Cleanup audio
            self.audio.cleanup()
            
            hl2ss_lnm.stop_subsystem_pv(HOST, hl2ss.StreamPort.PERSONAL_VIDEO)
            
            if self.keyboard_listener:
                self.keyboard_listener.join()
                
            print("Cleanup completed")
        except Exception as e:
            print(f"Cleanup error: {str(e)}")

if __name__ == "__main__":




    descriptor = HoloLensSceneDescription()
    descriptor.run()