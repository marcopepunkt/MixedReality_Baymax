from voice_triggered_detection import HoloLensVoiceDetection

from pynput import keyboard

class BayMax:
     
    def __init__(self):
        self.detector = HoloLensVoiceDetection()
        self.keyboard_listener = None

    def init_keyboard(self):
        """Initialize keyboard listener"""
        def on_press(key):
            global enable
            enable = key != keyboard.Key.esc
            return enable

        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()
        print("Keyboard listener initialized")

if __name__ == '__main__':

    app = BayMax()
    app.init_keyboard()

    while enable:
        if app.detector.listen():
            try: poses = app.detector.run()
            except Exception as e:
                print(f"Detector Failed")
                break

    app.detector.cleanup()

    # Stop keyboard events ----------------------------------------------------
    app.keyboard_listener.join()


