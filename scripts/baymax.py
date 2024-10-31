from voice_triggered_detection import HoloLensVoiceDetection

from pynput import keyboard

enable = True

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
    app.detector.start()

    while enable:
        event = app.detector.listen()
        if event:
            try: objects = app.detector.run()
            except Exception as e:
                print(f"Detector Failed")
                break

    app.detector.cleanup()

    # Stop keyboard events ----------------------------------------------------
    app.keyboard_listener.join()


