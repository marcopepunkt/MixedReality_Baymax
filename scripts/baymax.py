from triggered_detection import HoloLensDetection

from pynput import keyboard
from flask import jsonify, Flask
from utils import objects_to_json

flask_server = Flask(__name__)
app = HoloLensDetection(IP_ADDRESS="172.20.10.3")

@flask_server.route('/transform', methods=['GET'])
def trigger_event():
    try:
         # Run detector and capture objects
        print("Request from unity app arrived to the flask server!")
        objects, image_description = app.run()
        print("Detector ran successfully")
    except Exception as e:
        print("Detector Failed:", e)

    print("Final objects detected:", objects)
    return objects_to_json(objects, image_description)

if __name__ == '__main__':
    # Start the Processor -----------------------------------------------------
    app.start()
    try:
        flask_server.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        print("Stopping the Processor")
        # Cleanup the detector ------------------------------------------------ 
        app.cleanup()



