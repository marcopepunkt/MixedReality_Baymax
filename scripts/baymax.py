

from triggered_detection import HoloLensDetection

from pynput import keyboard
from flask import jsonify, Flask


flask_server = Flask(__name__)

@flask_server.route('/transform', methods=['GET'])
def trigger_event():
    try:
         # Run detector and capture objects
        objects = app.run()
        print("Detector ran successfully")
    except Exception as e:
        print("Detector Failed:", e)

    print("Objects detected:", objects)

    return jsonify(objects)


if __name__ == '__main__':
    # Start the Processor -----------------------------------------------------
    app = HoloLensDetection(IP_ADDRESS="192.168.0.31")
    app.start()

    try:
        flask_server.run(host="192.168.0.30", port=5000, debug=True)
    finally:

        print("Stopping the Processor")
        # Cleanup the detector ------------------------------------------------ 
        app.cleanup()



