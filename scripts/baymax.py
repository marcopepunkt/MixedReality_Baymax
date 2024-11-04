

from triggered_detection import HoloLensDetection

from pynput import keyboard
from flask import jsonify, Flask


flask_server = Flask(__name__)

@flask_server.route('/transform', methods=['GET'])
def trigger_event():
    try:
         # Run detector and capture objects
        #print(type(app))
        #
        print(app.height)
        objects = app.run()
        print("Detector ran successfully")
    except Exception as e:
        print("Detector Failed:", e)

    print("Objects detected:", objects)

    return jsonify(objects)


if __name__ == '__main__':
    # Start the Processor -----------------------------------------------------
    app = HoloLensDetection(IP_ADDRESS="169.254.174.24")
    app.start()

    try:
        flask_server.run(host="127.0.0.1", port=5000, debug=False)
    finally:

        print("Stopping the Processor")
        # Cleanup the detector ------------------------------------------------ 
        app.cleanup()



