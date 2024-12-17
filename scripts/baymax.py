from triggered_detection import HoloLensDetection
from flask import jsonify, Flask, request
from utils import objects_to_json, classes, objects_to_json_collisions
from scene_description import GeminiClient
from google_maps import GoogleMapsClient
import time
import argparse
import json

# add argparse arguments
parser = argparse.ArgumentParser(description="Parse API KEY")
parser.add_argument("--GEMINI_API_KEY", default="AIzaSyD9yl2md73tLb9XkkC56m4T4KVH8yFmsVg", help="API Key for Gemini")
parser.add_argument("--MAPS_API_KEY", default="AIzaSyALQcPwKRLlcEzmVIhbJ5954lstzc5XQBc", help="API Key for Google Maps")
parser.add_argument("--IP", default="192.168.1.245", help="hololense IP address")
args_cli = parser.parse_args()

flask_server = Flask(__name__)
app = HoloLensDetection(IP_ADDRESS=args_cli.IP)
gemini_client = GeminiClient(args_api_key=args_cli.GEMINI_API_KEY)
google_maps_client = GoogleMapsClient(args_api_key=args_cli.MAPS_API_KEY)

@flask_server.route('/initialize_streams', methods=['GET'])
def init_streams():
    #app.start()
    return jsonify({"result" : "Streams Initialized"})

@flask_server.route('/stop_streams', methods=['GET'])
def stop_streams():
    #app.cleanup()
    return jsonify({"result" : "Streams cleaed up"})

@flask_server.route('/calibrate_detector', methods=['GET'])
def calibrate_detector():
    print("Calibration Started")
    try:
        for _ in range(50): # Sometimes it takes a couple loops for the stream to set in
            if app.init_head_pose():
                print("Calibration Success")
                return jsonify({"result" : "Calibration Success"})
        
        print("Could not initialize head pose")
        return jsonify({"result" : "Head Pose Initialization Failed"})
    except Exception as e:
        print("Calibration Failed:", e)
        return jsonify({"result" : str(e)})

def get_scene_description(frame, detected_objects=None, user_prompt="Describe"):
    """Get scene description using Gemini"""
    try:
        if frame is None:
            return None
        # Default prompt for scene description
        prompt = user_prompt
        prompt += "I'm referring to the image. Take into account that I am blind. Provide a compact response with maximum 3 sentences."
        if user_prompt == "Describe":
            prompt = "Describe this photo to help understand the environment in 1-2 sentences and provide instructions for navigation."
            if detected_objects is not None:
                # If we have detected objects, use them in the prompt
                prompt = ("Describe this photo in 1-2 sentences to help understand the environment, for a blind person." +
                          "Report the closest objects, considering also the detected objects and their depths." +
                          "Provide instructions for navigation.")
                prompt += "I see "
                for obj in detected_objects:
                    prompt += classes[obj.label] + ", at depth " + str(obj.depth) + " ,"
                prompt = prompt[:-2] + "."
        # Get scene description from Gemini
        description = gemini_client.analyze_image(frame, prompt)
        print("\nGemini Scene Description:")
        print(description)  
        return description
    except Exception as e:
        print(f"Gemini description error: {str(e)}")
        return None
    
# @flask_server.route('/collision', methods=['GET'])
# def collision_event():
#     # make a short pause
    
#     try:
#          # Run detector and capture objects
#         print("Request from unity app arrived to the flask server!")
#         floor_detected, objects, _, _ = app.run_collision_cycle()
#         print("Detector ran successfully")
#         if floor_detected:
#             print("floor")
#             return objects_to_json_collisions(objects)
#         else:
#             print("No floor")
#             return ("", 204)
#     except Exception as e:
#         print("Detector Failed:", e)
#         return ("", 204)
    
@flask_server.route('/collision', methods=['GET'])
def collision_heading():
    # make a short pause
    try:
         # Run detector and capture objects
        print("Request from unity app arrived to the flask server!")
        floor_detected, objects , _ , heading_obj = app.run_collision_cycle()
        print("Detector ran successfully")
        if floor_detected:
            print("floor")
            if len(heading_obj + objects) == 0:
                return ("", 204)
            return_json = {"heading":objects_to_json_collisions(heading_obj),
                           "obstacles":objects_to_json_collisions(objects)}
            return jsonify(return_json)
        else:
            print("No floor")
            if len(objects) == 0:
                return ("", 204)
            return_json = {"obstacles":objects_to_json_collisions(objects)}
            return jsonify(return_json)
    except Exception as e:
        print("Detector Failed:", e)
        return ("", 204)

@flask_server.route('/api', methods=['GET', 'POST'])
def handle_speech():
    # TODO: put app.start() and app.cleanup() in here, so the stream starts new everytime
    if request.method == 'GET':
        return "API is working! Send a POST request to use this endpoint."

    # Handle POST request as usual
    speech_text = request.form.get('speechText')
    if not speech_text:
        return jsonify({'response': 'No speech text provided'})
    try:
        print("Request from unity app arrived to the flask server!")
        for _ in range(50):
            objects = app.run_detection_cycle()
            if app.latest_frame.any(): break

        gemini_description = None
        if app.latest_frame is not None:
            gemini_description = get_scene_description(app.latest_frame, objects, speech_text)
        #app.cleanup()
        return jsonify({'response': gemini_description})

    except Exception as e:
        print("Detector Failed:", e)
        return jsonify({'response': 'No frame recieved'})

@flask_server.route('/directions', methods=['GET', 'POST'])
def main_directions():
    if request.method == 'GET':
        return "API is working! Send a POST request to use this endpoint."

    # Handle POST request as usual
    address = request.form.get('speechText')
    if not address:
        return jsonify({'response': 'No speech text provided'}), 400

    try:
        print("Google maps request from unity app arrived to the flask server!")
        main_instructions, stop_coordinates = google_maps_client.get_main_directions(address)

        if main_instructions is not None and stop_coordinates is not None:
            response = jsonify(main_instructions=main_instructions, stop_coordinates=stop_coordinates)
            print(response.get_json())
            return response
        else:
            print("Directions Failed")
            return None

    except Exception as e:
        print("Directions Failed:", e)
        return None

@flask_server.route('/walking_directions', methods=['GET', 'POST'])
def walking_directions():
    if request.method == 'GET':
        return "API is working! Send a POST request to use this endpoint."

    # Handle POST request as usual
    gps_coords = request.form.get('speechText')
    if not gps_coords:
        return jsonify({'error': 'No speech text provided'}), 400

    try:
        print("Request from unity app arrived to the flask server!")
        subinstructions = google_maps_client.get_walking_directions(gps_coords)

        if subinstructions is not None:
            return jsonify(subinstructions=subinstructions)
        else:
            print("Directions Failed")
            return None

    except Exception as e:
        print("Directions Failed:", e)
        return None

@flask_server.route('/compare_gps', methods=['GET', 'POST'])
def compare_gps():
    if request.method == 'GET':
        return "API is working! Send a POST request to use this endpoint."

    # Handle POST request as usual
    target_lat = request.form.get('target_lat')
    target_lng = request.form.get('target_lng')

    if not target_lng or not target_lat:
        return jsonify({'error': 'at least one gps coordinate was not provided'}), 400

    try:
        print("Request from unity app arrived to the flask server!")
        distance_to_target = google_maps_client.compute_distance_to_target(target_lat, target_lng)

        if distance_to_target is not float('nan'):
            return jsonify(distance_to_target=distance_to_target)
        else:
            print("GPS comparison failed")
            return ("", 204)

    except Exception as e:
        print("GPS comparison failed:", e)
        return ("", 204)


if __name__ == '__main__':
    # Start the Processor -----------------------------------------------------
    app.start()  # TODO:Remove here and call init function
    try:
        flask_server.run(host="0.0.0.0", port=6000, debug=False)
    finally:
        print("Stopping the Processor")
        # Cleanup the detector ------------------------------------------------ 
        app.cleanup() # TODO:Remove here and call stop function
