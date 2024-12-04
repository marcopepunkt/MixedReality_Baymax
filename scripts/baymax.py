from triggered_detection import HoloLensDetection
from flask import jsonify, Flask, request
from utils import objects_to_json, classes
from scene_description import GeminiClient

flask_server = Flask(__name__)
app = HoloLensDetection(IP_ADDRESS="172.20.10.3")
gemini_client = GeminiClient()

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
    
@flask_server.route('/transform', methods=['GET'])
def trigger_event():
    try:
         # Run detector and capture objects
        print("Request from unity app arrived to the flask server!")
        objects = app.run_detection_cycle()
        print("Detector ran successfully")
        return objects_to_json(objects)
    except Exception as e:
        print("Detector Failed:", e)
        return None

@flask_server.route('/collision', methods=['GET'])
def collision_event():
    try:
         # Run detector and capture objects
        print("Request from unity app arrived to the flask server!")
        floor_detected, objects = app.run_collision_cycle()
        print("Detector ran successfully")
        return objects_to_json_collisions(objects)
    except Exception as e:
        print("Detector Failed:", e)
        return None

@flask_server.route('/api', methods=['GET', 'POST'])
def handle_speech():
    if request.method == 'GET':
        return "API is working! Send a POST request to use this endpoint."

    # Handle POST request as usual
    speech_text = request.form.get('speechText')
    if not speech_text:
        return jsonify({'error': 'No speech text provided'}), 400

    try:
        print("Request from unity app arrived to the flask server!")
        objects = app.run()
        # TODO: natural language processing uses gemini instead of open ai there. return here the object and depth information
        gemini_description = None
        if app.latest_frame is not None:
            gemini_description = get_scene_description(app.latest_frame, objects, speech_text)
        return jsonify({'response': gemini_description})

    except Exception as e:
        print("Detector Failed:", e)
        return None


if __name__ == '__main__':
    # Start the Processor -----------------------------------------------------
    app.start()
    try:
        flask_server.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        print("Stopping the Processor")
        # Cleanup the detector ------------------------------------------------ 
        app.cleanup()
