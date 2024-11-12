from triggered_detection import HoloLensDetection


from pynput import keyboard
from flask import jsonify, Flask
from utils import objects_to_json

flask_server = Flask(__name__)
app = HoloLensDetection(IP_ADDRESS="172.20.10.3")

from scene_description import GeminiClient
gemini_client = GeminiClient()
def get_scene_description(frame, detected_objects=None):
    """Get scene description using Gemini"""
    try:
        if frame is None:
            return None
        # Default prompt for scene description
        prompt = "Describe this photo to help understand the environment and assist with navigation."
        if detected_objects is not None:
            # If we have detected objects, use them in the prompt
            prompt = "Describe this photo to help understand the environment and assist with navigation. "
            prompt += "I see "
            for obj in detected_objects:
                prompt += obj.label + ", "
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
        
        objects, image_description = app.run()
        # TODO: natural language processing uses gemini instead of open ai there. return here the object and depth information
        # Get additional Gemini description 
        gemini_description = None
        if hasattr(app, 'latest_frame') and app.latest_frame is not None:
            gemini_description = get_scene_description(app.latest_frame)
        print("Detector ran successfully")
    except Exception as e:
        print("Detector Failed:", e)

    print("Final objects detected:", objects, 'with description:', gemini_description)
    return objects_to_json(objects, image_description, gemini_description)

if __name__ == '__main__':
    # Start the Processor -----------------------------------------------------
    app.start()
    try:
        flask_server.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        print("Stopping the Processor")
        # Cleanup the detector ------------------------------------------------ 
        app.cleanup()



