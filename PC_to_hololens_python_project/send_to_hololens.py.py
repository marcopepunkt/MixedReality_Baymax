from flask import Flask, jsonify, request
import random

app = Flask(__name__)

# Refined YOLO classes with precise priority levels
yolo_classes_with_priority = {
    "bus": 1,
    "car": 1,
    "motorcycle": 1,
    "pedestrian": 1,
    "stop_sign": 2,
    "traffic_light": 2,
    "fire_hydrant": 3
}

@app.route('/transform', methods=['GET'])
def get_classes():
    # Generate a random number of transformations between 1 and 3
    num_transformations = random.randint(1, 3)
    transformations = []

    for i in range(num_transformations):
        random_x = random.uniform(-1.8, 1.8)
        random_y = random.uniform(1.5, 2.0)
        random_z = random.uniform(1, 10)
        class_name = random.choice(list(yolo_classes_with_priority.keys()))  # Pick a random YOLO object
        priority = yolo_classes_with_priority[class_name]  # Use the fixed priority

        transform_data = {
            'class_name': class_name,
            'priority': priority,
            'x': random_x,
            'y': random_y,
            'z': random_z
        }

        transformations.append(transform_data)

    # Sort by priority first, then by z value
    transformations.sort(key=lambda t: (t['priority'], t['z']))

    # Return the list of transformations as JSON
    return jsonify(transformations)

@app.route('/api', methods=['GET', 'POST'])
def handle_speech():
    if request.method == 'GET':
        return "API is working! Send a POST request to use this endpoint."
    
    # Handle POST request as usual
    speech_text = request.form.get('speechText')
    if not speech_text:
        return jsonify({'error': 'No speech text provided'}), 400
    response_text = f"You said: {speech_text}. This is the server response!"
    return jsonify({'response': response_text})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 5000, debug=True)
