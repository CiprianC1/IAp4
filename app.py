from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import numpy as np
import onnxruntime as ort

app = Flask(__name__)

# Load the ONNX model
onnx_model_path = "emotion_model.onnx"
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Helper function to preprocess the image
def preprocess_image(base64_str):
    image_data = base64.b64decode(base64_str.split(",")[1])
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image).astype("float32") / 255.0  # Normalize
    image = np.transpose(image, (2, 0, 1))  # Channels-first format
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/analyze', methods=['POST'])
def analyze_frame():
    try:
        frame_data = request.json['frame']
        image = preprocess_image(frame_data)

        # Run the inference
        preds = session.run([output_name], {input_name: image})
        emotion_idx = np.argmax(preds[0])  # Get predicted emotion index
        emotions = ["Happy", "Sad", "Angry", "Neutral", "Surprised", "Disgust", "Fear"]
        emotion = emotions[emotion_idx]

        return jsonify({"emotion": emotion})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)
