from flask import Flask, request, jsonify, render_template
import base64
import io
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
import onnxruntime as ort
from bson.objectid import ObjectId
from pymongo import MongoClient
import os

app = Flask(__name__)

mongo_uri = f"mongodb://{os.environ['MONGO_SUPER_USER_NAME']}:{os.environ['MONGO_SUPER_USER_PASSWORD']}@{os.environ['MONGO_SERVER']}"
client = MongoClient(mongo_uri)
db = client.gallery
images_collection = db.images

emotion_list = ['adoration', 'affection', 'aggravation', 'agitation', 'agony', 'alarm', 'alienation', 'amazement', 'amusement', 'anger', 'anguish', 'annoyance', 'anxiety', 'apprehension', 'arousal', 'astonishment', 'attraction', 'bitterness', 'bliss', 'caring', 'cheerfulness', 'compassion', 'contempt', 'contentment', 'defeat', 'dejection', 'delight', 'depression', 'desire', 'despair', 'disappointment', 'disgust', 'dislike', 'dismay', 'displeasure', 'distress', 'dread', 'eagerness', 'ecstasy', 'elation', 'embarrassment', 'enjoyment', 'enthrallment', 'enthusiasm', 'envy', 'euphoria', 'exasperation', 'excitement', 'exhilaration', 'fear', 'ferocity', 'fondness', 'fright', 'frustration', 'fury', 'gaiety', 'gladness', 'glee', 'gloom', 'glumness', 'grief', 'grouchiness', 'grumpiness', 'guilt', 'happiness', 'hate', 'homesickness', 'hope', 'hopelessness', 'horror', 'hostility', 'humiliation', 'hurt', 'hysteria', 'infatuation', 'insecurity', 'insult', 'irritation', 'isolation', 'jealousy', 'jolliness', 'joviality', 'joy', 'jubilation', 'liking', 'loathing', 'loneliness', 'longing', 'love', 'lust', 'melancholy', 'misery', 'mortification', 'neglect', 'nervousness', 'optimism', 'outrage', 'panic', 'passion', 'pity', 'pleasure', 'pride', 'rage', 'rapture', 'regret', 'rejection', 'relief', 'remorse', 'resentment', 'revulsion', 'sadness', 'satisfaction', 'scorn', 'sentimentality', 'shame', 'shock', 'sorrow', 'spite', 'suffering', 'surprise', 'sympathy', 'tenderness', 'tenseness', 'terror', 'thrill', 'torment', 'triumph', 'uneasiness', 'unhappiness', 'vengefulness', 'woe', 'worry', 'wrath', 'zeal', 'zest']
label_to_idx = {emotion: idx for idx, emotion in enumerate(emotion_list)}

primary_emotion_mapping = {'adoration': 'Love','affection': 'Love','aggravation': 'Anger','agitation': 'Fear','agony': 'Sadness','alarm': 'Fear','alienation': 'Sadness','amazement': 'Surprise','amusement': 'Happiness','anger': 'Anger','anguish': 'Sadness','annoyance': 'Anger','anxiety': 'Fear','apprehension': 'Fear','arousal': 'Desire','astonishment': 'Surprise','attraction': 'Love','bitterness': 'Anger','bliss': 'Happiness','caring': 'Love','cheerfulness': 'Happiness','compassion': 'Love','contempt': 'Disgust','contentment': 'Happiness','defeat': 'Sadness','dejection': 'Sadness','delight': 'Happiness','depression': 'Sadness','desire': 'Desire','despair': 'Sadness','disappointment': 'Sadness','disgust': 'Disgust','dislike': 'Disgust','dismay': 'Sadness','displeasure': 'Disgust','distress': 'Sadness','dread': 'Fear','eagerness': 'Desire','ecstasy': 'Happiness','elation': 'Happiness','embarrassment': 'Fear','enjoyment': 'Happiness','enthrallment': 'Happiness','enthusiasm': 'Happiness','envy': 'Anger','euphoria': 'Happiness','exasperation': 'Anger','excitement': 'Happiness','exhilaration': 'Happiness','fear': 'Fear','ferocity': 'Anger','fondness': 'Love','fright': 'Fear','frustration': 'Anger','fury': 'Anger','gaiety': 'Happiness','gladness': 'Happiness','glee': 'Happiness','gloom': 'Sadness','glumness': 'Sadness','grief': 'Sadness','grouchiness': 'Anger','grumpiness': 'Anger','guilt': 'Sadness','happiness': 'Happiness','hate': 'Anger','homesickness': 'Sadness','hope': 'Happiness','hopelessness': 'Sadness','horror': 'Fear','hostility': 'Anger','humiliation': 'Sadness','hurt': 'Sadness','hysteria': 'Fear','infatuation': 'Love','insecurity': 'Fear','insult': 'Anger','irritation': 'Anger','isolation': 'Sadness','jealousy': 'Anger','jolliness': 'Happiness','joviality': 'Happiness','joy': 'Happiness','jubilation': 'Happiness','liking': 'Love','loathing': 'Disgust','loneliness': 'Sadness','longing': 'Desire','love': 'Love','lust': 'Desire','melancholy': 'Sadness','misery': 'Sadness','mortification': 'Sadness','neglect': 'Sadness','nervousness': 'Fear','optimism': 'Happiness','outrage': 'Anger','panic': 'Fear','passion': 'Desire','pity': 'Love','pleasure': 'Happiness','pride': 'Happiness','rage': 'Anger','rapture': 'Happiness','regret': 'Sadness','rejection': 'Sadness','relief': 'Happiness','remorse': 'Sadness','resentment': 'Anger','revulsion': 'Disgust','sadness': 'Sadness','satisfaction': 'Happiness','scorn': 'Disgust','sentimentality': 'Love','shame': 'Sadness','shock': 'Surprise','sorrow': 'Sadness','spite': 'Anger','suffering': 'Sadness','surprise': 'Surprise','sympathy': 'Love','tenderness': 'Love','tenseness': 'Fear','terror': 'Fear','thrill': 'Happiness','torment': 'Sadness','triumph': 'Happiness','uneasiness': 'Fear','unhappiness': 'Sadness','vengefulness': 'Anger','woe': 'Sadness','worry': 'Fear','wrath': 'Anger','zeal': 'Happiness','zest': 'Happiness'}

primary_emotions = ['Love', 'Anger', 'Fear', 'Sadness', 'Happiness', 'Surprise', 'Desire', 'Disgust']
primary_emotion_to_idx = {emotion: idx for idx, emotion in enumerate(primary_emotions)}

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=10, min_detection_confidence=0.5)

# Load ONNX model for emotion inference
onnx_model_path = "emotion_model.onnx"
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Load ONNX model for emotion colour inference (this model is smaller and has a higher accuracy) - deprecated
# onnx_model_colour = "emotion_model-8classes.onnx"
# session_colour = ort.InferenceSession(onnx_model_colour)
# input_name_colour = session_colour.get_inputs()[0].name
# output_name_colour = session_colour.get_outputs()[0].name

def preprocess_image(base64_str):
    image_data = base64.b64decode(base64_str.split(",")[1])
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return np.array(image)

def create_thumbnail(image_data):
    image_binary = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_binary))
    
    image.thumbnail((400, 400))
    
    thumbnail_io = io.BytesIO()
    image.save(thumbnail_io, format='JPEG')
    thumbnail_io.seek(0)
    
    return f"data:image/jpeg;base64,{base64.b64encode(thumbnail_io.getvalue()).decode()}"

@app.route('/analyse', methods=['POST'])
def analyze_frame():
    try:
        # Receive the image from the frontend
        data = request.json['frame']
        image = preprocess_image(data)

        # Process the image with MediaPipe
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        output = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract face points
                face_points = []
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                    face_points.append((x, y))

                # Compute convex hull
                hull = cv2.convexHull(np.array(face_points, dtype=np.int32)).tolist()

                # Preprocess face region for emotion detection
                face_crop = cv2.boundingRect(np.array(hull))
                x, y, w, h = face_crop
                min_x, min_y = min(x - 5, 0), min(y - 5, 0)
                max_x, max_y = min(x + w + 5, image.shape[1]), min(y + h + 5, image.shape[0])
                face_region = cv2.resize(image[min_y : max_y, min_x : max_x], (224, 224))  # Adjust to model's input size
                face_region = face_region.astype("float32") / 255.0
                face_region = np.transpose(face_region, (2, 0, 1))
                face_region = np.expand_dims(face_region, axis=0)

                # Emotion inference
                preds = session.run([output_name], {input_name: face_region})
                emotion_idx = int(np.argmax(preds[0]))  # Convert to integer for JSON serialization

                # Emotion colour inference - this feature is not used anymore, as it is infered from the specific emotion
                # preds_colour = session_colour.run([output_name_colour], {input_name_colour: face_region})
                # emotion_idx_colour = int(np.argmax(preds_colour[0]))
                # Append results
                output.append({
                    "primary_emotion_idx": primary_emotion_to_idx[primary_emotion_mapping[emotion_list[emotion_idx]]], # emotion_idx_colour
                    "emotion_label": emotion_list[emotion_idx],
                    "hull": hull
                })

        return jsonify(output)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/gallery', methods=['GET'])
def gallery():
    images = list(images_collection.find({}, {'thumbnail': 1}))
    return render_template('gallery.html', images=images)

@app.route('/image/<image_id>')
def get_image(image_id):
    image = images_collection.find_one({'_id': ObjectId(image_id)})
    if image:
        return jsonify({'image': image['full_image']})
    return jsonify({'error': 'Image not found'}), 404

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        image_data = request.json['image']
        
        thumbnail = create_thumbnail(image_data)
        
        result = images_collection.insert_one({
            'full_image': image_data,
            'thumbnail': thumbnail
        })
        
        return jsonify({
            'message': 'Image uploaded successfully',
            'id': str(result.inserted_id)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete/<image_id>', methods=['DELETE'])
def delete_image(image_id):
    try:
        result = images_collection.delete_one({'_id': ObjectId(image_id)})
        if result.deleted_count:
            return jsonify({'message': 'Image deleted successfully'})
        return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')

@app.route('/get_emotion_model', methods=['GET'])
def get_emotion_model():
    return app.send_static_file('emotion_model.onnx')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8888)
