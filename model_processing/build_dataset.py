import pandas as pd
import mediapipe as mp
import numpy as np
import json
import cv2
import os
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm

json_file_path = 'training_raw_data.json'

emotion_list = ['adoration', 'affection', 'aggravation', 'agitation', 'agony', 'alarm', 'alienation', 'amazement', 'amusement', 'anger', 'anguish', 'annoyance', 'anxiety', 'apprehension', 'arousal', 'astonishment', 'attraction', 'bitterness', 'bliss', 'caring', 'cheerfulness', 'compassion', 'contempt', 'contentment', 'defeat', 'dejection', 'delight', 'depression', 'desire', 'despair', 'disappointment', 'disgust', 'dislike', 'dismay', 'displeasure', 'distress', 'dread', 'eagerness', 'ecstasy', 'elation', 'embarrassment', 'enjoyment', 'enthrallment', 'enthusiasm', 'envy', 'euphoria', 'exasperation', 'excitement', 'exhilaration', 'fear', 'ferocity', 'fondness', 'fright', 'frustration', 'fury', 'gaiety', 'gladness', 'glee', 'gloom', 'glumness', 'grief', 'grouchiness', 'grumpiness', 'guilt', 'happiness', 'hate', 'homesickness', 'hope', 'hopelessness', 'horror', 'hostility', 'humiliation', 'hurt', 'hysteria', 'infatuation', 'insecurity', 'insult', 'irritation', 'isolation', 'jealousy', 'jolliness', 'joviality', 'joy', 'jubilation', 'liking', 'loathing', 'loneliness', 'longing', 'love', 'lust', 'melancholy', 'misery', 'mortification', 'neglect', 'nervousness', 'optimism', 'outrage', 'panic', 'passion', 'pity', 'pleasure', 'pride', 'rage', 'rapture', 'regret', 'rejection', 'relief', 'remorse', 'resentment', 'revulsion', 'sadness', 'satisfaction', 'scorn', 'sentimentality', 'shame', 'shock', 'sorrow', 'spite', 'suffering', 'surprise', 'sympathy', 'tenderness', 'tenseness', 'terror', 'thrill', 'torment', 'triumph', 'uneasiness', 'unhappiness', 'vengefulness', 'woe', 'worry', 'wrath', 'zeal', 'zest']

label_to_idx = {emotion: idx for idx, emotion in enumerate(emotion_list)}

primary_emotion_mapping = {'adoration': 'Love','affection': 'Love','aggravation': 'Anger','agitation': 'Fear','agony': 'Sadness','alarm': 'Fear','alienation': 'Sadness','amazement': 'Surprise','amusement': 'Happiness','anger': 'Anger','anguish': 'Sadness','annoyance': 'Anger','anxiety': 'Fear','apprehension': 'Fear','arousal': 'Desire','astonishment': 'Surprise','attraction': 'Love','bitterness': 'Anger','bliss': 'Happiness','caring': 'Love','cheerfulness': 'Happiness','compassion': 'Love','contempt': 'Disgust','contentment': 'Happiness','defeat': 'Sadness','dejection': 'Sadness','delight': 'Happiness','depression': 'Sadness','desire': 'Desire','despair': 'Sadness','disappointment': 'Sadness','disgust': 'Disgust','dislike': 'Disgust','dismay': 'Sadness','displeasure': 'Disgust','distress': 'Sadness','dread': 'Fear','eagerness': 'Desire','ecstasy': 'Happiness','elation': 'Happiness','embarrassment': 'Fear','enjoyment': 'Happiness','enthrallment': 'Happiness','enthusiasm': 'Happiness','envy': 'Anger','euphoria': 'Happiness','exasperation': 'Anger','excitement': 'Happiness','exhilaration': 'Happiness','fear': 'Fear','ferocity': 'Anger','fondness': 'Love','fright': 'Fear','frustration': 'Anger','fury': 'Anger','gaiety': 'Happiness','gladness': 'Happiness','glee': 'Happiness','gloom': 'Sadness','glumness': 'Sadness','grief': 'Sadness','grouchiness': 'Anger','grumpiness': 'Anger','guilt': 'Sadness','happiness': 'Happiness','hate': 'Anger','homesickness': 'Sadness','hope': 'Happiness','hopelessness': 'Sadness','horror': 'Fear','hostility': 'Anger','humiliation': 'Sadness','hurt': 'Sadness','hysteria': 'Fear','infatuation': 'Love','insecurity': 'Fear','insult': 'Anger','irritation': 'Anger','isolation': 'Sadness','jealousy': 'Anger','jolliness': 'Happiness','joviality': 'Happiness','joy': 'Happiness','jubilation': 'Happiness','liking': 'Love','loathing': 'Disgust','loneliness': 'Sadness','longing': 'Desire','love': 'Love','lust': 'Desire','melancholy': 'Sadness','misery': 'Sadness','mortification': 'Sadness','neglect': 'Sadness','nervousness': 'Fear','optimism': 'Happiness','outrage': 'Anger','panic': 'Fear','passion': 'Desire','pity': 'Love','pleasure': 'Happiness','pride': 'Happiness','rage': 'Anger','rapture': 'Happiness','regret': 'Sadness','rejection': 'Sadness','relief': 'Happiness','remorse': 'Sadness','resentment': 'Anger','revulsion': 'Disgust','sadness': 'Sadness','satisfaction': 'Happiness','scorn': 'Disgust','sentimentality': 'Love','shame': 'Sadness','shock': 'Surprise','sorrow': 'Sadness','spite': 'Anger','suffering': 'Sadness','surprise': 'Surprise','sympathy': 'Love','tenderness': 'Love','tenseness': 'Fear','terror': 'Fear','thrill': 'Happiness','torment': 'Sadness','triumph': 'Happiness','uneasiness': 'Fear','unhappiness': 'Sadness','vengefulness': 'Anger','woe': 'Sadness','worry': 'Fear','wrath': 'Anger','zeal': 'Happiness','zest': 'Happiness'}

with open(json_file_path, 'r') as file:
    data = json.load(file)

labels = []
urls = []

for image in data:
    labels.append(image['label'])
    urls.append(image['url'])

df = pd.DataFrame({
    'Label': labels,
    'URL': urls
})

df['Name'] = [str(i) + ".jpg" for i in range(len(df))]
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.2)

def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = np.array(bytearray(response.content), dtype=np.uint8)
        return cv2.imdecode(image, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def crop_face(image):
    try:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            face_points = [
                (int(lm.x * image.shape[1]), int(lm.y * image.shape[0]))
                for lm in face_landmarks.landmark
            ]
            hull = cv2.convexHull(np.array(face_points, dtype=np.int32)).tolist()
            face_crop = cv2.boundingRect(np.array(hull))
            x, y, w, h = face_crop
            
            min_x, min_y = max(x - 5, 0), max(y - 5, 0)
            max_x, max_y = min(x + w + 5, image.shape[1]), min(y + h + 5, image.shape[0])
            cropped_face = image[min_y:max_y, min_x:max_x]
            return cv2.resize(cropped_face, (224, 224))
        else:
            return None
    except Exception as e:
        print(f"Error processing face: {e}")
        return None

output_folder = 'faces'
os.makedirs(output_folder, exist_ok=True)
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
    url = row['URL']
    name = row['Name']

    image = download_image(url)
    
    if image is not None:
        face_image = crop_face(image)
        if face_image is not None:
            # Save the cropped face image
            output_path = os.path.join(output_folder, name)
            cv2.imwrite(output_path, face_image)
        else:
            df.drop(index, inplace=True)

df.to_csv('training_data.csv', index=False)