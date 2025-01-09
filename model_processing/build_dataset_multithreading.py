import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import requests
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.2)

def download_image(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = np.array(bytearray(response.content), dtype=np.uint8)
        return cv2.imdecode(image, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
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

def process_row(row, output_folder):
    url, name = row['URL'], row['Name']
    try:
        image = download_image(url)
        if image is not None:
            face_image = crop_face(image)
            if face_image is not None:
                output_path = os.path.join(output_folder, name)
                cv2.imwrite(output_path, face_image)
                return row 
    except Exception as e:
        print(f"Error processing row {name}: {e}")
    return None

if __name__ == "__main__":
    json_file_path = 'training_raw_data.json'

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

    output_folder = 'faces'
    os.makedirs(output_folder, exist_ok=True)

    processed_rows = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(process_row, row, output_folder)
            for _, row in df.iterrows()
        ]
        for future in tqdm(futures, total=len(futures), desc="Processing images"):
            result = future.result()
            if result is not None:
                processed_rows.append(result)

    processed_df = pd.DataFrame(processed_rows)
    processed_df.to_csv('training_data.csv', index=False)
