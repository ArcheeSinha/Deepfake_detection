from flask import Flask, render_template, request, redirect, url_for
import os
import hashlib
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}
app.config['CACHE_FOLDER'] = 'cache'

# Ensure the cache folder exists
os.makedirs(app.config['CACHE_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f'frame_{frame_number:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_number += 1
    cap.release()

def preprocess_frame(img_path):
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_frames(folder_path):
    preprocessed_frames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            frame = preprocess_frame(img_path)
            preprocessed_frames.append(frame)
    return np.vstack(preprocessed_frames)

def get_cache_key(video_path):
    with open(video_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return os.path.join(app.config['CACHE_FOLDER'], f'{file_hash}.pkl')

def save_cache(cache_key, result):
    with open(cache_key, 'wb') as f:
        pickle.dump(result, f)

def load_cache(cache_key):
    if os.path.exists(cache_key):
        with open(cache_key, 'rb') as f:
            return pickle.load(f)
    return None

def predict_video(video_path, model_path, output_folder):
    cache_key = get_cache_key(video_path)
    cached_result = load_cache(cache_key)
    if cached_result:
        return cached_result

    extract_frames(video_path, output_folder)
    model = load_model(model_path)
    preprocessed_frames = preprocess_frames(output_folder)
    predictions = model.predict(preprocessed_frames)
    average_prediction = np.mean(predictions)
    result = 'Fake' if average_prediction > 0.5 else 'Real'

    save_cache(cache_key, result)
    return result

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            
            model_path = 'C:/Users/91995/Desktop/deepfake_detection/deepfake_detection_model.h5'
            output_folder = 'extracted_frames'
            
            result = predict_video(video_path, model_path, output_folder)
            
            return render_template('result.html', result=result)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

