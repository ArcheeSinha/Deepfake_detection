import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def extract_frames(video_path, output_folder):
    if not os.path.exists(video_path):
        print(f"Error: The video file '{video_path}' does not exist.")
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open the video file '{video_path}'.")
        return
    
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f'frame_{frame_number:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_number += 1
    
    cap.release()
    print(f'Frames extracted to {output_folder}')

def preprocess_frame(img_path):
    img = load_img(img_path, target_size=(128, 128))  
match model input
    img_array = img_to_array(img) / 255.0 
    return np.expand_dims(img_array, axis=0)  

def predict_video(video_path, model_path, output_folder):
    extract_frames(video_path, output_folder)
    
    model = load_model(model_path)
    
    predictions = []
    for filename in os.listdir(output_folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(output_folder, filename)
            frame = preprocess_frame(img_path)
            prediction = model.predict(frame)
            predictions.append(prediction)
    
    if predictions:
        average_prediction = np.mean(predictions)
        result = 'Fake' if average_prediction > 0.2 else 'Real'
    else:
        result = 'Unknown'
    
    print(f'The video is predicted to be: {result}')
    return result


video_path = 'path_to_your_video.mp4'  
model_path = 'C:/Users/91995/Desktop/deepfake_detection/deepfake_detection_model.h5'
output_folder = 'C:/Users/91995/Desktop/deepfake_detection/extracted_frames'

predict_video(video_path, model_path, output_folder)

