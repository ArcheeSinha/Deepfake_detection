import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_and_preprocess_frames(directory, label):
    images = []
    labels = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img_path = os.path.join(directory, filename)
            img = load_img(img_path, target_size=(128, 128))  
            img_array = img_to_array(img) / 255.0  
            images.append(img_array)
            labels.append(label)  
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def preprocess_frames_and_save():

    real_frames_folder = 'C:\Users\91995\Desktop\deepfake_detection\REAL'
    fake_frames_folder = 'C:\Users\91995\Desktop\deepfake_detection\FAKE'  
    output_folder = 'C:/Users/91995/Desktop/deepfake_detection/extracted_frames'  

    X_real, y_real = load_and_preprocess_frames(real_frames_folder, 1)  
    X_fake, y_fake = load_and_preprocess_frames(fake_frames_folder, 0)  

    X = np.concatenate((X_real, X_fake), axis=0)
    y = np.concatenate((y_real, y_fake), axis=0)
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, 'X.npy'), X)
    np.save(os.path.join(output_folder, 'y.npy'), y)

preprocess_frames_and_save()

