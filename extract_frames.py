import cv2
import os

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
    print(f'Frames extracted to {output_folder}')

video_path = 'path_to_your_video.mp4' 
output_folder = 'C:/Users/91995/Desktop/deepfake_detection/extracted_frames' 

extract_frames(video_path, output_folder)