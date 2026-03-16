import cv2
import mediapipe as mp
import numpy as np
import glob
import joblib
from sklearn.ensemble import RandomForestClassifier

# PATHS BASED ON YOUR SYSTEM
# Change this line in train_model.py
UCF_PATH = "/home/sai/Desktop/minipjt/data/CricketShot/*.avi"
MODEL_NAME = "cricket_model.joblib"

mp_pose = mp.solutions.pose.Pose(static_image_mode=True)

def get_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        results = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            # Flatten 33 landmarks x (x,y,z) coordinates
            landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            features.append(np.array(landmarks).flatten())
    cap.release()
    return features

print("Starting feature extraction from UCF101...")
X = []
for video in glob.glob(UCF_PATH):
    X.extend(get_landmarks(video))

# Label professional shots as '1' (Correct Form)
y = [1] * len(X)

print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

joblib.dump(model, MODEL_NAME)
print(f"Model saved to {MODEL_NAME}!")