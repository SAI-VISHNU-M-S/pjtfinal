import cv2
import mediapipe as mp
import numpy as np
import os
import subprocess
import joblib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from .utils import calculate_angle, get_posture_feedback, classify_shot

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

try: my_model = joblib.load('cricket_model.joblib')
except: my_model = None

def process_video(video_path, output_path, report_path):
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    temp_output = output_path.replace(".mp4", "_temp.mp4")
    out = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    angles, custom_tips, frame_count = [], [], 0
    with mp_pose.Pose(min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lm = results.pose_landmarks.landmark
                ang = calculate_angle([lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y])
                angles.append(ang)
                custom_tips.extend(get_posture_feedback(lm))
            out.write(frame)
    cap.release()
    out.release()
    
    subprocess.run(['ffmpeg', '-y', '-i', temp_output, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_path])
    if os.path.exists(temp_output): os.remove(temp_output)

    avg_angle = int(np.mean(angles)) if angles else 0
    shot_type = classify_shot({"average_elbow_angle": avg_angle})
    
    biometrics = [f"Stroke: {shot_type}", f"Avg Extension: {avg_angle}°", f"Frames: {frame_count}"]
    final_feedback = biometrics + list(set(custom_tips))
    generate_pdf(report_path, final_feedback)
    return avg_angle, final_feedback

def generate_pdf(path, feedback):
    c = canvas.Canvas(path, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Posture Detection System Audit")
    y = 710
    c.setFont("Helvetica", 11)
    for f in feedback:
        c.drawString(100, y, f"• {f}")
        y -= 25
    c.save()