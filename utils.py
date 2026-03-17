import numpy as np

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return round(angle, 2)

def get_posture_feedback(landmarks):
    feedback = []
    head = landmarks[0]
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    if head.y > shoulder_y:
        feedback.append("Correction: Keep your head level; it is dropping below shoulder level.")
    knee_diff = abs(landmarks[25].y - landmarks[26].y)
    if knee_diff > 0.12:
        feedback.append("Correction: Balance your weight; your lead knee is collapsing.")
    return feedback if feedback else ["Technique is stable. Maintain this posture."]

def classify_shot(metrics):
    avg_elbow = metrics.get("average_elbow_angle", 0)
    if avg_elbow > 165: 
        return "Lofted Drive / Sixer (Maximum Extension)"
    elif 155 <= avg_elbow <= 165: 
        return "Cover / Straight Drive (Full Extension)"
    elif 140 <= avg_elbow < 155: 
        return "Defensive Stroke (Controlled Elbow)"
    elif 125 <= avg_elbow < 140: 
        return "Square Cut / Late Cut (Wrist Work)"
    elif 110 <= avg_elbow < 125: 
        return "Pull / Hook Shot (High-to-Low)"
    elif 90 <= avg_elbow < 110: 
        return "Close-Body Block (Compact Defense)"
    else: 
        return "Under-Edge / Late Reaction (Cramped Posture)"