import numpy as np

def calculate_angle(a, b, c):
    """Calculates the 2D angle between three landmarks for biomechanical analysis."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    # Standardizing angle calculation for joint extension
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(angle, 2)

def get_posture_feedback(landmarks):
    """
    Analyzes specific body segments for technical deviations.
    This works alongside your Custom UCF101 classifier.
    """
    feedback = []
    
    # 1. Head Position Check: Ensuring head stability
    # Landmark 0 is the Nose
    head = landmarks[0]
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    if head.y > shoulder_y:
        feedback.append("Correction: Keep your head level; it is dropping below shoulder level.")

    # 2. Knee & Stance Balance: Detecting weight transfer issues
    # Landmarks 25: L_Knee, 26: R_Knee
    knee_diff = abs(landmarks[25].y - landmarks[26].y)
    if knee_diff > 0.12:
        feedback.append("Correction: Balance your weight; your lead knee is collapsing.")

    return feedback if feedback else ["Technique is stable. Maintain this posture."]

def classify_shot(metrics):
    """
    Classifies the shot based on average elbow extension.
    Optimized for the Cricket AI Pro Academy benchmarks.
    """
    avg_elbow = metrics.get("average_elbow_angle", 0)
    if avg_elbow > 158:
        return "Cover / Straight Drive (Full Extension)"
    elif 140 <= avg_elbow <= 158:
        return "Defensive Stroke (Balanced Stance)"
    elif 115 <= avg_elbow < 140:
        return "Pull / Cut Shot (Controlled Extension)"
    else:
        return "Close to Body Block (Compact Posture)"