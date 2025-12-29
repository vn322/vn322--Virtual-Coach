# utils.py
import math
import numpy as np
from scipy.stats import entropy

def calculate_angle(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    dot = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cos_angle = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))

def calculate_angular_velocity(prev_angle, curr_angle, dt):
    return abs(curr_angle - prev_angle) / dt if dt > 0.001 else 0.0

def calculate_shannon_entropy(angles):
    if not angles:
        return 0.0
    filtered = [a for a in angles if 0 <= a <= 180]
    if not filtered:
        return 0.0
    hist, _ = np.histogram(filtered, bins=10, range=(0, 180))
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return entropy(probs, base=2)

def calculate_acceleration(prev_vel, curr_vel, dt):
    return abs(curr_vel - prev_vel) / dt if dt > 0.001 else 0.0