# pose_analyzer.py
import cv2
import mediapipe as mp
import numpy as np
import time
import math
from utils import (
    calculate_angle, calculate_angular_velocity,
    calculate_shannon_entropy, calculate_acceleration
)

class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.frame_count = 0
        self.flip_camera = False
        self.rotate_180 = False
        self.x_factor_history = []
        self.prev_angles = {}
        self.prev_velocities = {}
        self.prev_time = time.time()
        self.frame_data_log = []

    def process_frame(self, frame):
        # Применяем трансформации кадра
        if self.flip_camera:
            frame = cv2.flip(frame, 1)
        if self.rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        h, w = frame.shape[:2]
        data = {"frame_number": self.frame_count}

        # Если ключевые точки не найдены
        if not results.pose_landmarks:
            self.frame_count += 1
            self.frame_data_log.append(data)
            self._put_text(frame, f"{self.frame_count}", (10, 30))
            # Подпись в левом нижнем углу
            self._put_text(frame, "Ermakov.AV, 2025", (10, h - 20), (255, 255, 255))
            return frame, data

        annotated = frame.copy()
        self._draw_skeleton(annotated, results.pose_landmarks, h, w)

        landmarks = results.pose_landmarks.landmark
        data["avg_visibility"] = np.mean([lm.visibility for lm in landmarks])

        def get_coords(idx):
            lm = landmarks[idx]
            return (int(lm.x * w), int(lm.y * h))

        # --- 4 сустава: shoulder, elbow, hip, knee ---
        for side in ['left', 'right']:
            prefix = 'LEFT' if side == 'left' else 'RIGHT'
            color = (0, 255, 0) if side == 'left' else (0, 255, 255)  # зелёный / голубой

            # Shoulder
            try:
                elbow = get_coords(getattr(self.mp_pose.PoseLandmark, f'{prefix}_ELBOW').value)
                shoulder = get_coords(getattr(self.mp_pose.PoseLandmark, f'{prefix}_SHOULDER').value)
                hip = get_coords(getattr(self.mp_pose.PoseLandmark, f'{prefix}_HIP').value)
                angle = calculate_angle(elbow, shoulder, hip)
                data[f'shoulder_{side}_angle'] = angle
                self._put_text(annotated, f"{int(angle)}", (shoulder[0] + 10, shoulder[1] - 10), color)
            except: pass

            # Elbow
            try:
                shoulder = get_coords(getattr(self.mp_pose.PoseLandmark, f'{prefix}_SHOULDER').value)
                elbow = get_coords(getattr(self.mp_pose.PoseLandmark, f'{prefix}_ELBOW').value)
                wrist = get_coords(getattr(self.mp_pose.PoseLandmark, f'{prefix}_WRIST').value)
                angle = calculate_angle(shoulder, elbow, wrist)
                data[f'elbow_{side}_angle'] = angle
                self._put_text(annotated, f"{int(angle)}", (elbow[0] + 10, elbow[1] - 10), color)
            except: pass

            # Hip
            try:
                shoulder = get_coords(getattr(self.mp_pose.PoseLandmark, f'{prefix}_SHOULDER').value)
                hip = get_coords(getattr(self.mp_pose.PoseLandmark, f'{prefix}_HIP').value)
                knee = get_coords(getattr(self.mp_pose.PoseLandmark, f'{prefix}_KNEE').value)
                angle = calculate_angle(shoulder, hip, knee)
                data[f'hip_{side}_angle'] = angle
                self._put_text(annotated, f"{int(angle)}", (hip[0] + 10, hip[1] - 10), color)
            except: pass

            # Knee
            try:
                hip = get_coords(getattr(self.mp_pose.PoseLandmark, f'{prefix}_HIP').value)
                knee = get_coords(getattr(self.mp_pose.PoseLandmark, f'{prefix}_KNEE').value)
                ankle = get_coords(getattr(self.mp_pose.PoseLandmark, f'{prefix}_ANKLE').value)
                angle = calculate_angle(hip, knee, ankle)
                data[f'knee_{side}_angle'] = angle
                self._put_text(annotated, f"{int(angle)}", (knee[0] + 10, knee[1] - 10), color)
            except: pass

        # --- X-Factor: угол между линиями плеч и бёдер ---
        try:
            l_sh = get_coords(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            r_sh = get_coords(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            l_h = get_coords(self.mp_pose.PoseLandmark.LEFT_HIP.value)
            r_h = get_coords(self.mp_pose.PoseLandmark.RIGHT_HIP.value)

            sh_vec = np.array([r_sh[0] - l_sh[0], r_sh[1] - l_sh[1]])
            hip_vec = np.array([r_h[0] - l_h[0], r_h[1] - l_h[1]])
            dot = np.dot(sh_vec, hip_vec)
            norm_s = np.linalg.norm(sh_vec)
            norm_h = np.linalg.norm(hip_vec)
            x_factor = abs(math.degrees(math.acos(np.clip(dot / (norm_s * norm_h), -1.0, 1.0)))) if norm_s > 0 and norm_h > 0 else 0.0
            data["x_factor_angle"] = x_factor

            cx = int((l_sh[0] + r_sh[0] + l_h[0] + r_h[0]) // 4)
            cy = int((l_sh[1] + r_sh[1] + l_h[1] + r_h[1]) // 4)
            self._put_text(annotated, f"X:{int(x_factor)}", (cx - 15, cy), (255, 255, 255))

        except Exception as e:
            data["x_factor_angle"] = 0.0

        # --- Скорости, ускорения, энтропии ---
        curr_time = time.time()
        dt = curr_time - self.prev_time
        for key in list(data.keys()):
            if key.endswith('_angle'):
                curr = data[key]
                prev = self.prev_angles.get(key, curr)
                vel = calculate_angular_velocity(prev, curr, dt)
                acc = calculate_acceleration(self.prev_velocities.get(key.replace('_angle', '_velocity'), 0.0), vel, dt)
                data[key.replace('_angle', '_velocity')] = vel
                data[key.replace('_angle', '_acceleration')] = acc

                hist = getattr(self, f"{key}_hist", [])
                hist.append(curr)
                if len(hist) > 30:
                    hist = hist[-30:]
                data[key.replace('_angle', '_entropy')] = calculate_shannon_entropy(hist)
                setattr(self, f"{key}_hist", hist)

                self.prev_angles[key] = curr
                self.prev_velocities[key.replace('_angle', '_velocity')] = vel

        self.prev_time = curr_time
        self.frame_count += 1

        # Номер кадра в левом верхнем углу
        self._put_text(annotated, f"{self.frame_count}", (10, 30))
        # Подпись автора в левом нижнем углу — ✅ добавлена
        self._put_text(annotated, "Ermakov.AV, 2025", (10, h - 20), (255, 255, 255))

        self.frame_data_log.append(data)
        return annotated, data

    def _put_text(self, img, text, pos, color=(255, 255, 255)):
        """Отрисовка текста без PIL — только cv2"""
        x, y = pos
        # Тень для контраста
        cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        # Основной текст
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    def _draw_skeleton(self, frame, landmarks, h, w):
        """Рисует скелет: левая — зелёная, правая — голубая"""
        connections = self.mp_pose.POSE_CONNECTIONS
        for start_idx, end_idx in connections:
            start = landmarks.landmark[start_idx]
            end = landmarks.landmark[end_idx]
            if start.visibility < 0.5 or end.visibility < 0.5:
                continue
            pt1 = (int(start.x * w), int(start.y * h))
            pt2 = (int(end.x * w), int(end.y * h))
            color = (0, 255, 0) if self._is_left(start_idx) else (0, 255, 255)
            cv2.line(frame, pt1, pt2, color, 2)
            cv2.circle(frame, pt1, 3, (255, 255, 255), -1)

    def _is_left(self, idx):
        return idx in [11, 13, 15, 23, 25, 27]  # LEFT_* indices

    def get_summary_stats(self):
        return {"total_frames": len(self.frame_data_log)}

    def reset(self):
        self.__init__()