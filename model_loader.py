# model_loader.py
import os
import pandas as pd

def load_model(folder: str):
    csv_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.csv')])
    video_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.avi'))])

    if not csv_files or not video_files:
        raise FileNotFoundError("В папке должны быть как минимум один CSV и одно видео (mp4/avi).")

    def find_pref(files, pref):
        for f in files:
            if pref.lower() in f.lower():
                return f
        return files[0]

    csv_path = os.path.join(folder, find_pref(csv_files, "model"))
    video_path = os.path.join(folder, find_pref(video_files, "model"))

    df = pd.read_csv(csv_path)

    # ✅ Очистка от пустых строк (как в вашем data_1766228186.csv — 47 пустых строк!)
    df = df.dropna(subset=['frame_number']).reset_index(drop=True)
    angle_cols = [c for c in df.columns if c.endswith('_angle')]
    if angle_cols:
        df = df[~(df[angle_cols] == 0).all(axis=1)].reset_index(drop=True)

    mandatory = {'frame_number', 'shoulder_left_angle', 'x_factor_angle'}
    missing = mandatory - set(df.columns)
    if missing:
        raise ValueError(f"В CSV отсутствуют обязательные столбцы: {missing}")

    return df, video_path