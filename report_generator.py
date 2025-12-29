# report_generator.py
import os
import pandas as pd
import cv2

def save_video_with_overlay(path, frames, fps=30):
    if not frames: return
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames: out.write(f)
    out.release()

def save_csv_report(path, log):
    if not log: return
    df = pd.DataFrame(log)
    df.to_csv(path, index=False, float_format="%.3f")