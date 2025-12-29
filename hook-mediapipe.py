# hook-mediapipe.py
from PyInstaller.utils.hooks import collect_data_files

# Включаем все .tflite, .pb, .binarypb — особенно pose_landmark
datas = collect_data_files('mediapipe', include_py_files=False)