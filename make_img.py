import cv2
import os
import numpy as np
import shutil
# Opens the Video file


n_frames = 200
frames_idx = []
for i in range(0, n_frames):
  frames_idx.append(np.random.randint(0, 6000))

print(frames_idx)
camera = 'fisheye'
files = os.listdir(f'dst/{camera}')
print(files)
for j in files:
    folder = f"dst/{camera}/{j}"
    print(folder)
    for i in frames_idx:
        shutil.copy2(f"{folder}/{i}.jpg", f"for_label/{j}-00{i}.jpg")