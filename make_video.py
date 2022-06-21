import cv2
import numpy as np
import glob

img_array = []
for filename in glob.glob('macp/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('macp.mp4', fourcc, 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()