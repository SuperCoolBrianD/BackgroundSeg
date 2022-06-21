import cv2
import os
import numpy as np
# Opens the Video file



cap= cv2.VideoCapture(f'test.mp4')
i=0
if not os.path.exists("dst/fisheye/test"):
    os.mkdir('dst')
    os.mkdir('dst/fisheye')
    os.mkdir('dst/fisheye/test')
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite(f'dst/fisheye/test/{i}.jpg',frame)
    i+=1

cap.release()
cv2.destroyAllWindows()