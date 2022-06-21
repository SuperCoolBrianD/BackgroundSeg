import numpy as np
import cv2

cap = cv2.VideoCapture('test_1.mp4')
# see link https://docs.opencv.org/4.x/d2/d55/group__bgsegm.html#ga1a5838fa2d2697ac455b136bfcdb4600
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# fgbg = cv2.bgsegm.createBackgroundSubtractorGMG() # ok result lots of noise
# fgbg = cv2.bgsegm.createBackgroundSubtractorLSBP(mc=0) # bad result
# fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC() # improved detection accuracy but reduced range
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(nmixtures=8, history=400)

while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('frame',fgmask)
    cv2.imshow('image', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()