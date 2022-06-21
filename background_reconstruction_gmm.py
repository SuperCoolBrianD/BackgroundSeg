import cv2
import os
import numpy as np
from sort import *


def drawtrack(detections, img):
    """Draw bounding boxes for each detection from YOLO"""
    bbox = []
    for detection in detections:
        # print(detection)
        pt1 = (int(float(detection[0])), int(float(detection[1])))
        pt2 = (int(float(detection[2])), int(float(detection[3])))
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img, f"{detection[-1]} TrackID: {str(int(float(detection[-1])))}", (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
    return img


def morph_shape(val):
    if val == 0:
        return cv2.MORPH_RECT
    elif val == 1:
        return cv2.MORPH_CROSS
    elif val == 2:
        return cv2.MORPH_ELLIPSE


def dilatation(img, dilatation_size):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    dilatation_dst = cv2.dilate(img, element)
    return dilatation_dst
    # cv.imshow(title_dilation_window, dilatation_dst)


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def erosion(gray, idx):
    eroded = gray
    for i in range(idx):
        eroded = cv2.erode(eroded, None, iterations=i + 1)
    return eroded

# Opens the Video file
camera = 'fisheye'
files = os.listdir(f'Source/{camera}')
print(files)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

background = cv2.imread('background_macp.jpg', cv2.IMREAD_GRAYSCALE)
mot_tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.05)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(nmixtures=4, history=5000)
detectionidx = 0

cap= cv2.VideoCapture(f'macp.mp4')
i=0
ret, ppv_frame = cap.read()
ppv_frame = cv2.cvtColor(ppv_frame, cv2.COLOR_BGR2GRAY)
ret, prev_frame = cap.read()
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    background_sub = cv2.subtract(gray, background)
    # _, background_sub = cv2.threshold(background_sub, 20, 255, cv2.THRESH_BINARY)
    # background_sub = cv2.medianBlur(background_sub, 5)
    fgmask = fgbg.apply(frame)
    mask = fgmask
    # mask = cv2.addWeighted(fgmask, 0.5, background_sub, 0.5, 0)
    # mask = cv2.addWeighted(mask, 0.5, flow_sub_2, 0.5, 0)

    median = cv2.medianBlur(mask, 7)
    ret, thresh = cv2.threshold(median, 1, 255, 0)
    thresh = dilatation(thresh, 3)
    # thresh = erosion(thresh, 2)
    # masked = cv2.bitwise_and(frame, frame, mask=median)
    #
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = cv2.convexHull(contours)
    dets = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        x1, y1, x2, y2 = x, y, x + w, y + h
        a = w*h
        if a > 200:
            dets.append([x1, y1, x2, y2, 0 ,0])
            detection = frame[y:y + h, x:x + w, :]
            cv2.imwrite(f'detection/{detectionidx}.jpg', detection)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)
            detectionidx+=1

    if dets == []:
        dets = np.empty((0, 6))
    else:
        dets = np.array(dets, dtype='object_')
    # print(dets)
    try:
        trk = mot_tracker.update(dets)
    except ValueError:
        continue
    display = drawtrack(trk, display)
    # id = max(trk[:, -1])

    print(trk)
    # cv2.drawContours(display, contours, -1, (0, 255, 0), 3)
    # thresh = cv2.threshold(masked, 200, 255, cv2.THRESH_BINARY)[1]

    # cv2.imshow('2 stage flow image subtraction + background subtraction + median blur + thresholding + dilation', dila)
    cv2.imshow('display', display)
    # cv2.imshow('flow_sub', flow_sub)
    # cv2.imshow('display', display)
    # cv2.imshow('RGB', frame)
    # cv2.imshow('Gray', gray)
    cv2.imshow('median', median)
    cv2.imshow('thresh', thresh)
    cv2.imshow('fgmask', fgmask)
    cv2.imshow('background_sub', background_sub)
    cv2.waitKey(1)
    i+=1
    prev_frame = gray
    ppv_frame = prev_frame
print(id)
cap.release()


