import cv2
import os
import numpy as np
from sort import *
import torch
from PIL import Image
from torchvision import transforms
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


def res_classifer(image, categories, model):

    input_image = Image.fromarray(image)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    input_batch = input_batch.to('cuda')
    model.to('cuda')
    t = time.time()
    output = model(input_batch)
    print(time.time()-t)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    return categories[top5_catid[0]], top5_prob[0]


# Opens the Video file
camera = 'fisheye'
files = os.listdir(f'Source/{camera}')
print(files)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
background = cv2.imread('background.jpg', cv2.IMREAD_GRAYSCALE)
mot_tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.05)
detectionidx = 0
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
for j in files:
    folder = j.strip('.mp4')
    cap= cv2.VideoCapture(f'Source/{camera}/{j}')
    i=0
    ret, ppv_frame = cap.read()
    ppv_frame = cv2.cvtColor(ppv_frame, cv2.COLOR_BGR2GRAY)
    print('counting '+ j)
    ret, prev_frame = cap.read()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        background_sub = cv2.subtract(gray, background)
        _, background_sub = cv2.threshold(background_sub, 20, 255, cv2.THRESH_BINARY)

        flow_sub = cv2.subtract(gray, prev_frame)
        # flow_sub = cv2.cvtColor(flow_sub, cv2.COLOR_BGR2GRAY)
        _, flow_sub = cv2.threshold(flow_sub, 50, 255, cv2.THRESH_BINARY)

        flow_sub_2 = cv2.subtract(gray, ppv_frame)
        # # flow_sub_2 = cv2.cvtColor(flow_sub_2, cv2.COLOR_BGR2GRAY)
        _, flow_sub_2 = cv2.threshold(flow_sub_2, 50, 255, cv2.THRESH_BINARY)

        mask = cv2.addWeighted(flow_sub, 0.5, background_sub, 0.5, 0)
        # mask = cv2.addWeighted(mask, 0.5, flow_sub_2, 0.5, 0)

        median = cv2.medianBlur(mask, 5)
        ret, thresh = cv2.threshold(median, 1, 255, 0)
        thresh = dilatation(thresh, 12)
        # thresh = erosion(thresh, 2)
        # masked = cv2.bitwise_and(frame, frame, mask=median)
        # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
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
                cls, conf = res_classifer(detection, categories, model)
                # print(c)
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(display, f"{cls}, {conf:2f} ", (x, y+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            [255, 255, 0], 2)
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
        id = max(trk[:, -1])

        # print(trk)
        # cv2.drawContours(display, contours, -1, (0, 255, 0), 3)
        # thresh = cv2.threshold(masked, 200, 255, cv2.THRESH_BINARY)[1]

        # cv2.imshow('2 stage flow image subtraction + background subtraction + median blur + thresholding + dilation', dila)
        cv2.imshow('display', display)
        # cv2.imshow('flow_sub', flow_sub)
        # cv2.imshow('display', display)
        # cv2.imshow('RGB', frame)
        # cv2.imshow('Gray', gray)
        # cv2.imshow('flow', flow_sub_2)
        cv2.imshow('thresh', thresh)
        cv2.waitKey(1)
        i+=1
        prev_frame = gray
        ppv_frame = prev_frame
    print(id)
    cap.release()




"""
load 20 images

count pixel for each gray scale image




"""