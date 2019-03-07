# Load libraries
import json
from pprint import pprint
import glob
import cv2
import numpy as np
from random import shuffle

from generate_results_riseq import *
import time


gt_dir = "/home/fer/git_clone/scorer_scripts_v1_2/new_label/trainingdata_1000_to_1499.json"
img_dir = "/home/fer/Documents/DroneChallenge/Virtual_Qualifier/Test2/Data_Training/*"
pred_dir = "/home/fer/git_clone/scorer_scripts_v1_2/riseq_trainingset_results3.json"

gt_frames = json.load(open(gt_dir, 'r'))
pred_frames = json.load(open(pred_dir, 'r'))
print(len(pred_frames.keys()))

limit = 100

for i,key in enumerate(gt_frames.keys()):

    if i > limit:
        continue

    gt_labels = gt_frames[key][0]
    pred_labels = pred_frames[key][0]

    print(img_dir.replace('*',key))

    img  = cv2.imread(img_dir.replace('*',key))

    line_point = 2

    
    # draw ground thruth
    for i in range(0,len(gt_labels)-2, 2):
        cv2.line(img,(int(gt_labels[i]),int(gt_labels[i+1])),(int(gt_labels[i+2]),int(gt_labels[i+2+1])),(0,0,255),line_point)
    # last line
    cv2.line(img,(int(gt_labels[6]),int(gt_labels[7])),(int(gt_labels[0]),int(gt_labels[1])),(0,0,255),line_point)
    

    # draw prediction
    for i in range(0,len(pred_labels)-2, 2):
        cv2.line(img,(int(pred_labels[i]),int(pred_labels[i+1])),(int(pred_labels[i+2]),int(pred_labels[i+2+1])),(0,255,0),line_point)
    # last line
    cv2.line(img,(int(pred_labels[6]),int(pred_labels[7])),(int(pred_labels[0]),int(pred_labels[1])),(0,255,0),line_point)


    cv2.imshow('ground thruth', img)
    cv2.waitKey()


    i = i + 1
