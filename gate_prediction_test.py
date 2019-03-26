# Load libraries
import json
from pprint import pprint
import glob
import cv2
import numpy as np
from random import shuffle

from generate_results_riseq import *


img_file = glob.glob('testing/images/*.JPG')
img_keys = [img_i.split('/')[-1] for img_i in img_file]



# Instantiate a new detector
finalDetector = GenerateFinalDetections()
# load image, convert to RGB, run model and plot detections. 
limit = min(100,len(img_keys))
for img_key in img_keys[0:limit-1]:
    img =cv2.imread('testing/images/'+img_key)
    img_test =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bb_all = finalDetector.predict(img_test)
    bb_all = bb_all[0]
    bb_all.pop()   # drop last point (confidence)
    print(bb_all)
    # draw lines
    line_point = 3
    for i in range(0,len(bb_all)-2, 2):
    	cv2.line(img,(int(bb_all[i]),int(bb_all[i+1])),(int(bb_all[i+2]),int(bb_all[i+2+1])),(0,0,255),line_point)

    #draw last line
    cv2.line(img,(int(bb_all[6]),int(bb_all[7])),(int(bb_all[0]),int(bb_all[1])),(0,0,255),line_point)

    cv2.imshow("hola",img)
    cv2.waitKey()


