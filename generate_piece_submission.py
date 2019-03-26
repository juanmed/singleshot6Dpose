# Load libraries
import json
from pprint import pprint
import glob
import cv2
import numpy as np
from random import shuffle

from generate_results_riseq import *
import time


#img_file = glob.glob('testing/images/*.JPG')
img_folder = '/home/fer/Documents/DroneChallenge/Virtual_Qualifier/Test2/Data_Training/'
#img_keys = [img_i.split('/')[-1] for img_i in img_file]

gt_dir = "trainingdata_6000_to_6499.json"
gt_frames = json.load(open(gt_dir, 'r'))

# Instantiate a new detector
finalDetector = GenerateFinalDetections()
# load image, convert to RGB, run model and plot detections. 
time_all = []
pred_dict = {}
for k, img_key in enumerate(gt_frames.keys()):
    img =cv2.imread(img_folder+img_key)
    img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tic = time.monotonic()
    bb_all = finalDetector.predict(img)
    toc = time.monotonic()
    pred_dict[img_key] = bb_all
    time_all.append(toc-tic)

    if (k%100 == 0):
    	print ("Image: {}".format(k))

mean_time = np.mean(time_all)
ci_time = 1.96*np.std(time_all)
freq = np.round(1/mean_time,2)
    
print("Processed images: {}".format(len(time_all)))    
print('95% confidence interval for inference time is {0:.2f} +/- {1:.4f}.'.format(mean_time,ci_time))
print('Operating frequency from loading image to getting results is {0:.2f}.'.format(freq))

with open('predict_6000to6499_model5_416.json', 'w') as f:
    json.dump(pred_dict, f)