# Create flyable area ground thruth dataset
# Use as input json files from NDDS and outputs .txt with format compliant with singleshot6dpose

# Load libraries
import json
from pprint import pprint
import glob
import cv2
import numpy as np
from random import shuffle

from generate_results_riseq import *
import time


#img_file = glob.glob('testing/images/*.JPG')\
file_dir = '/home/fer/Documents/DroneChallenge/Virtual_Qualifier/Test2/NDDS_Data/Training_Data/Gate_Random_1/'
img_file = glob.glob(file_dir+ '*.json')
img_keys = [img_i.split('/')[-1] for img_i in img_file]

def view_gt(points,img_name):
    img = cv2.imread(img_name)
    height, width, depth = img.shape

    # draw cuboid points
    for i,point in enumerate(points):
        col1 = 28*i
        col2 = 255 - (28*i)
        col3 = np.random.randint(0,256)
        x = int(point[0])
        y = int(point[1])       
        cv2.circle(img, (x,y), 3, (col1, col2, col3), -1)
        cv2.putText(img, str(i), (x + 5, y + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (col1, col2, col3), 1)

        # next point
        if i < (points.shape[0]-1):
            x_next = points[i+1][0]
            y_next = points[i+1][1]
        else:
            x_next = points[0][0]
            y_next = points[0][1]

        cv2.line(img,(x,y),(x_next,y_next),(0,255,0),2)

    # create a window to display image
    wname = "Prediction"
    cv2.namedWindow(wname)
    # Show the image and wait key press
    cv2.imshow(wname, img)
    cv2.waitKey()

# load image, convert to RGB, run model and plot detections. 
gt_dict = {}
for k, img_key in enumerate(img_keys):

    # ignore camera intrinsics and object settings file
    if "camera_settings" in img_key:
        continue

    if "object_settings" in img_key:
        continue

    if "training" in img_key:
        continue

    # open ground truth file
    frames = json.load(open(file_dir+img_key, 'r'))

    # get projected cuboid
    cubo = frames['objects'][1]['projected_cuboid']
    cuboid = list(map(lambda a: [float(a[0]),float(a[1])],cubo))

    # get projected centroid
    cent = frames['objects'][1]['projected_cuboid_centroid']
    centroid = [float(cent[0]),float(cent[1])]

    # get bounding box
    bbox = frames['objects'][1]['bounding_box']
    tl = bbox['top_left']
    br = bbox['bottom_right'] 
    
    flyarea_corners = np.zeros((4,2), dtype = 'float32')

    for i,j in enumerate([5,4,7,6]):

        flyarea_corners[i][0] = cuboid[j][0]
        flyarea_corners[i][1] = cuboid[j][1]

    #view_gt(flyarea_corners, file_dir+img_key.replace('json','png'))

    #flyarea_corners[1][0] = p2[0] + int((p4[0]-p2[0])*offset_x_ratio)
    #flyarea_corners[1][1] = p2[1] + int((p1[1]-p2[1])*offset_x_ratio) 

    #flyarea_corners[2][0] = p1[0] + int((p3[0]-p1[0])*offset_x_ratio)
    #flyarea_corners[2][1] = p1[1] + int((p2[1]-p1[1])*offset_x_ratio)

    #flyarea_corners[3][0] = p3[0] + int((p1[0]-p3[0])*offset_x_ratio)
    #flyarea_corners[3][1] = p3[1] + int((p4[1]-p3[1])*offset_x_ratio) 


    gt_dict[img_key.replace('json','png')] = np.array([flyarea_corners.flatten()]).tolist()


    if (k%100 == 0):
        print ("Image: {}".format(k))

print("Total images: {}".format(k+1))
with open(file_dir+'training_gt.json', 'w') as f:
    json.dump(gt_dict, f)