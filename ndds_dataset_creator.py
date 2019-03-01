# This file takes a folder containing images created from NDDS and 
# creates label files compatible with singleshot6DPose

import numpy as np 
import cv2
import json
import argparse
import glob, os
import time

def parse_args():
    """Use argparse to get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Input path. Folder with NDDS images.')
    parser.add_argument('-o', help='Output path. Folder directory to store output images.')
    parser.add_argument('-p', help="Percent of data to use as training. From 0.0 to 1.0 .")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    convert_labels(args.i, args.o, float(args.p))

def test_image(img_f, centroid, cuboid, o_range,bbox):
	img = cv2.imread(img_f)
	height, width, depth = img.shape

	# draw cuboid points
	for i,point in enumerate(cuboid):
		col1 = 28*i
		col2 = 255 - (28*i)
		col3 = np.random.randint(0,256)
		x = int(point[0])
		y = int(point[1])		
		cv2.circle(img, (x,y), 3, (col1, col2, col3), -1)
		cv2.putText(img, str(i+1), (x + 5, y + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (col1, col2, col3), 1)

	# draw centroid
	x = int(centroid[0])
	y = int(centroid[1])	
	cv2.circle(img, (x,y), 3, (col1, col2, col3), -1)

	#draw bounding box
	tl = bbox['top_left']
	br = bbox['bottom_right']

	tl[0] = int(tl[0])
	tl[1] = int(tl[1])
	br[0] = int(br[0])
	br[1] = int(br[1])
	# limit bounding box to image size
	#tl[0] = max(0,tl[0])
	#tl[0] = min(height,tl[0])
	#tl[1] = max(0,tl[1])
	#tl[1] = min(width,tl[1])
	#br[0] = max(0,br[0])
	#br[0] = min(height,br[0])
	#br[1] = max(0,br[1])
	#br[1] = min(width,br[1])

	# draw bounding box
	cv2.line(img,(tl[1],tl[0]), (br[1],tl[0]), (0,255,0),2)
	cv2.line(img,(tl[1],tl[0]), (tl[1],br[0]), (0,255,0),2)
	cv2.line(img,(tl[1],br[0]), (br[1],br[0]), (0,255,0),2)
	cv2.line(img,(br[1],tl[0]), (br[1],br[0]), (0,255,0),2)

	#cv2.line(img,(tl[0],tl[1]), (br[0],tl[1]), (0,255,0),2)
	#cv2.line(img,(tl[0],tl[1]), (tl[0],br[1]), (0,255,0),2)
	#cv2.line(img,(tl[0],br[1]), (br[0],br[1]), (0,255,0),2)
	#cv2.line(img,(br[0],tl[1]), (br[0],br[1]), (0,255,0),2)	

	# draw line representing range
	cv2.line(img,(tl[1],tl[0]),(tl[1]+int(o_range[0]),tl[0]+int(o_range[1])),(255,0,0),1)

	# create a window to display image
	wname = "Prediction"
	cv2.namedWindow(wname)
	# Show the image and wait key press
	cv2.imshow(wname, img)
	cv2.waitKey()
	#cv2.destroyAllWindows()

def convert_labels(label_path, det_path, percent):

	path, dirs, files = os.walk(label_path).__next__()

	files = [x for x in files if "json" in x]
	nfiles = int(len(files)-2)  # total no of json files. subtract camera and object settins file
	print("Desired Percent: {}, total files {}".format(percent,nfiles))
	# open train.txt and test.txt files
	ftrain = open(label_path+"train.txt", 'a')
	ftest = open(label_path+"test.txt", 'a')

	i = 0
	count = 0
	test_im = 0
	train_im = 0
	# Traverse all .json files

	t1 = time.time()

	for json_file in glob.iglob(os.path.join(label_path,"*.json")):

		# ignore camera intrinsics and object settings file
		if "camera_settings" in json_file:
			continue

		if "object_settings" in json_file:
			continue


		# limit files to analyze. For debugging purposes.
		#if i >= 30:
		#	break

		#print(json_file)

		try:
			loc, suffix, ext = json_file.split(".")
		except:
			loc, ext = json_file.split(".")
			suffix = None

		loc = loc.split("/")
		name = loc[-1]
		save_dir = ""
		for folder in loc[0:len(loc)-1]:
			save_dir = save_dir + folder+"/"

		# corresponding image name
		if suffix is not None:
			img_file = save_dir+name+"."+suffix+".png"
		else:
			img_file = save_dir+name+".png"

		# load image
		img = cv2.imread(img_file)
		height, width, depth = img.shape
		#print(img.shape)
  

		# get data containers
		frames = json.load(open(json_file, 'r'))

		# get projected cuboid
		cubo = frames['objects'][0]['projected_cuboid']
		cuboid = list(map(lambda a: [float(a[0]),float(a[1])],cubo))

		# get projected centroid
		cent = frames['objects'][0]['projected_cuboid_centroid']
		centroid = [float(cent[0]),float(cent[1])]

		# get bounding box
		bbox = frames['objects'][0]['bounding_box']
		tl = bbox['top_left']
		br = bbox['bottom_right']
		# limit bounding box to image size
		#tl[0] = max(0,tl[0])
		#tl[0] = min(height,tl[0])
		#tl[1] = max(0,tl[1])
		#tl[1] = min(width,tl[1])
		#br[0] = max(0,br[0])
		#br[0] = min(height,br[0])
		#br[1] = max(0,br[1])
		#br[1] = min(width,br[1])

		y_range = float(br[0] - tl[0])
		x_range = float(br[1] - tl[1])
		#print(x_range,y_range)

		# view bounding box and cuboid over image
		# for debugging purposes
		test_image(img_file, centroid, cuboid, [x_range,y_range],bbox)


		# create corresponding label file


		if suffix is not None:
			label_file = save_dir+name+"."+suffix+".txt"
		else:
			label_file = save_dir+name+".txt"

		#print("Label file name: {}".format(label_file))
		f = open(label_file,"w")
		f.write("0 ")
		
		# write centroid coordinates
		f.write("{:.6f} ".format(centroid[0]/width))
		f.write("{:.6f} ".format(centroid[1]/height))

		# write projected cuboid coordinates to file
		# but in the order singleshot6dpose expects...
		for j in [3,0,7,4,2,1,6,5]:
			element = cuboid[j]
			f.write("{:.6f} ".format(element[0]/width))
			f.write("{:.6f} ".format(element[1]/height))

		# write x,y range
		f.write("{:.6f} ".format(x_range/width))
		f.write("{:.6f} ".format(y_range/height))
		f.close()

		i = i + 1
		if ( (i % 100) == 0):
			t2 = time.time()
			print("Processed images: {} in {:.6f}".format(i,t2-t1))
			t1 = t2

		#print("i : {}".format(i))
		count = count + 1

		# create image file name
		if suffix is not None:
			img_dir  = save_dir+"JPEGImages/"+name+"."+suffix+".png\n"
		else:
			img_dir = save_dir+"JPEGImages/"+name+".png\n"

		#print("Img dir: "+img_dir)

		if (count > int(nfiles*percent)):
			# write on test.txt file
			ftest.write(img_dir)
			test_im = test_im +1
		else:
			# write image name on train.txt
			ftrain.write(img_dir)
			train_im = train_im + 1
	
	ftest.close()
	ftrain.close()

	print("Test images:  {}".format(test_im))
	print("Train images: {}".format(train_im))
	print("Total images: {}".format(test_im+train_im))


if __name__ == '__main__':
	main()
