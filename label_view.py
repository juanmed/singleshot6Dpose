# This script takes a label file from LINEMOD and displays
# the training points over the corresponding image
# This is just to understand how the label file should be created.


import cv2
import numpy as np

import os


label_dir = "Box_Random/labels/000007.txt" #"LINEMOD/ape/labels/000000.txt"
image_dir = "Box_Random/JPEGImages/000007.png" #"LINEMOD/ape/JPEGImages/000000.jpg"


#label_dir = "LINEMOD/ape/labels/000000.txt"
#image_dir = "LINEMOD/ape/JPEGImages/000000.jpg"

def main():
	with open(label_dir) as lb_f:
		labels = lb_f.readlines()
		#print(lb_f.readlines())

	#label = label.split()

	img = cv2.imread(image_dir)
	print(img.shape)

	# create a window to display image
	wname = "Prediction"
	cv2.namedWindow(wname)	

	for k,label in enumerate(labels):
		label = label.split()
		# draw bounding cube
		p = list()
		for j,i in enumerate(np.arange(1,19,2)):
			col1 = 28*j
			col2 = 255 - (28*j)
			col3 = np.random.randint(0,256)
			x = int(float(label[i])*img.shape[1])
			y = int(float(label[i+1])*img.shape[0])
			print("label: {} point{}: {},{}".format(k,j,x,y))
			p.append([x,y])
			cv2.circle(img,(x,y), 3, (col1,col2,col3), -1)
			#if j==0:
			#	cv2.putText(img, str(k), (x + 5, y + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (col1, col2, col3), 2)
			cv2.putText(img, str(j+1), (x + 5, y + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (col1, col2, col3), 1)

		# Get each predicted point and the centroid
		p1 = p[1]
		p2 = p[2]
		p3 = p[3]
		p4 = p[4]
		p5 = p[5]
		p6 = p[6]
		p7 = p[7]
		p8 = p[8]
		center = p[0] 

		# Draw cube lines around detected object
		# draw front face
		line_point = 1
		cv2.line(img,(p1[0],p1[1]),(p2[0],p2[1]), (0,255,0),line_point)
		cv2.line(img,(p2[0],p2[1]),(p4[0],p4[1]), (0,255,0),line_point)
		cv2.line(img,(p4[0],p4[1]),(p3[0],p3[1]), (0,255,0),line_point)
		cv2.line(img,(p3[0],p3[1]),(p1[0],p1[1]), (0,255,0),line_point)
		
		# draw back face
		cv2.line(img,(p5[0],p5[1]),(p6[0],p6[1]), (0,255,0),line_point)
		cv2.line(img,(p7[0],p7[1]),(p8[0],p8[1]), (0,255,0),line_point)
		cv2.line(img,(p6[0],p6[1]),(p8[0],p8[1]), (0,255,0),line_point)
		cv2.line(img,(p5[0],p5[1]),(p7[0],p7[1]), (0,255,0),line_point)

		# draw right face
		cv2.line(img,(p2[0],p2[1]),(p6[0],p6[1]), (0,255,0),line_point)
		cv2.line(img,(p1[0],p1[1]),(p5[0],p5[1]), (0,255,0),line_point)
		
		# draw left face
		cv2.line(img,(p3[0],p3[1]),(p7[0],p7[1]), (0,255,0),line_point)
		cv2.line(img,(p4[0],p4[1]),(p8[0],p8[1]), (0,255,0),line_point)

		# draw x and y range
		x_r = int(float(label[19])*img.shape[1])
		y_r = int(float(label[20])*img.shape[0])

		#cv2.line(img,(int(float(label[17])*img.shape[1]),int(float(label[18])*img.shape[0])),(int(float(label[17])*img.shape[1])+x_r,int(float(label[18])*img.shape[0])+y_r), (0,255,0),1)

	# Show the image and wait key press
	cv2.imshow(wname, img)
	cv2.waitKey()




if __name__ == '__main__':
	main()