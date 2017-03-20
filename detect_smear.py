'''

Solution to Assignment 2 for Geospatial Vision and Visualization

Authors: Arun, Himanshu, Shivani, Suraj

Objective: Finding the smear on camera if it exists given number of images taken one after another 

'''

#Using Numpy, Scipy and Opencv libraries for python

import os, argparse
import numpy as np
import imutils, cv2
from skimage.filters import threshold_adaptive

#defining global variables
debug=True 

#Function to detect if there is smear on the images
def detect_smear( directory=os.getcwd(), extension="jpg"):
	# Access all JPG files in directory
	allfiles=os.listdir(directory)
	
	#Finding the list of images
	imlist=[filename for filename in allfiles if  filename[-4:] in [ extension, extension.upper()]]

	h = w = 500
	#Create a numpy array of floats to store the average (assume RGB images)
	average_img=np.zeros((h,w,3),np.float)

	# Build up average pixel intensities, casting each image as an array of floats
	for im in imlist:
		img 		= cv2.imread(os.path.join(directory, im))
		img 		= imutils.resize(img, width=500)
		img 		= cv2.GaussianBlur(img, (3,3), 0)
		imarr 		= np.array(img,dtype=np.float)
		average_img	= average_img+imarr/len(imlist)

	# Round values in array and cast as 8-bit integer
	average_img = np.array(np.round(average_img),dtype=np.uint8)

	# Generate, save and preview final image
	cv2.imwrite("Average.jpg",average_img)
	
	if debug: cv2.imshow("Average image",average_img); cv2.waitKey(0)

	# Grayscale the average image, 
	grey_average_img = cv2.cvtColor(average_img,cv2.COLOR_BGR2GRAY)
	
	# Apply adaptive thresholding and scale it with 255
	warped = threshold_adaptive(grey_average_img, 250, offset = 10)
	warped_average_image = warped.astype("uint8") * 255

	# Detect edges in the adaptive thresholded (warped_average_image) image 
	edge_detected_image = cv2.Canny(warped_average_image, 75, 200)

	if debug: cv2.imshow('Adaptive Thresholded image',warped_average_image); cv2.waitKey(0)

	# Find contours in the edge detected image, sort them based on their size with largest first
	(_, cnts, _) = cv2.findContours(edge_detected_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	
	#Finding the largest contours
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

	#import pdb;pdb.set_trace()
	# Iterate over the contours, approximate them and if a contour has 4 points, take it as the screen
	list = []
	mask_img=np.zeros((h,w,1),np.float)
	oimg 		= cv2.imread(os.path.join(directory, imlist[0]))
	oimg 		= imutils.resize(oimg, width=500)
	
	#Checking if the contour area is big enough to be called as smear
	for c in cnts:
		peri = cv2.arcLength(c, True)
		#Based on Douglas Peucker algorithm
		approx = cv2.approxPolyDP(c, 0.02 * peri, True) 
		#print(cv2.contourArea)
		(x,y),radius = cv2.minEnclosingCircle(c)
		if abs(cv2.contourArea(c)-3.14*radius**2) < 300 and cv2.contourArea(c)>300:
			cv2.drawContours(oimg, [approx], -1, (255, 255, 0), 2)
			cv2.drawContours(mask_img, [approx], -1, (255, 255, 255), -1)
			list.append(c)
	
	
	
	# Show thw contour detected image
	if debug: cv2.imshow('FinalResult',oimg); cv2.waitKey(0)
	
	if debug: cv2.imshow('mask_img',mask_img); cv2.waitKey(0)
	
	if len(list) == 0: return False
	
	# Generate, save and preview the image with the detected smear
	cv2.imwrite('FinalResult.jpg',oimg)
	cv2.imwrite('mask_img.jpg',mask_img)
	return True

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	#extension
	parser.add_argument("-e", "--extension", default='.jpg', help="file extension of images")
	
	#Directory
	parser.add_argument("-d", "--directory", help="directory of the images")
	
	#Execution Mode
	parser.add_argument("-r", "--release", type=bool, help="debug or release mode")
	
	#Others if required
	
	#Making the parse
	args = parser.parse_args()
	
	#Checking if the directory is provided
	if not args.directory:
		parser.print_help()
		sys.exit(1)
	
	#Assigning values to the variables
	extension = args.extension
	directory = args.directory
	mode = args.release
	if mode: debug=False
	
	print "Smear Detection started"
	
	#Calling the detect smear function
	smear_detected = detect_smear(directory, extension)
		
	if(smear_detected):
		print("Smear is detected. Please find the mask_img.jpg to check the location of smear")
	else:
		print("Smear not detected!")
	