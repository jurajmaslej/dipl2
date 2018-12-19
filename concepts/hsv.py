import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from os.path import join
class Hsv:
	
	def __init__(self, filename, path=None):
		
		if path:
			print('fname in hsv ', join(path, filename))
			self.img = cv.imread(join(path, filename))
		else:	
			self.img = cv.imread(filename)
		
	def to_hsv(self):
		# Convert BGR to HSV
		#print (self.img)
		#print(self.img)
		hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
		'''
		print(hsv.shape)
		print(np.max(hsv[0]))
		print(np.max(hsv[1]))
		print(np.max(hsv[2]))
		print(np.min(hsv[0]))
		print(np.min(hsv[1]))
		print(np.min(hsv[2]))
		print(hsv[0])
		'''
		return hsv
		
	def blue_mask(self, hsv, low, high, ind):
		blue = np.uint8([[[0, 0, 255]]])
		hsv_blue = cv.cvtColor(blue,cv.COLOR_BGR2HSV)
		
		hsv_low_blue =  np.array(low)		#([220,50,50])
		hsv_high_blue =  np.array(high)		#([260, 255, 255])
		
		# Threshold the HSV image to get only blue colors
		mask = cv.inRange(hsv, hsv_low_blue, hsv_high_blue)
		
		# Bitwise-AND mask and original image
		res = cv.bitwise_and(self.img, self.img, mask= mask)
		
		cv.imshow('frame',self.img)
		cv.imshow('mask',mask)
		cv.imwrite('mask_new.jpg', mask)
		cv.imshow('res',res)
		cv.imwrite('res_new.jpg',res)
		return res
		
h = Hsv('fisheye_1414615708319_1403623620709_00013_20140624_172851_jpg')
hsv = h.to_hsv()
#h.new_blue_mask(hsv)
ind = 0
h.blue_mask(hsv, [100,50,50], [140,255,255], 1)
# [86, 31, 4], [220, 88, 50]
#for low, high in ([[100,50,50],[140,255,255]], [[100,50,50],[140,255,255]], [[100,50,50],[140,255,255]], [[100,50,50],[140,255,255]], [[100,50,50],[140,255,255]]) :
#	h.blue_mask(hsv)  ##,[110,50,50],[130,255,255])