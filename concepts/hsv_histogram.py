from __future__ import division
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from os.path import join
from os import listdir

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
	
	def load_img(self, img):
		return cv.imread(img)
	
	def histogram_rgb(self, image):
		# extract a 3D RGB color histogram from the image,
		# using 8 bins per channel, normalize, and update
		# the index
		hist = cv.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
		hist = cv.normalize(hist, None).flatten()
		#print(hist)
		return hist
	
	def histogram_hsv(self, image):
		# extract a 3D RGB color histogram from the image,
		# using 8 bins per channel, normalize, and update
		# the index
		hist = cv.calcHist([image], [0, 1], None, [180, 256], [0, 180, 0, 256])
		hist = cv.normalize(hist, None).flatten()
		#print(hist)
		return hist
	
	def compare_histograms(self, h1, h2):
		#print '   correl method'
		#result = cv.compareHist(h1, h2, cv.HISTCMP_CORREL)
		#print(result)
		#print '   chisq method'
		result = cv.compareHist(h1, h2, cv.HISTCMP_CHISQR)
		return result
		
	def histogram_list(self, folder):
		histo_list = []
		for img in listdir(folder):
			image = cv.imread(folder + '/' + img)
			histo = self.histogram_rgb(image)
			histo_list.append(histo)
		return histo_list
	
	def histo_err(self, histo_to_judge, histo_list):
		err = 0
		for hs in histo_list:
			err += self.compare_histograms(histo_to_judge, hs)
			print(err)
		return err/len(histo_list)

	def compare_img_to_histos(self, img, grass_histo_list, sky_histo_list):
		'''
		return True if near to sky histograms,
		return False if near to grass histograms
		'''
		histo_to_judge = self.histogram_rgb(img)
		grass_err = self.histo_err(histo_to_judge, grass_histo_list)
		print ('total grass err ' + str(grass_err))
		sky_err = self.histo_err(histo_to_judge, sky_histo_list)
		print ('total sky err ' + str(sky_err))
		return sky_err < grass_err
		
	def histogram_all(self, image):
		grass_histo_list = self.histogram_list('grass_examples')
		sky_histo_list = self.histogram_list('sky_examples')
		print self.compare_img_to_histos(self.img, grass_histo_list, sky_histo_list)
		
			
h = Hsv('fisheye_orig_sky_jpg')
h.histogram_all(None)
h = Hsv('fisheye_orig_grass2_jpg')
h.histogram_all(None)


#img2 = h.load_img('fisheye_orig_grass_jpg')
#img3 = h.load_img('fisheye_orig_grass2_jpg')
'''
img1 = h.load_img('hsv_grass.jpg')
img2 = h.load_img('hsv_grass2.jpg')
img3 = h.load_img('hsv_sky.jpg')
img4 = h.load_img('hsv_sky2.jpg')


hist_grass = h.histogram_hsv(img1)
hist_grass2 = h.histogram_hsv(img2)
hist_sky = h.histogram_hsv(img3)
hist_sky2 = h.histogram_hsv(img4)
print 'grass vs sky'
h.compare_histograms(hist_sky, hist_grass)
print 'grass vs grass'
h.compare_histograms(hist_grass2, hist_grass)
'''

#hsv = h.to_hsv()
#h.new_blue_mask(hsv)
#ind = 0
#h.blue_mask(hsv, [100,50,50], [140,255,255], 1)
# [86, 31, 4], [220, 88, 50]
#for low, high in ([[100,50,50],[140,255,255]], [[100,50,50],[140,255,255]], [[100,50,50],[140,255,255]], [[100,50,50],[140,255,255]], [[100,50,50],[140,255,255]]) :
#	h.blue_mask(hsv)  ##,[110,50,50],[130,255,255])