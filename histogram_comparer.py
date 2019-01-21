from __future__ import division
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from os.path import join
from os import listdir

class Hsv:
	
	def __init__(self):
		'''
		really deletes some shit
		changed error evaluation method, 
		now (29.11. 01:50) almost perfect
		'''
		self.grass_histo_list = self.histogram_list('grass_examples')
		self.sky_histo_list = self.histogram_list('sky_examples')
	
	def load_img(self, img):
		return cv.imread(img)
	
	def histogram_rgb(self, image):
		# extract a 3D RGB color histogram from the image,
		# using 8 bins per channel, normalize, and update
		# the index
		hist = cv.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
		hist = cv.normalize(hist, None).flatten()
		return hist
	
	def histogram_hsv(self, image):
		# extract a 3D RGB color histogram from the image,
		# using 8 bins per channel, normalize, and update
		# the index
		try:
			hist = cv.calcHist([image], [0, 1], None, [8, 16], [0, 180, 0, 256])
		#print('len hist not flattend ' + str(len(hist)))
		#print('len hist[0] not flattend ' + str(len(hist[0])))
			hist = cv.normalize(hist, None).flatten()
			return hist
		except:
			print('hist not created')
			pass
	
	def compare_histograms(self, h1, h2):
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
		err = 1000000
		for hs in histo_list:
			new_err = self.compare_histograms(histo_to_judge, hs)
			if new_err < err:
				err = new_err
			#print(err)
		return err

	def compare_img_to_histos(self, img):
		'''
		return True if near to sky histograms,
		return False if near to grass histograms
		'''
		histo_to_judge = self.histogram_rgb(img)
		grass_err = self.histo_err(histo_to_judge, self.grass_histo_list)
		#print ('total grass err ' + str(grass_err))
		sky_err = self.histo_err(histo_to_judge, self.sky_histo_list)
		#print ('total sky err ' + str(sky_err))
		return sky_err < grass_err
		
	def histogram_all(self, image):
		return self.compare_img_to_histos(image)
		
			
#h = Hsv('fisheye_orig_sky_jpg')
#h.histogram_all(None)
#h = Hsv('fisheye_orig_grass2_jpg')
#h.histogram_all(None)