from helper import Helper
from main_loader import Main_loader
from histogram_comparer import Hsv

import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


class Trivial_algo(Helper):
	
	def __init__(self, main_folder, init_loader=True):
		Helper.__init__(self)
		self.main_folder = main_folder
		self.histograms = Hsv()
		self.histo_dict = {}
		if init_loader:
			main_loader = Main_loader({'koliba': 'SYNOPs/BA_Koliba_SYNOP_2014-2016.txt', 'airport':'SYNOPs/BA_Letisko_SYNOP_2014-2016.txt' }, None)
			main_loader.create_synops_dict()
			main_loader.load_obj('imgs')
			main_loader.strip_to_datetime()
		
	def iterate_year(self):
		years = os.listdir(self.main_folder)
		#print(years)
		for y in years:
			self.iterate_month(y)
			return
	
	def iterate_month(self, year):
		months = [self.main_folder + '/' + year +'/' + m for m in os.listdir(self.main_folder + '/' + year)]
		#print (months)
		for m in months:
			print('month ' + str(m))
			self.iterate_days(m)
			return
			
	def iterate_days(self, year_month):
		'''
		now has also funcionality of create_standardized_dict and iterate_days_for_histo_creation
		'''
		#fisheyes/2014/06/24/fisheye_1414641474324_1403630832716_00013_20140624_192851_jpg
		days = os.listdir(year_month)
		y_m_d = [year_month + '/' + d for d in days]
		for img_dir in y_m_d:
			for img in os.listdir(img_dir):
				path = img_dir + '/' + img
				img_read = cv.imread(path)
				histo = self.histograms.histogram_hsv(img_read)
				standard_key = self.standardize_date(img)
				self.hist_dict[]
				
				#return
				
				
t_algo = Trivial_algo('fisheyes', False)
t_algo.iterate_year()