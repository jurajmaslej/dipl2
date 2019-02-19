# PATH: init_work2/DATA/data/ImagesCameraFisheye/ImagesCameraFisheye
# NEW PATH: init_work2/fisheyes
import os
import numpy as np
import cv2 as cv
import json
from matplotlib import pyplot as pltcv
import datetime
import csv

from histogram_comparer import Hsv
from helper import Helper
import pickle

class Load_camera(Helper):
	
	def __init__(self, folder):
		Helper.__init__(self)
		print 'Loading fiseyes'
		self.main_folder = folder
		self.camera = {}
		self.path = ''
		self.img_c = 0
		self.imgs = dict()
		self.h = Hsv()
		self.key_c = 0
		self.key_set = set()
		#self.w = csv.writer(open("imgs_dict.csv", "w"))
	
	def iterate_year(self):
		years = os.listdir(self.main_folder)
		#print(years)
		for y in years:
			self.iterate_month(y)
	
	def iterate_month(self, year):
		months = [self.main_folder + '/' + year +'/' + m for m in os.listdir(self.main_folder + '/' + year)]
		#print (months)
		for m in months:
			print('month ' + str(m))
			#self.iterate_days_for_histo_creation(m)
			#self.iterate_days(m)
			self.create_standardized_dict(m)
			
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
				try:
					standard_key = self.standardize_date(img)
					if standard_key in self.key_set:
						self.key_c += 1
					self.key_set.add(standard_key)
					if 'resized' not in path:
						img_read = cv.imread(path)
						img_read = cv.resize(img_read, (150,150))
						# do histograms and possible datetimes
						if not int(path[-10:-8]) in range(6,20):
							os.remove(path)
						elif self.h.histogram_all(img_read) is False:
							#print('removing smth at path ' + path)
							os.remove(path)
						else:
							hsv = cv.cvtColor(img_read, cv.COLOR_BGR2HSV)
							new_name = img[:img.find('.')] + '.png'
							cv.imwrite(img_dir + '/resized_hsv_' + new_name, hsv)
							img_read, hsv = None, None
							isfile = os.path.isfile(path)
							os.remove(path)
							
							# add entry to dict object
							standard_key = self.standardize_date(img)
							if standard_key:
								self.imgs[standard_key] = hsv
					else:
							print('resized in path')
				except:
					print('bad file at ' + path)
					pass
			print('day ' + str(img_dir))
		#return
		
	def iterate_days_for_histo_creation(self, year_month):
		#fisheyes/2014/06/24/fisheye_1414641474324_1403630832716_00013_20140624_192851_jpg
		'''
		really deletes some shit
		'''
		days = os.listdir(year_month)
		#print(os.listdir(year_month + '/' + days[0]))
		y_m_d = [year_month + '/' + d for d in days]
		for img_dir in y_m_d:
			for img in os.listdir(img_dir):
				path = img_dir + '/' + img
				if self.h.histogram_all(cv.imread(path)) is False:
					print('removing smth at path ' + path)
					os.remove(path)
			#print(img_dir)
			return
			
	def round_time(self, dt=None, round_to=60):
		"""Round a datetime object to any time lapse in seconds
		dt : datetime.datetime object, default now.
		roundTo : Closest number of seconds to round to, default 1 minute.
		Author: Thierry Husson 2012 - Use it as you want but don't blame me.
		"""
		if dt == None : dt = datetime.datetime.now()
		seconds = (dt.replace(tzinfo=None) - dt.min).seconds
		rounding = (seconds + round_to/2) //round_to * round_to
		return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)
			
	def standardize_date(self, img_name):
		'''
		get date&time from filename, convert it to standard tstamp ==> round to hours
		'''
		date = img_name[-22:-7]
		try:
			rounded_date = str(self.round_time(datetime.datetime(int(date[0:4]),  int(date[4:6]), int(date[6:8]), int(date[9:11]),  int(date[11:13]), int(date[13:15]),  int('0000')), round_to=3600))
			standardized_date = rounded_date.replace('-','').replace(' ','').replace(':','')[:-2]	#remove seconds,cause synops dont have 'em
			return standardized_date
		except:
			pass
			return None
			
	def create_standardized_dict(self, year_month, one_photo_per_hour = False):
		'''
		create dict with keys same as dict for synops
		called from iterate_month
		
		one ph per hour: we have multiple shots falling to same hour, which one to choose?
		14.2. using format:
		do vstack
		[[first shot],[second shot]........]
		TODO: change further code to reflect this change
		'''
		print('standardize_dict')
		days = os.listdir(year_month)
		y_m_d = [year_month + '/' + d for d in days]
		ind = 0
		indl = dict()
		for img_dir in y_m_d:
			for img in os.listdir(img_dir):
				path = img_dir + '/' + img
				standard_key = self.standardize_date(img)
				if standard_key in self.key_set:
					
					if one_photo_per_hour:
						self.imgs[standard_key] = cv.imread(path)
					else:
						self.imgs[standard_key] = np.vstack((self.imgs[standard_key], [cv.imread(path)]))
					ind += 1
					indl[standard_key] += 1 
				else:
					self.imgs[standard_key] = [cv.imread(path)]
					indl[standard_key] = 0
				self.key_set.add(standard_key)
				
	def write_to_file(self):
		print('creating dict')
		
		with open('imgs_dict.csv', 'w') as f:
			for key in self.imgs.keys():
				f.write("%s,%s\n"%(key,self.imgs[key]))

	def save_obj(self, obj, name):
		print 'saving object'
		with open('obj/'+ name + '.pkl', 'wb') as f:
			pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
			

#print lc.round_time(datetime.datetime(2012,12,31,23,44,59,1234),roundTo=60*60)