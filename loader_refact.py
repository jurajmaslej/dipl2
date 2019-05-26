import os
import numpy as np
import cv2 as cv
import json
from matplotlib import pyplot as pltcv
import csv

#from histogram_comparer import Hsv
#from helper import Helper
import pickle
import pandas as pd

from datetime import datetime
from tabulate import tabulate

import ast
import numpy as np


class Loader():		#Helper optional
	
	def __init__(self, path=None, addr=None, result_addr=None, new_load=False, new_save=False, w=150, h=150):
		'''
		will load photos and synops according to txt file 
		'''
		self.path = path
		self.addr = addr
		self.result_addr = result_addr
		self.dt_dic = dict()
		self.w = w
		self.h = h

	def check_datafile_entry(self, fline):
		line = fline.strip()
		items = line.split(' ')
		if len(items) != 3:
			raise ValueError('File format not as expeceted, line not parsed into date, photo, synop')
		return items
	
	def create_dict_photo_synop(self):
		with open(self.path, 'r') as data_file:
			for fline in data_file:
				try:
					items = self.check_datafile_entry(fline)
				except ValueError as err:
					print(err)
					continue
				dt_object = datetime.strptime(items[0],'%Y%m%d%H%M')
				self.dt_dic[dt_object] = list([items[1], items[2]])
					
	def rewrite_photo_dataset(self):
		with open(self.path, 'r') as data_file:
			for fline in data_file:
				try:
					items = self.check_datafile_entry(fline)
				except ValueError as err:
					print(err)
					continue
				try:
					self.load_photo(items[1], w, h, do_rgb=False)
				except:
					print('not loaded')
					print(self.addr + '/' + items[1])
					
	
	def load_photo(self, fname,w,h, do_rgb=False, photo_crop=False):
		img_read = cv.imread(self.addr + '/' + fname)
		if photo_crop:
			img_read = self.photo_cropping(img_read)
		if do_rgb is False:
			img_read = cv.cvtColor(img_read, cv.COLOR_BGR2HSV)
		cv.imwrite(self.result_addr + '/' + fname, img_read)
		return img_read
	
	def photo_cropping(self, img):
		img_read = img_read[:50, :50]  # cropping here
		
	def add_photos_to_data(self):
		cnt = 1
		for k, v in self.dt_dic.iteritems():
			img_read = cv.imread(self.addr + '/' + v[0])
			if img_read is not None:
				self.dt_dic[k].append(img_read)
			else:
				print('NONE')
		
	def dict_to_dframe(self):
		df = pd.DataFrame.from_dict(self.dt_dic, orient='index').reset_index()
		df.rename(columns={'index':'date', 0:'name',1:'synop'}, inplace=True)
		df.to_csv('df_refact.csv')
		return df
	
	def load_csv_df(self,fname):
		self.main_df = pd.read_csv(fname)
		self.train, self.validate, self.test = np.split(self.main_df.sample(frac=1), [int(.6*len(self.main_df)), int(.8*len(self.main_df))])
		return self.main_df, self.train, self.validate, self.test
	

class Loader_test(Loader):
	def add_photos_to_data(self):
		return None
	
	def rewrite_photo_dataset(self):
		return None
	
# doing cropping right now, check load photo
#l = Loader('bright2/list_9to8.txt', 'bright2', 'pano_cropped_hsv2', new_load = True, new_save=False, w=224, h=224)

# need to run these two only if creating new dataframe
#l.add_photos_to_data()	
#l.dict_to_dframe()

#l.load_csv_df('df_cropped.csv')



#to rotate
'''
M = cv2.getRotationMatrix2D(center, angle180, scale)
rotated180 = cv2.warpAffine(img, M, (w, h))
'''