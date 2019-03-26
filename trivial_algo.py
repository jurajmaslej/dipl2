from helper import Helper
from main_loader import Main_loader
from histogram_comparer import Hsv

import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score

class Trivial_algo(Helper):
	
	def __init__(self, main_folder, init_loader=True):
		Helper.__init__(self)
		self.main_folder = main_folder
		self.histograms = Hsv()
		self.hist_dict = {}
		if init_loader:
			self.main_loader = Main_loader({'koliba': 'SYNOPs/BA_Koliba_SYNOP_2014-2016.txt', 'airport':'SYNOPs/BA_Letisko_SYNOP_2014-2016.txt' }, None)
			self.main_loader.create_synops_dict()
			self.main_loader.load_obj('imgs')
			self.main_loader.strip_to_datetime()
		
	def iterate_year(self):
		years = os.listdir(self.main_folder)
		for y in years:
			print(y)
			if int(y) < 2017:
				self.iterate_month(y)
			#return
	
	def iterate_month(self, year):
		months = [self.main_folder + '/' + year +'/' + m for m in os.listdir(self.main_folder + '/' + year)]
		#print (months)
		for m in months:
			print('month ' + str(m))
			self.iterate_days(m)
			
			
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
				if histo is None:
					print path
					break
				standard_key = self.standardize_date(img)
				if standard_key in self.hist_dict.keys():
					self.hist_dict[standard_key] = np.vstack((self.hist_dict[standard_key], [histo]))
				else:
					self.hist_dict[standard_key] = [histo]
				if img_read.shape != (150,150,3):
					print(img_read.shape)
					
	def iterate_dataset(self):
		print(len(set(self.hist_dict.keys())))
		for k in self.hist_dict.keys():
			print(k)
			print(self.main_loader.synops_avg[k])
			
	def create_tree(self):
		clf = tree.DecisionTreeClassifier()
		
		common_keys = set(self.hist_dict.keys()).intersection(set(self.main_loader.synops_avg.keys()))
		train = set()
		validate= set()
		for i in common_keys:
			if '201502'  in i:
				validate.add(i)
			else:
				train.add(i)
		print('train size ' + str(len(train)))
		print('validate size ' + str(len(validate)))
		hist_sorted = [self.hist_dict[key] for key in sorted(train)]	# access histograms from hist_dict based on cmn keys with synops
		synops_sorted = [int(self.main_loader.synops_koliba[key]) for key in sorted(train)]
		hist_sorted_v =  [self.hist_dict[key] for key in sorted(validate)]
		synops_sorted_v = [int(self.main_loader.synops_koliba[key]) for key in sorted(validate)]
		clf.fit(hist_sorted, synops_sorted)
		
		y_pred = list(clf.predict(hist_sorted_v))
		y_true = synops_sorted_v
		print(accuracy_score(y_true, y_pred))
		self.absolute_results(y_true, y_pred)
		
	def flatten_dset(self):
		common_keys = set(self.hist_dict.keys()).intersection(set(self.main_loader.synops_avg.keys()))
		train = set()
		validate= set()
		for i in common_keys:
			if '201502'  in i:
				validate.add(i)
			else:
				train.add(i)
		print('train size ' + str(len(train)))
		print('validate size ' + str(len(validate)))
		
		hist_sorted = list()
		synops_sorted = list()
		for k in sorted(train):
			for v in self.hist_dict[k]:
				hist_sorted.append(v)
				synops_sorted.append(int(self.main_loader.synops_koliba[k]))
				
		hist_sorted_v = list()
		synops_sorted_v = list()
		for k in sorted(validate):
			for v in self.hist_dict[k]:
				hist_sorted_v.append(v)
				synops_sorted_v.append(int(self.main_loader.synops_koliba[k]))
				
		print('sizes:')
		print(len(hist_sorted))
		print(len(synops_sorted))
		print(len(hist_sorted_v))
		print(len(synops_sorted_v))
		
		self.fit_tree(hist_sorted, synops_sorted, hist_sorted_v, synops_sorted_v)
		
	def fit_tree(self, hist_sorted, synops_sorted, hist_sorted_v, synops_sorted_v):
		clf = tree.DecisionTreeClassifier()
		print(len(hist_sorted))
		print('llen')
		print(len(hist_sorted[0]))
		print(hist_sorted[0].shape)
		clf.fit(hist_sorted, synops_sorted)
		y_pred = list(clf.predict(hist_sorted_v))
		y_true = synops_sorted_v
		print(accuracy_score(y_true, y_pred))
		self.absolute_results(y_true, y_pred)
		
	def absolute_results(self, y_true, y_predict):
		err = 0
		rel_err = 0
		for t,p in zip(y_true, y_predict):
			if t != p:
				err+=1
			rel_err += abs(t - p)
			
		rel_err = rel_err / len(y_true)
		print('abs error count on size ' + str(len(y_true)))
		print(err)
		print('relative err count')
		print(rel_err)
		
		
				
t_algo = Trivial_algo('fisheyes', True)
t_algo.iterate_year()
t_algo.flatten_dset()
#t_algo.iterate_dataset()
#t_algo.create_tree()
