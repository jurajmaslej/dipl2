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

class Loader():		#Helper optional
	
	def __init__(self, path, addr, result_addr, new_load=False):
		'''
		will load photos and synops according to txt file 
		'''
		self.path = path
		self.addr = addr
		self.result_addr = result_addr
		
		self.dt_dic = dict()
		
		if new_load:
			with open(path, 'r') as data_file:
				for fline in data_file:
					line = fline.strip()
					items = line.split(' ')
					if len(items) != 3:
						print 'faulty line in input file'
						continue
					dt_object = datetime.strptime(items[0],'%Y%m%d%H%M')
					self.dt_dic[dt_object] = list([items[1], items[2]])
					self.load_photo(items[1], new_save=False)
					#break

	def load_photo(self, fname, new_save=False):
		img_read = cv.imread(self.addr + '/' + fname)
		img_read = cv.resize(img_read, (150,150))
		hsv = cv.cvtColor(img_read, cv.COLOR_BGR2HSV)
		if new_save:
			cv.imwrite(self.result_addr + '/' + fname, hsv)
		return hsv
	
	def add_photos_to_data(self):
		for k, v in self.dt_dic.iteritems():
			img_read = cv.imread(self.addr + '/' + v[0])
			self.dt_dic[k].append(img_read)
		
	def dict_to_dframe(self):
		df = pd.DataFrame.from_dict(self.dt_dic, orient='index')
		df.rename(columns={0:'name',1:'synop', 2:'data'}, inplace=True)
		df.to_csv('df.csv')
		
	def load_csv_df(self,fname):
		self.main_df = pd.DataFrame.from_csv(fname)
		self.train, self.validate, self.test = np.split(self.main_df.sample(frac=1), [int(.6*len(self.main_df)), int(.8*len(self.main_df))])
		#print(tabulate(df, headers='keys', tablefmt='psql'))
		
l = Loader('bright/list.txt', 'bright', 'formatted_pics', new_load = True)

# need to run these two only if creating new dataframe
#l.add_photos_to_data()	
#l.dict_to_dframe()

l.load_csv_df('df.csv')