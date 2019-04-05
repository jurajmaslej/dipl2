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
	
	def __init__(self, path, addr, result_addr, new_load=False, new_save=False, w=150, h=150):
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
					if new_save:
						self.load_photo(items[1], w, h)
					#break

	def load_photo(self, fname,w,h):
		img_read = cv.imread(self.addr + '/' + fname)
		img_read = cv.resize(img_read, (w,h))
		hsv = cv.cvtColor(img_read, cv.COLOR_BGR2HSV)
		cv.imwrite(self.result_addr + '/' + fname, hsv)
		return hsv
	
	def add_photos_to_data(self):
		for k, v in self.dt_dic.iteritems():
			img_read = cv.imread(self.addr + '/' + v[0])
			self.dt_dic[k].append(img_read)
		
	def dict_to_dframe(self):
		df = pd.DataFrame.from_dict(self.dt_dic, orient='index')
		df.rename(columns={0:'name',1:'synop', 2:'data'}, inplace=True)
		print(df.data.shape)
		df['d1'] = df.data.apply(lambda x: x[:,:,0])
		df['d2'] = df.data.apply(lambda x: x[:,:,1])
		df['d3'] = df.data.apply(lambda x: x[:,:,2])
		df.to_csv('df.csv')

	def from_np_array(self, array_string):
		array_string = ','.join(array_string.replace('[ ', '[').split())
		#print('arr str')
		#print(type(array_string))
		return np.array(ast.literal_eval(array_string))
		
	def load_csv_df(self,fname):
		#self.main_df = pd.read_csv(fname, converters={'d1': self.from_np_array})
		self.main_df = pd.read_csv(fname)
		#print(tabulate(self.main_df, headers='keys', tablefmt='psql'))
		
		#a= self.main_df.d1.iloc[0]
		#array_string = ','.join(a.replace('[ ', '[').split())
		#b = ast.literal_eval(array_string)
		#print(self.main_df.d1.iloc[0])
		self.train, self.validate, self.test = np.split(self.main_df.sample(frac=1), [int(.6*len(self.main_df)), int(.8*len(self.main_df))])
		#
		return self.main_df, self.train, self.validate, self.test
		
l = Loader('bright2/list.txt', 'bright2', 'formatted_pics2_224', new_load = True, new_save=True, w=224, h=224)

# need to run these two only if creating new dataframe
l.add_photos_to_data()	
#l.dict_to_dframe()

#l.load_csv_df('df.csv')