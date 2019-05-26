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
		#self.path = path
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
					#print(items[1])
					#print(self.dt_dic)
					#return
					if new_save:
						try:
							self.load_photo(items[1], w, h, do_rgb=False)
						except:
							print('not loaded')
							print(self.addr + '/' + items[1])
					

	def load_photo(self, fname,w,h, do_rgb=False):
		img_read = cv.imread(self.addr + '/' + fname)
		#M = cv.getRotationMatrix2D((112,28), 180, 1)
		#img_read = cv.warpAffine(img_read, M, (224, 56))	# cropped pixels from bottom
		#img_read = img_read[:50, :50]  # cropping here
		#img_read = cv.resize(img_read, (w,h))
		#print(self.result_addr + '/' + fname)
		if do_rgb is False:
			img_read = cv.cvtColor(img_read, cv.COLOR_BGR2HSV)
		cv.imwrite(self.result_addr + '/' + fname, img_read)
		return img_read
	
	def add_photos_to_data(self):
		cnt = 1
		for k, v in self.dt_dic.iteritems():
			#print(self.addr + '/' + v[0])
			img_read = cv.imread(self.addr + '/' + v[0])
			if img_read is not None:
				self.dt_dic[k].append(img_read)
			else:
				print('NONE')
		
	def dict_to_dframe(self):
		df = pd.DataFrame.from_dict(self.dt_dic, orient='index')
		df.rename(columns={0:'name',1:'synop', 2:'data'}, inplace=True)
		print(df.data.shape)
		df['d1'] = df.data.apply(lambda x: x[:,:,0])
		df['d2'] = df.data.apply(lambda x: x[:,:,1])
		df['d3'] = df.data.apply(lambda x: x[:,:,2])
		df.to_csv('df_9to8.csv')

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
	
# doing cropping right now, check load photo
l = Loader('bright2/list_9to8.txt', 'bright2', 'pano_cropped_hsv2', new_load = True, new_save=False, w=224, h=224)

# need to run these two only if creating new dataframe
l.add_photos_to_data()	
l.dict_to_dframe()

#l.load_csv_df('df_cropped.csv')



#to rotate
'''
M = cv2.getRotationMatrix2D(center, angle180, scale)
rotated180 = cv2.warpAffine(img, M, (w, h))
'''