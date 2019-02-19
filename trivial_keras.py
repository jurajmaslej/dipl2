from helper import Helper
from main_loader import Main_loader
from histogram_comparer import Hsv

import os
import numpy as np
import cv2 as cv

from sklearn import tree
from sklearn.metrics import accuracy_score

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

class Trivial_algo(Helper):
	
	def __init__(self, main_folder, init_loader=True, create_syn_dict=True, create_cam_dict=False):
		Helper.__init__(self)
		self.main_folder = main_folder
		self.histograms = Hsv()
		self.hist_dict = {}
		if init_loader:
			self.main_loader = Main_loader({'koliba': 'SYNOPs/BA_Koliba_SYNOP_2014-2016.txt', 'airport':'SYNOPs/BA_Letisko_SYNOP_2014-2016.txt' }, None)
			if create_syn_dict:
				self.main_loader.create_synops_dict()
			if create_cam_dict:
				self.main_loader.create_camera_dict()
			self.main_loader.load_obj('imgs')
			self.main_loader.strip_to_datetime()
			
	def prepare_dataset(self, sample_size=0.25, print_shapes=False):
		'''
		img dataset size : 3216*150*150*
		sample_size: percentage of data to use, 1 == all 
		'''
		self.synops_airport = self.one_hot_encode(self.main_loader.synops_airport)
		self.synops_koliba = self.one_hot_encode(self.main_loader.synops_koliba)
		x_all = list()
		y_all = list()
		for ts, img_list in self.main_loader.imgs_dict.items():
			if ts in self.synops_airport.keys():
				for img in img_list:
					x_all.append(img)
					y_all.append(self.synops_airport[ts])
		x_all_np = np.array(x_all)[:int(len(x_all)*sample_size)]
		y_all_np = np.array(y_all)[:int(len(y_all)*sample_size)]
		self.x_train = x_all_np[:int(x_all_np.shape[0]*0.8)]
		self.x_val = x_all_np[int(x_all_np.shape[0]*0.8):]
		self.y_train = y_all_np[:int(y_all_np.shape[0]*0.8)]
		self.y_val = y_all_np[int(y_all_np.shape[0]*0.8):]
		if print_shapes:
			print(x_all_np.shape)
			print(y_all_np.shape)
			print(self.x_train.shape)
			print(self.y_train.shape)
			print(self.x_val.shape)
			print(self.y_val.shape)
			
	def one_hot_encode(self, synop_dict):
		empty = np.array([0,]*8)
		for key, synop in synop_dict.items():
			try:
				syn_int = int(synop)
				encoded = empty.copy()
				encoded[syn_int - 1] = 1
				synop_dict[key] = encoded
				#self.main_loader.synops_airport[key] = to_categorical(self.main_loader.synops_airport[key])
			except:
				#print('bad synop')
				#print(str(key) + ': ' +str(synop))
				pass
		return synop_dict
	
	def buil_trivial_model(self):
		#create model
		model = Sequential()

		#add model layers
		model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(150,150,3)))
		model.add(Conv2D(32, kernel_size=3, activation='relu'))
		model.add(Conv2D(32, kernel_size=4, activation='relu'))
		model.add(Flatten())
		model.add(Dense(8, activation='softmax'))
		
		#compile model using accuracy to measure model performance
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		
		#train the model
		model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val), epochs=5)
		
t_algo = Trivial_algo('fisheyes', True)
t_algo.prepare_dataset()
t_algo.buil_trivial_model()