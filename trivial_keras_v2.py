from __future__ import division

from loader import Loader
from histogram_comparer import Hsv
from save_model import Save_model

from os import listdir
import numpy as np
import cv2 as cv
import pandas as pd
from tabulate import tabulate

from sklearn.metrics import accuracy_score

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

#resnet
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import GlobalAveragePooling2D
from keras.layers import Dropout
from keras import Model

class Trivial_keras():
	
	def __init__(self):
		l = Loader('bright/list.txt', 'bright', 'formatted_pics', new_load = False)
		self.loader_df, self.train, self.validate, self.test = l.load_csv_df('df.csv')
		self.hst_script = Hsv(examples_init=False)
		self.save_model = Save_model()
		
	def load_photos(self, path):
		'''
		DO's:
		join column with histograms of hsv150*150*3 photos to self.loader_df, thus create self.df
		'''
		pht_dic = dict()
		for img in listdir(path):
			im_name = path +'/'+ img
			img_read = cv.imread(im_name)
			histo = self.hst_script.histogram_hsv(img_read)
			pht_dic[img] = [img_read]
		formatted_df = pd.DataFrame.from_dict(pht_dic, orient='index').reset_index()
		self.df = pd.merge(self.loader_df[['name','synop']], formatted_df, left_on=['name'], right_on=['index'], how='inner')
		self.df.drop(columns=['index'], inplace=True)
		self.df.rename(columns={0:'hst'}, inplace=True)
		#print(self.df.hst.iloc[0].shape)
	
	def pprint(self):
		print(tabulate(self.df.query('index<2'), headers='keys', tablefmt='psql'))
		
	def split_dataset(self):
		train, test = np.split(self.df.sample(frac=1), [int(.8*len(self.df))])
		#print(train.shape)
		#print(test.shape)
		return train, test
	
	def do_one_hot(self):
		self.df.synop= self.df.synop.apply(lambda x: self.one_hot_encode(x))
		
	def one_hot_encode(self, synop):
		empty = np.array([0,]*8)
		syn_int = int(synop)
		encoded = empty.copy()
		encoded[syn_int - 1] = 1
		return encoded
	
	def buil_trivial_model(self, train, test, save=False):
		train_reshape = np.array([x for x in train.hst.values])[:60] 
		test_reshape = np.array([x for x in test.hst.values])[:8] 
		train_synp =  np.array([x for x in train.synop.values])[:60] 
		test_synp =  np.array([x for x in test.synop.values])[:8] 
		#create model
		model = Sequential()

		#add model layers
		model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(224,224,3)))
		#model.add(Conv2D(32, kernel_size=3, activation='relu'))
		#model.add(Conv2D(32, kernel_size=4, activation='relu'))
		model.add(Flatten())
		model.add(Dense(8, activation='softmax'))
		
		#compile model using accuracy to measure model performance
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		
		#train the model
		history_callback = model.fit(train_reshape, train_synp, validation_data=(test_reshape, test_synp), epochs=10)
		print(type(history_callback))
		print(history_callback)
		if save:
			self.save_model.save_keras('test1', model, history_callback, save_model_plot=True)
		
	def resnet50_model(self, train, test):
		train_reshape = np.array([x for x in train.hst.values])
		test_reshape = np.array([x for x in test.hst.values])
		train_synp =  np.array([x for x in train.synop.values])
		test_synp =  np.array([x for x in test.synop.values])
		print(train_synp.shape)
		base_model = ResNet50(input_shape= (224,224,3), weights='imagenet', pooling='max')
		base_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		base_model.fit(train_reshape, train_synp, validation_data=(test_reshape, test_synp), epochs=5)
		
	def resnet50_model2(self, train, test, save=False):
		train_reshape = np.array([x for x in train.hst.values])[:60] 
		test_reshape = np.array([x for x in test.hst.values])[:8] 
		train_synp =  np.array([x for x in train.synop.values])[:60] 
		test_synp =  np.array([x for x in test.synop.values])[:8] 
		print(train_synp.shape)
		base_model = ResNet50(input_shape= (224,224,3), weights='imagenet', include_top=False)
		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		x = Dropout(0.7)(x)
		predictions = Dense(8, activation='softmax')(x)
		model = Model(inputs= base_model.inputs, outputs= predictions)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		history_callback = model.fit(train_reshape, train_synp, validation_data=(test_reshape, test_synp), epochs=1)
		print(type(history_callback))
		print(history_callback)
		if save:
			self.save_model.save_keras('test1', model, history_callback)
		

t_alg = Trivial_keras()
t_alg.load_photos('formatted_pics_224')
t_alg.do_one_hot()
#t_alg.pprint()
train, test = t_alg.split_dataset()	
t_alg.buil_trivial_model(train, test, save=True)
#t_alg.resnet50_model2(train, test, save=True)
	