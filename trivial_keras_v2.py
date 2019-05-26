from __future__ import division


# import my files
from loader import Loader
from histogram_comparer import Hsv
from save_model import Save_model
from keras_metrics import Keras_metrics

# system and libraries
from os import listdir
import numpy as np
import cv2 as cv
import pandas as pd
from tabulate import tabulate

from sklearn.metrics import accuracy_score

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from keras import backend as K
from math import ceil

#preprocess for imagenet, keras
from keras.applications.imagenet_utils import preprocess_input

#resnet
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import GlobalAveragePooling2D
from keras.layers import Dropout
from keras import Model

import tensorflow as tf



class Trivial_keras():
	
	def __init__(self, data_path):
		l = Loader(result_addr=data_path, new_load = False)
		self.loader_df, self.train, self.validate, self.test = l.load_csv_df('df.csv')
		self.hst_script = Hsv(examples_init=False)
		self.save_model = Save_model()
		self.custom_metric_obj = Keras_metrics()
		
	def load_photos(self, path):
		'''
		DO's:
		join column with histograms of hsv150*150*3 photos to self.loader_df, thus create self.df
		'''
		pht_dic = dict()
		for img in listdir(path):
			im_name = path +'/'+ img
			img_read = cv.imread(im_name)
			#histo = self.hst_script.histogram_hsv(img_read)
			img_read = np.array(img_read, dtype=np.float64)
			img_read = preprocess_input(img_read)
			pht_dic[img] = [img_read]
		formatted_df = pd.DataFrame.from_dict(pht_dic, orient='index').reset_index()
		self.df = pd.merge(self.loader_df[['name','synop']], formatted_df, left_on=['name'], right_on=['index'], how='inner')
		self.df.drop(columns=['index'], inplace=True)
		self.df.rename(columns={0:'hst'}, inplace=True)
		#print(self.df.hst.iloc[0].shape)
	
	def pprint(self):
		print(tabulate(self.df.query('index<2'), headers='keys', tablefmt='psql'))
		
	def split_dataset(self, perc):
		sample = self.df.sample(frac=perc)
		train, test = np.split(sample, [int(0.6*len(sample))])
		return train, test
	
	def do_one_hot(self, merged=False):
		if merged:
			self.df.synop= self.df.synop.apply(lambda x: self.one_hot_encode_merge_classes(x, 3))
			return
		self.df.synop= self.df.synop.apply(lambda x: self.one_hot_encode(x))
		
	def one_hot_encode(self, synop):
		empty = np.array([0,]*NUM_CLASSES)
		syn_int = int(synop)
		encoded = empty.copy()
		encoded[syn_int - 1] = 1
		return encoded
	
	def one_hot_encode_merge_classes(self, synop, merged_num_cl = None):  # merged num cl 3
		empty = np.array([0,]*merged_num_cl)
		syn_int = int(synop)
		#print((NUM_CLASSES / merged_num_cl)
		if merged_num_cl:
			syn_int = int(ceil(syn_int / (NUM_CLASSES / merged_num_cl)))
			#print(new_cl)
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
		
	def resnet50_model_trivial(self, train, test):
		train_reshape = np.array([x for x in train.hst.values])
		test_reshape = np.array([x for x in test.hst.values])
		train_synp =  np.array([x for x in train.synop.values])
		test_synp =  np.array([x for x in test.synop.values])
		print(train_synp.shape)
		base_model = ResNet50(input_shape= (224,224,3), weights='imagenet', pooling='max')
		base_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		base_model.fit(train_reshape, train_synp, validation_data=(test_reshape, test_synp), epochs=5)
		
	def custom_loss(self, y_true, y_pred):
		#y_true = tf.cast(y_true, tf.int64)
		#y_pred = tf.cast(y_pred, tf.int64)
		return self.custom_metric_obj.loss2(y_true, y_pred)
	
	def custom_metric(self, y_true, y_pred):
		return self.custom_metric_obj.loss2(y_true, y_pred, metric=True)
	
	def one_class_acc(self, y_true, y_pred):
		return self.custom_metric_obj.single_class_accuracy(y_true, y_pred)
	
	def trainable_setup(self, base_model):
		# https://github.com/keras-team/keras/issues/9214
		trainable_list = []
		trainable1 = []
		for layer in base_model.layers:
			trainable1.append(layer.trainable)
		#print(trainable1)
		
		for layer in base_model.layers:
			if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
				layer.trainable = True
				trainable_list.append(True)
				K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
				K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
			else:
				layer.trainable = False
				trainable_list.append(False)
		#print(trainable_list)
		c= 0
		for i,j in zip(trainable1, trainable_list):
			c+=1
		print('trainable changed on : ' + str(c))
		return base_model
		
	def resnet50_model(self, train, test, epochs=10, complex_setup=False, weights='imagenet', dropout=True, save=False):
		
		train_reshape = np.array([x for x in train.hst.values]) 
		test_reshape = np.array([x for x in test.hst.values]) 
		train_synp =  np.array([x for x in train.synop.values]) 
		test_synp =  np.array([x for x in test.synop.values]) 
		dataset_len = train_reshape.shape[0] + test_reshape.shape[0]
		print('running on: {}'.format(dataset_len))
		base_model = ResNet50(input_shape= (224,224,3), weights=weights, include_top=False)
		if complex_setup:
			base_model = self.trainable_setup(base_model)
			
		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		if dropout:
			x = Dropout(0.7)(x)
		predictions = Dense(int(NUM_CLASSES/3), activation='softmax')(x)
		model = Model(inputs= base_model.inputs, outputs= predictions)
		model.compile(optimizer='adam', loss=self.custom_loss, metrics=['categorical_accuracy','accuracy',self.custom_metric, self.one_class_acc])
		history_callback = model.fit(train_reshape, train_synp, validation_data=(test_reshape, test_synp), epochs=epochs)
		
		# add proper TEST sample there
		y_prediction = model.predict(test_reshape)
		y_train_pred = model.predict(train_reshape)
		
		if save:
			self.save_model.save_keras('test_epochs{}_size{}_weights-{}'.format(str(epochs), str(1900), weights), None, history_callback, save_model_plot=True)
			self.save_model.plot_conf_matrix(test_synp, y_prediction)
			self.save_model.plot_conf_matrix(train_synp, y_train_pred, train_matrix=True)
			
		
DATA_PATH = 'formatted_pics2_224_rgb_cropped'
NUM_CLASSES = 9
FRAC=0.05
t_alg = Trivial_keras(DATA_PATH)
t_alg.load_photos(DATA_PATH)
t_alg.do_one_hot(merged=True)

#t_alg.pprint()
train, test = t_alg.split_dataset(0.4)	
#t_alg.buil_trivial_model(train, test, save=True)
t_alg.resnet50_model(train, test, epochs=1, complex_setup=False, weights='imagenet', save=True)
	