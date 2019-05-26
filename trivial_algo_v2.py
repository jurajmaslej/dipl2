from __future__ import division

from loader import Loader
from histogram_comparer import Hsv
from plot_rbf_parameters import Rbf_params

from os import listdir
import numpy as np
import cv2 as cv

from sklearn import tree
from sklearn import svm
from sklearn.metrics import accuracy_score

import pandas as pd
from tabulate import tabulate

class Trivial_algo():
	
	def __init__(self, rgb=False):
		if rgb:
			l = Loader('bright2/list.txt', 'bright2', 'formatted_pics2_224_rgb', new_load = False)
		else:
			l = Loader('bright/list.txt', 'bright', 'formatted_pics', new_load = False)
		self.loader_df, self.train, self.validate, self.test = l.load_csv_df('df.csv')
		self.hst_script = Hsv(examples_init=False)
		
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
			pht_dic[img] = [histo]
		formatted_df = pd.DataFrame.from_dict(pht_dic, orient='index').reset_index()
		self.df = pd.merge(self.loader_df[['name','synop']], formatted_df, left_on=['name'], right_on=['index'], how='inner')
		self.df.drop(columns=['index'], inplace=True)
		self.df.rename(columns={0:'hst'}, inplace=True)
		#print(self.df.hst.iloc[0].shape)
		#print(tabulate(self.df.query('index<2'), headers='keys', tablefmt='psql'))
	
	def split_dataset(self):
		train, test = np.split(self.df.sample(frac=1), [int(.8*len(self.df))])
		#print(train.shape)
		#print(test.shape)
		return train, test
	
	def create_tree(self, train, test):
		clf = tree.DecisionTreeClassifier()
		#print(len(list(train.hst.values)))
		clf.fit(list(train.hst.values), list(train.synop.values))
		y_train_pred = clf.predict(list(train.hst.values))
		diff_train = train.synop.values - y_train_pred
		y_test_pred = clf.predict(list(test.hst.values))
		diff_test = abs(test.synop.values - y_test_pred)
		print('test dataset len {}'.format(test.synop.values.shape[0]))
		print('sum test err '+ str(diff_test.sum()))
		print('relative err {}'.format(diff_test.sum()/test.synop.values.shape[0]))
		
	def create_svm(self, train, test):
		clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
		clf.fit(list(train.hst.values), list(train.synop.values))
		y_train_pred = clf.predict(list(train.hst.values))
		diff_train = train.synop.values - y_train_pred
		y_test_pred = clf.predict(list(test.hst.values))
		diff_test = abs(test.synop.values - y_test_pred)
		print('test dataset len {}'.format(test.synop.values.shape[0]))
		print('sum test err '+ str(diff_test.sum()))
		print('relative err {}'.format(diff_test.sum()/test.synop.values.shape[0]))
		
	def rbf_params(self, train, test):
		self.rbf = Rbf_params()
		self.rbf.do_stuff(train, test)
		
	
	
	
t_alg = Trivial_algo()
t_alg.load_photos('pano_cropped')
train, test = t_alg.split_dataset()
print(train.hst.iloc[0])
#t_alg.create_tree(train, test)
#t_alg.create_svm(train, test)
#t_alg.rbf_params(train, test)