from load_synops import Load_synops
from load_camera import Load_camera
from helper import Helper

import csv
import numpy as np
import pickle


class Main_loader(Helper):
	
	def __init__(self, fnames_synops, fnames_camera):
		Helper.__init__(self)
		self.fnames_synops = fnames_synops
		
	def create_synops_dict(self):
		self.synops_airport = Load_synops(self.fnames_synops['airport']).synops
		self.synops_koliba = Load_synops(self.fnames_synops['koliba']).synops
		self.synops_avg = self.avg_synops(self.synops_airport, self.synops_koliba)
		
	def create_camera_dict(self):
		lc = Load_camera('fisheyes')
		lc.iterate_year()
		lc.save_obj(lc.imgs, 'imgs' )
		print(lc.key_c)
		print('key set ' + str(len(lc.key_set)))
		
	def load_obj(self, name):
		with open('obj/' + name + '.pkl', 'rb') as f:
			self.imgs_dict = pickle.load(f)
		
	def compare_dicts(self, synops):
		imgs_keys = set(self.imgs_dict.keys())
		synops_keys = set(synops.keys())
		print('	imgs keys ' + str(len(imgs_keys)))
		print('	synops keys ' + str(len(synops_keys)))
		print('	intersect : ' + str(len(imgs_keys.intersection(synops_keys))))
		
	def strip_to_datetime(self):
		#print('imgs dict keyset')
		#print(self.imgs_dict.keys())
		#print('size of keyset ' + str(len(self.imgs_dict.keys())))
		self.imgs_dict = self.daytime_only(self.imgs_dict)
		print(self.err_count)
		
'''
print('main loader exec')
main_loader = Main_loader({'koliba': 'SYNOPs/BA_Koliba_SYNOP_2014-2016.txt', 'airport':'SYNOPs/BA_Letisko_SYNOP_2014-2016.txt' }, None)
#main_loader.create_camera_dict()

main_loader.create_synops_dict()
main_loader.load_obj('imgs')

main_loader.strip_to_datetime()
#print(main_loader.synops_airport.values())

print 'compare airport'
main_loader.compare_dicts(main_loader.synops_airport)
print 'compare koliba'
main_loader.compare_dicts(main_loader.synops_koliba)
print 'compare avg'
main_loader.compare_dicts(main_loader.synops_avg)
'''