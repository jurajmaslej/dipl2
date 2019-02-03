import numpy as np
import cv2
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.models import Model

from os import listdir
from os.path import isfile, join, abspath

import ker2
import ker3

class Simple_conv:
	
	def __init__(self):
		two_up =  abspath(join(__file__ ,"../.."))
		print(two_up)
		self.path_to_img = two_up + '/dobre_fotky_jpg/'
		self.my_path = 'dobre_fotky_edges/'
		
	def load_imgs(self):	
		images = []
		onlyfiles = [f for f in listdir(self.path_to_img) if isfile(join(self.path_to_img, f))]
		for name in onlyfiles:
			if name[-4:] == 'jpeg':
				print('name ', name)
				ker2.run_ker2(self.path_to_img + name)
				ker3.create_model(name, name[:-4] + 'txt', self.path_to_img, self.my_path)
				return
			
conv = Simple_conv()
conv.load_imgs()