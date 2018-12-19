import os
import numpy as np
import cv2 as cv

#resized_hsv_fisheye_1414593023678_1403623027411_00013_20140624_171851_jp.png
# /home/juraj/Desktop/juro/programovanie/dipl/dipl/init_work2

def open_img(fname):
	path = '/home/juraj/Desktop/juro/programovanie/dipl/dipl/init_work2/fisheyes/2014/06/24/' + fname
	img_read = cv.imread(path)
	print(len(img_read))
	print(len(img_read[0]))
	print(img_read[0][0])
	for i in img_read:
		for j in i:
			print j,
		print('\n')
	
open_img('resized_hsv_fisheye_1414593023678_1403623027411_00013_20140624_171851_jp.png')