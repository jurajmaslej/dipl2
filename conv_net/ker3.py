

import numpy as np
import cv2
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.models import Model
from os.path import dirname as up_one_dir

from os import listdir
from os.path import isfile, join, abspath

def create_model(img, img_txt, dir_of_images, dir_save_to):
	img_path = dir_of_images + img
	img_txt_path = dir_of_images + img_txt
	sample_inp = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
	sample_out = np.loadtxt(img_txt_path, dtype=np.float32)
	rows,cols = sample_inp.shape
	sample_inp = np.array(sample_inp).reshape((rows,cols,1))
	sample_out = np.array(sample_out).reshape((rows,cols,1))
	sample_inp.shape
	sample_out.shape
	samples_inp = np.array([sample_inp])
	samples_out = np.array([sample_out])

	inp = Input(shape=(None,None,1))
	out = Conv2D(1, (3, 3), kernel_initializer='normal', use_bias=False, padding='same')(inp)
	model = Model(inputs=inp, outputs=out)
	model.summary()

	model.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])	#kompilovanie modelu, mse = mean squared error, optimizer -> ako hladat spravne vahy
	num_epochs = 100
	
	
	for i in range(50,80,100):
		num_epochs = i
		model.fit(samples_inp, samples_out, batch_size=1, epochs = num_epochs)

		model.layers[1].get_weights()

		model.evaluate(samples_inp, samples_out, verbose=True)
		model.metrics_names

		output_images = model.predict(samples_inp)

		output_image = output_images[0].reshape((rows,cols))

		output_image = abs(output_image);
		output_image = cv2.normalize(output_image,None,0,255,cv2.NORM_MINMAX)
		output_image = np.uint8(output_image)
		name = img[:-4] + str(num_epochs) + '_grayscale.jpg'
		print('saving in ', join(dir_save_to, name))
		print(name)
		print(dir_save_to)
		cv2.imwrite(join(dir_save_to, name), output_image)
	#model.save('sobel.h5')

	#quit()
