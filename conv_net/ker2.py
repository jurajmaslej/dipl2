#C:> activate keras
#(keras) C:> set TF_CPP_MIN_LOG_LEVEL=2
#(keras) C:> python
# >>>

import numpy as np
import cv2
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.models import Model


def run_ker2(img_name):
	inp = Input(shape=(None,None,1)) # objekt kt pohlti vstupny obrazok
	out = Conv2D(1, (3, 3), kernel_initializer='normal', use_bias=False, padding='same')(inp)
	model = Model(inputs=inp, outputs=out)  # 2D konvolucna vrstva, prva 1 = iba jedna vrstva, kazda vrstva ma vlastny kernel, kernel rozmerov 3*3, padding = same -> zachovat okraje

	len(model.layers)
	#print model.layers[0].get_weights()
	#print model.layers[1].get_weights()

	w = np.array([[			#vahy nastavene na sobelov operator
		[[[-1]],[[0]],[[1]]],
		[[[-2]],[[0]],[[2]]],
		[[[-1]],[[0]],[[1]]]
		]])   #vahy do kernelu
	w_T = np.array([[			#vahy nastavene na sobelov operator
		[[[-1]],[[-2]],[[-1]]],
		[[[0]],[[0]],[[0]]],
		[[[1]],[[2]],[[1]]]
		]])
	#w = w_T
	model.layers[1].set_weights(w)
	model.layers[1].get_weights()  # momentalne 2 vrstvy

	image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
	image.shape
	rows,cols = image.shape

	input_image = np.array(image).reshape((rows,cols,1))		#reshape, pridat tretiu suradnicu
	input_images = np.array([input_image])

	output_images = model.predict(input_images)

	output_image = output_images[0].reshape((rows,cols))
	np.amax(output_image)
	np.amin(output_image)
	txt_fname = img_name[:-5] + '.txt'
	np.savetxt(txt_fname, output_image, fmt='%f')

	output_image = abs(output_image);
	output_image = cv2.normalize(output_image,None,0,255,cv2.NORM_MINMAX)
	output_image = np.uint8(output_image)
	cv2.imwrite('edge-horizon.jpg',output_image)

	#cv2.imshow('lena',output_image)
	#cv2.waitKey(0)
	#cv2.destroyWindow('lena')

#quit()
