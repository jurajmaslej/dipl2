import numpy as np
from keras import backend as K
import tensorflow as tf

class Keras_metrics:
	
	def __init__(self):
		pass
	
	def loss1(self, yt, yp, tensors=True):
		if tensors:
			yt = K.eval(yt) 
			yp = K.eval(yp) 
		yt_ind = np.argmax(yt)
		yp_ind = np.argmax(yp)
		diff = abs(yt_ind - yp_ind)
		diff_sq = diff**2
		return diff_sq
	
	def loss2(self, yt, yp, metric=False):
		## 8-0 = 8, 8**2 = 64 -> to je max. rozdiel pre 9 tried
		## metrika(64 - aktualny rozdiel) da funkciu, kt. stupa cim lepsie klasifikujem
		## lebo cim lepsie klasifikujem, tym akt. rozdiel bude klesat
		yt_ind = K.argmax(yt)
		yp_ind = K.argmax(yp)
		if metric:
			return 1 - K.square(yp-yt)   
		return K.square(yp - yt)
	

	def single_class_accuracy(self, y_true, y_pred):
		INTERESTING_CLASS_ID = 4
		# https://stackoverflow.com/questions/41458859/keras-custom-metric-for-single-class-accuracy
		class_id_true = K.argmax(y_true, axis=-1)
		class_id_preds = K.argmax(y_pred, axis=-1)
		# Replace class_id_preds with class_id_true for recall here
		accuracy_mask = K.cast(K.equal(class_id_preds, INTERESTING_CLASS_ID), 'int32')
		class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
		class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
		return class_acc
	
		
#km = Keras_metrics()
#yt = np.array([0,0,0,1])
#yp = np.array([1,0,0,0])
#km.loss1(yt, yp, tensors=False)