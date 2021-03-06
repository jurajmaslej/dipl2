from os import makedirs
from datetime import datetime
import numpy as np

from keras.models import model_from_json
from keras.utils import plot_model
from json import dump
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class Save_model:

	def __init__(self):
		pass
	
	def save_keras(self, dir_name, model=None, history=None, save_model_plot=False):
		self.dir_name = 'logs/' + dir_name + datetime.now().strftime('%c')
		makedirs(self.dir_name)
		if model:
			model_json = model.to_json()
			with open("model.json", "w") as json_file:
				json_file.write(model_json)
			# serialize weights to HDF5
			model.save_weights("{}/model.h5".format(self.dir_name))
			print("Saved model to disk")
			
		if history:
			self.plot_model_scores(history)
			with open('{}/history.json'.format(self.dir_name), 'w') as f:
				dump(history.history, f)
				
		if save_model_plot and model:
			plot_model(model, to_file='{}/model.png'.format(self.dir_name))
			
			
	def plot_model_scores(self, history):
		# Plot training & validation accuracy values
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.grid()
		plt.title('Model accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Test'], loc='upper left')
		plt.savefig('{}/acc.png'.format(self.dir_name))
		
		plt.figure()
		# Plot training & validation loss values
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.grid()
		plt.title('Model loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Test'], loc='upper left')
		plt.savefig('{}/loss.png'.format(self.dir_name))
		
		plt.figure()
		# categ acc
		plt.plot(history.history['categorical_accuracy']) 
		plt.plot(history.history['val_categorical_accuracy'])
		plt.grid()
		plt.title('categ acc')
		plt.ylabel('categ_acc')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Test'], loc='upper left')
		plt.savefig('{}/cat_acc.png'.format(self.dir_name))
		
		plt.figure()
		# custom metric same as loss at this moment
		plt.plot(history.history['custom_metric'])
		plt.plot(history.history['val_custom_metric'])
		plt.grid()
		plt.title('custom metric')
		plt.ylabel('custom_metric')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Test'], loc='upper left')
		plt.savefig('{}/custom_metric.png'.format(self.dir_name))
		
		plt.figure()
		# custom metric same as loss at this moment
		plt.plot(history.history['one_class_acc'])
		plt.plot(history.history['val_one_class_acc'])
		plt.grid()
		plt.title('one class acc')
		plt.ylabel('one class acc')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Test'], loc='upper left')
		plt.savefig('{}/one_class_acc.png'.format(self.dir_name))
		
	def plot_conf_matrix(self, yt, yp, train_matrix=False): #yt - ytrue, yp - ypredicted
		plt.figure()
		confm = confusion_matrix(yt.argmax(axis=1), yp.argmax(axis=1))
		if train_matrix:
			print('train matrix')
		print(confm)
		plt.plot(confm)
		plt.savefig('confm.png')
		
		plt.figure()
		plt.figure(figsize=(8, 6))
		plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
		plt.imshow(confm, interpolation='nearest', cmap=plt.cm.hot,
				norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
		plt.xlabel('x')
		plt.ylabel('y')
		plt.colorbar()
		#plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
		#plt.yticks(np.arange(len(C_range)), C_range)
		if train_matrix:
			plt.title('train_mtrix')
			plt.savefig('train_mtrx.png')
		else:
			plt.title('valid_mtrix')
			plt.savefig('valid_mtrx.png')
		#plt.show()
		
		
			