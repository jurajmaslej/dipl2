from os import makedirs
from datetime import datetime
import numpy as np

from keras.models import model_from_json
from keras.utils import plot_model
from json import dump
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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
			
	def plot_model_scores(self, history):
		# Plot training & validation accuracy values
		print('plt started')
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.grid()
		print('plt started2')
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
		
	def plot_conf_matrix(self, yt, yp): #yt - ytrue, yp - ypredicted
		confusion_matrix(yt.argmax(axis=1), yp.argmax(axis=1))
		plt.figure()
		plt.plot(confusion_matrix)
		print(confusion_matrix)
		plt.savefig('conf_matrix.png')
		
			