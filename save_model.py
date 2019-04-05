from os import makedirs
from datetime import datetime
import numpy as np

from keras.models import model_from_json
from keras.utils import plot_model
from json import dump
import matplotlib.pyplot as plt

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
			with open('{}/history.json'.format(self.dir_name), 'w') as f:
				dump(history.history, f)
				
		if save_model_plot and model:
			plot_model(model, to_file='{}/model.png'.format(self.dir_name))
			self.plot_model_scores(history)
			
	def plot_model_scores(self, history):
		# Plot training & validation accuracy values
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('Model accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Test'], loc='upper left')
		plt.savefig('{}/train_valid_acc.png'.format(self.dir_name))
		
		plt.figure()
		# Plot training & validation loss values
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('Model loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Test'], loc='upper left')
		plt.savefig('{}/train_valid_loss.png'.format(self.dir_name))
			