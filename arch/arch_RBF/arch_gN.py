from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
from matplotlib.backends.backend_pgf import PdfPages



class ARCH_gN():	
	def __init__(self):
		return

	def generator_model_dcgan_gN(self):
		# FOr regular comarisons when 2m-n = 0
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.075, seed=None) 
		# FOr comparisons when 2m-n > 0 
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
		# FOr comparisons when 2m-n < 0 
		init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None)
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)
		
		inputs = tf.keras.Input(shape=(self.noise_dims,))

		enc0 = tf.keras.layers.Dense(32*32*3, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(inputs)
		enc0 = tf.keras.layers.LeakyReLU()(enc0)

		enc_res = tf.keras.layers.Reshape([32,32,3])(enc0)

		enc1 = tf.keras.layers.Conv2D(1024, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc_res) #16x16x64
		enc1 = tf.keras.layers.BatchNormalization()(enc1)
		enc1 = tf.keras.layers.LeakyReLU()(enc1)

		enc2 = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc1) #8x8x128
		enc2 = tf.keras.layers.BatchNormalization()(enc2)
		enc2 = tf.keras.layers.LeakyReLU()(enc2)

		enc3 = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc2) #4x4x256
		enc3 = tf.keras.layers.BatchNormalization()(enc3)
		enc3 = tf.keras.layers.LeakyReLU()(enc3)


		enc4 = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc3) #2x2x128
		enc4 = tf.keras.layers.BatchNormalization()(enc4)
		enc4 = tf.keras.layers.LeakyReLU()(enc4)

		enc5 = tf.keras.layers.Conv2D(int(self.latent_dims), 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc4) #1x1xlatent

		enc = tf.keras.layers.Flatten()(enc5)
		enc = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(enc)

		model = tf.keras.Model(inputs = inputs, outputs = enc)

		return model


	def show_result_gN(self,images = None,num_epoch = 0,show = False,save = False,path = 'result.png'):
		print("\n Gaussian Stats : True mean {} True Cov {} \n Fake mean {} Fake Cov {}".format(np.mean(self.reals,axis = 0), np.cov(self.reals,rowvar = False), np.mean(self.fakes, axis = 0), np.cov(self.fakes,rowvar = False) ))
		if self.res_flag:
			self.res_file.write("\n Gaussian Stats : True mean {} True Cov {} \n Fake mean {} Fake Cov {}".format(np.mean(self.reals, axis = 0), np.cov(self.reals,rowvar = False), np.mean(self.fakes, axis = 0), np.cov(self.fakes,rowvar = False) ))

	def save_for_paper_gN(self, images = None, num_epoch = 0, path = 'result.png'):
		print("Gaussian Stats : True mean {} True Cov {} \n Fake mean {} Fake Cov {}".format(np.mean(self.reals,axis = 0), np.cov(self.reals,rowvar = False), np.mean(self.fakes, axis = 0), np.cov(self.fakes,rowvar = False) ))
		if self.res_flag:
			self.res_file.write("Gaussian Stats : True mean {} True Cov {} \n Fake mean {} Fake Cov {}".format(np.mean(self.reals, axis = 0), np.cov(self.reals,rowvar = False), np.mean(self.fakes, axis = 0), np.cov(self.fakes,rowvar = False) ))

	def FID_gN(self):
		#### Function is called FID, but it computes W^{2,2} as there is no inception embedding
		if self.FID_load_flag == 0:
			self.FID_load_flag = 1	
			self.fid_image_dataset = self.train_dataset


		if self.mode == 'fid':
			print(self.checkpoint_dir)
			self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			print('Models Loaded Successfully')

		with tf.device(self.device):
			for data_batch in self.fid_image_dataset:
				noise = tf.random.normal([self.batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)
				preds = self.generator(noise, training=False)
				preds = preds.numpy()

				try:
					self.act1 = np.concatenate([self.act1,data_batch], axis = 0)
					self.act2 = np.concatenate([self.act2,preds], axis = 0)
				except:
					self.act1 = data_batch
					self.act2 = preds
			self.eval_FID()
			return
