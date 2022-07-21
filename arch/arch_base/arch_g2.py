from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class ARCH_g2():
	def __init__(self):
		print("Creating 2-D Gaussian architectures for base cases")
		return

	def generator_model_dense_g2(self):
		init_fn = tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None)
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.noise_dims,))

		enc1 = tf.keras.layers.Dense(64, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(inputs)
		enc1 = tf.keras.layers.LeakyReLU()(enc1)

		enc2 = tf.keras.layers.Dense(32, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(enc1)
		# enc2 = tf.keras.layers.Activation( activation = 'tanh')(enc2)
		enc2 = tf.keras.layers.LeakyReLU()(enc2)

		enc3 = tf.keras.layers.Dense(16, kernel_initializer=init_fn, use_bias = False)(enc2)
		# enc3 = tf.keras.layers.Activation( activation = 'tanh')(enc3)
		enc3 = tf.keras.layers.LeakyReLU()(enc3)

		enc4 = tf.keras.layers.Dense(self.output_size, kernel_initializer=init_fn, use_bias = False)(enc3)

		model = tf.keras.Model(inputs = inputs, outputs = enc4)

		return model


	def discriminator_model_dense_g2(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(10, use_bias=False, input_shape=(2,), kernel_initializer=init_fn))
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(20, use_bias=False, kernel_initializer=init_fn))
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(5, use_bias=False, kernel_initializer=init_fn))
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(1))
		return model

	def show_result_g2(self, images = None, num_epoch = 0, path = 'result.png', show = False, save = True):
		basis = np.expand_dims(np.linspace(-10., 10., int(1e4), dtype=np.float32), axis = 1)

		if self.total_count.numpy() == 1 or self.total_count.numpy()%10000 == 0:
			np.save(path+'_reals.npy',np.array(self.reals))
			np.save(path+'_fakes.npy',np.array(self.fakes))
		
		if self.colab == 1:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
		else:
			from matplotlib.backends.backend_pgf import FigureCanvasPgf
			matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "serif",  # use serif/main font for text elements
				"font.size":10,	
				"font.serif": [], 
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		with PdfPages(path+'_Classifier.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(False)
			ax1.get_yaxis().set_visible(False)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim(bottom=self.MIN,top=self.MAX)
			ax1.scatter(self.reals[:,0],self.reals[:,1], c='r', linewidth = 0.8,  marker = '.', alpha = 0.8)
			ax1.scatter(self.fakes[:,0],self.fakes[:,1], c='g', linewidth = 0.8,  marker = '.', alpha = 0.8)
			fig1.tight_layout()
			pdf.savefig(fig1)
			plt.close(fig1)