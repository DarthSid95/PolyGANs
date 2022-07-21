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

class ARCH_mnist():

	def __init__(self):
		print("CREATING ARCH_AAE CLASS")
		return


	#### Not Used. But the architecutre exists if needed
	def encdec_model_dcgan_mnist(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.output_size,self.output_size,1)) #64x64x3

		enc1 = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(inputs) #32x32x64
		enc1 = tf.keras.layers.BatchNormalization()(enc1)
		enc1 = tf.keras.layers.LeakyReLU()(enc1)

		enc2 = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc1) #16x16x128
		enc2 = tf.keras.layers.BatchNormalization()(enc2)
		enc2 = tf.keras.layers.LeakyReLU()(enc2)

		enc3 = tf.keras.layers.Conv2D(512, 4, strides=1, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc2) #8x8x256
		enc3 = tf.keras.layers.BatchNormalization()(enc3)
		enc3 = tf.keras.layers.LeakyReLU()(enc3)

		enc4 = tf.keras.layers.Conv2D(1024, 4, strides=1, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc3) #4x4x128
		enc4 = tf.keras.layers.BatchNormalization()(enc4)
		enc4 = tf.keras.layers.LeakyReLU()(enc4)

		dense = tf.keras.layers.Flatten()(enc4)
		dense = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(dense)


		enc = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(dense)
		encoded = tf.keras.Input(shape=(self.latent_dims,))



		den = tf.keras.layers.Dense(1024*int(self.output_size/4)*int(self.output_size/4), kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(encoded)
		enc_res = tf.keras.layers.Reshape([int(self.output_size/4),int(self.output_size/4),1024])(den)

		denc5 = tf.keras.layers.Conv2DTranspose(512, 4, strides=2,padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc_res) #2x2x128
		denc5 = tf.keras.layers.BatchNormalization()(denc5)
		denc5 = tf.keras.layers.LeakyReLU()(denc5)

		denc4 = tf.keras.layers.Conv2DTranspose(256, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=True, bias_initializer = bias_init_fn)(denc5) #4x4x128
		denc4 = tf.keras.layers.BatchNormalization()(denc4)
		# denc4 = tf.keras.layers.Dropout(0.5)(denc4)
		denc1 = tf.keras.layers.LeakyReLU()(denc4)

		out = tf.keras.layers.Conv2DTranspose(1, 4,strides=1,padding='same', kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(denc1) #64x64x3
		out =  tf.keras.layers.Activation( activation = 'tanh')(out)


		self.Decoder = tf.keras.Model(inputs=encoded, outputs=out)
		self.Encoder = tf.keras.Model(inputs=inputs, outputs=enc)
		return

	def encdec_model_mnist(self):


		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.output_size,self.output_size,1)) #64x64x3

		input_res =tf.keras.layers.Reshape([int(self.output_size*self.output_size)])(inputs)

		dec0 = tf.keras.layers.Dense(512, kernel_initializer=init_fn, use_bias = True)(input_res)
		dec0 = tf.keras.layers.BatchNormalization()(dec0)
		dec0 = tf.keras.layers.LeakyReLU()(dec0)

		dec1 = tf.keras.layers.Dense(256, kernel_initializer=init_fn)(dec0)
		dec1 = tf.keras.layers.BatchNormalization()(dec1)
		dec1 = tf.keras.layers.LeakyReLU()(dec1)

		dec2 = tf.keras.layers.Dense(128, kernel_initializer=init_fn)(dec1)
		dec2 = tf.keras.layers.BatchNormalization()(dec2)
		dec2 = tf.keras.layers.LeakyReLU()(dec2)

		dec3 = tf.keras.layers.Dense(64, kernel_initializer=init_fn)(dec2)
		dec3 = tf.keras.layers.BatchNormalization()(dec3)
		dec3 = tf.keras.layers.LeakyReLU()(dec3)


		dense = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(dec3)


		enc = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(dense)

		encoded = tf.keras.Input(shape=(self.latent_dims,))


		dec0 = tf.keras.layers.Dense(64, kernel_initializer=init_fn, use_bias = True)(encoded)
		dec0 = tf.keras.layers.BatchNormalization()(dec0)
		dec0 = tf.keras.layers.LeakyReLU()(dec0)

		dec1 = tf.keras.layers.Dense(128, kernel_initializer=init_fn)(dec0)
		dec1 = tf.keras.layers.BatchNormalization()(dec1)
		dec1 = tf.keras.layers.LeakyReLU()(dec1)

		dec2 = tf.keras.layers.Dense(256, kernel_initializer=init_fn)(dec1)
		dec2 = tf.keras.layers.BatchNormalization()(dec2)
		dec2 = tf.keras.layers.LeakyReLU()(dec2)

		dec3 = tf.keras.layers.Dense(512, kernel_initializer=init_fn)(dec2)
		dec3 = tf.keras.layers.BatchNormalization()(dec3)
		dec3 = tf.keras.layers.LeakyReLU()(dec3)

		out_enc = tf.keras.layers.Dense(int(self.output_size*self.output_size), kernel_initializer=init_fn)(dec3)
		out_enc = tf.keras.layers.Activation( activation = 'tanh')(out_enc)
		out = tf.keras.layers.Reshape([int(self.output_size),int(self.output_size),1])(out_enc)

		self.Decoder = tf.keras.Model(inputs=encoded, outputs=out)
		self.Encoder = tf.keras.Model(inputs=inputs, outputs=enc)



	def discriminator_model_mnist(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)
		model = tf.keras.Sequential()
		model.add(layers.Dense(256, use_bias=False, input_shape=(self.latent_dims,), kernel_initializer=init_fn))
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(512, use_bias=False, kernel_initializer=init_fn))
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(128, use_bias=False, kernel_initializer=init_fn))
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(1))
		return model

	def save_images_FID(self):
		def data_preprocess(image):
			with tf.device('/CPU'):
				image = tf.image.grayscale_to_rgb(image)
			return image


		if self.FID_load_flag == 0:
			### First time FID call setup
			self.FID_load_flag = 1	
			random_points = tf.keras.backend.random_uniform([self.FID_num_samples], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
			print(random_points)
			if self.FID_num_samples <50000:
				self.fid_train_images = self.fid_train_images[random_points]

			## self.fid_train_images has the names to be read. Make a dataset with it
			self.fid_image_dataset = tf.data.Dataset.from_tensor_slices(self.fid_train_images)
			self.fid_image_dataset = self.fid_image_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset = self.fid_image_dataset.batch(self.FID_num_samples)

		with tf.device(self.device):
			for image_batch in self.fid_image_dataset:
				for i in range(self.FID_num_samples):
					real = image_batch[i,:,:,:]
					tf.keras.preprocessing.image.save_img(self.FIDRealspath+str(i)+'.png', real,  scale=True)
			for i in range(self.FID_num_samples):	
				preds = self.Decoder(self.get_noise(2), training=False)
				preds = tf.image.grayscale_to_rgb(preds)
				preds = preds.numpy()
				fake = preds[0,:,:,:]
				tf.keras.preprocessing.image.save_img(self.FIDFakespath+str(i)+'.png', fake,  scale=True)
		return



	def MNIST_Classifier(self):
		self.FID_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(299,299,3), classes=1000)

	def FID_mnist(self):
		def data_preprocess(image):
			with tf.device('/CPU'):
				image = tf.image.resize(image,[299,299])
				image = tf.image.grayscale_to_rgb(image)
			return image


		if self.FID_load_flag == 0:
			### First time FID call setup
			self.FID_load_flag = 1	

			random_points = tf.keras.backend.random_uniform([self.FID_num_samples], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
			print(random_points)
			self.fid_train_images = self.fid_train_images[random_points]
			self.fid_image_dataset = tf.data.Dataset.from_tensor_slices(self.fid_train_images)
			if self.FID_kind != 'latent':
				self.fid_image_dataset = self.fid_image_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset = self.fid_image_dataset.batch(self.fid_batch_size)
			self.MNIST_Classifier()


		if self.mode == 'fid':
			print(self.checkpoint_dir)
			self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			print('Models Loaded Successfully')

		with tf.device(self.device):
			for image_batch in self.fid_image_dataset:
				if self.FID_kind == 'latent':
					## Measure FID on the latent Gaussians 
					preds = self.Encoder(image_batch)
					act1 = self.get_noise(tf.constant(100))
					act2 = preds
				else:
					preds = self.Decoder(self.get_noise(tf.constant(100)), training=False)
					preds = tf.image.resize(preds, [299,299])
					preds = tf.image.grayscale_to_rgb(preds)
					preds = preds.numpy()

					act1 = self.FID_model.predict(image_batch)
					act2 = self.FID_model.predict(preds)
					
				try:
					self.act1 = np.concatenate([self.act1,act1], axis = 0)
					self.act2 = np.concatenate([self.act2,act2], axis = 0)
				except:
					self.act1 = act1
					self.act2 = act2
			self.eval_FID()
			return

