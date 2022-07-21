from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class ARCH_mnist():

	def __init__(self):
		print("Creating MNIST architectures for base cases ")
		return


	def generator_model_dense_mnist(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.noise_dims,)) #100x1

		dec0 = tf.keras.layers.Dense(64, kernel_initializer=init_fn, use_bias = True)(inputs)
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

		model = tf.keras.Model(inputs = inputs, outputs = out)

		return model

	
	def discriminator_model_dense_mnist(self):
		init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn, input_shape=[int(self.output_size), int(self.output_size), 1]))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())
		
		# model.add(layers.Dropout(0.3))

		model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn, input_shape=[int(self.output_size), int(self.output_size), 1]))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())
		

		model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())
		

		model.add(layers.Flatten())
		
		model.add(layers.Dense(50))
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(1))
		return model


	def generator_model_dcgan_mnist(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(int(self.output_size/4)*int(self.output_size/4)*256, use_bias=False, input_shape=(self.noise_dims,),kernel_initializer=init_fn))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Reshape((int(self.output_size/4), int(self.output_size/4), 256)))
		assert model.output_shape == (None, int(self.output_size/4), int(self.output_size/4), 256) # Note: None is the batch size

		model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=init_fn))
		assert model.output_shape == (None, int(self.output_size/4), int(self.output_size/4), 128)
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn))
		assert model.output_shape == (None, int(self.output_size/2), int(self.output_size/2), 64)
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2DTranspose(32, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=init_fn))
		assert model.output_shape == (None, int(self.output_size/2), int(self.output_size/2), 32)
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn,))
		assert model.output_shape == (None, int(self.output_size), int(self.output_size), 1)
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=init_fn))
		assert model.output_shape == (None, int(self.output_size), int(self.output_size), 1)
		model.add(layers.Activation( activation = 'tanh'))

		return model

	def discriminator_model_dcgan_mnist(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn, input_shape=[int(self.output_size), int(self.output_size), 1]))
		model.add(layers.LeakyReLU())
		model.add(layers.BatchNormalization())

		model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn, input_shape=[int(self.output_size), int(self.output_size), 1]))
		model.add(layers.LeakyReLU())
		model.add(layers.BatchNormalization())

		model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn))
		model.add(layers.LeakyReLU())
		model.add(layers.BatchNormalization())

		model.add(layers.Flatten())
		
		model.add(layers.Dense(50))
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(1))

		return model



	def MNIST_Classifier(self):
		self.FID_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(299,299,3), classes=1000)

	def FID_mnist(self):

		def data_preprocess(image):
			with tf.device('/CPU'):
				image = tf.image.resize(image,[299,299]) ### USed to be 80 during RumiGAN days. Now 299
				image = tf.image.grayscale_to_rgb(image)
			return image

		if self.FID_load_flag == 0:
			### First time FID call setup
			self.FID_load_flag = 1
			random_points = tf.keras.backend.random_uniform([min(self.fid_train_images.shape[0],self.FID_num_samples)], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
			print(random_points)
			self.fid_train_images = self.fid_train_images[random_points]
			self.fid_image_dataset = tf.data.Dataset.from_tensor_slices(self.fid_train_images)
			self.fid_image_dataset = self.fid_image_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset = self.fid_image_dataset.batch(self.fid_batch_size)

			self.MNIST_Classifier()


		if self.mode == 'fid':
			print(self.checkpoint_dir)
			self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			print('Models Loaded Successfully')

		with tf.device(self.device):
			for image_batch in self.fid_image_dataset:
				noise = self.get_noise([self.fid_batch_size, self.noise_dims])
				preds = self.generator(noise, training=False)
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

