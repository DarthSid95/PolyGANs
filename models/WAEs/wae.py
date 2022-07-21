from __future__ import print_function
import os, sys, time, argparse
import numpy as np
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
import tensorflow_probability as tfp
from matplotlib.backends.backend_pgf import PdfPages
from itertools import product as cart_prod

import math
import tensorflow as tf
from absl import app
from absl import flags
from scipy.interpolate import interp1d

from gan_topics import *


'''***********************************************************************************
********** WAEFeR ********************************************************************
***********************************************************************************'''
### self.gan = WAE and self.topic = PolyGAN and self.loss = 'RBF'
class WAE_PolyGAN(GAN_WAE, RBFSolver):

	def __init__(self,FLAGS_dict):

		GAN_WAE.__init__(self,FLAGS_dict)

		''' Set up the RBF Network common to PolyGAN and PolyGAN-WAE'''
		RBFSolver.__init__(self)

		self.first_iteration_flag = 1
	

		if self.colab and (self.data in ['mnist', 'celeba', 'cifar10', 'svhn', 'church', 'bedroom']):
			self.bar_flag = 0
		else:
			self.bar_flag = 1

		self.lambda_PGP = 100.


	def create_models(self):
		with tf.device(self.device):
			self.total_count = tf.Variable(0,dtype='int64')
			eval(self.encdec_model)
			self.discriminator_RBF = self.discriminator_model_RBF()
			

			print("Model Successfully made")

			print("\n\n ENCODER MODEL: \n\n")
			print(self.Encoder.summary())
			print("\n\n DECODER MODEL: \n\n")
			print(self.Decoder.summary())
			print("\n\n DISCRIMINATOR RBF: \n\n")
			print(self.discriminator_RBF.summary())


			if self.res_flag == 1:
				with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
					# Pass the file handle in as a lambda function to make it callable
					fh.write("\n\n ENCODER MODEL: \n\n")
					self.Encoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DECODER MODEL: \n\n")
					self.Decoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DISCRIMINATOR RBF: \n\n")
					self.discriminator_RBF.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
		return

	def create_optimizer(self):

		with tf.device(self.device):
			lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=100000, decay_rate=0.95, staircase=True)
			#### Added for IMCL rebuttal
			self.E_optimizer = tf.keras.optimizers.Adam(self.lr_AE_Enc)
			self.D_optimizer = tf.keras.optimizers.Adam(self.lr_AE_Dec)
			self.G_optimizer = tf.keras.optimizers.Nadam(self.lr_G)
			# self.Disc_optimizer = tf.keras.optimizers.Adam(self.lr_D)
			if self.data in ['mnist']:
				decay_steps = 5000
				decay_rate = 0.95
			elif self.data in ['cifar10']:
				decay_steps = 5000
				decay_rate = 0.98
			elif self.data in ['celeba', 'church', 'bedroom']:
				decay_steps = 20000
				decay_rate = 0.99



			if self.loss == 'RBF':

				self.Enc_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_AE_Enc, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
				self.Dec_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_AE_Dec, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
				self.G_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=5000, decay_rate=0.98, staircase=True)
				### Added for IMCL rebuttal
				if self.data == 'cifarrr10':
					self.E_optimizer = tf.keras.optimizers.Adam(self.Enc_lr_schedule)
					self.D_optimizer = tf.keras.optimizers.Adam(self.Dec_lr_schedule)
					self.G_optimizer = tf.keras.optimizers.Adam(self.G_lr_schedule)	
				else:
					self.E_optimizer = tf.keras.optimizers.Adam(self.lr_AE_Enc)
					self.D_optimizer = tf.keras.optimizers.Adam(self.lr_AE_Dec)
					self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G)			
		print("Optimizers Successfully made")	
		return	

	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(E_optimizer = self.E_optimizer,
								 D_optimizer = self.D_optimizer,
								 G_optimizer = self.G_optimizer,
								 Encoder = self.Encoder,
								 Decoder = self.Decoder,
								 discriminator_RBF = self.discriminator_RBF, \
								 locs = self.locs,
								 total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
				print("Model restored...")
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.Encoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Encoder.h5')
					self.Decoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Decoder.h5')
					# self.discriminator_RBF = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator_RBF.h5')
					print("Model restored...")
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
			


	def save_epoch_h5models(self):
		self.Encoder.save(self.checkpoint_dir + '/model_Encoder.h5', overwrite = True)
		self.Decoder.save(self.checkpoint_dir + '/model_Decoder.h5', overwrite = True)
		self.discriminator_RBF.save(self.checkpoint_dir + '/model_discriminator_RBF.h5', overwrite = True)
		return


	def pretrain_step_GAN(self,reals_all):
		self.AE_loss = tf.constant(0.)
		## Actually Pretrain GAN. - Will make a sperate flag nd control if it does infact work out
		with tf.device(self.device):
			self.fakes_enc = target_noise = self.get_noise(self.batch_size)
			self.reals = reals_all

		with tf.GradientTape() as G_tape:

			self.reals_enc = self.Encoder(self.reals, training = True)
			# print(self.reals_enc.numpy())

			self.real_output,self.lambda_x_terms_1 = self.discriminator_RBF(target_noise,training = True)
			self.fake_output,self.lambda_x_terms_2 = self.discriminator_RBF(self.reals_enc,training=True)

			with G_tape.stop_recording():
				Centres, Weights, Lamb_Weights = self.find_rbf_centres_weights()
				self.discriminator_RBF.set_weights([Centres,Weights,Centres,Lamb_Weights])

				if self.first_iteration_flag:
					self.first_iteration_flag = 0 
					self.lamb = tf.constant(0.1)
					self.D_loss = self.G_loss = tf.constant(0.)
					return

				self.find_lambda()

			self.divide_by_lambda()

			eval(self.loss_func)

			self.G_grads = G_tape.gradient(self.G_loss, self.Encoder.trainable_variables)
			self.G_optimizer.apply_gradients(zip(self.G_grads, self.Encoder.trainable_variables))


	def pretrain_step_AE(self,reals_all):
		with tf.device(self.device):
			self.reals = reals_all

		with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:

			self.reals_enc = self.Encoder(self.reals, training = True)
			self.reals_dec = self.Decoder(self.reals_enc, training = True)

			self.loss_AE()
			self.E_grads = enc_tape.gradient(self.AE_loss, self.Encoder.trainable_variables)
			self.D_grads = dec_tape.gradient(self.AE_loss, self.Decoder.trainable_variables)
			self.E_optimizer.apply_gradients(zip(self.E_grads, self.Encoder.trainable_variables))
			self.D_optimizer.apply_gradients(zip(self.D_grads, self.Decoder.trainable_variables))

			self.D_loss = self.G_loss = tf.constant(0.)


	def train_step(self,reals_all):
		with tf.device(self.device):
			self.fakes_enc = target_noise = self.get_noise(self.batch_size)
			self.reals = reals_all

		with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as G_tape:

			self.reals_enc = self.Encoder(self.reals, training = True)
			self.reals_dec = self.Decoder(self.reals_enc, training = True)

			self.loss_AE()
			self.E_grads = enc_tape.gradient(self.AE_loss, self.Encoder.trainable_variables)
			self.D_grads = dec_tape.gradient(self.AE_loss, self.Decoder.trainable_variables)
			self.E_optimizer.apply_gradients(zip(self.E_grads, self.Encoder.trainable_variables))
			self.D_optimizer.apply_gradients(zip(self.D_grads, self.Decoder.trainable_variables))


			self.real_output,self.lambda_x_terms_1 = self.discriminator_RBF(target_noise, training = True)
			self.fake_output,self.lambda_x_terms_2 = self.discriminator_RBF(self.reals_enc,training =True)
						
			if self.first_iteration_flag:
				self.first_iteration_flag = 0 
				self.lamb = tf.constant(0.1)
				self.D_loss = self.G_loss = tf.constant(0)
				Centres, Weights, Lamb_Weights = self.find_rbf_centres_weights()
				self.discriminator_RBF.set_weights([Centres,Weights,Centres,Lamb_Weights])
				return

			self.find_lambda()
			self.divide_by_lambda()
			eval(self.loss_func)

			self.G_grads = G_tape.gradient(self.G_loss, self.Encoder.trainable_variables)
			self.G_optimizer.apply_gradients(zip(self.G_grads, self.Encoder.trainable_variables))

			Centres, Weights, Lamb_Weights = self.find_rbf_centres_weights()
			self.discriminator_RBF.set_weights([Centres,Weights,Centres,Lamb_Weights])

					

	def loss_RBF(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output) 
		self.D_loss = 1 * (-loss_real + loss_fake)

		if (2*self.rbf_m - self.rbf_n) >= 0:
			self.G_loss = loss_real - loss_fake
		elif (2*self.rbf_m - self.rbf_n) < 0:
			self.G_loss = -1*(loss_real - loss_fake)



	def loss_AE(self):
		loss_AE_reals = tf.reduce_mean(tf.abs(self.reals - self.reals_dec))
		self.AE_loss =  loss_AE_reals 



'''***********************************************************************************
********** WAE ********************************************************************
***********************************************************************************'''
### self.gan = WAE and self.topic = WAEMMD and self.loss = 'RBFG', 'IMQ', 'SW', 'CW'
class WAE_WAEMMD(GAN_WAE):

	def __init__(self,FLAGS_dict):

		GAN_WAE.__init__(self,FLAGS_dict)

		if self.colab and self.data in ['mnist', 'celeba','cifar10']:
			self.bar_flag = 0
		else:
			self.bar_flag = 1

		self.GAN_pretrain_epochs = 0
		self.AE_pretrain_epochs = 0

		self.D_loss = tf.constant(0.)

		if self.loss == 'CW':
			if self.data in ['mnist', 'cifar10', 'svhn']:
				self.lambda_CW = 1.0
			elif self.data in ['fmnist']:
				self.lambda_CW = 10.0
			elif self.data in ['celeba', 'ukiyoe', 'church']:
				self.lambda_CW = 5.0 

		if self.loss == 'SW':
			if self.data in ['mnist',]:
				self.lambda_SW = 0.1
				L=500
			elif self.data in ['celeba', 'ukiyoe']: 
				self.lambda_SW = 0.1
				L=500 
			elif self.data in ['church']:
				self.lambda_SW = 0.1 
				L=500
			elif self.data in ['cifar10']:
				self.lambda_SW = 0.1
				L=500

			self.theta=[w/np.sqrt((w**2).sum()) for w in np.random.normal(size=(L,self.latent_dims))]

	
	def create_models(self):
		with tf.device(self.device):
			self.total_count = tf.Variable(0,dtype='int64')
			eval(self.encdec_model)
			self.discriminator = eval(self.disc_model)

			print("Model Successfully made")

			print("\n\n ENCODER MODEL: \n\n")
			print(self.Encoder.summary())
			print("\n\n DECODER MODEL: \n\n")
			print(self.Decoder.summary())
			# print("\n\n DISCRIMINATOR MODEL: \n\n")
			# print(self.discriminator.summary())

			if self.res_flag == 1:
				with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
					# Pass the file handle in as a lambda function to make it callable
					fh.write("\n\n ENCODER MODEL: \n\n")
					self.Encoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DECODER MODEL: \n\n")
					self.Decoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					# fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
					# self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
		return


	def create_optimizer(self):
		with tf.device(self.device):
			lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=1000000, decay_rate=0.95, staircase=True)

			if self.loss == 'SW':
				optimizer = tf.keras.optimizers.Adam
				self.E_optimizer = optimizer(self.lr_AE_Enc)
				self.D_optimizer = optimizer(self.lr_AE_Dec)
				self.G_optimizer = optimizer(self.lr_G,0.5,0.9)
			else:
				optimizer = tf.keras.optimizers.Adam
				self.E_optimizer = optimizer(self.lr_AE_Enc)
				self.D_optimizer = optimizer(self.lr_AE_Dec)
				self.G_optimizer = optimizer(self.lr_G)

			print("Optimizers Successfully made")
		return

	def create_load_checkpoint(self):
		self.checkpoint = tf.train.Checkpoint(E_optimizer = self.E_optimizer,
								 D_optimizer = self.D_optimizer,
								 Encoder = self.Encoder,
								 Decoder = self.Decoder,
								 total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.Encoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Encoder.h5')
					self.Decoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Decoder.h5')
					# self.discriminator = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
			

	def save_epoch_h5models(self):
		self.Encoder.save(self.checkpoint_dir + '/model_Encoder.h5', overwrite = True)
		self.Decoder.save(self.checkpoint_dir + '/model_Decoder.h5', overwrite = True)
		return

	def train_step(self,reals_all):
		with tf.device(self.device):
			self.fakes_enc = target_noise = self.get_noise(self.batch_size)
			self.reals = reals_all

		with tf.GradientTape() as dec_tape, tf.GradientTape() as G_tape:

			self.reals_enc = self.Encoder(self.reals, training = True)
			self.reals_dec = self.Decoder(self.reals_enc, training = True)

			self.loss_AE()
			eval(self.loss_func)

			self.D_grads = dec_tape.gradient(self.AE_loss, self.Decoder.trainable_variables)
			self.G_grads = G_tape.gradient(self.G_loss, self.Encoder.trainable_variables)
			self.D_optimizer.apply_gradients(zip(self.D_grads, self.Decoder.trainable_variables))
			self.G_optimizer.apply_gradients(zip(self.G_grads, self.Encoder.trainable_variables))


	def loss_RBFG(self):
		#### Code Courtest: https://github.com/siddharth-agrawal/Generative-Moment-Matching-Networks

		def makeScaleMatrix(num_gen, num_orig):

			# first 'N' entries have '1/N', next 'M' entries have '-1/M'
			s1 =  tf.constant(1.0 / num_gen, shape = [self.batch_size, 1])
			s2 = -tf.constant(1.0 / num_orig, shape = [self.batch_size, 1])
			return tf.concat([s1, s2], axis = 0)

		sigma = [1,2,5,10,20,50]

		X = tf.concat([self.fakes_enc, self.reals_enc], axis = 0)
		# print(X)

		# dot product between all combinations of rows in 'X'
		XX = tf.matmul(X, tf.transpose(X))

		# dot product of rows with themselves
		X2 = tf.reduce_sum(X * X, 1, keepdims = True)

		# exponent entries of the RBF kernel (without the sigma) for each
		# combination of the rows in 'X'
		# -0.5 * (x^Tx - 2*x^Ty + y^Ty)
		exponent = XX - 0.5 * X2 - 0.5 * tf.transpose(X2)
		# print(exponent)

		# scaling constants for each of the rows in 'X'
		s = makeScaleMatrix(tf.cast(self.batch_size,'float32'), tf.cast(self.batch_size,'float32'))

		# scaling factors of each of the kernel values, corresponding to the
		# exponent values
		S = tf.matmul(s, tf.transpose(s))

		loss = 0

		# for each bandwidth parameter, compute the MMD value and add them all
		for i in range(len(sigma)):

			# kernel values for each combination of the rows in 'X' 
			kernel_val = tf.exp((1.0 / sigma[i]) * exponent)
			loss += tf.reduce_sum(S * kernel_val)
			# print(kernel_val)
			# print(loss)

		self.G_loss = tf.sqrt(loss)
		self.D_loss = tf.constant(0.0)

		return 

	#################################################################

	def loss_CW(self):
		###3 Code Courtesy https://github.com/gmum/cwae
		def euclidean_norm_squared(X, axis=None):
			return tf.reduce_sum(tf.square(X), axis=axis)
		
		D = tf.cast(self.latent_dims, tf.float32)#tf.cast(tf.shape(self.reals_enc)[1], tf.float32)
		N = tf.cast(self.batch_size, tf.float32)#tf.cast(tf.shape(self.reals_enc)[0], tf.float32)
		y = tf.pow(4/(3*N), 0.4)

		K = 1/(2*D-3)

		A1 = euclidean_norm_squared(tf.subtract(tf.expand_dims(self.reals_enc, 0), tf.expand_dims(self.reals_enc, 1)), axis=2)
		A = (1/(N**2)) * tf.reduce_sum((1/tf.sqrt(y + K*A1)))

		B1 = euclidean_norm_squared(self.reals_enc, axis=1)
		B = (2/N)*tf.reduce_sum((1/tf.sqrt(y + 0.5 + K*B1)))

		tensor_cw_distance = (1/tf.sqrt(1+y)) + A - B

		self.G_loss = self.AE_loss + self.lambda_CW*tf.math.log(tensor_cw_distance)
		return 


	#################################################################

	def loss_IMQ(self):
		###3 Code Courtesy https://github.com/hiwonjoon/wae-wgan/blob/master/wae_mmd.py

		n = tf.cast(self.batch_size,tf.float32)
		C_base = 2.*self.fakes_enc.shape[1]

		z = self.fakes_enc
		z_tilde = self.reals_enc

		z_dot_z = tf.matmul(z,z,transpose_b=True) #[B,B} matrix where its (i,j) element is z_i \dot z_j.
		z_tilde_dot_z_tilde = tf.matmul(z_tilde,z_tilde,transpose_b=True)
		z_dot_z_tilde = tf.matmul(z,z_tilde,transpose_b=True)

		dist_z_z = \
			(tf.expand_dims(tf.linalg.diag_part(z_dot_z),axis=1)\
				+ tf.expand_dims(tf.linalg.diag_part(z_dot_z),axis=0))\
			- 2*z_dot_z
		dist_z_tilde_z_tilde = \
			(tf.expand_dims(tf.linalg.diag_part(z_tilde_dot_z_tilde),axis=1)\
				+ tf.expand_dims(tf.linalg.diag_part(z_tilde_dot_z_tilde),axis=0))\
			- 2*z_tilde_dot_z_tilde
		dist_z_z_tilde = \
			(tf.expand_dims(tf.linalg.diag_part(z_dot_z),axis=1)\
				+ tf.expand_dims(tf.linalg.diag_part(z_tilde_dot_z_tilde),axis=0))\
			- 2*z_dot_z_tilde

		L_D = 0.
		#with tf.control_dependencies([
		#    tf.assert_non_negative(dist_z_z),
		#    tf.assert_non_negative(dist_z_tilde_z_tilde),
		#    tf.assert_non_negative(dist_z_z_tilde)]):

		for scale in [1.0]:
			C = tf.cast(C_base*scale,tf.float32)

			k_z = C / (C + dist_z_z + 1e-8)
			k_z_tilde = C / (C + dist_z_tilde_z_tilde + 1e-8)
			k_z_z_tilde = C / (C + dist_z_z_tilde + 1e-8)

			loss = 1/(n*(n-1))*tf.reduce_sum(k_z)\
				+ 1/(n*(n-1))*tf.reduce_sum(k_z_tilde)\
				- 2/(n*n)*tf.reduce_sum(k_z_z_tilde)

			L_D += loss

		self.G_loss = self.AE_loss + L_D
		self.D_loss = tf.constant(0.0)

	#####################################################################

	def loss_SW(self):
		##### Code Courtesy https://github.com/skolouri/swae

		# Let projae be the projection of the encoded samples
		# projae = tf.dot(self.reals_enc,tf.transpose(theta))
		# projae = tf.linalg.matmul(self.reals_enc, self.theta, transpose_a=False, transpose_b=True)
		projae = tf.keras.backend.dot(tf.cast(self.reals_enc,'float32'), tf.transpose(tf.cast(self.theta,'float32')))

		# Let projz be the projection of the $q_Z$ samples
		# projz = tf.dot(z,tf.transpose(theta))
		# projz = tf.linalg.matmul(self.fakes_enc, self.theta, transpose_a=False, transpose_b=True)
		projz = tf.keras.backend.dot(tf.cast(self.fakes_enc,'float32'), tf.transpose(tf.cast(self.theta,'float32')))

		# Calculate the Sliced Wasserstein distance by sorting 
		# the projections and calculating the L2 distance between
		W2=(tf.nn.top_k(tf.transpose(projae),k=tf.cast(self.batch_size, 'int32')).values-tf.nn.top_k(tf.transpose(projz),k=tf.cast(self.batch_size, 'int32')).values)**2

		SWLoss = self.lambda_SW * tf.reduce_mean(W2)

		bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

		bce_loss = bce(self.reals_dec, self.reals)

		self.G_loss = self.AE_loss + bce_loss + SWLoss


	#####################################################################

	def loss_AE(self):
		loss_AE_reals = tf.reduce_mean(tf.abs(self.reals - self.reals_dec))
		self.AE_loss =  loss_AE_reals 



