from __future__ import print_function
import os, sys, time, argparse
import numpy as np
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
import tensorflow_probability as tfp
from matplotlib.backends.backend_pgf import PdfPages

import math
import tensorflow as tf
from absl import app
from absl import flags

from gan_topics import *

'''***********************************************************************************
********** Baseline WGANs ************************************************************
***********************************************************************************'''
### self.gan = 'WGAN' and self.topic = 'Base' and self.loss = 'GP', 'LP', 'ALP', 'R1', 'R2'.
class WGAN_Base(GAN_Base):

	def __init__(self,FLAGS_dict):

		GAN_Base.__init__(self,FLAGS_dict)

		self.lambda_GP = 0.1 
		self.lambda_ALP = 10.0 
		self.lambda_LP = 0.1 
		self.lambda_R1 = 0.5
		self.lambda_R2 = 0.5

	#################################################################

	def create_optimizer(self):
		with tf.device(self.device):
			if  self.loss == 'ALP' :
				self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=100, decay_rate=0.9, staircase=True)
				self.G_optimizer = tf.keras.optimizers.Adam(self.lr_schedule, self.beta1, self.beta2)
			else:
				self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2)
			self.Disc_optimizer = tf.keras.optimizers.Adam(self.lr_D, self.beta1, self.beta2)
			print("Optimizers Successfully made")	
		return	

	#################################################################

	def save_epoch_h5models(self):
		self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
		self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)
		return

	#################################################################

	def train_step(self,reals_all):
		for i in tf.range(self.Dloop):
			with tf.device(self.device):
				noise = self.get_noise([self.batch_size, self.noise_dims])
			self.reals = reals_all

			with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
				
				self.fakes = self.generator(noise, training=True)

				self.real_output = self.discriminator(self.reals, training = True)
				self.fake_output = self.discriminator(self.fakes, training = True)
				eval(self.loss_func)

			self.D_grads = disc_tape.gradient(self.D_loss, self.discriminator.trainable_variables)
			self.Disc_optimizer.apply_gradients(zip(self.D_grads, self.discriminator.trainable_variables))

			if self.loss == 'base':
				wt = []
				for w in self.discriminator.get_weights():
					w = tf.clip_by_value(w, -0.1,0.1) #0.01 for [0,1] data, 0.1 for [0,10]
					wt.append(w)
				self.discriminator.set_weights(wt)
			if i >= (self.Dloop - self.Gloop):
				self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
				self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))


	#################################################################

	def loss_base(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output) 

		self.D_loss = 1 * (-loss_real + loss_fake)
		self.G_loss = 1 * (loss_real - loss_fake)

	#################################################################

	def loss_GP(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.gradient_penalty()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_GP * self.gp 
		self.G_loss = 1 * (loss_real - loss_fake)

	def gradient_penalty(self):
		if self.data in ['g2', 'gmm8', 'gN']:
			alpha = tf.random.uniform([self.batch_size, 1], 0., 1.)
		else:
			alpha = tf.random.uniform([self.batch_size, 1, 1, 1], 0., 1.)
		diff = tf.cast(self.fakes,dtype='float32') - tf.cast(self.reals,dtype='float32')
		inter = tf.cast(self.reals,dtype='float32') + (alpha * diff)
		with tf.GradientTape() as t:
			t.watch(inter)
			pred = self.discriminator(inter, training = True)
		grad = t.gradient(pred, [inter])[0]
		if self.data in ['g2', 'gmm8', 'gN']:
			slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
		else:
			slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
		self.gp = tf.reduce_mean((slopes - 1.)**2)
		return 


	#################################################################

	def loss_R1(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.gradient_penalty_R1()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_R1 * self.gp 
		self.G_loss = 1 * (loss_real - loss_fake)

	def gradient_penalty_R1(self):
		# Code Courtesy of Paper Author dterjek's GitHub
		# URL : https://github.com/dterjek/adversarial_lipschitz_regularization/blob/master/wgan_alp.py
		if self.data in ['g2', 'gmm8', 'gN']:
			alpha = tf.random.uniform([self.batch_size, 1], 0., 1.)
		else:
			alpha = tf.random.uniform([self.batch_size, 1, 1, 1], 0., 1.)
		inter = tf.cast(self.reals,dtype='float32')
		with tf.GradientTape() as t:
			t.watch(inter)
			pred = self.discriminator(inter, training = True)
		grad = t.gradient(pred, [inter])[0]
		if self.data in ['g2', 'gmm8', 'gN']:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1])
		else:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1, 2, 3])
		self.gp = tf.reduce_mean(slopes)
		return 

	#################################################################

	def loss_R2(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.gradient_penalty_R2()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_R2 * self.gp 
		self.G_loss = 1 * (loss_real - loss_fake)

	def gradient_penalty_R2(self):
		# Code Courtesy of Paper Author dterjek's GitHub
		# URL : https://github.com/dterjek/adversarial_lipschitz_regularization/blob/master/wgan_alp.py
		if self.data in ['g2', 'gmm8', 'gN']:
			alpha = tf.random.uniform([self.batch_size, 1], 0., 1.)
		else:
			alpha = tf.random.uniform([self.batch_size, 1, 1, 1], 0., 1.)
		inter = tf.cast(self.fakes,dtype='float32')
		with tf.GradientTape() as t:
			t.watch(inter)
			pred = self.discriminator(inter, training = True)
		grad = t.gradient(pred, [inter])[0]
		if self.data in ['g2', 'gmm8', 'gN']:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1])
		else:
			slopes = tf.reduce_sum(tf.square(grad), axis=[1, 2, 3])
		self.gp = tf.reduce_mean(slopes)
		return 

	#################################################################

	def loss_LP(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.lipschitz_penalty()

		self.D_loss = -loss_real + loss_fake + self.lambda_LP * self.lp 
		self.G_loss = loss_real - loss_fake

	def lipschitz_penalty(self):
		# Code Courtesy of Paper Author dterjek's GitHub
		# URL : https://github.com/dterjek/adversarial_lipschitz_regularization/blob/master/wgan_alp.py

		self.K = 1
		self.p = 2

		if self.data in ['g2', 'gmm8', 'gN']:
			epsilon = tf.random.uniform([tf.shape(self.reals)[0], 1], 0.0, 1.0)
		else:
			epsilon = tf.random.uniform([tf.shape(self.reals)[0], 1, 1, 1], 0.0, 1.0)
		x_hat = epsilon * self.fakes + (1 - epsilon) * self.reals

		with tf.GradientTape() as t:
			t.watch(x_hat)
			D_vals = self.discriminator(x_hat, training = False)
		grad_vals = t.gradient(D_vals, [x_hat])[0]

		#### args.p taken from github as default p=2
		dual_p = 1 / (1 - 1 / self.p) if self.p != 1 else np.inf

		grad_norms = tf.norm(grad_vals, ord=dual_p, axis=1, keepdims=True)
		self.lp = tf.reduce_mean(tf.maximum(grad_norms - self.K, 0)**2)

	#################################################################

	def loss_ALP(self):
		
		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.adversarial_lipschitz_penalty()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_ALP * self.alp 
		self.G_loss = 1 * (loss_real - loss_fake)


	def adversarial_lipschitz_penalty(self):
		def normalize(x, ord):
			return x / tf.maximum(tf.norm(x, ord=ord, axis=1, keepdims=True), 1e-10)
		# Code Courtesy of Paper Author dterjek's GitHub
		# URL : https://github.com/dterjek/adversarial_lipschitz_regularization/blob/master/wgan_alp.py
		self.eps_min = 0.1
		self.eps_max = 10.0
		self.xi = 10.0
		self.ip = 1
		self.p = 2
		self.K = 5 #was 1. made 5 for G2 compares

		samples = tf.concat([self.reals, self.fakes], axis=0)
		if self.data in ['g2', 'gmm8', 'gN']:
			noise = tf.random.uniform([tf.shape(samples)[0], 1], 0, 1)
		else:
			noise = tf.random.uniform([tf.shape(samples)[0], 1, 1, 1], 0, 1)

		eps = self.eps_min + (self.eps_max - self.eps_min) * noise

		with tf.GradientTape(persistent = True) as t:
			t.watch(samples)
			validity = self.discriminator(samples, training = False)

			d = tf.random.uniform(tf.shape(samples), 0, 1) - 0.5
			d = normalize(d, ord=2)
			t.watch(d)
			for _ in range(self.ip):
				samples_hat = tf.clip_by_value(samples + self.xi * d, clip_value_min=-1, clip_value_max=1)
				validity_hat = self.discriminator(samples_hat, training = False)
				dist = tf.reduce_mean(tf.abs(validity - validity_hat))
				grad = t.gradient(dist, [d])[0]
				# print(grad)
				d = normalize(grad, ord=2)
			r_adv = d * eps

		samples_hat = tf.clip_by_value(samples + r_adv, clip_value_min=-1, clip_value_max=1)

		d_lp                   = lambda x, x_hat: tf.norm(x - x_hat, ord=self.p, axis=1, keepdims=True)
		d_x                    = d_lp

		samples_diff = d_x(samples, samples_hat)
		samples_diff = tf.maximum(samples_diff, 1e-10)

		validity      = self.discriminator(samples    , training = False)
		validity_hat  = self.discriminator(samples_hat, training = False)
		validity_diff = tf.abs(validity - validity_hat)

		alp = tf.maximum(validity_diff / samples_diff - self.K, 0)
		# alp = tf.abs(validity_diff / samples_diff - args.K)

		nonzeros = tf.greater(alp, 0)
		count = tf.reduce_sum(tf.cast(nonzeros, tf.float32))

		self.alp = tf.reduce_mean(alp**2)
		# alp_loss = args.lambda_lp * reduce_fn(alp ** 2)

	#####################################################################

	def loss_AE(self):
		mse = tf.keras.losses.MeanSquaredError()
		loss_AE_reals = tf.reduce_mean(tf.abs(self.reals - self.reals_dec))#mse(self.reals, self.reals_dec)
		self.AE_loss =  loss_AE_reals  


'''***********************************************************************************
********** Baseline GMMN ************************************************************
***********************************************************************************'''
### self.gan = 'WGAN' and self.topic = 'GMMN' and self.loss = 'RBFG', 'IMQ'
class WGAN_GMMN(GAN_Base):

	def __init__(self,FLAGS_dict):
		GAN_Base.__init__(self,FLAGS_dict)

	#################################################################

	def create_optimizer(self):
		with tf.device(self.device):
			self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2)
			print("Optimizers Successfully made")	
		return	

	#################################################################

	def save_epoch_h5models(self):
		self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
		return

	#################################################################

	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer, \
							generator = self.generator, \
							total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
		return

	#################################################################

	def train_step(self,reals_all):

		with tf.device(self.device):
			noise = self.get_noise([self.batch_size, self.noise_dims])
		self.reals = reals_all

		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			
			self.fakes = self.generator(noise, training=True)

			if self.data in ['mnist']:
				self.reals = tf.reshape(self.reals, [self.reals.shape[0], self.reals.shape[1]*self.reals.shape[2]*self.reals.shape[3]])
				self.fakes = tf.reshape(self.fakes, [self.fakes.shape[0], self.fakes.shape[1]*self.fakes.shape[2]*self.fakes.shape[3]])

			eval(self.loss_func)

			self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
			self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))


	#################################################################

	def loss_RBFG(self):
		#### Code Courtest: https://github.com/siddharth-agrawal/Generative-Moment-Matching-Networks

		def makeScaleMatrix(num_gen, num_orig):

			# first 'N' entries have '1/N', next 'M' entries have '-1/M'
			s1 =  tf.constant(1.0 / num_gen, shape = [self.batch_size, 1])
			s2 = -tf.constant(1.0 / num_orig, shape = [self.batch_size, 1])
			return tf.concat([s1, s2], axis = 0)

		sigma = [1, 5,10,20,50]

		X = tf.concat([self.reals, self.fakes], axis = 0)
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


		self.G_loss = tf.sqrt(loss)
		self.D_loss = tf.constant(0.0)

		return 

	#################################################################

	def loss_IMQ(self):
		#### Code Courtesy https://github.com/hiwonjoon/wae-wgan/blob/master/wae_mmd.py

		n = tf.cast(self.batch_size,tf.float32)
		C_base = 2.*self.reals.shape[1]

		z = self.reals
		z_tilde = self.fakes

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

		for scale in [1.0]:
			C = tf.cast(C_base*scale,tf.float32)

			k_z = C / (C + dist_z_z + 1e-8)
			k_z_tilde = C / (C + dist_z_tilde_z_tilde + 1e-8)
			k_z_z_tilde = C / (C + dist_z_z_tilde + 1e-8)

			loss = 1/(n*(n-1))*tf.reduce_sum(k_z)\
				+ 1/(n*(n-1))*tf.reduce_sum(k_z_tilde)\
				- 2/(n*n)*tf.reduce_sum(k_z_z_tilde)

			L_D += loss

		self.G_loss = L_D
		self.D_loss = tf.constant(0.0)

	#####################################################################

	def loss_AE(self):
		mse = tf.keras.losses.MeanSquaredError()
		loss_AE_reals = tf.reduce_mean(tf.abs(self.reals - self.reals_dec))#mse(self.reals, self.reals_dec)
		self.AE_loss =  loss_AE_reals  


'''***********************************************************************************
********** WGAN ELEGANT WITH LATENT **************************************************
***********************************************************************************'''
### self.gan = WAE and self.topic = PolyGAN and self.loss = 'RBF'
class WGAN_PolyGAN(GAN_Base, RBFSolver):

	def __init__(self,FLAGS_dict):

		GAN_Base.__init__(self,FLAGS_dict)

		''' Set up the RBF Series Solver common to PolyGAN and its WAE variant'''
		RBFSolver.__init__(self)

		self.postfix = {0: f'{0:3.0f}', 1: f'{0:2.4e}', 2: f'{0:2.4e}'}
		self.bar_format = '{n_fmt}/{total_fmt} |{bar}| {rate_fmt}  Batch: {postfix[0]} ETA: {remaining}  Elapsed Time: {elapsed}  D_Loss: {postfix[1]}  G_Loss: {postfix[2]}'

		self.first_iteration_flag = 1

	def create_models(self):
		with tf.device(self.device):
			self.total_count = tf.Variable(0,dtype='int64')
			self.generator = eval(self.gen_model)
			self.discriminator_RBF = self.discriminator_model_RBF()

			print("Model Successfully made")
			print("\n\n GENERATOR MODEL: \n\n")
			print(self.generator.summary())
			print("\n\n DISCRIMINATOR RBF: \n\n")
			print(self.discriminator_RBF.summary())

			if self.res_flag == 1 and self.resume != 1:
				with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
					# Pass the file handle in as a lambda function to make it callable
					fh.write("\n\n GENERATOR MODEL: \n\n")
					self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DISCRIMINATOR RBF: \n\n")
					self.discriminator_RBF.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
		return

	def create_optimizer(self):
		with tf.device(self.device):
			self.lr_G_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=500, decay_rate=0.9, staircase=True)
			self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G) #Nadam
		print("Optimizers Successfully made")
		return


	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer, \
							generator = self.generator, \
							discriminator_RBF = self.discriminator_RBF, \
							total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
					self.discriminator_RBF = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator_RBF.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
		return


	def train_step(self,reals_all):
		with tf.device(self.device):
			noise = self.get_noise([self.batch_size, self.noise_dims])
		self.reals = reals_all

		with tf.GradientTape() as gen_tape:

			self.fakes = self.generator(noise, training=True)
				
			self.real_output,self.lambda_x_terms_1 = self.discriminator_RBF(self.reals, training = True)
			self.fake_output,self.lambda_x_terms_2 = self.discriminator_RBF(self.fakes, training = True)

			# print(self.real_output, self.fake_output)
			with gen_tape.stop_recording():
				Centres, Weights, Lamb_Weights = self.find_rbf_centres_weights()
				self.discriminator_RBF.set_weights([Centres,Weights,Centres,Lamb_Weights])

				if self.first_iteration_flag:
					self.first_iteration_flag = 0 
					self.lamb = tf.constant(0.1)
					self.D_loss = self.G_loss = tf.constant(0)
					return

				self.find_lambda()
			self.divide_by_lambda()
			
			eval(self.loss_func)
			self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
			self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))


	def loss_RBF(self):
		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)

		self.D_loss = 1 * (-loss_real + loss_fake)
		if (2*self.rbf_m - self.rbf_n) >= 0:
			self.G_loss = loss_real - loss_fake
		elif (2*self.rbf_m - self.rbf_n) < 0:
			self.G_loss = -1*(loss_real - loss_fake)


