from __future__ import print_function
import os, sys, time, argparse
from datetime import date
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import math
from absl import app
from absl import flags
import json

from gan_data import *
from gan_src import *

import tensorflow_probability as tfp
tfd = tfp.distributions
from matplotlib.backends.backend_pgf import PdfPages
from scipy.interpolate import interp1d
mse = tf.keras.losses.MeanSquaredError()
from ops import *


'''
GAN_topic is the Overarching class file, where corresponding parents are instantialized, along with setting up the calling functions for these and files and folders for resutls, etc. data reading is also done from here. Sometimes display functions, architectures, etc may be modified here if needed (overloading parent classes)
'''


'''***********************************************************************************
********** The PHS Radial Basis Function Solver **************************************
***********************************************************************************'''
class RBFSolver():

	def __init__(self):
		from itertools import product as cart_prod

		## For 1-D and 2-D Gaussians, no latent projections are needed. So latent dims are true dims.
		if self.data in ['g1']:
			self.latent_dims = 1
		if self.data in ['g2', 'gmm8']:
			self.latent_dims = 2
		if self.data in ['gN']:
			self.latent_dims = self.GaussN

		self.rbf_n = self.latent_dims

		## Defining the Solution cases based on m and n
		if self.rbf_n%2 == 1:
			if self.rbf_m < ((self.rbf_n+1)/2) :
				self.poly_case = 0
			else:
				self.poly_case = 1 ## odd_n, for all m
		else:
			if self.rbf_m <= ((self.rbf_n/2) - 1):
				self.poly_case = 2 ## even_n, negtive 2m-n
			else:
				self.poly_case = 3 ## even_n, positive 2m-n
			self.rbf_eta = self.rbf_n/2



		## Target data is for evaluateion of S_pd
		## Generator data is for evaluateion of S_pg
		if self.gan in ['WGAN']:
			self.target_data = 'self.reals'
			self.generator_data = 'self.fakes'
		elif self.gan in ['WAE']:
			self.target_data = 'self.fakes_enc'
			self.generator_data = 'self.reals_enc'
		return

	def discriminator_model_RBF(self):

		if self.gan not in ['WAE'] and self.data not in ['g2','gmm8', 'gN']:
			inputs = tf.keras.Input(shape=(self.output_size,self.output_size,1))
			inputs_res = tf.keras.layers.Reshape(target_shape=[self.output_size*self.output_size])(inputs)
		else:
			inputs = tf.keras.Input(shape=(self.latent_dims,))
			inputs_res = inputs

		
		num_centers = 2*self.batch_size
		# num_centers = 2*self.N_centers ### For Supp Experiment

		Out = RBFLayer(num_centres=num_centers, output_dim=1, order_m = self.rbf_m, batch_size = self.batch_size)(inputs_res)

		lambda_term = RBFLayer(num_centres=num_centers, output_dim=1, order_m = self.rbf_m, batch_size = self.batch_size,  rbf_pow = -self.rbf_n)(inputs_res)

		model = tf.keras.Model(inputs=inputs, outputs= [Out,lambda_term])

		return model

	def find_rbf_centres_weights(self):
		C_d = eval(self.target_data) #SHould be NDxn
		C_g = eval(self.generator_data)

		if self.gan not in ['WAE'] and self.data not in ['g2','gmm8', 'gN']:
			C_d = tf.reshape(C_d, [C_d.shape[0], C_d.shape[1]*C_d.shape[2]*C_d.shape[3]])
			C_g = tf.reshape(C_g, [C_g.shape[0], C_g.shape[1]*C_g.shape[2]*C_g.shape[3]])

		Centres = np.concatenate((C_d,C_g), axis = 0)

		D_d = (-1/C_d.shape[0])*np.ones([self.batch_size])
		D_g = (1/(C_g.shape[0]))*np.ones([self.batch_size])
		W_lamb = 1*tf.ones_like(D_d)
		Weights = np.concatenate((D_d,D_g), axis = 0)
		Lamb_Weights = np.concatenate((W_lamb,W_lamb), axis = 0)
		return Centres, Weights, Lamb_Weights


	def find_lambda(self):
		### The approx order of lambda
		pow_val = np.ceil(np.log10(np.amax(np.abs(self.real_output.numpy()), axis = 0)))
		if (2*self.rbf_m - self.rbf_n) < 0 or (2*self.rbf_m - self.rbf_n) > 1 or self.data in ['mnist']:
			self.lamb = tf.constant(1.0* (10.0 ** (pow_val)))
		else:
			self.lamb = tf.constant(1.0)

	### Use for negative 2m-n
	def find_lambda_exact(self):

		if self.poly_case == 1 or self.poly_case == 2 or self.poly_case == 3: 
			try:
				approx_ord_1 = ((4*self.rbf_m + self.rbf_n + 1) ** ((4*self.rbf_m + self.rbf_n + 1)/2 ))/((2*self.rbf_m + self.rbf_n + 1)**((2*self.rbf_m + self.rbf_n + 1)/2 ))
			except:
				try:
					approx_ord_1 = ((4*self.rbf_m - self.rbf_n + 1) ** ((4*self.rbf_m -self.rbf_n + 1)/2 ))/((2*self.rbf_m - self.rbf_n + 1) ** ((2*self.rbf_m - self.rbf_n + 1)/2))
				except:
					approx_ord_1 = ((2*self.rbf_m - self.rbf_n) ** ((2*self.rbf_m -self.rbf_n)/2 ))/((1*self.rbf_m - tf.floor(self.rbf_n/2)) ** ((1*self.rbf_m - tf.floor(self.rbf_n/2))/2))
				
			approx_ord_2 = tf.math.exp(tf.cast(self.rbf_m, dtype = 'float32'))

			approx_ord_3 = tf.sqrt( (12*self.rbf_m + 3*self.rbf_n + 2)/(6*self.rbf_m + 3*self.rbf_n + 2))

			approx_ord_4 = 2 * (tf.cast(self.batch_size, dtype = 'float32') ** (3/2)) * (self.rbf_n ** (-self.rbf_m/2))

			apprx_ord = approx_ord_1 * approx_ord_2 * approx_ord_3 * approx_ord_4 

			if self.rbf_n > 64:
				if (2*self.rbf_m - self.rbf_n) <= 1:
					self.lamb = (self.rbf_m-(self.rbf_n/2))**((2*self.rbf_m-self.rbf_n))
				else:
					self.lamb = tf.constant((10.0**(2*self.rbf_m - self.rbf_n)))

			elif self.rbf_n <= 64 and self.rbf_n >= 25:
				self.lamb = tf.divide(2**(3*self.rbf_n),apprx_ord)
			else:
				self.lamb = tf.divide(tf.reduce_sum(self.lambda_x_terms_2) + tf.reduce_sum(self.lambda_x_terms_1),apprx_ord)#,self.rbf_m**self.rbf_n)

				self.lamb = tf.sqrt(self.lamb)

				if (2*self.rbf_m - self.rbf_n) > 4 and self.data in ['gmm8', 'g2', 'gN']:
					self.lamb *= tf.constant((2.0**((2*self.rbf_m - self.rbf_n/2))))
				else:
					self.lamb *= tf.constant((10.0**(2*self.rbf_m - self.rbf_n)))
		self.lamb = tf.constant(self.lamb)

	def divide_by_lambda(self):
		self.real_output = tf.divide(self.real_output, self.lamb)
		self.fake_output = tf.divide(self.fake_output, self.lamb)


'''***********************************************************************************
********** Custom RBF Layer **********************************************************
***********************************************************************************'''
class RBFLayer(tf.keras.layers.Layer):
	""" Layer of Gaussian RBF units.
	# Example
	```python
		model = Sequential()
		model.add(RBFLayer(10,
						   initializer=InitCentersRandom(X),
						   betas=1.0,
						   input_shape=(1,)))
		model.add(Dense(1))
	```
	# Arguments
		output_dim: number of hidden units (i.e. number of outputs of the
					layer)
		initializer: instance of initiliazer to initialize centers
		betas: float, initial value for betas
	"""

	def __init__(self, num_centres, output_dim, order_m, batch_size, rbf_pow = None, initializer=None, **kwargs):

		self.m = order_m
		self.output_dim = output_dim #1 for us
		self.num_hidden = num_centres #N for us 
		self.rbf_pow =rbf_pow
		# self.unif_weight = 1/batch_size
		if not initializer:
			self.initializer = tf.keras.initializers.RandomUniform(0.0, 1.0)
		else:
			self.initializer = initializer
		super(RBFLayer, self).__init__(**kwargs)


	def build(self, input_shape):
		print(input_shape) ## Should be NB x n
		self.n = input_shape[1]
		self.centers = self.add_weight(name='centers',
									   shape=(self.num_hidden, input_shape[1]), ## Nxn
									   initializer=self.initializer,
									   trainable=True)
		self.rbf_weights = self.add_weight(name='rbf_weights',
									 shape=(self.num_hidden,), ## N,1
									 initializer='ones',
									 trainable=True)

		super(RBFLayer, self).build(input_shape)

	def call(self, X):
		X = tf.expand_dims(X, axis = 2) ## X in Nonexnx1
		C = tf.expand_dims(self.centers, axis = 2) ## Nxnx1
		C = tf.expand_dims(C, axis = 0)
		C_tiled = tf.tile(C, [tf.shape(X)[0],1,1,1])
		X = tf.expand_dims(X, axis = 1)
		X_tiled = tf.tile(X, [1,self.num_hidden,1,1])
		Tau = C_tiled - X_tiled ## NonexNxnx1 = NonexNxnx1 - NonexNxnx1
		# print('Tau', Tau)
		if self.rbf_pow == None:
			order = (2*self.m) - self.n
		else:
			order = self.rbf_pow
		
		if self.n%2 == 1 or (self.n%2 == 0 and (2*self.m-self.n)<0) :
			norm_tau = tf.norm(Tau, ord = 'euclidean', axis = 2)
			ord_tensor = order*tf.ones_like(norm_tau)
			Phi = tf.pow(norm_tau, ord_tensor) ## Nx1
		else:
			norm_tau = tf.norm(Tau, ord = 'euclidean', axis = 2)
			ord_tensor = order*tf.ones_like(norm_tau)
			Phi = 1*tf.multiply(tf.pow(norm_tau, ord_tensor),tf.math.log(norm_tau+10.0**(-100)))##Nx1

		W = tf.expand_dims(self.rbf_weights, axis = 1)
		D = tf.squeeze(tf.linalg.matmul(W, Phi, transpose_a=True, transpose_b=False),axis = 2)
		return D


	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)

	def get_config(self):
		# have to define get_config to be able to use model_from_json
		config = {
			'output_dim': self.output_dim
		}
		base_config = super(RBFLayer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


'''***********************************************************************************
********** GAN Baseline setup ********************************************************
***********************************************************************************'''
class GAN_Base(GAN_SRC, GAN_DATA_Base):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_SRC class - defines all fundamental ops and metric functions'''
		GAN_SRC.__init__(self,FLAGS_dict)
		''' Set up the GAN_DATA class'''
		GAN_DATA_Base.__init__(self)

	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.gen_model = 'self.generator_model_'+self.arch+'_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.arch+'_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.batch_size)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'


	def get_data(self):
		# with tf.device('/CPU'):
		self.train_data = eval(self.gen_func)

		self.num_batches = int(np.floor((self.train_data.shape[0] * self.reps)/self.batch_size))
		''' Set PRINT and SAVE iters if 0'''
		self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
		self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

		self.train_dataset = eval(self.dataset_func)
		self.train_dataset_size = self.train_data.shape[0]

		print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size,  self.num_batches,self.print_step, self.save_step))


	###### WGAN-RBF overloads this function. 
	def create_models(self):
		with tf.device(self.device):
			self.total_count = tf.Variable(0,dtype='int64')
			self.generator = eval(self.gen_model)
			self.discriminator = eval(self.disc_model)

			if self.res_flag == 1:
				with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
					# Pass the file handle in as a lambda function to make it callable
					fh.write("\n\n GENERATOR MODEL: \n\n")
					self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
					self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))

			print("Model Successfully made")

			print(self.generator.summary())
			print(self.discriminator.summary())
		return		


	###### WGAN-FS overloads this function. Need a better way to execute it.... The overload has to do for now..... 
	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer,
								 Disc_optimizer = self.Disc_optimizer,
								 generator = self.generator,
								 discriminator = self.discriminator,
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
					self.discriminator = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))
		return


	def get_noise(self, shape):

		elif self.noise_kind == 'gaussian':
			noise = tf.random.normal(shape, mean = self.noise_mean, stddev = self.noise_stddev)

		return noise

	def train(self):
		start = int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1
		for epoch in range(start,self.num_epochs):
			if self.pbar_flag:
				bar = self.pbar(epoch)   
			start = time.time()
			batch_count = tf.Variable(0,dtype='int64')
			start_time =0

			for image_batch in self.train_dataset:
				# print(image_batch.shape)
				self.total_count.assign_add(1)
				batch_count.assign_add(1)
				start_time = time.time()
				with tf.device(self.device):
					self.train_step(image_batch)
					self.eval_metrics()
						

				train_time = time.time()-start_time

				if self.pbar_flag:
					bar.postfix[0] = f'{batch_count.numpy():6.0f}'
					bar.postfix[1] = f'{self.D_loss.numpy():2.4e}'
					bar.postfix[2] = f'{self.G_loss.numpy():2.4e}'
					bar.update(self.batch_size.numpy())
				if (batch_count.numpy() % self.print_step.numpy()) == 0 or self.total_count <= 2:
					if self.res_flag:
						self.res_file.write("Epoch {:>3d} Batch {:>3d} in {:>2.4f} sec; D_loss - {:>2.4f}; G_loss - {:>2.4f} \n".format(epoch,batch_count.numpy(),train_time,self.D_loss.numpy(),self.G_loss.numpy()))

				self.print_batch_outputs(epoch)

				# Save the model every SAVE_ITERS iterations
				if (self.total_count.numpy() % self.save_step.numpy()) == 0:
					if self.save_all:
						self.checkpoint.save(file_prefix = self.checkpoint_prefix)
					else:
						self.manager.save()

			if self.pbar_flag:
				bar.close()
				del bar
			tf.print('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
			self.save_epoch_h5models()


	def save_epoch_h5models(self):

		self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)

		if self.loss == 'RBF':
			self.discriminator_RBF.save(self.checkpoint_dir +'/model_discriminator_RBF.h5',overwrite=True)
		else:
			self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)
		return


	def print_batch_outputs(self,epoch):

		if (self.total_count.numpy() <= 5) and self.data in [ 'g2']:
			self.generate_and_save_batch(epoch)
		if ((self.total_count.numpy() % 100) == 0 and self.data in [ 'g2']):
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)

	def eval_sharpness(self):
		i = 0
		for train_batch in self.train_dataset:
			noise = self.get_noise([self.batch_size, self.noise_dims])
			preds = self.generator(noise, training = False)

			sharp = self.find_sharpness(preds)
			base_sharp = self.find_sharpness(train_batch)
			try:
				sharp_vec.append(sharp)
				base_sharp_vec.append(base_sharp)

			except:
				sharp_vec = [sharp]
				base_sharp_vec = [base_sharp]
			i += 1
			if i == 10:
				break
		###### Sharpness averaging measure
		sharpness = np.mean(np.array(sharp_vec))
		baseline_sharpness = np.mean(np.array(base_sharp_vec))

		return baseline_sharpness, sharpness



	def test(self):
		num_interps = 10
		if self.mode == 'test':
			num_figs = 20#int(400/(2*num_interps))
		else:
			num_figs = 9
		# there are 400 samples in the batch. to make 10x10 images, 
		for j in range(num_figs):
			path = self.impath+'_TestingInterpolationV2_'+str(self.total_count.numpy())+'_TestCase_'+str(j)+'.png'
			for i in range(num_interps):
				start = self.get_noise([1, self.noise_dims])
				end = self.get_noise([1, self.noise_dims]) 
				stack = np.vstack([start, end])

				linfit = interp1d([1,num_interps+1], stack, axis=0)
				interp_latents = linfit(list(range(1,num_interps+1)))

				cur_interp_figs = self.generator(interp_latents)

				sharpness = self.find_sharpness(cur_interp_figs)
				try:
					sharpness_vec.append(sharpness)
				except:
					shaprpness_vec = [sharpness]
				try:
					batch_interp_figs = np.concatenate((batch_interp_figs,cur_interp_figs),axis = 0)
				except:
					batch_interp_figs = cur_interp_figs

			images = (batch_interp_figs + 1.0)/2.0
			size_figure_grid = num_interps
			images_on_grid = self.image_grid(input_tensor = images, grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(num_interps,num_interps))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'INTERPOLATED IMAGES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			plt.savefig(path)
			plt.close()
			del batch_interp_figs

		###### Interpol samples - Sharpness
		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("Interpolation Sharpness - " + str(overall_sharpness))
		if self.res_flag:
			self.res_file.write("Interpolation Sharpness - "+str(overall_sharpness))



'''***********************************************************************************
********** WAE-GAN Setup *************************************************************
***********************************************************************************'''
class GAN_WAE(GAN_SRC, GAN_DATA_WAE):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_SRC class - defines all GAN architectures'''

		GAN_SRC.__init__(self,FLAGS_dict)

		''' Set up the GAN_DATA class'''
		GAN_DATA_WAE.__init__(self)
		# eval('GAN_DATA_'+FLAGS.topic+'.__init__(self,data)')

	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.encdec_model = 'self.encdec_model_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.batch_size)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		if self.loss == 'FS':
			self.disc_model = 'self.discriminator_model_FS()' 
			self.DEQ_func = 'self.discriminator_ODE()'

		''' Define dataset and tf.data function. batch sizing done'''
		# self.get_data()
		# print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size, self.num_batches,self.print_step, self.save_step))

	def get_data(self):
		# with tf.device('/CPU'):
		self.train_data = eval(self.gen_func)

		# self.batch_size_big = tf.constant(self.batch_size*self.Dloop,dtype='int64')
		self.num_batches = int(np.floor((self.train_data.shape[0] * self.reps)/self.batch_size))
		''' Set PRINT and SAVE iters if 0'''
		self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
		self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

		self.train_dataset = eval(self.dataset_func)
		self.train_dataset_size = self.train_data.shape[0]

		print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size,  self.num_batches,self.print_step, self.save_step))


	def get_noise(self,batch_size):
		if self.noise_kind == 'gaussian_trunc':
			noise = tfp.distributions.TruncatedNormal(loc=0., scale=0.3, low=-1., high=1.).sample([batch_size, self.latent_dims])

		if self.noise_kind == 'gaussian':
			noise = tf.random.normal([batch_size, self.latent_dims], mean = 0.0, stddev = 1.0)

		if self.noise_kind == 'gaussian_s2':
			noise = tf.random.normal([batch_size, self.latent_dims], mean = 0.0, stddev = np.sqrt(2))

		if self.noise_kind == 'gaussian_s4':
			noise = tf.random.normal([batch_size, self.latent_dims], mean = 0.0, stddev = 2)
		return noise


	def train(self):    
		start = int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1 
		for epoch in range(start,self.num_epochs):
			if self.pbar_flag:
				bar = self.pbar(epoch)
			start = time.time()
			batch_count = tf.Variable(0,dtype='int64')
			start_1 = 0

			for image_batch in self.train_dataset:
				self.total_count.assign_add(1)
				batch_count.assign_add(self.Dloop)
				start_1 = time.time()
				
				with tf.device(self.device):
					## By Default, we dont need to pretrain either the GAN or the AE 
					## particularly, but the code leaves the option open just in case.
					if epoch <= self.GAN_pretrain_epochs or epoch <= self.AE_pretrain_epochs:
						if epoch <= self.GAN_pretrain_epochs:
							self.pretrain_step_GAN(image_batch)
						if epoch <= self.AE_pretrain_epochs:
							self.pretrain_step_AE(image_batch)
					else:
						self.train_step(image_batch)
						self.eval_metrics()
				
				train_time = time.time()-start_1

				if self.pbar_flag:
					bar.postfix[0] = f'{batch_count.numpy():4.0f}'
					bar.postfix[1] = f'{self.D_loss.numpy():2.4e}'
					bar.postfix[2] = f'{self.G_loss.numpy():2.4e}'
					bar.postfix[3] = f'{self.AE_loss.numpy():2.4e}'
					bar.update(self.batch_size.numpy())

				if (batch_count.numpy() % self.print_step.numpy()) == 0 or self.total_count <= 2:
					if self.res_flag:
						self.res_file.write("Epoch {:>3d} Batch {:>3d} in {:>2.4f} sec; D_loss - {:>2.4f}; G_loss - {:>2.4f}; AE_loss - {:>2.4f} \n".format(epoch,batch_count.numpy(),train_time,self.D_loss.numpy(),self.G_loss.numpy(),self.AE_loss.numpy()))

				self.print_batch_outputs(epoch)

				# Save the model every SAVE_ITERS iterations
				if (self.total_count.numpy() % self.save_step.numpy()) == 0:
					if self.save_all:
						self.checkpoint.save(file_prefix = self.checkpoint_prefix)
					else:
						self.manager.save()
		
			if self.pbar_flag:
				bar.close()
				del bar

			tf.print ('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
			self.save_epoch_h5models()

	def print_batch_outputs(self,epoch):		
		if (self.total_count.numpy() <= 2) or ((self.total_count.numpy() % self.save_step.numpy()) == 0) or ((self.total_count.numpy() % 250) ==0):
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % 1000) == 0:
			self.test()


	def test(self):
		###### Kurtosis and Skewness: 
		### Implementation courtesy "Cramer-Wold autoencoder", Knop et al., 2020

		for image_batch in self.interp_dataset:
			encoded = self.Encoder(image_batch, training = False).numpy()
			try:
				encoded_all = np.concatenate((encoded_all,encoded), axis=0)
			except:
				encoded_all = encoded

		self.eval_MardiaStats(encoded_all)
		if self.mode =='test':
			print("\n Final skewness score - "+str(self.skewness))
			print("\n Final Kurtosis score - "+str(self.kurtosis))

		###### Test code to print random samples by decoding Latent space vectors
		###### Sharpness is also computed over bathces of radom images
		for i in range(10):
			path = self.impath+'_Testing_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			noise = self.get_noise(self.batch_size)
			images = self.Decoder(noise)

			sharpness = self.find_sharpness(images)
			try:
				sharpness_vec.append(sharpness)
			except:
				shaprpness_vec = [sharpness]

			images = (images + 1.0)/2.0
			size_figure_grid = self.num_to_print
			images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(7,7))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			plt.savefig(path)
			plt.close()

		###### Random Samples - Sharpness averaging 
		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("\n Random Sharpness - " + str(overall_sharpness))
		if self.res_flag:
			self.res_file.write("\n Random Sharpness - "+str(overall_sharpness))


		###### Benchmark Samples - Sharpness computed on training data 
		i = 0
		for image_batch in self.train_dataset:
			i+=1
			sharpness = self.find_sharpness(image_batch)
			try:
				sharpness_vec.append(sharpness)
			except:
				shaprpness_vec = [sharpness]
			if i==100:
				break

		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("\n Dataset Sharpness 10k samples - " + str(overall_sharpness))
		if self.res_flag:
			self.res_file.write("\n Dataset Sharpness 10k samples - "+str(overall_sharpness))


		###### Reconstruction Error computation
		for image_batch in self.recon_dataset:				
			path = self.impath+'_TestingRecon_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			images = self.Decoder(self.Encoder(image_batch))

			### Bacth wise compute. So we compute moving average across batches
			try:
				recon_loss = 0.5*(recon_loss) + 0.5*tf.reduce_mean(tf.abs(image_batch - images))
			except:
				recon_loss = tf.reduce_mean(tf.abs(image_batch - images))

			### Printig recon images
			images = (images + 1.0)/2.0
			size_figure_grid = self.num_to_print
			images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(7,7))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			plt.savefig(path)
			plt.close()

			###### Recon - Ground Truth
			path = self.impath+'_TestingReconOrg_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			images = image_batch
			images = (images + 1.0)/2.0
			size_figure_grid = self.num_to_print
			images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(7,7))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			plt.savefig(path)
			plt.close()
			break

		### Print metric value/write to file
		if self.mode == 'test':
			print("\n Reconstruction error - " + str(recon_loss))
		if self.res_flag:
			self.res_file.write("\n Reconstruction error - " + str(recon_loss))


		####### Interpolation test
		num_interps = self.num_to_print
		if self.mode == 'test':
			num_figs = int(400/(2*num_interps))
		else:
			num_figs = 9
		# there are 400 samples in the batch. to make 10x10 images, 
		for image_batch in self.interp_dataset:
			for j in range(num_figs):
				path = self.impath+'_TestingInterpolationV2_'+str(self.total_count.numpy())+'_TestCase_'+str(j)+'.png'
				current_batch = image_batch[2*num_interps*j:2*num_interps*(j+1)]
				image_latents = self.Encoder(current_batch)
				for i in range(num_interps):
					start = image_latents[i:1+i].numpy()
					end = image_latents[num_interps+i:num_interps+1+i].numpy()
					stack = np.vstack([start, end])

					linfit = interp1d([1,num_interps+1], stack, axis=0)
					interp_latents = linfit(list(range(1,num_interps+1)))
					cur_interp_figs = self.Decoder(interp_latents)

					sharpness = self.find_sharpness(cur_interp_figs)

					try:
						sharpness_vec.append(sharpness)
					except:
						shaprpness_vec = [sharpness]
					cur_interp_figs_with_ref = np.concatenate((current_batch[i:1+i],cur_interp_figs.numpy(),current_batch[num_interps+i:num_interps+1+i]), axis = 0)
					# print(cur_interp_figs_with_ref.shape)
					try:
						batch_interp_figs = np.concatenate((batch_interp_figs,cur_interp_figs_with_ref),axis = 0)
					except:
						batch_interp_figs = cur_interp_figs_with_ref

				images = (batch_interp_figs + 1.0)/2.0
				# print(images.shape)
				size_figure_grid = num_interps
				images_on_grid = self.image_grid(input_tensor = images, grid_shape = (size_figure_grid,size_figure_grid+2),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
				fig1 = plt.figure(figsize=(num_interps,num_interps+2))
				ax1 = fig1.add_subplot(111)
				ax1.cla()
				ax1.axis("off")
				if images_on_grid.shape[2] == 3:
					ax1.imshow(np.clip(images_on_grid,0.,1.))
				else:
					ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

				label = 'INTERPOLATED IMAGES AT ITERATION '+str(self.total_count.numpy())
				plt.title(label)
				plt.tight_layout()
				plt.savefig(path)
				plt.close()
				del batch_interp_figs

		###### Interpolated samples - Sharpness
		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("\n Interpolation Sharpness - " + str(overall_sharpness))
		if self.res_flag:
			self.res_file.write("\n Interpolation Sharpness - "+str(overall_sharpness))








