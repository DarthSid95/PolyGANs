from __future__ import print_function
import os, sys, time, argparse
from datetime import date
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from absl import app
from absl import flags
import json
import glob
from tqdm.autonotebook import tqdm
import shutil
from cleanfid import fid
import ot

import tensorflow_probability as tfp
tfd = tfp.distributions

##FOR FID
from numpy import cov
from numpy import trace
from scipy.linalg import sqrtm
import scipy as sp
from numpy import iscomplexobj
from numpy.linalg import norm as norml2

from ops import *

class GAN_Metrics():

	def __init__(self):

		self.KID_flag = 0
		self.W22_flag = 0
		self.FID_flag = 0
		self.sharp_flag = 0
		self.recon_flag = 0
		self.GradGrid_flag = 0
		self.metric_counter_vec = []

		if 'W22' in self.metrics:				
			self.W22_flag = 1
			self.W22_vec = []

			if self.data in ['g1', 'g2', 'gN']:
				self.W22_steps = 10
			else:
				self.W22_flag = 1
				self.W22_steps = 50
				print('W22 is not an accurate metric on this datatype')
		

		if 'FID' in self.metrics:
			self.FID_flag = 1
			self.FID_load_flag = 0
			self.FID_vec = []
			self.FID_vec_new = []

			if self.data in ['mnist']:
				self.FID_steps = 500 #was 500, make 2500 to run on Colab 
				if self.gan == 'WAE':
					self.FID_steps = 500
				if self.mode == 'metrics':
					self.FID_num_samples = 10000
				else:
					self.FID_num_samples = 5000 
			elif self.data in ['cifar10']:
				self.FID_steps = 500
				if self.mode == 'metrics':
					self.FID_num_samples = 10000
				else:
					self.FID_num_samples = 5000
			elif self.data in ['celeba', 'church']:
				self.FID_steps = 5000 #2500 for Rumi
				if self.mode == 'metrics':
					self.FID_num_samples = 10000
				else:
					self.FID_num_samples = 5000
			elif self.data in ['gN']:
				self.FID_steps = 200#50
			else:
				self.FID_flag = 0
				print('FID cannot be evaluated on this dataset')

		if 'recon' in self.metrics:
			self.recon_flag = 1
			self.recon_vec = []
			self.FID_vec_new = []

			if self.data in ['mnist']:
				self.recon_steps = 500
			elif self.data in ['cifar10']:
				self.recon_steps = 1500
			elif self.data in ['celeba', 'chruch']:
				self.recon_steps = 1500 
			else:
				self.recon_flag = 0
				print('Reconstruction cannot be evaluated on this dataset')

		if 'GradGrid' in self.metrics:
			if self.data in ['g2', 'gmm8']:
				self.GradGrid_flag = 1
				self.GradGrid_steps = 50
			else:
				print("Cannot plot Gradient grid. Not a 2D dataset")

		if 'sharpness' in self.metrics:
			self.sharp_flag = 1
			self.sharp_vec = []
			self.sharp_steps = 1000


		if 'KID' in self.metrics:
			self.KID_flag = 1
			self.FID_load_flag = 0 ### FID functions load data
			self.KID_vec = []
			self.KID_vec_new = []

			if self.data in ['mnist']:
				self.KID_steps = 500 #was 500, make 2500 to run on Colab 
				if self.gan == 'WAE':
					self.KID_steps = 500
				if self.mode in ['metrics']:
					self.FID_num_samples = 10000 ### FID functions load data
				else:
					self.FID_num_samples = 5000 #was 5k
			elif self.data in ['cifar10']:
				self.KID_steps = 500
				if self.mode in ['metrics'] and self.testcase != 'single':
					self.FID_num_samples = 50000
				elif self.testcase != 'single':
					self.FID_num_samples = 5000
				else:
					self.FID_num_samples = 5000
			elif self.data in ['celeba', 'church']:
				self.KID_steps = 5000 
				if self.mode in ['metrics']:
					self.FID_num_samples = 5000
				else:
					self.FID_num_samples = 5000
			elif self.data in ['gN']:
				self.KID_steps = 200
			else:
				self.KID_flag = 0
				print('KID cannot be evaluated on this dataset')




	def eval_metrics(self):
		update_flag = 0

		if self.W22_flag and ((self.total_count.numpy()%self.W22_steps == 0 or self.total_count.numpy() <= 10 or self.total_count.numpy() == 1)  or self.mode in ['metrics', 'model_metrics']):
			update_flag = 1
			self.update_W22()
			if self.mode != 'metrics':
				np.save(self.metricpath+'W22.npy',np.array(self.W22_vec))
				self.print_W22()

		if self.FID_flag and (self.total_count.numpy()%self.FID_steps == 0 or self.mode == 'metrics' or self.total_count.numpy() < 2):
			update_flag = 1
			self.update_FID()
			if self.mode != 'metrics':
				np.save(self.metricpath+'FID.npy',np.array(self.FID_vec))
				self.print_FID()

		if self.KID_flag and (self.total_count.numpy()%self.KID_steps == 0 or self.mode in ['metrics', 'model_metrics'] or self.total_count.numpy() < 2):
			update_flag = 1
			self.update_KID()
			if self.mode != 'metrics':
				np.save(self.metricpath+'KID.npy',np.array(self.KID_vec))

		if self.sharp_flag and (self.total_count.numpy()%self.sharp_steps == 0 or self.mode == 'metrics'):
			update_flag = 1
			self.update_sharpness()
			if self.mode != 'metrics':
				np.save(self.metricpath+'Sharpness_all.npy',np.array(self.sharp_vec))
			else:
				np.save(self.metricpath+'Sarpness_MetricsEval.npy',np.array(self.sharp_vec))


		if self.recon_flag and ((self.total_count.numpy()%self.recon_steps == 0 or self.total_count.numpy() == 1)  or self.mode == 'metrics'):
			update_flag = 1
			self.eval_recon()
			if self.mode != 'metrics':
				np.save(self.metricpath+'recon.npy',np.array(self.recon_vec))
				self.print_recon()

		if self.GradGrid_flag and ((self.total_count.numpy()%self.GradGrid_steps == 0 or self.total_count.numpy() == 1) or self.mode == 'metrics'):
			update_flag = 1
			self.print_GradGrid()

		if self.res_flag and update_flag:
			self.res_file.write("Metrics avaluated at Iteration " + str(self.total_count.numpy()) + '\n')


	def update_W22(self):
		if self.topic == 'PolyGAN' and self.gan == 'WGAN':
			if self.data in ['g1','g2']:
				self.eval_W22(self.reals,self.fakes)
			else:
				self.estimate_W22(self.reals,self.fakes)
		elif self.gan == 'WAE':
			self.eval_W22(self.fakes_enc,self.reals_enc)
		else:
			if self.data in ['g1','g2', 'gN']:
				self.eval_W22(self.reals,self.fakes)
			else:
				self.estimate_W22(self.reals,self.fakes)

	def eval_W22(self,act1,act2):
		mu1, sigma1 = act1.numpy().mean(axis=0), cov(act1.numpy(), rowvar=False)
		mu2, sigma2 = act2.numpy().mean(axis=0), cov(act2.numpy(), rowvar=False)
		ssdiff = np.sum((mu1 - mu2)**2.0)
		# calculate sqrt of product between cov
		if self.data not in ['g1', 'gmm2']:
			covmean = sqrtm(sigma1.dot(sigma2))
		else:
			covmean = np.sqrt(sigma1*sigma2)
		# check and correct imaginary numbers from sqrt
		if iscomplexobj(covmean):
			covmean = covmean.real
		# calculate score
		if self.data not in ['g1', 'gmm2']:
			self.W22_val = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
		else:
			self.W22_val = ssdiff + sigma1 + sigma2 - 2.0 * covmean
		self.W22_vec.append([self.W22_val, self.total_count.numpy()])
		if self.mode in ['metrics', 'model_metrics']:
			print("Final W22 score - "+str(self.W22_val))
			self.res_file.write("Final W22 score - "+str(self.W22_val))

		if self.res_flag:
			self.res_file.write("W22 score - "+str(self.W22_val))
		return

	def estimate_W22(self,target_sample, gen_sample, q=2, p=2):
		target_sample = tf.cast(target_sample, dtype = 'float32').numpy()
		gen_sample = tf.cast(gen_sample, dtype = 'float32').numpy()
		target_weights = np.ones(target_sample.shape[0]) / target_sample.shape[0]
		gen_weights = np.ones(gen_sample.shape[0]) / gen_sample.shape[0]

		x = target_sample[:, None, :] - gen_sample[None, :, :]

		M = tf.norm(x, ord=q, axis = 2)**p / p
		# print(target_sample.shape, gen_sample.shape, M.shape)
		T = ot.emd2(target_weights, gen_weights, M.numpy())
		self.W22_val = W = ((M.numpy() * T).sum())**(1. / p)

		self.W22_vec.append([self.W22_val, self.total_count.numpy()])
		if self.mode in ['metrics', 'model_metrics']:
			print("Final W22 score - "+str(self.W22_val))
			self.res_file.write("Final W22 score - "+str(self.W22_val))

		if self.res_flag:
			self.res_file.write("W22 score - "+str(self.W22_val))
		return


	def print_W22(self):
		path = self.metricpath
		if self.colab==1 or self.latex_plot_flag==0:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		vals = list(np.array(self.W22_vec)[:,0])
		locs = list(np.array(self.W22_vec)[:,1])
		

		with PdfPages(path+'W22_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = r'$\mathcal{W}^{2,2}(p_d,p_g)$ Vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)



	def update_FID(self):
		#FID Funcs vary per dataset. We therefore call the corresponding child func foundin the arch_*.py files
		if self.FID_kind == 'clean':
			self.save_images_FID()
			self.eval_CleanFID()
		else:
			eval(self.FID_func)

	def eval_FID(self):
		mu1, sigma1 = self.act1.mean(axis=0), cov(self.act1, rowvar=False)
		mu2, sigma2 = self.act2.mean(axis=0), cov(self.act2, rowvar=False)
		# calculate sum squared difference between means
		ssdiff = np.sum((mu1 - mu2)**2.0)
		# calculate sqrt of product between cov
		covmean = sqrtm(sigma1.dot(sigma2))
		# check and correct imaginary numbers from sqrt
		if iscomplexobj(covmean):
			covmean = covmean.real
		# calculate score
		self.fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
		self.FID_vec.append([self.fid, self.total_count.numpy()])
		if self.mode == 'metrics':
			print("Final FID score - "+str(self.fid))
			self.res_file.write("Final FID score - "+str(self.fid))

		if self.res_flag:
			self.res_file.write("FID score - "+str(self.fid))
		return

	def eval_CleanFID(self):
		mode = 'legacy_tensorflow'
		if self.data == 'cifar10' and self.testcase != 'single':
			self.fid = fid.compute_fid(self.FIDfakes_dir, dataset_name="cifar10", dataset_res=32, dataset_split="train", mode="clean")
		else:
			self.fid = fid.compute_fid(self.FIDfakes_dir, self.FIDreals_dir, mode = "legacy_tensorflow")


		self.FID_vec.append([self.fid, self.total_count.numpy()])
		if self.mode in ['metrics', 'model_metrics']:
			print("Final FID score - "+str(self.fid))
			self.res_file.write("Final FID score - "+str(self.fid))
		if self.res_flag:
			self.res_file.write("FID score - "+str(self.fid))
		return

	def print_FID(self):
		path = self.metricpath
		#Colab has issues with latex. Better not waste time printing on Colab. Use the NPY later, offline,...
		if self.latex_plot_flag:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})
		else:
			from matplotlib.backends.backend_pdf import PdfPages

		vals = list(np.array(self.FID_vec)[:,0])
		locs = list(np.array(self.FID_vec)[:,1])

		with PdfPages(path+'FID_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'FID vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

	def update_KID(self):
		if 'FID' not in self.metrics:
			self.save_images_FID()
			self.eval_CleanKID()


	def eval_CleanKID(self):
		self.kid = fid.compute_kid(self.FIDfakes_dir, self.FIDreals_dir)
		self.KID_vec.append([self.kid, self.total_count.numpy()])
		if self.mode in ['metrics', 'model_metrics']:
			print("Final KID score - "+str(self.kid))
			self.res_file.write("Final KID score - "+str(self.kid))

		if self.res_flag:
			self.res_file.write("KID score - "+str(self.kid))
		return

	def print_KID(self):
		path = self.metricpath
		#Colab has issues with latex. Better not waste time printing on Colab. Use the NPY later, offline,...
		if self.latex_plot_flag:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})
		else:
			from matplotlib.backends.backend_pdf import PdfPages

		vals = list(np.array(self.KID_vec)[:,0])
		locs = list(np.array(self.KID_vec)[:,1])

		with PdfPages(path+'KID_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'KID vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)


	def eval_recon(self):
		# print('Evaluating Recon Loss\n')
		mse = tf.keras.losses.MeanSquaredError()
		for image_batch in self.recon_dataset:
			# print("batch 1\n")
			recon_images = self.Decoder(self.Encoder(image_batch, training= False) , training = False)
			try:
				recon_loss = 0.5*(recon_loss) + 0.5*tf.reduce_mean(tf.abs(image_batch - recon_images)) 
			except:
				recon_loss = tf.reduce_mean(tf.abs(image_batch - recon_images))

		self.recon_vec.append([recon_loss, self.total_count.numpy()])
		if self.mode == 'metrics':
			print("Final Reconstruction error - "+str(recon_loss))
			if self.res_flag:
				self.res_file.write("Final Reconstruction error - "+str(recon_loss))

		if self.res_flag:
			self.res_file.write("Reconstruction error - "+str(recon_loss))

	def print_recon(self):
		path = self.metricpath
		#Colab has issues with latex. Better not waste time printing on Colab. Use the NPY later, offline,...
		if self.colab==1 or self.latex_plot_flag==0:
			from matplotlib.backends.backend_pdf import PdfPages
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		vals = list(np.array(self.recon_vec)[:,0])
		locs = list(np.array(self.recon_vec)[:,1])

		with PdfPages(path+'recon_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'Reconstruction Error')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)


	def print_GradGrid(self):

		path = self.metricpath + str(self.total_count.numpy()) + '_'

		if self.colab==1 or self.latex_plot_flag==0:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size": 12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		
		from itertools import product as cart_prod

		x = np.arange(self.MIN,self.MAX+0.1,0.1)
		y = np.arange(self.MIN,self.MAX+0.1,0.1)

		# X, Y = np.meshgrid(x, y)
		prod = np.array([p for p in cart_prod(x,repeat = 2)])
		# print(x,prod)

		X = prod[:,0]
		Y = prod[:,1]

		# print(prod,X,Y)
		# print(XXX)

		with tf.GradientTape() as disc_tape:
			prod = tf.cast(prod, dtype = 'float32')
			disc_tape.watch(prod)
			if self.loss == 'RBF':
				d_vals = self.discriminator_RBF(prod,training = False)
			else:
				d_vals = self.discriminator(prod,training = False)
		grad_vals = disc_tape.gradient(d_vals, [prod])[0]

		#Flag to control normalization of D(x) values for printing on the contour plot
		Normalize_Flag = False
		try:
			# print(d_vals[0])
			
			if Normalize_Flag and ((min(d_vals[0]) <= -2) or (max(d_vals[0]) >= 2)):
				### IF NORMALIZATION IS NEEDED
				d_vals_sub = d_vals[0] - min(d_vals[0])
				d_vals_norm = d_vals_sub/max(d_vals_sub)
				d_vals_norm -= 0.5
				d_vals_norm *= 3
				d_vals_new = np.reshape(d_vals_norm,(x.shape[0],y.shape[0])).transpose()
			else:
				### IF NORMALIZATION IS NOT NEEDED
				d_vals_norm = d_vals[0]
				d_vals_new = np.reshape(d_vals_norm,(x.shape[0],y.shape[0])).transpose()
		except:
			d_vals_new = np.reshape(d_vals,(x.shape[0],y.shape[0])).transpose()
		# print(d_vals_new)
		dx = grad_vals[:,1]
		dy = grad_vals[:,0]
		# print(XXX)
		n = -1
		color_array = np.sqrt(((dx-n)/2)**2 + ((dy-n)/2)**2)

		with PdfPages(path+'GradGrid_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim(bottom=self.MIN,top=self.MAX)
			ax1.quiver(X,Y,dx,dy,color_array)
			ax1.scatter(self.reals[:1000,0], self.reals[:1000,1], c='r', linewidth = 1, label='Real Data', marker = '.', alpha = 0.1)
			ax1.scatter(self.fakes[:1000,0], self.fakes[:1000,1], c='g', linewidth = 1, label='Fake Data', marker = '.', alpha = 0.1)
			pdf.savefig(fig1)
			plt.close(fig1)

		with PdfPages(path+'Contour_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(False)
			ax1.get_yaxis().set_visible(False)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim([self.MIN,self.MAX])
			ax1.contour(x,y,d_vals_new,15,linewidths = 1.0, alpha = 0.6 )
			ax1.scatter(self.reals[:,0], self.reals[:,1], c='r', linewidth = 1, marker = '.', alpha = 0.75)
			ax1.scatter(self.fakes[:,0], self.fakes[:,1], c='g', linewidth = 1, marker = '.', alpha = 0.75)
			# cbar = fig1.colorbar(cs, shrink=1., orientation = 'horizontal')
			pdf.savefig(fig1)
			plt.close(fig1)

	def update_sharpness(self):

		self.baseline_sharpness, self.sharpness = self.eval_sharpness()
		if self.mode == 'metrics':
			print("\n Final Sharpness score - "+str(self.sharpness))
			print(" \n Baseline Sharpness score - "+str(self.baseline_sharpness))
			if self.res_flag:
				self.res_file.write("\n Final Sharpness score - "+str(self.sharpness))
				self.res_file.write("\n Baseline Sharpness score - "+str(self.baseline_sharpness))

		if self.res_flag:
			self.res_file.write("\n Sharpness score - "+str(self.sharpness))
			self.res_file.write("\n Baseline Sharpness score - "+str(self.baseline_sharpness))

	def find_sharpness(self,input_ims):
		def laplacian(input, ksize, mode=None, constant_values=None, name=None):
			"""
			Apply Laplacian filter to image.
			Args:
			  input: A 4-D (`[N, H, W, C]`) Tensor.
			  ksize: A scalar Tensor. Kernel size.
			  mode: A `string`. One of "CONSTANT", "REFLECT", or "SYMMETRIC"
				(case-insensitive). Default "CONSTANT".
			  constant_values: A `scalar`, the pad value to use in "CONSTANT"
				padding mode. Must be same type as input. Default 0.
			  name: A name for the operation (optional).
			Returns:
			  A 4-D (`[N, H, W, C]`) Tensor.
			"""

			input = tf.convert_to_tensor(input)
			ksize = tf.convert_to_tensor(ksize)

			tf.debugging.assert_none_equal(tf.math.mod(ksize, 2), 0)

			ksize = tf.broadcast_to(ksize, [2])

			total = ksize[0] * ksize[1]
			index = tf.reshape(tf.range(total), ksize)
			g = tf.where(
				tf.math.equal(index, tf.math.floordiv(total - 1, 2)),
				tf.cast(1 - total, input.dtype),
				tf.cast(1, input.dtype),
			)

			channel = tf.shape(input)[-1]
			shape = tf.concat([ksize, tf.constant([1, 1], ksize.dtype)], axis=0)
			g = tf.reshape(g, shape)
			shape = tf.concat([ksize, [channel], tf.constant([1], ksize.dtype)], axis=0)
			g = tf.broadcast_to(g, shape)
			return tf.nn.depthwise_conv2d(input, g, [1, 1, 1, 1], padding="VALID")

		# import tensorflow_io as tfio
		lap_img = laplacian(input_ims,3)
		if input_ims.shape[3] == 3:
			reduction_axis = [1,2,3]
		else:
			reduction_axis = [1,2]
		var = tf.square(tf.math.reduce_std(lap_img, axis = reduction_axis))
		var_out = np.mean(var)
		# print(var_out)
		return var_out



		

