from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import glob
from absl import flags
import csv

from scipy import io as sio

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import PdfPages


### Need to prevent tfds downloads bugging out? check
import urllib3
urllib3.disable_warnings()


FLAGS = flags.FLAGS

'''***********************************************************************************
********** Base Data Loading Ops *****************************************************
***********************************************************************************'''
class GAN_DATA_ops:

	def __init__(self):
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data)'
		self.noise_mean = 0.0
		self.noise_stddev = 1.00
		#Default Number of repetitions of a dataset in tf.dataset mapping
		self.reps = 1
		if self.loss == 'RBF':
			self.reps_centres = int(np.ceil(self.N_centers//self.batch_size))

		if self.data == 'g2':
			self.MIN = -1
			self.MAX = 1.2
			self.noise_dims = 100
			self.output_size = 2
			self.noise_mean = 0.0
			self.noise_stddev = 1.0
		elif self.data == 'gN':
			self.MIN = 0
			self.MAX = 1
			self.noise_dims = 100
			self.output_size = self.GaussN
			self.noise_stddev = 1.0
			self.noise_mean = 0.0
		elif self.data == 'gmm8':
			self.noise_dims = 100
			self.output_size = 2
			self.noise_mean = 0.0
			self.noise_stddev = 1.0
		else:
			self.noise_dims = 100
			if self.data in ['celeba', 'church']:
				self.output_size = eval('self.'+self.data+'_size')
			elif self.data in ['mnist']:
				self.output_size = 28
			elif self.data in ['cifar10']:
				self.output_size = 32



	def mnist_loader(self):
		(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
		train_images = train_images.reshape(train_images.shape[0],28,28, 1).astype('float32')
		train_labels = train_labels.reshape(train_images.shape[0], 1).astype('float32')
		train_images = (train_images - 127.5) / 127.5
		test_images = test_images.reshape(test_images.shape[0],28,28, 1).astype('float32')
		test_images = (test_images - 127.5) / 127.5

		return train_images, train_labels, test_images, test_labels


	def celeba_loader(self):
		if self.colab:
			try:
				with open("data/CelebA/Colab_CelebA_Names.txt","r") as names:
					true_files = np.array([line.rstrip() for line in names])
					print("Data File Found. Reading filenames")
			except:
				true_files = sorted(glob.glob('/content/colab_data_faces/img_align_celeba/*.jpg'))
				print("Data File Created. Saving filenames")
				with open("data/CelebA/Colab_CelebA_Names.txt","w") as names:
					for name in true_files:
						names.write(str(name)+'\n')
		else:
			try:
				with open("data/CelebA/CelebA_Names.txt","r") as names:
					true_files = np.array([line.rstrip() for line in names])
					print("Data File Found. Reading filenames")
			except:
				true_files = sorted(glob.glob('data/CelebA/img_align_celeba/*.jpg'))
				print("Data File Created. Saving filenames")
				with open("data/CelebA/CelebA_Names.txt","w") as names:
					for name in true_files:
						names.write(str(name)+'\n')
		train_images = np.expand_dims(np.array(true_files),axis=1)

		attr_file = 'data/CelebA/list_attr_celeba.csv'

		with open(attr_file,'r') as a_f:
			data_iter = csv.reader(a_f,delimiter = ',',quotechar = '"')
			data = [data for data in data_iter]
		# print(data,len(data))
		label_array = np.asarray(data)

		return train_images, label_array


	def cifar10_loader(self):

		(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
		train_images = train_images.reshape(train_images.shape[0],32,32, 3).astype('float32')
		train_labels = train_labels.reshape(train_images.shape[0], 1).astype('float32')
		train_images = (train_images - 127.5) / 127.5
		test_images = test_images.reshape(test_images.shape[0],32,32, 3).astype('float32')
		test_labels = test_labels.reshape(test_images.shape[0], 1).astype('float32')
		test_images = (test_images - 127.5) / 127.5

		return train_images, train_labels, test_images, test_labels


	def church_loader(self):
		if self.colab:
			try:
				with open("data/LSUN/Colab_LSUN_Church_Names.txt","r") as names:
					true_files = np.array([line.rstrip() for line in names])
					print("Data File Found. Reading filenames")

				with open("data/LSUN/Colab_LSUN_Church_Val_Names.txt","r") as names:
					val_files = np.array([line.rstrip() for line in names])
					print("Validation File Found. Reading filenames")
			except:
				true_files = sorted(glob.glob('/content/local_data/LSUN/church_outdoor_train/*.jpg'))
				print("Data File Created. Saving filenames")
				with open("data/LSUN/Colab_LSUN_Church_Names.txt","w") as names:
					for name in true_files:
						names.write(str(name)+'\n')

				val_files = sorted(glob.glob('/content/local_data/LSUN/church_outdoor_val/*.jpg'))
				print("Validation File Created. Saving filenames")
				with open("data/LSUN/Colab_LSUN_Church_Val_Names.txt","w") as val_names:
					for name in val_files:
						val_names.write(str(name)+'\n')
		else:
			try:
				with open("data/LSUN/LSUN_Church_Names.txt","r") as names:
					true_files = np.array([line.rstrip() for line in names])
					print("Data File Found. Reading filenames")

				with open("data/LSUN/LSUN_Church_Val_Names.txt","r") as names:
					val_files = np.array([line.rstrip() for line in names])
					print("Validation File Found. Reading filenames")
			except:
				true_files = sorted(glob.glob('data/LSUN/church_outdoor_train/*.jpg'))
				print("Data File Created. Saving filenames")
				with open("data/LSUN/LSUN_Church_Names.txt","w") as names:
					for name in true_files:
						names.write(str(name)+'\n')

				val_files = sorted(glob.glob('data/LSUN/church_outdoor_val/*.jpg'))
				print("Validation File Created. Saving filenames")
				with open("data/LSUN/LSUN_Church_Names.txt","w") as val_names:
					for name in val_files:
						val_names.write(str(name)+'\n')

		train_images = np.expand_dims(np.array(true_files),axis=1)
		val_images = np.expand_dims(np.array(val_files),axis=1)

		return train_images, val_images

'''
GAN_DATA functions are specific to the topic, ELeGANt, RumiGAN, PRDeep or DCS. Data reading and dataset making functions per data, with init having some specifics generic to all, such as printing instructions, noise params. etc.
'''
'''***********************************************************************************
********** GAN_DATA_Baseline *********************************************************
***********************************************************************************'''
class GAN_DATA_Base(GAN_DATA_ops):

	def __init__(self):#,data,testcase,number,out_size):
		# self.gen_func = 'self.gen_func_'+data+'()'
		# self.dataset_func = 'self.dataset_'+data+'(self.train_data)'
		self.noise_mean = 0.0
		self.noise_stddev = 1.00
		GAN_DATA_ops.__init__(self)#,data,testcase,number,out_size)

	def gen_func_mnist(self):
		# self.output_size = int(28)

		train_images, train_labels, test_images, test_labels = self.mnist_loader()
		self.test_images = test_images 


		self.fid_train_images = train_images
		self.reps = int(60000.0/train_images.shape[0])

		return train_images

	def dataset_mnist(self,train_data,batch_size):

		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(50000)
		train_dataset = train_dataset.batch(batch_size,drop_remainder=True)
		train_dataset = train_dataset.prefetch(10)

		if self.loss == 'RBF':
			self.center_dataset = tf.data.Dataset.from_tensor_slices((train_data))
			self.center_dataset = self.center_dataset.repeat(self.reps_centres)
			self.center_dataset = self.center_dataset.shuffle(50000)
			self.center_dataset = self.center_dataset.batch(self.N_centers,drop_remainder=True)
			self.center_dataset = self.center_dataset.prefetch(10)
			
		return train_dataset


	def gen_func_g2(self):

		self.MIN = -5.5
		self.MAX = 10.5
		self.data_centres = tf.random.normal([500*self.N_centers, 2], mean =np.array([3.5,3.5]), stddev = np.array([1.25,1.25]))
		data = tf.random.normal([500*self.batch_size.numpy(), 2], mean =np.array([3.5,3.5]), stddev = np.array([1.25,1.25]))

	def dataset_g2(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(4)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(5)

		if self.loss == 'RBF':
			self.center_dataset = tf.data.Dataset.from_tensor_slices((self.data_centres))
			self.center_dataset = self.center_dataset.shuffle(4)
			self.center_dataset = self.center_dataset.batch(self.N_centers)
			self.center_dataset = self.center_dataset.prefetch(5)

		return train_dataset


	def gen_func_gmm8(self):
		## Cirlce - [0,1]
		scaled_circ = 0.35
		offset = 0.5
		locs = [[scaled_circ*1.+offset, 0.+offset], \
				[0.+offset, scaled_circ*1.+offset], \
				[scaled_circ*-1.+offset,0.+offset], \
				[0.+offset,scaled_circ*-1.+offset], \
				[scaled_circ*1*0.7071+offset, scaled_circ*1*0.7071+offset], \
				[scaled_circ*-1*0.7071+offset, scaled_circ*1*0.7071+offset], \
				[scaled_circ*1*0.7071+offset, scaled_circ*-1*0.7071+offset], \
				[scaled_circ*-1*0.7071+offset, scaled_circ*-1*0.7071+offset] ]
		self.MIN = -0. 
		self.MAX = 1.0 
		stddev_scale = [.02, .02, .02, .02, .02, .02, .02, .02]

		gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=probs),components_distribution=tfd.MultivariateNormalDiag(loc=locs,scale_identity_multiplier=stddev_scale))
		self.data_centres = gmm.sample(sample_shape=(int(500*self.N_centers)))
		return gmm.sample(sample_shape=(int(500*self.batch_size.numpy())))

	def dataset_gmm8(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(4)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(5)

		if self.loss == 'RBF':
			self.center_dataset = tf.data.Dataset.from_tensor_slices((self.data_centres))
			self.center_dataset = self.center_dataset.shuffle(4)
			self.center_dataset = self.center_dataset.batch(self.N_centers)
			self.center_dataset = self.center_dataset.prefetch(5)
		return train_dataset


	def gen_func_gN(self):

		self.data_centres = tf.random.normal([200*self.N_centers, self.GaussN], mean = 0.7*np.ones((1,self.output_size)), stddev = 0.2*np.ones((1,self.output_size)))

		return tf.random.normal([200*self.batch_size, self.GaussN], mean = 0.7*np.ones((1,self.output_size)), stddev = 0.2*np.ones((1,self.output_size)))

	def dataset_gN(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(4)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(5)

		if self.loss == 'RBF':
			self.center_dataset = tf.data.Dataset.from_tensor_slices((self.data_centres))
			self.center_dataset = self.center_dataset.shuffle(4)
			self.center_dataset = self.center_dataset.batch(self.N_centers)
			self.center_dataset = self.center_dataset.prefetch(5)

		return train_dataset


'''***********************************************************************************
********** GAN_DATA_WAE **************************************************************
***********************************************************************************'''
class GAN_DATA_WAE(GAN_DATA_ops):

	def __init__(self):#,data,testcase,number,out_size):
		GAN_DATA_ops.__init__(self)#,data,testcase,number,out_size)


	def gen_func_mnist(self):
		train_images, train_labels, test_images, test_labels = self.mnist_loader()
		self.fid_train_images = train_images
		self.test_images = test_images 
		return train_images

	def dataset_mnist(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(40)
		train_dataset = train_dataset.batch(batch_size,drop_remainder=True)
		train_dataset = train_dataset.prefetch(10)

		recon_dataset = tf.data.Dataset.from_tensor_slices((self.test_images[0:10000:100]))
		self.recon_dataset = recon_dataset.batch(100,drop_remainder=True)

		interp_dataset = tf.data.Dataset.from_tensor_slices((self.test_images[0:10000:25]))
		interp_dataset = interp_dataset.shuffle(10)
		self.interp_dataset = interp_dataset.batch(400,drop_remainder=True)
		return train_dataset


	def gen_func_celeba(self):

		train_images, data_array = self.celeba_loader()
		self.fid_train_images = train_images
		return train_images

	def dataset_celeba(self,train_data,batch_size):	
		def data_reader_faces(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([218,178,3])
				image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
				image = tf.image.resize(image,[self.output_size,self.output_size])
				image = tf.subtract(image,127.0)
				image = tf.divide(image,127.0)
			return image

		train_dataset = tf.data.Dataset.from_tensor_slices((train_data[400:]))
		train_dataset = train_dataset.map(data_reader_faces,num_parallel_calls=int(self.num_parallel_calls))
		train_dataset = train_dataset.shuffle(500)
		train_dataset = train_dataset.batch(batch_size,drop_remainder=True)
		train_dataset = train_dataset.prefetch(15)

		recon_dataset = tf.data.Dataset.from_tensor_slices((train_data[0:100]))
		recon_dataset=recon_dataset.map(data_reader_faces,num_parallel_calls=int(self.num_parallel_calls))
		self.recon_dataset = recon_dataset.batch(100,drop_remainder=True)

		interp_dataset = tf.data.Dataset.from_tensor_slices((train_data[0:400]))
		interp_dataset=interp_dataset.map(data_reader_faces,num_parallel_calls=int(self.num_parallel_calls))
		self.interp_dataset = interp_dataset.batch(400,drop_remainder=True)

		return train_dataset

	def gen_func_church(self):
		train_images, val_images = self.church_loader()

		self.fid_train_images = train_images
		self.test_images = val_images

		return train_images

	def dataset_church(self,train_data,batch_size):	

		def data_reader_faces(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image  = tf.image.crop_to_bounding_box(image, 0, 0, 256,256)
				image = tf.image.resize(image,[self.output_size,self.output_size])
				image = tf.subtract(image,127.0)
				image = tf.divide(image,127.0)
			return image

		train_dataset = tf.data.Dataset.from_tensor_slices((train_data[400:]))
		train_dataset=train_dataset.map(data_reader_faces,num_parallel_calls=int(self.num_parallel_calls))
		train_dataset = train_dataset.shuffle(500)
		train_dataset = train_dataset.batch(batch_size,drop_remainder=True)
		train_dataset = train_dataset.prefetch(15)

		recon_dataset = tf.data.Dataset.from_tensor_slices((self.test_images[0:100]))
		recon_dataset=recon_dataset.map(data_reader_faces,num_parallel_calls=int(self.num_parallel_calls))
		self.recon_dataset = recon_dataset.batch(100,drop_remainder=True)

		interp_dataset = tf.data.Dataset.from_tensor_slices((train_data[0:400]))
		interp_dataset=interp_dataset.map(data_reader_faces,num_parallel_calls=int(self.num_parallel_calls))
		self.interp_dataset = interp_dataset.batch(400,drop_remainder=True)

		return train_dataset


	def gen_func_cifar10(self):
		# self.output_size = int(28)

		train_images, train_labels, test_images, test_labels = self.cifar10_loader()

		self.test_images = test_images
		self.fid_train_images = train_images

		return train_images

	def dataset_cifar10(self,train_data,batch_size):

		def data_to_grey(image):
			image = tf.image.rgb_to_grayscale(image)
			return image
		
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(10)
		train_dataset = train_dataset.batch(batch_size,drop_remainder=True)
		train_dataset = train_dataset.prefetch(5)

		recon_dataset = tf.data.Dataset.from_tensor_slices((self.test_images))	
		self.recon_dataset = recon_dataset.batch(100,drop_remainder=True)

		interp_dataset = tf.data.Dataset.from_tensor_slices((self.test_images[0:400]))	
		self.interp_dataset = interp_dataset.batch(400,drop_remainder=True)

		return train_dataset




