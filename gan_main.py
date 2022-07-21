from __future__ import print_function
import os, sys, time, argparse, signal, json, struct
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.python import debug as tf_debug
import traceback

print(tf.__version__)
from absl import app
from absl import flags


def signal_handler(sig, frame):
	print('\n\n\nYou pressed Ctrl+C! \n\n\n')
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

'''Generic set of FLAGS.'''
FLAGS = flags.FLAGS
flags.DEFINE_float('lr_G', 0.0001, """learning rate for generator""")
flags.DEFINE_float('lr_D', 0.0001, """learning rate for discriminator""")
flags.DEFINE_float('beta1', 0.5, """beta1 for Adam""")
flags.DEFINE_float('beta2', 0.9, """beta2 for Adam""")
flags.DEFINE_integer('colab', 0, """ set 1 to run code in a colab friendy way """)
flags.DEFINE_integer('batch_size', 100, """Batch size.""")
flags.DEFINE_integer('paper', 1, """1 for saving images for a paper""")
flags.DEFINE_integer('resume', 1, """1 vs 0 for Yes Vs. No""")
flags.DEFINE_integer('saver', 1, """1-Save events for Tensorboard. 0 O.W.""")
flags.DEFINE_integer('res_flag', 1, """1-Write results to a file. 0 O.W.""")
flags.DEFINE_integer('pbar_flag', 1, """1-Display Progress Bar, 0 O.W.""")
flags.DEFINE_integer('latex_plot_flag', 1, """1-Plot figs with latex, 0 O.W.""")
flags.DEFINE_integer('out_size', 32, """Image output reshape size""")
flags.DEFINE_list('metrics', '', 'CSV for the metrics to evaluate. KLD, FID, PR')
flags.DEFINE_integer('save_all', 0, """1-Save all the models. 0 for latest 10""") #currently functions as save_all internally
flags.DEFINE_integer('seed', 42, """Initialize the random seed of the run (for reproducibility).""")
flags.DEFINE_integer('num_epochs', 200, """Number of epochs to train for.""")
flags.DEFINE_integer('Dloop', 1, """Number of loops to run for D.""")
flags.DEFINE_integer('Gloop', 1, """Number of loops to run for G.""")
flags.DEFINE_integer('num_parallel_calls', 5, """Number of parallel calls for dataset map function""")
flags.DEFINE_string('run_id', 'default', """ID of the run, used in saving.""")
flags.DEFINE_string('log_folder', 'default', """ID of the run, used in saving.""")
flags.DEFINE_string('mode', 'train', """Operation mode: train, test, fid """)
flags.DEFINE_string('topic', 'PolyGAN', """Base, PolyGAN, GMMN, WAEMMD or WAE""")
flags.DEFINE_string('data', 'mnist', """Type of Data to run for - g2, gN, gmm8, mnist, celeba, cifar10 and church""")
flags.DEFINE_string('gan', 'wgan', """Type of GAN for""")
flags.DEFINE_string('loss', 'RBF', """Type of Loss function to use - RBF for PolyGAN""")
flags.DEFINE_string('GPU', '0', """GPU's made visible '0', '1', or '0,1' """)
flags.DEFINE_string('device', '0', """Which GPU device to run on: 0,1 or -1(CPU)""")
flags.DEFINE_string('noise_kind', 'gaussian', """Type of Noise for WAE latent prior or for SpiderGAN""")
flags.DEFINE_string('arch', 'dcgan', """dense vs resnet vs dcgan""")

flags.DEFINE_integer('celeba_size', 64, """ Output size for CelebA data""")
flags.DEFINE_integer('cifar10_size', 32, """ Output size for CIFAR-10 data""")
flags.DEFINE_integer('church_size', 64, """ Output size for LSUN Churches data""")
flags.DEFINE_integer('sn_flag', 1, """ set 1 to use spectral normalization """)


# ''' Flags for PolyGAN RBF'''
flags.DEFINE_integer('rbf_m', 2, """Gradient order for RBF. The m in k=2m-n [2]""") #
flags.DEFINE_integer('GaussN', 3, """ N for Gaussian [3] """)
flags.DEFINE_integer('N_centers', 100, """ N for number of centres in PolyRBF [100]""")

# GAN pretraining for WAE, AE pretraining, etc.
flags.DEFINE_integer('GAN_pretrain_epochs', 0, """Num of GAN pre-training Epochs, if needed [0]""")
flags.DEFINE_integer('AE_pretrain_epochs', 0, """Num of AE pre-training Epochs, if needed [0]""")
flags.DEFINE_string('FID_kind', 'clean', """default or clean (uses clean-FID) [clean]""")

flags.DEFINE_float('lr_AE_Enc', 0.005, """learning rate""")
flags.DEFINE_float('lr_AE_Dec', 0.005, """learning rate""")


FLAGS(sys.argv)
from models import *


if __name__ == '__main__':
	'''Enable Flags and various tf declarables on GPU processing '''
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU #'0' or '0,1', or '0,1,2' or '1,2,3'
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	print('Visible Physical Devices: ',physical_devices)
	for gpu in physical_devices:
		print(gpu)
		tf.config.experimental.set_memory_growth(gpu, True)
	tf.config.threading.set_inter_op_parallelism_threads(6)
	tf.config.threading.set_intra_op_parallelism_threads(6)

	
	# Level | Level for Humans | Level Description                  
	# ------|------------------|------------------------------------ 
	# 0     | DEBUG            | [Default] Print all messages       
	# 1     | INFO             | Filter out INFO messages           
	# 2     | WARNING          | Filter out INFO & WARNING messages 
	# 3     | ERROR            | Filter out all messages
	tf.get_logger().setLevel('ERROR')
	# tf.debugging.set_log_device_placement(True)
	# if FLAGS.colab and FLAGS.data == 'celeba':
	os.environ["TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD"] = "500G"
	if FLAGS.colab:
		import warnings
		warnings.filterwarnings("ignore")



	''' Set random seed '''
	np.random.seed(FLAGS.seed)
	tf.random.set_seed(FLAGS.seed)

	FLAGS_dict = FLAGS.flag_values_dict()

	###	EXISTING Variants:
	##	(1) WGAN - 
	##		(A) Base
	##		(B) PolyGAN
	##		(C) GMMN
	##
	##	(4) WAE - 
	##		(A) Base
	##		(B) PolyGAN
	##		(C) WAEMMD
	### -----------------


	gan_call = FLAGS.gan + '_' + FLAGS.topic + '(FLAGS_dict)'

	print('trying')
	gan = eval(gan_call)
	gan.initial_setup()
	gan.get_data()
	gan.create_models()
	gan.create_optimizer()
	gan.create_load_checkpoint()
	print('Setup Completed')

	if gan.mode == 'train':
		print(gan.mode)
		gan.train()
		if gan.data not in [ 'g2', 'gN', 'gmm8']:
			gan.test()
	if gan.mode == 'h5_from_checkpoint':
		gan.h5_from_checkpoint()
	if gan.mode == 'test':
		gan.test()
	if gan.mode == 'metrics':
		gan.eval_metrics()


###############################################################################  
	
	
	print('Completed.')
