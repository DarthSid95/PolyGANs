The Optimal GAN Discriminator is a High-dimensional Interpolator.
====================

## Introduction

This is the Code submission accompanying the SIAM SIMODS 2024 Submission "The Optimal Discriminator in Higher-order Gradient-regularized Generative Adversarial Networks." All codes are in Tensorflow2.0 Keras, and can be implemented on either GPU, CPU and on Colab or locally. A public version of this code will be released with the camera-ready version of the paper

## PolyGAN RBF

The central component of PolyGANs is the radial basis function implementation, and can be found in the ``gan_topics.py`` file, defined under the ``RBFSolver`` and ``RBFLayer`` classes. A reviewer interested in getting to the core of these methods is requested to check this part of the code to get insights on the RBF disciminator implementation, weight and centre computation, lambda_d computation, etc. 

## PolyGAN RBF Network Architecture

A slice of the RBF Network code is given below. In TF, the code for builing the RBF would look like this:

```
def discriminator_model_RBF(self):
    if self.gan not in ['WAE'] and self.data not in ['g2','gmm8', 'gN']:
        inputs = tf.keras.Input(shape=(self.output_size,self.output_size,1))
        inputs_res = tf.keras.layers.Reshape(target_shape=[self.output_size*self.output_size])(inputs)
    else:
        inputs = tf.keras.Input(shape=(self.latent_dims,))
        inputs_res = inputs

    ### A FLAG to control 'N', the number of centers
    num_centers = 2*self.N_centers 

    ### A Custom RBFLayer Implementation
    ### Please Check L167 of ``gan_topics.py`` 
    ### Note the argument ``order_m = self.rbf_m``, which is the m in k=2m-n
    Out = RBFLayer(num_centres=num_centers, output_dim=1, order_m = self.rbf_m, batch_size = self.batch_size)(inputs_res)


    ### \lambda_d^* can be computed with another RBF layer
    ### Note the new arguement rbf_pow = -self.rbf_n (cf. Appendix B)
    lambda_term = RBFLayer(num_centres=num_centers, output_dim=1, order_m = self.rbf_m, batch_size = self.batch_size,  rbf_pow = -self.rbf_n)(inputs_res)

    model = tf.keras.Model(inputs=inputs, outputs= [Out,lambda_term])

    return model
```

## PolyGAN Anaconda Environment


This codebase consists of the TensorFlow2.x implementation PolyGAN and PolyGAN-WAE. The baseline comparisons are with multiple WGAN, GMMN and WAE variants as described in the paper. All results were optained when training on TF2.5. Other version of TF might cause instability due to library deprications. Use with care. 

Dependencies can be installed via anaconda. The ``PolyGAN_GPU_TF25.yml`` file list the dependencies to setup the GPU system based environment. To install from the yml file, please run ``conda env create --file PolyGAN_GPU_TF25.yml`` : 

```
GPU accelerated TensorFlow2.0 Environment:
dependencies:
  - cudatoolkit=11.3.1
  - cudnn=8.2.1=cuda11.3_0
  - opencv=3.4.2
  - pip=20.0.2
  - python=3.6.10
  - pytorch=1.4.0=py3.6_cpu_0
  - pip:
    - absl-py==0.9.0
    - clean-fid==0.1.15
    - h5py==2.10.0
    - matplotlib==3.1.3
    - numpy==1.19.5
    - pot==0.8.0
    - tensorflow-addons==0.13.0
    - tensorflow-datasets==4.4.0
    - tensorflow-estimator==2.5.0
    - tensorflow-gpu==2.5.0
    - tensorflow-probability==0.13.0
    - torch==1.10.0+cu113
    - torchaudio==0.10.0+cu113
    - torchvision==0.11.1+cu113
    - tqdm==4.42.1
```
If a GPU is unavailable, the CPU only environment can be built  with ``PolyGAN_CPU.yml``. This setting is meant to run evaluation code;. Training on the CPU environment is not advisable.

```
CPU based TensorFlow2.0 Environment:
- pip=20.0.2
- python=3.6.10
- opencv=3.4.2
- pytorch=1.4.0=py3.6_cpu_0
- pip:
    - absl-py==0.9.0
    - h5py==2.10.0
    - clean-fid==0.1.15
    - ipython==7.15.0
    - ipython-genutils==0.2.0
    - matplotlib==3.1.3
    - numpy==1.18.1
    - scikit-learn==0.22.1
    - scipy==1.4.1
    - pot==0.8.0
    - tensorboard==2.0.2
    - tensorflow-addons==0.6.0
    - tensorflow-datasets==3.0.1
    - tensorflow-estimator==2.0.1
    - tensorflow==2.0.0
    - tensorflow-probability==0.8.0
    - tqdm==4.42.1
    - gdown==3.12
```

If a Conda environment with the required dependecies already exists, or you wish to use your own environment for any particular reason, we suggest making a clone called PolyGAN to maintain consistent code-running experiemnt with the exisitng bash files: ``conda create --name PolyGAN --clone <Your_Env_Name>``

The ``clean-fid`` and ``pot`` libraries are used for metric computation (cf. Appendix E).

Codes were tested locally and on Google Colab. Implementation Jupyter Notebooks to recrete the results shown in the paper will be included with the camera-ready version of the paper. The current version of the code allows implementing locally, the training codes for the various models considered.

In case of runtime issues occuring due to hardware/driver incompatibitlity, please refer the associated user-manuals of NVIDIA CUDA, CudNN, PyTorch or TensorFlow to install dependecies from source.


## Training Data

MNIST and CIFAR-10 are loaded from TensorFlow-Datasets. The CelebA datasets must be downloaded by running the following code:
```
python download_celeba.py
```
Code for downloading LSUN is well document in the official repository: `https://github.com/fyu/lsun`
. The datasets are relative large. Please keep trach of internet bandwidth availability. 


## Training  

The code provides training procedures on MNIST, CIFAR-10, CelebA and LSUN-Churches datasets. We include training codes for all basic experinets presented in the paper.

1) **Running ``train_*.sh`` bash files**: There are two bash files to run an example instacne of trainging: ``train_PolyGAN_GPU.sh`` and ``train_Baseline_GPU.sh``. The fastest was to train a model is by running these bash files. Around 3-4 sample functions calls are provided in the above bash files.  Uncomment the desired command to train for the associated testcase.
```
bash bash_files/train/train_PolyGAN_GPU.sh
bash bash_files/train/train_Baseline_GPU.sh
```
(While code is provided for running experiments on GPU, their CPU counterparts can be made by seting the appropriate flags, as discussed in the next section.)
2) **Manually running ``gan_main.py``**: Aternatively, you can train any model of your choice by running ``gan_main.py`` with custom flags and modifiers. The list of flags and their defauly values are are defined in  ``gan_main.py``.    

3) **Training on Colab**: The experiments in the submission was trained on Google Colab (although it is *not* optimal for training on large models for long durations, pandemic related closures of other compute facilities made the use of Colab unavoidable). For those familiar with the approach, this repository could be cloned to your google drive and steps (1) or (2) could be used for training. CelebA must be downloaded to you current instance on Colab as reading data from GoogleDrive currently causes a Timeout error.  Setting the flags ``--colab 1``,  ``--latex_plot_flag 0``, and ``--pbar_flag 0`` is advisable. The ``colab`` flag modifies CelebA data-handling code to read data from the local folder, rather than ``PolyGANs/data/CelebA/``. The ``latex_plot_flag`` flag removes code dependency on latex for plot labels, since the Colab isntance does not native include this. (Alternatively, you could install texlive_full in your colab instance). Lastly, turning off the ``pbar_flag`` was found to prevent the browser from eating too much RAM when training the model. **The .ipynb file for training on Colab, along with instructions of modifications necessary for running on Colab will be included with the public release of the paper**. 

## Flags

The default values of the various flags used in the bash files are discussed in ``gan_main.py``. While most flags pertain to training parameters, a few control the actula GAN trining algorithm:

1) ``--gan``: Can be set to ``WGAN`` or ``WAE`` based on which set of experiments are to be executed. The corresponding file from ``models/GANs/`` or ``models/WAEs/`` will be imported.
2) ``--topic``: The kind of GAN varinat to consider. ``Base`` calls the class associated with trainig baselines WGAN variants (GP, LP, Rd (or R1 as the oirignal authors call it), Rg (or R2), and ALP. ``PolyGAN`` results in calling those classes associated with PolyGAN training. ``GMMN`` can be called for GMMN-IMQ and GMMN-RBFG variants. To run WAE variants, we set the topic to ``PolyGAN`` or ``WAEMMD`` for the proposed, or baselines variants, respectively. 
3) ``--loss``: Choice of GAN loss. For WGAN, there is ``base``,``GP``, ``LP``,``ALP``, ``R1`` and ``R2`` available for training. The proposed loss variant is ``RBF``. For GMMN, the options are ``RBFG`` (Gaussian Kernel) and ``IMQ`` (Inverse Multiquadratic Kernel). For WAE, the choices are ``RBF``, ``RBFG``, ``IMQ``, ``SW`` (Sliced Wassersten) and ``CW`` (Cramer Wold).
4) ``--mode``: Choose between ``train``, ``test``, or ``metrics``. While ``test`` genetares interpolation and reconstruction results, the ``metrics`` mode allows for evaluating FID, sharpness, etc. 
5) ``--metrics``: A comma seperated string of metrics to compute. Currently, it supports ``FID,sharpness``. When passing multiple, please avoid blank spaces.
6) ``--data``: Target data to train on. Support ``mnist``,``cifar10``,``celeba``, and ``church``. (All lowercase only)
7) ``--latent_dims``: The letent space dimensionality to use in WAE variants. for 2D and nD Gaussians, this is atuomatically assumed to be the dimensionality of the data itself.
8) ``--rbf_m``: The gradient pentaty order, brought to life via the order of the polyharmonic RBF order 2m-n. Optimal performance when m = ceil(n/2)
9) ``--GPU``: A list indicating the visble GPU devices to CUDA. Set to ``0``, ``0,1``, ``<empyt>``, etc. based on compute available.
10) ``--device``: The device that TensorFlow places its variables on. Begins iteration from ``0`` all the way up to ``n``, when `n` devies are proved in ``--GPU``. Set ``--GPU`` as empty string and ``--device`` to ``-1`` to train on CPU. 



----------------------------------
----------------------------------


----------------------------------
----------------------------------
