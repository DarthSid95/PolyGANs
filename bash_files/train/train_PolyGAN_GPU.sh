set -e
if [ "${CONDA_DEFAULT_ENV}" != "PolyGAN" ]; then
	echo 'You are not in the <PolyGAN> environment. Attempting to activate the PolyGAN environment via conda. Please run "conda activate PolyGAN" and try again if this fails. If the PolyGAN environment has not been installed, please refer the README.md file for further instructions.'
	condaActivatePath=$(which activate)
	source ${condaActivatePath} PolyGAN
fi

# An example of running PolyGAN and PolyGAN-WAE
# Data can be changed to mnist, cifar10, celeba and church when the ``gan`` flag is set to WAE
# Data can be changed to g2, gmm8, gN when the ``gan`` flag is set to WGAN
# Check paper for appropriate value for latent_dims
# We suggest setting rbf_m to ceil(n/2)


### Train call for PolyGAN on g2 learning 
python ./gan_main.py  --run_id 'new' --resume 0 --GPU '0' --device '0' --topic 'PolyGAN' --mode 'train' --data 'g2' --noise_kind 'gaussian' --gan 'WGAN' --loss 'RBF' --arch 'dense' --latent_dims 2 --saver 1 --num_epochs 5 --res_flag 1 --lr_G 0.01 --lr_D 0.0 --paper 1 --batch_size '500' --metrics 'W22,GradGrid' --colab 0 --pbar_flag 1 --latex_plot_flag 0 --rbf_m 1 

### Train call for PolyGAN for gmm8 learning
python ./gan_main.py  --run_id 'new' --resume 0 --GPU '0' --device '0' --topic 'PolyGAN' --mode 'train' --data 'gmm8' --noise_kind 'gaussian' --gan 'WGAN' --loss 'RBF' --arch 'dense' --latent_dims 2 --saver 1 --num_epochs 40 --res_flag 1 --lr_G 0.01 --lr_D 0.0 --paper 1 --batch_size '500' --metrics 'W22,GradGrid' --colab 0 --pbar_flag 1 --latex_plot_flag 0 --rbf_m 1 

### Train call for PolyGAN for gN learning
python ./gan_main.py  --run_id 'new' --resume 0 --GPU '0' --device '0' --topic 'PolyGAN' --mode 'train' --data 'gN' --GaussN 6 --noise_kind 'gaussian' --gan 'WGAN' --loss 'RBF' --arch 'dcgan' --latent_dims 11 --saver 1 --num_epochs 40 --res_flag 1 --lr_G 0.05 --lr_D 0.0 --paper 1 --batch_size '500' --metrics 'W22'--colab 0 --pbar_flag 1 --latex_plot_flag 0 --rbf_m 3

### Train call for PolyGAN for image-space matching with mnist
python ./gan_main.py  --run_id 'new' --resume 0 --GPU '0' --device '0' --topic 'PolyGAN' --mode 'train' --data 'mnist' --noise_kind 'gaussian' --gan 'WGAN' --loss 'RBF' --arch 'dcgan' --latent_dims 784 --saver 1 --num_epochs 250 --res_flag 1 --lr_G 0.05 --lr_D 0.0 --paper 1 --batch_size '100' --metrics 'FID' --FID_kind 'clean'  --colab 0 --pbar_flag 0 --latex_plot_flag 0 --rbf_m 392 

### Train call for PolyGAN-WAE on cifar-10
 python ./gan_main.py  --run_id 'new' --resume 0 --GPU '0' --device '0' --topic 'PolyGAN' --mode 'train' --data 'cifar10' --latent_dims 64 --noise_kind 'gaussian' --gan 'WAE' --loss 'RBF' --arch 'dcgan' --saver 1 --num_epochs 250 --res_flag 1 --lr_G 0.0025 --lr_D 0.00 --lr_AE_Enc 0.0075 --lr_AE_Dec 0.0075 --paper 1 --batch_size '100' --metrics 'KID,FID,recon' --FID_kind 'clean' --models_for_metrics 1 --colab 0 --pbar_flag 1 --latex_plot_flag 0 --rbf_m 32 --AE_pretrain_epochs 0 --GAN_pretrain_epochs 2
