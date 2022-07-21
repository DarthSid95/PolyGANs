set -e
if [ "${CONDA_DEFAULT_ENV}" != "PolyGAN" ]; then
	echo 'You are not in the <PolyGAN> environment. Attempting to activate the PolyGAN environment via conda. Please run "conda activate PolyGAN" and try again if this fails. If the PolyGAN environment has not been installed, please refer the README.md file for further instructions.'
	condaActivatePath=$(which activate)
	source ${condaActivatePath} PolyGAN
fi

# An example of running WGAN code on gmm8 for baselines:
# Losses can be chaned to GP, LP, ALP, R1 and R2.
# Data can be changed to g2, gmm8 

### Train call for Base WGAN-LP or GMMN-RBFG on gmm8
python ./gan_main.py  --run_id 'new' --resume 0 --GPU '2' --device '0' --topic 'Base' --mode 'train' --data 'gmm8' --latent_dims 1 --noise_kind 'gaussian' --gan 'WGAN' --loss 'LP' --arch 'dense' --saver 1 --num_epochs 50 --res_flag 1 --lr_G 0.002 --lr_D 0.0075 --Dloop 1 --paper 1 --batch_size '500' --Dloop 1 --metrics 'W22,GradGrid' --colab 0 --pbar_flag 1 --latex_plot_flag 0 




# An example of running GMMN code on gmm8 for baselines:
# Losses can be chaned to IMQ, RBFG.
# Data can be changed to g2, gmm8 

### Train call for Base GMMN-RBFG on gmm8
python ./gan_main.py  --run_id 'new' --resume 0 --GPU '0' --device '0' --topic 'GMMN' --mode 'train' --data 'gmm8' --latent_dims 1 --noise_kind 'gaussian' --gan 'WGAN' --loss 'RBFG' --arch 'dense' --saver 1 --num_epochs 40 --res_flag 1 --lr_G 0.002 --lr_D 0.0 --Dloop 1 --paper 1 --batch_size '500' --Dloop 1 --metrics 'W22,GradGrid' --colab 0 --pbar_flag 1 --latex_plot_flag 0 


# An example of running WAEMMD code on gmm8 for baselines:
# Losses can be chaned to IMQ, RBFG, SW (for SWAE) and CW (for CWAE).
# Data can be changed to mnist, cifar10, celeba and church
# Check paper for appropriate value for latent_dims

### Train call for WAEMMD with CWAE loss on CIFAR-10
 python ./gan_main.py  --run_id 'new' --resume 0 --GPU '0' --device '0' --topic 'WAEMMD' --mode 'train' --data 'cifar10' --latent_dims 64 --noise_kind 'gaussian' --gan 'WAE' --loss 'CW' --arch 'dcgan' --saver 1 --num_epochs 250 --res_flag 1 --lr_G 0.005 --lr_D 0.00 --lr_AE_Enc 0.002 --lr_AE_Dec 0.002 --paper 1 --batch_size '100' --metrics 'KID,FID,recon' --FID_kind 'clean' --models_for_metrics 1 --colab 0 --pbar_flag 1 --latex_plot_flag 0 --AE_pretrain_epochs 0 --GAN_pretrain_epochs 2
