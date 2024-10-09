#!/bin/bash

#$-l rt_F=1
#$ -l h_rt=168:00:00
#$-j y
#$-cwd

qrsh -g gce50978 -l rt_F=1 -l h_rt=12:00:00
source /etc/profile.d/modules.sh

module load python/3.11/3.11.2
module load cuda/11.7/11.7.1
module load cudnn/8.9/8.9.2
module load nccl/2.14/2.14.3-1
module load intel-mpi/2021.8

MODEL_FLAGS="--start_ema=0.999 --save_check_period=1000 --eval_interval=5000 --eval_fid=True --eval_similarity=False --check_dm_performance=False --compute_ema_fids=True --gan_fake_inner_type=model --gan_fake_outer_type=target_model_sg --gan_training=True --g_learning_period=2 --resume_checkpoint=/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/resume_checkpoint/euler_edm_stop_grad_random_17_M_bs_384_from_edm_dsm_1.0/model050000.pt --microbatch=11"

python3.11 image_sample.py $MODEL_FLAGS --out_dir /groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/GAN/uncond/GAN_bs_264_ema_0.9999_diff_aug --model_path=/groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/GAN/uncond/GAN_bs_264_ema_0.9999/model081000.pt --eval_num_samples=50000 --batch_size=1000 --stochastic_seed=True --save_format=npz --ind_1=1 --ind_2=0 --class_cond=False --use_MPI=True --sampler=exact --sampling_steps=1 --device_id=1


