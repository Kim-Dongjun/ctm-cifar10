#!/bin/bash

#$-l rt_F=1
#$ -l h_rt=168:00:00
#$-j y
#$-cwd

source /etc/profile.d/modules.sh

module load python/3.11/3.11.2
module load cuda/11.7/11.7.1
module load cudnn/8.9/8.9.2
module load nccl/2.14/2.14.3-1
module load hpcx/2.12

MODEL_FLAGS="--global_batch_size=528 --microbatch=11 --gan_different_augment=True --start_ema=0.9999 --save_interval=1000 --eval_interval=1000 --eval_fid=False --eval_similarity=True --check_dm_performance=True --compute_ema_fids=True --gan_fake_inner_type=model --gan_fake_outer_type=target_model_sg --gan_training=True --g_learning_period=2"
CKPT_FLAGS="--out_dir /groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/GAN/uncond/GAN_bs_528_ema_0.9999_diff_aug/ --resume_checkpoint=/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/resume_checkpoint/euler_edm_stop_grad_random_17_M_bs_384_from_edm_dsm_1.0/model050000.pt"

mpiexec -n 4 python3.11 cm_train.py $MODEL_FLAGS $CKPT_FLAGS