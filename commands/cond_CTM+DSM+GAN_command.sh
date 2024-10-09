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

MODEL_FLAGS="--global_batch_size=528 --microbatch=11 --class_cond=True --start_ema=0.999 --gan_different_augment=True --save_interval=1000 --eval_interval=1000 --eval_fid=False --eval_similarity=True --check_dm_performance=False --compute_ema_fids=False --gan_fake_inner_type=model --gan_fake_outer_type=target_model_sg --gan_training=True --g_learning_period=2"
CKPT_FLAGS="--out_dir /groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/GAN/cond/GAN_bs_528_ema_0.999_diff_aug/ --resume_checkpoint=/groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/GAN/cond/GAN_bs_264/model027000.pt --teacher_model_path=/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/author_ckpt/edm-cifar10-32x32-cond-vp.pkl --data_dir=/home/acf15618av/dataset/CIFAR10-cond"

mpiexec -n 4 python3.11 cm_train.py $MODEL_FLAGS $CKPT_FLAGS