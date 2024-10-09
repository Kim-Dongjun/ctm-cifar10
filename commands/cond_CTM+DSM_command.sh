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

MODEL_FLAGS="--save_interval=10000 --check_dm_performance=False --eval_fid=True --eval_similarity=False --eval_interval=10000 --microbatch 16 --global_batch_size=256 --class_cond=True --teacher_model_path=/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/author_ckpt/edm-cifar10-32x32-cond-vp.pkl --data_dir=/home/acf15618av/dataset/CIFAR10-cond"

mpiexec -n 4 python3.11 cm_train.py $MODEL_FLAGS --out_dir /groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/ablation-V100/CTM/cond/default/