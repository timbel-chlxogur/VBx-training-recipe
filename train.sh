# @Created Time:   2019-03-02
# @Author: Shuai Wang
# @Email: wsstriving@gmail.com
# @Last Modified Time: 2019-08-18

#!/bin/bash

# if [ "$#" -ne 2 ]; then
#   echo "Illegal number of parameters. Expecting egs directory and output directory."
#   exit 1
# fi

if [ "$#" -lt 2 -o "$#" -gt 3 ]; then
  echo "Usage: $0 <train_egs_dir> [<val_egs_dir>] <out_dir>"
  exit 1
fi

# egs_dir=$1
# out_dir=$2

train_egs_dir=$1
if [ "$#" -eq 3 ]; then
  val_egs_opt="--val-egs-dir $2"
  out_dir=$3
else               # val egs 미지정 → train dir의 diagnostic egs 사용(현행 동작)
  val_egs_opt=""
  out_dir=$2
fi

AVAIL_GPUS=$(local/utils/free-gpus.sh)
# ngpus=2 TODO : 태혁
ngpus=1

# linear | arc_margin | sphere |add_margin
metric="add_margin"
model="ResNet101"
embed_dim=256

export CUDA_VISIBLE_DEVICES=${AVAIL_GPUS:0:$((${ngpus}*2-1))}

# horovod stuff
export NCCL_SOCKET_IFNAME=bond0.6
export CUDA_LAUNCH_BLOCKING=1
export HOROVOD_GPU_ALLREDUCE=NCCL

. ./path.sh


# final model trained using 6x RTX 2080Ti, modify batchsize accordingly TODO : 태혁
# horovodrun -np ${ngpus} --mpi --autotune \
#     export CUDA_VISIBLE_DEVICES=0 \
#     python local/train_pytorch_dnn.py --model ${model} \
#         --num-targets 8178 \
#         --dir ${out_dir}/${model}_${metric}_embed${embed_dim}_${ngpus}gpu \
#         --metric ${metric} \
#         --egs-dir ${egs_dir} \
#         --minibatch-size 12 \
#         --embed-dim ${embed_dim} \
#         --warmup-epochs 0 \
#         --initial-effective-lrate 0.01 \
#         --final-effective-lrate 0.00005 \
#         --initial-margin-m 0.05 \
#         --final-margin-m 0.2 \
#         --optimizer SGD \
#         --momentum 0.9 \
#         --optimizer-weight-decay 0.0001 \
#         --preserve-model-interval 30 \
#         --num-epochs 3 \
#         --apply-cmn no \
#         --fix-margin-m 2

train_egs=exp/egs_train
num_targets=$(wc -l < ${train_egs}/info/spk2int)

export CUDA_VISIBLE_DEVICES=0
# python local/train_pytorch_dnn.py --model ${model} \
#     --num-targets ${num_targets} \
#     --dir ${out_dir}/${model}_${metric}_embed${embed_dim}_${ngpus}gpu \
#     --metric ${metric} \
#     --egs-dir ${train_egs_dir} \
#     ${val_egs_opt} \
#     --minibatch-size 32 \
#     --embed-dim ${embed_dim} \
#     --warmup-epochs 1 \
#     --initial-effective-lrate 0.1 \
#     --final-effective-lrate 0.00005 \
#     --initial-margin-m 0.05 \
#     --final-margin-m 0.2 \
#     --optimizer SGD \
#     --momentum 0.9 \
#     --optimizer-weight-decay 0.0001 \
#     --preserve-model-interval 30 \
#     --num-epochs 10 \
#     --apply-cmn no \
#     --fix-margin-m 6 \
#     --trials-path data/all_combined_aug_and_clean_valid/cosine_trials.txt

python local/train_pytorch_dnn.py --model ${model} \
    --num-targets ${num_targets} \
    --dir ${out_dir}/${model}_${metric}_embed${embed_dim}_${ngpus}gpu \
    --metric ${metric} \
    --egs-dir ${train_egs_dir} \
    ${val_egs_opt} \
    --minibatch-size 32 \
    --embed-dim ${embed_dim} \
    --warmup-epochs 1 \
    --initial-effective-lrate 0.1 \
    --final-effective-lrate 0.00005 \
    --initial-margin-m 0.05 \
    --final-margin-m 0.15 \
    --optimizer SGD \
    --momentum 0.9 \
    --optimizer-weight-decay 0.0001 \
    --preserve-model-interval 30 \
    --num-epochs 10 \
    --apply-cmn no \
    --fix-margin-m 4 \
    --trials-path data/all_combined_aug_and_clean_valid/cosine_trials.txt

