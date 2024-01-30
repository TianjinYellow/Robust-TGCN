#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH --cpus-per-task=10
#SBATCH -o ./results/hk_totalclean.out


source /home/liuyu/miniconda3/etc/profile.d/conda.sh
conda activate RTGCN

module purge
module load 2021
module load CUDA/11.3.1



for n in 3 12
do

python main.py --pre_len $n --model_name TGCN --max_epochs 5000 --learning_rate 0.001 --weight_decay 0.01 --batch_size 32 --hidden_dim 128 --loss mse_with_regularizer --settings supervised --gpus 1 --kl_gamma 5e-6 --data hk
done


