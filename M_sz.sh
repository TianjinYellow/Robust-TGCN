#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 4-00:00:00
#SBATCH --cpus-per-task=10
#SBATCH -o ./results/M_sz15min.out


source /home/liuyu/miniconda3/etc/profile.d/conda.sh
conda activate RTGCN

module purge
module load 2021
module load CUDA/11.3.1


for noise_t in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
python main.py --model_name TGCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 1e-4 --noise_type missing --batch_size 32 --hidden_dim 100 --loss mse_with_regularizer --pre_len 1 --settings supervised --gpus 1 --kl_gamma 1e-5 --data shenzhen --noise_test --noise_sever 2 --noise_ratio $noise_t --noise_ratio_node 0.4
done



for noise_n in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
python main.py --pre_len 1 --model_name TGCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 1e-4 --noise_type missing --batch_size 32 --hidden_dim 100 --loss mse_with_regularizer --settings supervised --gpus 1 --kl_gamma 1e-5 --data shenzhen --noise_test --noise_sever 2 --noise_ratio 0.4 --noise_ratio_node $noise_n
done

