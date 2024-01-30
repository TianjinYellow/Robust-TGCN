#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00
#SBATCH --cpus-per-task=10
#SBATCH -o ./results/para1.out


source /home/liuyu/miniconda3/etc/profile.d/conda.sh
conda activate RTGCN

module purge
module load 2021
module load CUDA/11.3.1
cd ..

for w in  5e-4 1e-3 5e-3 1e-2 5e-2 1e-1
do

python main.py --model_name TGCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0.1 --noise_type missing --batch_size 32 --hidden_dim 64 --loss mse_with_regularizer --pre_len 3 --settings supervised --gpus 1 --kl_gamma $w --data losloop --noise_test --noise_sever 8 --noise_ratio_test $noise_t --noise_ratio_node_test 0.1 --noise --noise_ratio 0.1 --noise_ratio_node 0.1
done
