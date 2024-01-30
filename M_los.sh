#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 3-00:00:00
#SBATCH --cpus-per-task=10
#SBATCH -o ./losloop_varyLayers1.out


source /home/liuyu/miniconda3/etc/profile.d/conda.sh
conda activate RTGCN

module purge
module load 2021
module load CUDA/11.3.1


# for noise_n in 0.1 0.2 0.3 0.4 0.5
# do
# python main.py --pre_len 3 --model_name TGCN --noise_type missing --max_epochs 3000 --learning_rate 0.001 --weight_decay 0.01 --batch_size 32 --hidden_dim 128 --loss mse_with_regularizer --settings supervised --gpus 1 --kl_gamma 5e-6 --data hk --noise_test --noise --noise_ratio 0.08 --noise_sever 8 --noise_ratio_test $noise_n --noise_ratio_node_test 0.1
# done
for n in 6 8 10
do
for noise_n in 0
do
python main.py --pre_len 3 --model_name TGCN --noise_type missing --N_layers $n --max_epochs 3000 --learning_rate 0.001 --weight_decay 0.01 --batch_size 32 --hidden_dim 128 --loss mse_with_regularizer --settings supervised --gpus 1 --kl_gamma 5e-6 --data losloop --noise_test --noise 1 --noise_ratio 0.08 --noise_sever 8 --noise_ratio_test 0.1 --noise_ratio_node_test $noise_n
done
done


