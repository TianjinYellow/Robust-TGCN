#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 3-00:00:00
#SBATCH --cpus-per-task=10
#SBATCH -o ./results/META-LA_RTGCN_0.10.4.out


source /home/liuyu/miniconda3/etc/profile.d/conda.sh
conda activate RTGCN

module purge
module load 2021
module load CUDA/11.3.1



for noise_n in 0.1 0.4
do

python main.py --pre_len 12 --model_name TGCN --max_epochs 5000 --learning_rate 0.001 --weight_decay 0.01 --batch_size 32 --hidden_dim 128 --loss mse_with_regularizer --settings supervised --gpus 1 --kl_gamma 5e-6 --data MeterLA --noise_test --noise_ratio 0.08 --noise_sever 8 --noise_ratio_test $noise_n --noise_ratio_node_test 0.1 --data MeterLA --traffic_df_filename ./data/metr-la.h5  --sensor_ids_filename ./data/sensor_graph/graph_sensor_ids.txt --distances_filename data/sensor_graph/distances_la_2012.csv --normalized_k 0.1 --noise 1 --dim_input 2


done


#for noise_n in 0 0.1 0.2 0.3 0.4 0.5 0.6
#do
#python main.py --pre_len 3 --model_name TGCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 1e-5 --noise_type missing --batch_size 32 --hidden_dim 64 --loss mse_with_regularizer --settings supervised --gpus 1 --kl_gamma 1e-5 --data losloop --noise_test --noise_sever 2 --noise_ratio 0.4 --noise_ratio_node $noise_n
#done

#python main.py --pre_len 12 --model_name TGCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0.01 --batch_size 32 --hidden_dim 128 --loss mse_with_regularizer --settings supervised --gpus 1 --kl_gamma 5e-6 --data MeterLA --noise_test --noise_ratio 0.08 --noise_sever 8 --noise_ratio_test $noise_n --noise_ratio_node_test 0.1 --data PemsBAY --traffic_df_filename ./data/pems-bay.h5 --sensor_ids_filename ./data/sensor_graph/adj_mx_bay.pkl --distances_filename data/sensor_graph/graph_sensor_locations.csv --normalized_k 0.1 --noise 2 --dim_input 2
