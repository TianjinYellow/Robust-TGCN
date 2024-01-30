@echo off
for %%p in (1,2,3,4) do (
    for %%n in (0,0.1,0.2,0.3,0.4,0.5) do (
        python main.py --model_name TGCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --hidden_dim 64 --loss mse_with_regularizer --pre_len %%p --settings supervised --gpus 1 --kl_gamma 0.0 --data shenzhen --noise --noise_sever 2 --noise_ratio %%n --noise_ratio_node 0.2
    )
)
for %%p in (1,2,3,4) do (
    for %%n in (0,0.1,0.2,0.3,0.4,0.5) do (
        python main.py --pre_len %%p --model_name TGCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --hidden_dim 64 --loss mse_with_regularizer --settings supervised --gpus 1 --kl_gamma 0.0 --data shenzhen --noise --noise_sever 2 --noise_ratio 0.2 --noise_ratio_node %%n
    )
)
pause