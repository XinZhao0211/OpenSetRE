for dataset in tacred
do
  for seed in 42
  do
    for replace_ratio in 0.20
    do
      CUDA_VISIBLE_DEVICES=7 python -u train_ours.py \
             --train_file               data/${dataset}/train_dp.json \
             --dev_file                 data/${dataset}/dev.json \
             --test_file                data/${dataset}/test.json \
             --id_relations_file        data/${dataset}/id_relations.json \
             --dev_ood_relations_file   data/${dataset}/dev_ood_relations.json \
             --test_ood_relations_file  data/${dataset}/test_ood_relations.json \
             --load checkpoints/${dataset}/energy_seed${seed}.pt \
             --save checkpoints/${dataset}/ours_weight0.1_ratio${replace_ratio}_seed${seed}.pt \
             --confidence_type energy \
             --epochs 1 \
             --learning_rate 2e-5 \
             --batch_size 16 \
             --max_len 128 \
             --hidden_dim 256 \
             --tem 1.0 \
             --replace_ratio ${replace_ratio} \
             --loss_weight 0.05 \
             --seed ${seed}
             
    done
  done
done