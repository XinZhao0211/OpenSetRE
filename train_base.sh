for dataset in fewrel2.0
do
  for confidence_type in msp
  do
    for seed in 0
    do
      CUDA_VISIBLE_DEVICES=1 python -u train_base.py \
             --train_file         data/${dataset}/train.json \
             --dev_file           data/${dataset}/dev.json \
             --test_file          data/${dataset}/test.json \
             --id_relations_file  data/${dataset}/id_relations.json \
             --dev_ood_relations_file     data/${dataset}/dev_ood_relations.json \
             --test_ood_relations_file     data/${dataset}/test_ood_relations.json \
             --save checkpoints/${dataset}/${confidence_type}_seed${seed}.pt \
             --confidence_type ${confidence_type} \
             --epochs 5 \
             --learning_rate 3e-5 \
             --batch_size 32 \
             --max_len 128 \
             --hidden_dim 256 \
             --tem 1.0 \
             --mask_ratio 0.25 \
             --seed ${seed} \
             >log/${dataset}/${confidence_type}_seed${seed}.out

    done
  done
done