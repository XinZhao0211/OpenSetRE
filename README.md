# OpenSetRE
This is the code for ACL2023 "Open Set Relation Extraction via Unknown-Aware Training".


## 数据
我们使用了Fewrel2.0和TACRED数据集，
数据见目录```data/{fewrel2.0/tacred}```。
```
relations.json          # 所有关系列表
id_relations.json       # ID关系列表
dev_ood_relations.json  # 验证集中OOD关系列表
test_ood_relations.json # 测试集中OOD关系列表
train_dp.json           # 训练集数据
dev.json                # 验证集数据
test.json               # 测试集数据
```
其中给出了数据格式的demo。


## 训练
训练脚本为```train_base.sh```和```train_ours.sh```。

```train_base.sh```使用交叉墒损失训练K分类模型，作为我们方法的基座模型。

```train_ours.sh```用于训练我们提出的模型。

参数介绍
```
--train_file               data/${dataset}/train_dp.json \
--dev_file                 data/${dataset}/dev.json \
--test_file                data/${dataset}/test.json \
--id_relations_file        data/${dataset}/id_relations.json \
--dev_ood_relations_file   data/${dataset}/dev_ood_relations.json \
--test_ood_relations_file  data/${dataset}/test_ood_relations.json \
--load 基座模型路径 \
--save 模型保存路径 \
--confidence_type energy \
--epochs 1 \
--learning_rate 3e-5 \
--batch_size 16 \
--max_len 128 \
--hidden_dim 256 \
--tem 1.0 \
--replace_ratio token替换比例 \
--loss_weight 损失权重 \
--seed 42
```
