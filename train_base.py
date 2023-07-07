# -*- coding: utf-8 -*-
import sys
import numpy as np
import random
import os
import argparse
import time
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import BertTokenizer

from utils_re import set_seed
from dataset_re import REDataset, load_json_file, get_rel2id
from bert_encoder import BERTSentenceEncoder
from validate_test import validate, test

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # begin{argument}
    parser = argparse.ArgumentParser(description='Train a Classifier and Detector', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, choices=['fewrel'], default='fewrel', help='Dataset.')
    parser.add_argument('--model', '-m', type=str, default='bert-base-uncased', choices=['bert-base-uncased'], help='Choose architecture.')
    # Input File
    parser.add_argument('--train_file', type=str, default='')
    parser.add_argument('--dev_file', type=str, default='')
    parser.add_argument('--test_file', type=str, default='')
    parser.add_argument('--id_relations_file', type=str, default='')
    parser.add_argument('--dev_ood_relations_file', type=str, default='')
    parser.add_argument('--test_ood_relations_file', type=str, default='')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=8, help='Number of epochs to train.')
    parser.add_argument('--step_size', type=int, default=4, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-5, help='The initial learning rate.')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='Batch size.')
    parser.add_argument('--max_len', '-ml', type=int, default=128, help='Max length.')
    parser.add_argument('--test_bs', type=int, default=16)
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0001, help='Weight decay (L2 penalty).')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--tem', type=float, default=1.0)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--confidence_type', type=str, default='')
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default='./checkpoints/best.pt', help='Folder to save checkpoints.')
    # Acceleration
    parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
    parser.add_argument('--seed', type=int, default=42)

    # parse
    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}
    print(state)
    set_seed(args.seed)
    # end{argument}

    # begin{dataset}
    rel2id, num_classes = get_rel2id(args.id_relations_file, args.dev_ood_relations_file, args.test_ood_relations_file)
    tokenizer = BertTokenizer.from_pretrained(args.model)

    train_data = load_json_file(args.train_file)
    train_dataset = REDataset(train_data, args.max_len, tokenizer, num_classes, rel2id)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    dev_data = load_json_file(args.dev_file)
    dev_dataset = REDataset(dev_data, args.max_len, tokenizer, num_classes, rel2id)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)

    test_data = load_json_file(args.test_file)
    test_dataset = REDataset(test_data, args.max_len, tokenizer, num_classes, rel2id)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    test_id, test_ood = 0, 0
    for data in test_data:
        if rel2id[data['relation']] < num_classes:
            test_id += 1
        else:
            test_ood += 1
    print(num_classes, rel2id)
    print('TRAIN:', len(train_dataset))
    print('DEV:', len(dev_dataset))
    print('TEST:', len(test_dataset))
    print('ID:', test_id)
    print('OOD:', test_ood)
    # end{dataset}

    # begin{model}
    net = BERTSentenceEncoder(args.model, num_classes, args.hidden_dim, args.tem).cuda()
    optimizer = torch.optim.AdamW(
        net.parameters(),
        args.learning_rate,
        # weight_decay=args.decay
    )
    # end{model}

    print('Beginning Training\n')
    best_score = -1
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs+1):
        begin_epoch = time.time()

        net.train()
        for batch_id, (unique_id, input_ids, input_mask, target) in enumerate(train_dataloader):
            optimizer.zero_grad()

            input_ids, input_mask, target = input_ids.cuda(), input_mask.cuda(), target.cuda()
            x, feature = net.forward_virtual(input_ids, input_mask)

            # cross entropy
            loss = loss_func(x, target)
            # backward
            loss.backward()
            optimizer.step()
            if batch_id % 20 == 0:
                print('[TRAIN] epoch: {0:4} step: {1:4} | loss: {2:2.5f}'.format(epoch, batch_id, loss.item()))

        dev_auroc, dev_fpr95, thresh = validate(net, dev_dataloader, args.confidence_type, num_classes)
        test(net, test_dataloader, args.confidence_type, thresh, num_classes)

        if dev_auroc > best_score:
            best_score = dev_auroc
            torch.save(net.state_dict(), args.save)
            print('save model...')

        print('Epoch: {0:3d} | Time: {1:5d}'.format(
            epoch,
            int(time.time() - begin_epoch)
        ))

    print('Beginning Testing\n')
    net.load_state_dict(torch.load(args.save))
    dev_auroc, dev_fpr95, thresh = validate(net, dev_dataloader, args.confidence_type, num_classes)
    test(net, test_dataloader, args.confidence_type, thresh, num_classes)
    