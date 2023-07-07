# -*- coding: utf-8 -*-
import sys
import numpy as np
import random
import copy
import os
import argparse
import time
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.backends import cudnn

from utils_re import set_seed
from dataset_re import REDataset, load_json_file, get_rel2id, load_out_vocab, convert_bert_input
from bert_encoder import BERTSentenceEncoder
from validate_test import validate, test

import warnings
warnings.filterwarnings('ignore')


def tokenize_by_word(words, tokenizer):
    sub_words = []
    keys = []
    index = 0
    for word in words:
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append((index, index + len(sub)))
        index += len(sub)

    return words, sub_words, keys


def find_replace_words(gradient, word_embeddings, sub_word_dict):
    sub_word_cnt = gradient.shape[0]

    word_subwordids = [(word_candidate, sub_word_candidate[1]) for word_candidate, sub_word_candidate in sub_word_dict[sub_word_cnt].items()]
    word_candidates = [item[0] for item in word_subwordids]
    sub_word_ids = torch.LongTensor([item[1] for item in word_subwordids])

    sub_word_embedding = word_embeddings[sub_word_ids]
    increase_score = torch.sum(torch.mul(sub_word_embedding, gradient), dim=[1, 2])
    _, topidx = torch.topk(increase_score, k=2)

    return word_candidates[topidx[0]]


def pad_entity(example):
    tokens = example.ori_tokens
    pos1, pos1_end = example.head_span
    pos2, pos2_end = example.tail_span
    if pos1 < pos2:
        new_tokens = tokens[:pos1] + ['#'] + tokens[pos1:pos1_end + 1] + ['#'] + tokens[pos1_end + 1:pos2] \
                     + ['@'] + tokens[pos2:pos2_end + 1] + ['@'] + tokens[pos2_end + 1:]
    else:
        new_tokens = tokens[:pos2] + ['@'] + tokens[pos2:pos2_end + 1] + ['@'] + tokens[pos2_end + 1:pos1] \
                     + ['#'] + tokens[pos1:pos1_end + 1] + ['#'] + tokens[pos1_end + 1:]
    return ' '.join(new_tokens)


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
    parser.add_argument('--replace_ratio', type=float, default=0.5)
    parser.add_argument('--loss_weight', type=float, default=0.1)
    parser.add_argument('--confidence_type', type=str, default='')
    parser.add_argument('--without', type=str, default='')
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default='./checkpoints/best.pt', help='Folder to save checkpoints.')
    parser.add_argument('--load', type=str, default='./checkpoints/best.pt', help='Folder to load checkpoints.')
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
    id2rel = {v: k for k, v in rel2id.items()}
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

    sub_word_dict = load_out_vocab('data/vocab_100k.txt', tokenizer)
    # end{dataset}

    # begin{model}
    net = BERTSentenceEncoder(args.model, num_classes, args.hidden_dim, args.tem).cuda()
    net.load_state_dict(torch.load(args.load))
    net.logistic_regression = nn.Linear(1, 2).cuda()
    net.logistic_regression.weight.data = torch.tensor([[0.0], [1.0]]).cuda()
    # net.logistic_regression.weight.requires_grad = False
    net.logistic_regression.bias.data = torch.tensor([7.0, 0.0]).cuda()
    # net.logistic_regression.bias.requires_grad = False
    cudnn.benchmark = True  # fire on all cylinders
    optimizer = torch.optim.AdamW(
        net.parameters(),
        args.learning_rate,
        # weight_decay=args.decay
    )

    best_score = -1
    loss_func = nn.CrossEntropyLoss()
    print('Beginning Training\n')
    for epoch in range(1, args.epochs+1):
        begin_epoch = time.time()

        net.train()
        for batch_id, (unique_id, input_ids, input_mask, target) in enumerate(train_dataloader):
            optimizer.zero_grad()

            input_ids, input_mask, target = input_ids.cuda(), input_mask.cuda(), target.cuda()
            token_embeds = net.bert.get_input_embeddings().weight[input_ids]
            token_embeds = torch.tensor(token_embeds, requires_grad=True)
            logits, _ = net.forward_embedding(token_embeds, input_ids, input_mask)

            energy = torch.mean(torch.logsumexp(logits, 1))
            energy.backward()
            grad = token_embeds.grad
            attribution = torch.abs(torch.sum(torch.mul(token_embeds, grad), dim=-1))
            attribution = attribution / torch.sum(attribution, dim=-1, keepdim=True)

            ood_examples, id_examples = [], []
            for enum_idx, sample_idx in enumerate(unique_id):
                sample_idx = int(sample_idx)
                # decode_sent = [tokenizer.decode([input_ids[enum_idx][idx]]) for idx in range(torch.sum(input_mask[enum_idx]))]
                example = copy.deepcopy(train_dataset.examples[sample_idx])
                tokens = [token.lower() for token in example.ori_tokens]
                head_entity = list(range(example.head_span[0], example.head_span[1] + 1))
                tail_entity = list(range(example.tail_span[0], example.tail_span[1] + 1))

                # print('Target:', target[enum_idx])
                # print('Tokens:', tokens)
                # print('Sent:', ' '.join(tokens))
                # print('Head entity:', head_entity, [tokens[idx] for idx in head_entity])
                # print('Tail entity:', tail_entity, [tokens[idx] for idx in tail_entity])
                # print('Energy: {0:.4f}'.format(energy))

                # gradient
                words, sub_words, keys = tokenize_by_word(example.tokens, tokenizer)
                words_spans = [(word.lower(), span) for word, span in zip(words, keys) if word != '#' and word != '@']
                if len(words) - len(words_spans) != 4:
                    print('difference not equal 4...')
                    continue
                if len(tokens) != len(words_spans):
                    print('len(tokens) != len(words_spans)')
                    continue

                gradient_score = {idx: (word, float(torch.sum(attribution[enum_idx][span[0] + 1: span[1] + 1])), grad[enum_idx][span[0] + 1: span[1] + 1])
                                  for idx, (word, span) in enumerate(words_spans)}

                # dependency parsing
                dp_path_tokens = example.dp_path_tokens
                dp_path_ids = [item[0] for item in dp_path_tokens]
                dp_score = {idx: (word, len(words_spans) / len(dp_path_tokens) if idx in dp_path_ids else 1) for idx, (word, span) in enumerate(words_spans)}
                # print('[DP]:', [item[1] for item in dp_path_tokens])

                # TF-IDF
                tfidf_score = {idx: (word, train_dataset.tfidf_dict[int(target[enum_idx])][word] if word in train_dataset.tfidf_dict[int(target[enum_idx])] else 0)
                               for idx, (word, span) in enumerate(words_spans)}
                # print('[TF-IDF]:', tfidf_tokens)

                # for idx in range(len(words_spans)):
                #     print('{0:>2} -> {1:>20} | {2:.8f}, {3:.8f}, {4:.8f}'.format(idx, gradient_score[idx][0], dp_score[idx][1], tfidf_score[idx][1], gradient_score[idx][1]))
                if args.without == 'grad':
                    sig_score = [(idx, word, dp_score[idx][1] * tfidf_score[idx][1], gradient_score[idx][2])
                                 for idx, (word, span) in enumerate(words_spans)]
                elif args.without == 'dp':
                    sig_score = [(idx, word, gradient_score[idx][1] * tfidf_score[idx][1], gradient_score[idx][2])
                                 for idx, (word, span) in enumerate(words_spans)]
                elif args.without == 'tfidf':
                    sig_score = [(idx, word, gradient_score[idx][1] * dp_score[idx][1], gradient_score[idx][2])
                                 for idx, (word, span) in enumerate(words_spans)]
                else:
                    sig_score = [(idx, word, gradient_score[idx][1] * dp_score[idx][1] * tfidf_score[idx][1], gradient_score[idx][2])
                                 for idx, (word, span) in enumerate(words_spans)]
                sig_score.sort(key=lambda item: item[2], reverse=True)
                replace_num = int(len(words_spans) * args.replace_ratio)
                for idx in range(replace_num):
                    try:
                        replace_word = find_replace_words(sig_score[idx][3], net.bert.get_input_embeddings().weight, sub_word_dict)
                        words_spans[sig_score[idx][0]] = [replace_word, words_spans[sig_score[idx][0]][1]]
                    except Exception as e:
                        # print(e)
                        print('[ERROR] find replace word..')
                    # print('{0:>2} {1:>20}  {2:.8f}'.format(sig_score[idx][0], sig_score[idx][1], sig_score[idx][2]), replace_word)
                example.ori_tokens = [word for (word, span) in words_spans]
                ood_examples.append(example)
                id_examples.append(train_dataset.examples[sample_idx])

            optimizer.zero_grad()

            ood_input_ids, ood_input_mask = convert_bert_input(ood_examples, tokenizer, args.max_len)
            ood_input_ids = torch.LongTensor(ood_input_ids).cuda()
            ood_input_mask = torch.LongTensor(ood_input_mask).cuda()
            ood_logits = net(ood_input_ids, ood_input_mask)
            ood_energy = torch.logsumexp(ood_logits, 1)
            # print('OOD Energy: {0:.4f}'.format(ood_energy))

            id_logits = net(input_ids, input_mask)
            id_energy = torch.logsumexp(id_logits, 1)

            input_for_lr = torch.cat((id_energy, ood_energy), -1)
            labels_for_lr = torch.cat((torch.ones(len(id_energy)).cuda(), torch.zeros(len(ood_energy)).cuda()), -1)
            output1 = net.logistic_regression(input_for_lr.view(-1, 1))
            lr_reg_loss = loss_func(output1, labels_for_lr.long())

            ce_loss = loss_func(id_logits, target)

            loss = (1-args.loss_weight) * ce_loss + args.loss_weight * lr_reg_loss
            loss.backward()
            optimizer.step()
            if batch_id % 100 == 0:
                print('[TRAIN] epoch: {0:4} step: {1:4} | loss: {2:2.5f} | '
                      'ce_loss: {3:2.5f} | reg_loss: {4:2.5f} | ID: {5:2.5f} | OOD: {6:2.5f}'.format(epoch,
                                                                                                     batch_id,
                                                                                                     loss.item(),
                                                                                                     ce_loss.item(),
                                                                                                     lr_reg_loss.item(),
                                                                                                     torch.mean(id_energy),
                                                                                                     torch.mean(ood_energy)))

            if (batch_id + 1) % 100 == 0:
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
    