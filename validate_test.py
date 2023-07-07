import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import torch.nn.functional as F


def fpr_at_recall95(labels, scores):
    recall_point = 0.95
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    # Sort label-score tuples by the score in descending order.
    indices = np.argsort(scores)[::-1]    #降序排列
    sorted_labels = labels[indices]
    sorted_scores = scores[indices]
    n_match = sum(sorted_labels)
    n_thresh = recall_point * n_match
    thresh_index = np.argmax(np.cumsum(sorted_labels) >= n_thresh)
    FP = np.sum(sorted_labels[:thresh_index] == 0)
    TN = np.sum(sorted_labels[thresh_index:] == 0)
    return float(FP) / float(FP + TN), (sorted_scores[thresh_index-1] + sorted_scores[thresh_index])/2


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = roc_auc_score(labels, examples)
    # aupr = sk.average_precision_score(labels, examples)
    fpr, thresh = fpr_at_recall95(labels, examples)

    return auroc, fpr, thresh


def validate(net, dataloader, mode, num_classes):
    net.eval()
    gold, predict, confidence_msp, confidence_energy = [], [], [], []

    with torch.no_grad():
        for unique_id, input_ids, input_mask, target in dataloader:
            input_ids, input_mask, target = input_ids.cuda(), input_mask.cuda(), target.cuda()

            # forward
            output = net(input_ids, input_mask)

            target = target.cpu().tolist()
            pred = output.data.max(1)[1].cpu().tolist()
            msp = F.softmax(output, dim=-1).max(1)[0].cpu().tolist()
            energy = torch.logsumexp(output, 1).cpu().tolist()

            gold.extend(target)
            predict.extend(pred)
            confidence_msp.extend(msp)
            confidence_energy.extend(energy)

    # acc = accuracy_score(gold, predict)
    if mode == 'msp':
        confidence_score = confidence_msp
    elif mode == 'energy':
        confidence_score = confidence_energy
    else:
        raise Exception('Not supported mode...')

    # AUROC & FPR95
    pos_ = []
    neg_ = []
    for g, c in zip(gold, confidence_score):
        if g < num_classes:
            pos_.append(c)
        else:
            neg_.append(c)
    auroc, fpr95, thresh = get_measures(pos_, neg_)

    # if mode == 'msp':
    #     sorted_confidence = sorted(confidence_msp)
    # elif mode == 'energy':
    #     sorted_confidence = sorted(confidence_energy)
    # else:
    #     raise Exception('Not supported mode...')
    # thresh_idx = int(len(sorted_confidence) * 0.05)
    # thresh = sorted_confidence[thresh_idx]

    print('[DEV] AUROC: {0:.4f} | FPR95: {1:.4f}| Thresh: {2:.4f}'.format(auroc, fpr95, thresh))
    net.train()
    return auroc, fpr95, thresh


def test(net, dataloader, mode, thresh, num_classes):
    net.eval()
    gold, predict, confidence_msp, confidence_energy = [], [], [], []

    with torch.no_grad():
        for unique_id, input_ids, input_mask, target in dataloader:
            input_ids, input_mask, target = input_ids.cuda(), input_mask.cuda(), target.cuda()

            # forward
            output = net(input_ids, input_mask)

            target = target.cpu().tolist()
            pred = output.data.max(1)[1].cpu().tolist()
            msp = F.softmax(output, dim=-1).max(1)[0].cpu().tolist()
            energy = torch.logsumexp(output, 1).cpu().tolist()

            gold.extend(target)
            predict.extend(pred)
            confidence_energy.extend(energy)
            confidence_msp.extend(msp)

    if mode == 'msp':
        confidence_score = confidence_msp
    elif mode == 'energy':
        confidence_score = confidence_energy
    else:
        raise Exception('Not supported mode...')

    # AUROC & FPR95
    pos_ = []
    neg_ = []
    for g, c in zip(gold, confidence_score):
        if g < num_classes:
            pos_.append(c)
        else:
            neg_.append(c)
    auroc, fpr95, _ = get_measures(pos_, neg_)

    # ID ACC
    id_gold_predict = [(g, p) for g, p in zip(gold, predict) if g < num_classes]
    id_gold = [item[0] for item in id_gold_predict]
    id_predict = [item[1] for item in id_gold_predict]
    id_acc = accuracy_score(id_gold, id_predict)

    # ACC & TNR95
    for idx in range(len(gold)):
        gold[idx] = min(gold[idx], num_classes)
        if confidence_score[idx] < thresh:
            predict[idx] = num_classes
    acc = accuracy_score(gold, predict)
    # cm = confusion_matrix(gold, predict)
    # tn = cm[-1][-1]
    # tnr95 = tn / cm[-1].sum() if cm[-1].sum() != 0 else 0

    print('[TEST] AUROC: {0:.4f} | FPR95: {1:.4f} | ACC: {2:.4f} | ID-ACC: {3:.4f}'.format(auroc, fpr95, acc, id_acc))
    net.train()
    return auroc, fpr95, acc
