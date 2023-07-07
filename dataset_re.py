import json
from collections import defaultdict
import random
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class InputExample(object):
    def __init__(self, unique_id, tokens, ori_tokens, head_span, tail_span, label, dp_path_tokens):
        self.unique_id = unique_id
        self.tokens = tokens
        self.ori_tokens = ori_tokens
        self.head_span = head_span
        self.tail_span = tail_span
        self.label = label
        self.dp_path_tokens = dp_path_tokens

    def __str__(self):
        return 'Unique Id: ' + str(self.unique_id) + '\n' + \
               'Text: ' + str(self.tokens) + '\n' + \
               'Head: ' + str(self.head_span) + '\n' + \
               'Tail: ' + str(self.tail_span) + '\n' + \
               'Label: ' + str(self.label)


class InputFeatures(object):
    def __init__(self, unique_id, tokens, input_ids, input_mask, label):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label = label


def load_json_file(file_path):
    data = json.load(open(file_path, 'r', encoding='utf-8'))
    return data


def get_rel2id(id_relations_file, dev_ood_relations_file, test_ood_relations_file):
    id_relations = load_json_file(id_relations_file)
    dev_ood_relations = load_json_file(dev_ood_relations_file)
    test_ood_relations = load_json_file(test_ood_relations_file)

    rel2id = {}
    idx = 0

    for relation in id_relations + dev_ood_relations + test_ood_relations:
        rel2id[relation] = idx
        idx += 1

    return rel2id, len(id_relations)


class REDataset(Dataset):
    def __init__(self, data, max_len, tokenizer, num_classes, rel2id):
        self.data = data
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.corpus = [''] * num_classes

        self.examples = []
        unique_id = 0
        for instance in self.data:
            tokens = instance['tokens']
            relation_id = rel2id[instance['relation']]
            # TF-IDF corpus
            if relation_id < len(self.corpus):
                self.corpus[relation_id] += ' ' + ' '.join(tokens).lower()

            pos1 = instance['h'][2][0][0]
            pos2 = instance['t'][2][0][0]
            pos1_end = instance['h'][2][0][-1]
            pos2_end = instance['t'][2][0][-1]
            head_span = [pos1, pos1_end]
            tail_span = [pos2, pos2_end]
            if pos1 < pos2:
                new_tokens = tokens[:pos1] + ['#'] + tokens[pos1:pos1_end + 1] + ['#'] + tokens[pos1_end + 1:pos2] \
                             + ['@'] + tokens[pos2:pos2_end + 1] + ['@'] + tokens[pos2_end + 1:]
            else:
                new_tokens = tokens[:pos2] + ['@'] + tokens[pos2:pos2_end + 1] + ['@'] + tokens[pos2_end + 1:pos1] \
                             + ['#'] + tokens[pos1:pos1_end + 1] + ['#'] + tokens[pos1_end + 1:]
            self.examples.append(InputExample(unique_id, new_tokens, tokens, head_span, tail_span, relation_id, instance['dp_path'] if 'dp_path' in instance else None))
            unique_id += 1

        self.features = self.convert_examples_to_features()

        # TF-IDF
        try:
            vectorizer = TfidfVectorizer(stop_words=set(ENGLISH_STOP_WORDS))
            self.tfidf = vectorizer.fit_transform(self.corpus).toarray()
            self.vocab = list(vectorizer.get_feature_names_out())
            print('[TF-IDF]:', self.tfidf.shape)

            self.tfidf_dict = defaultdict(dict)
            for i in range(self.tfidf.shape[0]):
                for j in range(self.tfidf.shape[1]):
                    self.tfidf_dict[i][self.vocab[j]] = self.tfidf[i][j]
        except Exception:
            print('Ignore TfIdf...')

    def __getitem__(self, index):
        unique_id = torch.tensor(self.features[index].unique_id).long()
        input_ids = torch.LongTensor(self.features[index].input_ids)
        input_mask = torch.LongTensor(self.features[index].input_mask)
        label = torch.tensor(self.features[index].label).long()
        return unique_id, input_ids, input_mask, label

    def __len__(self):
        return len(self.features)

    def convert_examples_to_features(self):
        features = []
        # features_by_class = defaultdict(list)
        for example in self.examples:
            bert_tokens = self.tokenizer.tokenize(' '.join(example.tokens))
            if len(bert_tokens) > self.max_len - 2:
                bert_tokens = bert_tokens[:self.max_len - 2]
            tokens = ['[CLS]'] + bert_tokens + ['[SEP]']
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length
            while len(input_ids) < self.max_len:
                input_ids.append(0)
                input_mask.append(0)
            assert len(input_ids) == self.max_len
            assert len(input_mask) == self.max_len
            features.append(
                InputFeatures(
                    unique_id=example.unique_id,
                    tokens=tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    label=example.label,
                )
            )
            # by class
            # features_by_class[example.label].append(
            #     InputFeatures(
            #         unique_id=example.unique_id,
            #         tokens=tokens,
            #         input_ids=input_ids,
            #         input_mask=input_mask,
            #         label=example.label,
            #     )
            # )
        return features


def load_out_vocab(vocab_file, tokenizer):
    lines = open(vocab_file, 'r', encoding='utf-8').readlines()
    sub_word_dict = defaultdict(dict)
    for idx, line in enumerate(lines):
        word = line.split('\t')[0].lower()
        sub_word = tokenizer.tokenize(word)
        sub_word_dict[len(sub_word)][word] = (sub_word, tokenizer.convert_tokens_to_ids(sub_word))

    return sub_word_dict


def convert_bert_input(examples, tokenizer, max_len):
    batch_input_ids, batch_input_mask = [], []
    for example in examples:
        tokens = example.ori_tokens
        pos1, pos1_end = example.head_span
        pos2, pos2_end = example.tail_span
        if pos1 < pos2:
            new_tokens = tokens[:pos1] + ['#'] + tokens[pos1:pos1_end + 1] + ['#'] + tokens[pos1_end + 1:pos2] \
                         + ['@'] + tokens[pos2:pos2_end + 1] + ['@'] + tokens[pos2_end + 1:]
        else:
            new_tokens = tokens[:pos2] + ['@'] + tokens[pos2:pos2_end + 1] + ['@'] + tokens[pos2_end + 1:pos1] \
                         + ['#'] + tokens[pos1:pos1_end + 1] + ['#'] + tokens[pos1_end + 1:]
        example.tokens = new_tokens

        bert_tokens = tokenizer.tokenize(' '.join(example.tokens))
        if len(bert_tokens) > max_len - 2:
            bert_tokens = bert_tokens[:max_len - 2]
        tokens = ['[CLS]'] + bert_tokens + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length
        while len(input_ids) < max_len:
            input_ids.append(0)
            input_mask.append(0)
        assert len(input_ids) == max_len
        assert len(input_mask) == max_len
        batch_input_ids.append(input_ids)
        batch_input_mask.append(input_mask)
    return batch_input_ids, batch_input_mask


if __name__ == '__main__':
    pass
