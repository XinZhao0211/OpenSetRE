import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F


class Similarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(F.normalize(x), F.normalize(y).T)


def extract_entity(sequence_output, e_mask):
    extended_e_mask = e_mask.unsqueeze(-1)
    extended_e_mask = extended_e_mask.float() * sequence_output
    out = torch.sum(extended_e_mask, dim=-2)
    return out


class BERTSentenceEncoder(nn.Module):
    def __init__(self, bert_model, num_classes, hidden_dim=256, tem=1.0):
        super(BERTSentenceEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.tem = tem
        # self.sim = Similarity()
        self.bert = BertModel.from_pretrained(bert_model)
        self.proj = nn.Sequential(
            nn.Linear(2 * 768, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, input_ids, input_mask):
        out, _ = self.forward_virtual(input_ids, input_mask)
        return out

    def forward_virtual(self, input_ids, input_mask):
        out = self.bert(input_ids, attention_mask=input_mask).last_hidden_state
        head_mask = torch.zeros(input_ids.shape).long()
        tail_mask = torch.zeros(input_ids.shape).long()

        n, s = input_ids.shape
        for i in range(n):
            for j in range(s):
                if input_ids[i, j] == 1001:
                    head_mask[i, j] = 1
                    break
        for i in range(n):
            for j in range(s):
                if input_ids[i, j] == 1030:
                    tail_mask[i, j] = 1
                    break

        head = extract_entity(out, head_mask.cuda())
        tail = extract_entity(out, tail_mask.cuda())
        out = torch.cat([head, tail], dim=-1)
        out = self.proj(out)
        return self.fc(out) / self.tem, out

    def forward_embedding(self, token_embeds, input_ids, input_mask):
        out = self.bert(inputs_embeds=token_embeds, attention_mask=input_mask).last_hidden_state
        head_mask = torch.zeros(input_ids.shape).long()
        tail_mask = torch.zeros(input_ids.shape).long()

        n, s = input_ids.shape
        for i in range(n):
            for j in range(s):
                if input_ids[i, j] == 1001:
                    head_mask[i, j] = 1
                    break
        for i in range(n):
            for j in range(s):
                if input_ids[i, j] == 1030:
                    tail_mask[i, j] = 1
                    break

        head = extract_entity(out, head_mask.cuda())
        tail = extract_entity(out, tail_mask.cuda())
        out = torch.cat([head, tail], dim=-1)
        out = self.proj(out)
        return self.fc(out) / self.tem, out
