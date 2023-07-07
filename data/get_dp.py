import json
import stanza
import networkx as nx


nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse', tokenize_pretokenized=True, download_method=None)


def get_dp_path_tokens(tokens, head_entity, tail_entity):
    doc = nlp([tokens])

    edges = []
    sent = doc.sentences[0]
    for word in sent.words:
        edges.append((word.id, word.head))

    graph = nx.Graph(edges)
    path_token_ids = []
    for e1 in head_entity:
        for e2 in tail_entity:
            spt = nx.shortest_path(graph, source=e1 + 1, target=e2 + 1)
            for pt in spt:
                if pt not in path_token_ids:
                    path_token_ids.append(pt)

    return [(idx - 1, sent.words[idx - 1].text) for idx in path_token_ids]


def get_dp(data_file, new_data_file):
    dataset = json.load(open(data_file, 'r', encoding='utf-8'))
    for idx, data in enumerate(dataset):
        head_entity = data['h'][2][0]
        tail_entity = data['t'][2][0]
        dp_path_tokens = get_dp_path_tokens(data['tokens'], head_entity, tail_entity)
        data['dp_path'] = dp_path_tokens
        if idx % 100 == 0:
            print(idx)

    with open(new_data_file, 'w', encoding='utf-8') as w:
        json.dump(dataset, w, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    get_dp('tacred/train.json', 'tacred/train_dp.json')
