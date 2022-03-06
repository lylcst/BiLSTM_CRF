from __future__ import unicode_literals
import argparse
import os
import json

import torch
from torch import nn
from torch.nn import init
import numpy as np
from data_loader import prepare_data
from model.bilstm_crf import BiLSTMCRF

START_TAG = "<START_TAG>"
END_TAG = "<END_TAG>"
PAD = "<PAD>"
UNK = "<UNK>"
token2idx, tag2idx = {}, {}


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch BiLSTM+CRF Sequence Labeling')
    parser.add_argument('--model_path', type=str, default='model_output/best.model', metavar='S',
                        help='model name')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--embedding-size', type=int, default=512, metavar='N',
                        help='embedding size')
    parser.add_argument('--hidden-size', type=int, default=1024, metavar='N',
                        help='hidden size')
    parser.add_argument('--rnn-layer', type=int, default=1, metavar='N',
                        help='RNN layer num')
    parser.add_argument('--with-layer-norm', action='store_true', default=False,
                        help='whether to add layer norm after RNN')
    parser.add_argument('--dropout', type=float, default=0, metavar='RATE',
                        help='dropout rate')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed')
    parser.add_argument('--vocab', nargs='+', required=False, metavar='SRC_VOCAB TGT_VOCAB',
                        default=["data/token2idx.json", "data/tag2idx.json"],
                        help='src vocab and tgt vocab')
    return parser.parse_args()


def separate_char(text):
    result = ""
    for t in text:
        result += t + " "
    result = result.rstrip()
    return result


def tag_parser(string, tags):
    item = {"string": string, "entities": [], "recog_label": "model"}
    entity_name = ""
    flag = []
    visit = False
    for char, tag in zip(string, tags):
        if tag[0] == "B":
            if entity_name != "":
                x = dict((a, flag.count(a)) for a in flag)
                y = [k for k, v in x.items() if max(x.values()) == v]
                item["entities"].append({"word": entity_name, "type": y[0]})
                flag.clear()
                entity_name = ""
            entity_name += char
            flag.append(tag[2:])
        elif tag[0] == "I":
            entity_name += char
            flag.append(tag[2:])
        else:
            if entity_name != "":
                x = dict((a, flag.count(a)) for a in flag)
                y = [k for k, v in x.items() if max(x.values()) == v]
                item["entities"].append({"word": entity_name, "type": y[0]})
                flag.clear()
            flag.clear()
            entity_name = ""

    if entity_name != "":
        x = dict((a, flag.count(a)) for a in flag)
        y = [k for k, v in x.items() if max(x.values()) == v]
        item["entities"].append({"word": entity_name, "type": y[0]})

    return item


def predict_text(text, model, tag2idx, token2idx, device):
    if isinstance(text, str):
        text = [text]
    idx2tag = {}
    for k, v in tag2idx.items():
        idx2tag[v] = k
    text = map(separate_char, text)
    model.eval()
    seq = prepare_data(text, token2idx, PAD, UNK, device=device)
    mask = torch.ne(seq, float(token2idx[PAD])).float()
    length = mask.sum(0)
    _, idx = length.sort(0, descending=True)
    seq = seq[:, idx]
    mask = mask[:, idx]
    best_paths = model.predict(seq, mask)
    tags = [[idx2tag[i] for i in t] for t in best_paths]
    return tags


def predict(texts):
    args = get_args()

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    path = os.path.dirname(__file__)
    args.vocab[0] = os.path.join(path, args.vocab[0])
    args.vocab[1] = os.path.join(path, args.vocab[1])

    assert os.path.isfile(args.vocab[0]) and os.path.isfile(args.vocab[1])

    with open(args.vocab[0], "r", encoding="utf-8") as fp:
        token2idx = json.load(fp)
    with open(args.vocab[1], "r", encoding="utf-8") as fp:
        tag2idx = json.load(fp)

    idx2tag = {}
    for k, v in tag2idx.items():
        idx2tag[v] = k
    idx2token = {}
    for k, v in token2idx.items():
        idx2token[v] = k

    model = BiLSTMCRF(token2idx, tag2idx, len(token2idx), len(tag2idx), args.embedding_size, args.hidden_size, args.rnn_layer, args.dropout,
                      args.with_layer_norm, PAD).to(device)

    args.model_path = os.path.join(path, args.model_path)
    if use_cuda:
        model.load_state_dict(torch.load(args.model_path))
    else:
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    tags = predict_text(texts, model, tag2idx, token2idx, device)
    res = tag_parser(texts, tags[0])

    return res


def main():
    texts = ["患者3月前因“直肠癌”于在我院于全麻上行直肠癌根治术(dixon术),手术过程顺利。",
             "患者1个月前无明显诱因出现下腹部不适,进食硬质食物时有哽噎感,哽咽感常在吞咽水后缓解消失,进食流质无明显不适。当时无发热、呕吐,无乏力,无声音嘶哑,无呕血黑便,无头晕头痛,无腹泻腹胀,无黄疸"]

    items = []
    for text in texts:
        res = predict(text)
        items.append(res)

    print(items)


if __name__ == "__main__":
    main()