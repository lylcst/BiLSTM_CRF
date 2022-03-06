#-*-coding:utf-8-*- 
# author lyl
from __future__ import unicode_literals
import argparse
import os
import loguru
import json
import sys
import linecache
import pdb

import torch
from torch import nn, optim
from torch.nn import init
from torch.utils.data import DataLoader
import numpy as np
from model.bilstm_crf import BiLSTMCRF
from data_loader import SequenceLabelingDataset, prepare_data

START_TAG = "<START_TAG>"
END_TAG = "<END_TAG>"
PAD = "<PAD>"
UNK = "<UNK>"

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch BiLSTM+CRF Sequence Labeling')
    parser.add_argument('--model_name', type=str, default='model', metavar='S',
                        help='model name')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',
                        help='test batch size')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train')
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
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed')
    parser.add_argument('--save-interval', type=int, default=30, metavar='N',
                        help='save interval')
    parser.add_argument('--valid-interval', type=int, default=60, metavar='N',
                        help='valid interval')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='log interval')
    parser.add_argument('--patience', type=int, default=30, metavar='N',
                        help='patience for early stop')
    parser.add_argument('--vocab', nargs='+', metavar='SRC_VOCAB TGT_VOCAB',
                        default=["data/token2idx.json", "data/tag2idx.json"],
                        help='src vocab and tgt vocab')
    parser.add_argument('--trainset', type=str, default=os.path.join('data', 'train.txt'), metavar='TRAINSET',
                        help='trainset path')
    parser.add_argument('--testset', type=str, default=os.path.join('data', 'test.txt'), metavar='TESTSET',
                        help='testset path')
    return parser.parse_args()


def compute_forward(model, seq, tags, mask):
    loss = model.neg_log_likelihood(seq, tags, mask)
    batch_size = seq.size(1)
    loss /= batch_size
    loss.backward()
    return loss.item()


def evaluate(model, testset_loader, token2idx, idx2tag, device):
    def get_entity(tags):
        entity = []
        prev_entity = "O"
        start = -1
        end = -1
        for i, tag in enumerate(tags):
            if tag[0] == "O":
              if prev_entity != "O":
                entity.append((start, end))
              prev_entity = "O"
            if tag[0] == "B":
              if prev_entity != "O":
                entity.append((start, end))
              prev_entity = tag[2:]
              start = end = i
            if tag[0] == "I":
              if prev_entity == tag[2:]:
                end = i
        return entity

    model.eval()
    correct_num = 0
    predict_num = 0
    truth_num = 0
    with torch.no_grad():
        for bidx, batch in enumerate(testset_loader):
            seq = prepare_data(batch[0], token2idx, PAD, UNK, device)
            mask = torch.ne(seq, float(token2idx[PAD])).float()
            length = mask.sum(0)
            _, idx = length.sort(0, descending=True)
            seq = seq[:, idx]
            mask = mask[:, idx]
            best_path = model.predict(seq, mask)
            ground_truth = [batch[1][i].strip().split(" ") for i in idx]
            for hyp, gold in zip(best_path, ground_truth):
                hyp = list(map(lambda x: idx2tag[x], hyp))
                predict_entities = get_entity(hyp)
                gold_entities = get_entity(gold)
                correct_num += len(set(predict_entities) & set(gold_entities))
                predict_num += len(set(predict_entities))
                truth_num += len(set(gold_entities))
    # calculate F1 on entity
    precision = correct_num / predict_num if predict_num else 0
    recall = correct_num / truth_num if truth_num else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    model.train()
    return f1, precision, recall


def main():
    args = get_args()
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    if len(args.vocab) != 2:
        print("ERROR: invalid vocab arguments -> {}".format(args.vocab), file=sys.stderr)
        exit(-1)

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

    loguru.logger.info("Loading data")
    trainset = SequenceLabelingDataset(args.trainset)
    testset = SequenceLabelingDataset(args.testset)
    trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    testset_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True)

    loguru.logger.info("Building model")
    model = BiLSTMCRF(token2idx,
                      tag2idx,
                      len(token2idx),
                      len(tag2idx),
                      args.embedding_size,
                      args.hidden_size,
                      args.rnn_layer,
                      args.dropout,
                      args.with_layer_norm,
                      PAD).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 保存模型的路径
    if not os.path.exists(args.model_name):
        os.makedirs(args.model_name)

    loguru.logger.info("Starting training")

    model.train()
    step = 0
    best_f1 = 0
    patience = 0
    early_stop = False
    for eidx in range(1, args.epochs + 1):
        if eidx == 2:
          model.debug = True
        if early_stop:
            loguru.logger.info("Early stop. epoch {} step {} best f1 {}".format(eidx, step, best_f1))
            sys.exit(0)
        loguru.logger.info("Start epoch {}".format(eidx))
        for bidx, batch in enumerate(trainset_loader):
            seq = prepare_data(batch[0], token2idx, PAD, UNK, device)
            tags = prepare_data(batch[1], tag2idx, END_TAG, UNK, device)
            mask = torch.ne(seq, float(token2idx[PAD])).float()
            length = mask.sum(0)
            _, idx = length.sort(0, descending=True)
            seq = seq[:, idx]
            tags = tags[:, idx]
            mask = mask[:, idx]
            optimizer.zero_grad()
            loss = compute_forward(model, seq, tags, mask)
            optimizer.step()
            step += 1
            if step % args.log_interval == 0:
                loguru.logger.info("epoch {} step {} batch {} loss {}".format(eidx, step, bidx, loss))
            if step % args.save_interval == 0:
                torch.save(model.state_dict(), os.path.join(args.model_name, "newest.model"))
                torch.save(optimizer.state_dict(), os.path.join(args.model_name, "newest.optimizer"))
            if step % args.valid_interval == 0:
                f1, precision, recall = evaluate(model, testset_loader, token2idx, idx2tag, device)

                loguru.logger.info("[valid] epoch {} step {} f1 {} precision {} recall {}".format(eidx, step, f1, precision, recall))
                if f1 > best_f1:
                    patience = 0
                    best_f1 = f1
                    torch.save(model.state_dict(), os.path.join(args.model_name, "best.model"))
                    torch.save(optimizer.state_dict(), os.path.join(args.model_name, "best.optimizer"))
                else:
                    patience += 1
                    if patience == args.patience:
                        early_stop = True


if __name__ == '__main__':
    main()