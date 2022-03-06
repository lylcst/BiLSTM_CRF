#-*-coding:utf-8-*- 
# author lyl
import torch
import torch.nn as nn
from model.crf_layer import CRFLayer


class BiLSTMCRF(nn.Module):
  def __init__(self,
               token2idx,
               tag2idx,
               vocab_size,
               tag_size,
               embedding_size,
               hidden_size,
               num_layers,
               dropout,
               with_ln,
               pad='<PAD>'):
    super(BiLSTMCRF, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=token2idx[pad])
    self.dropout = nn.Dropout(dropout)
    self.bilstm = nn.LSTM(input_size=embedding_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          bidirectional=True)
    self.with_ln = with_ln
    if with_ln:
      self.layer_norm = nn.LayerNorm(hidden_size)
    self.hidden2tag = nn.Linear(hidden_size * 2, tag_size)
    self.crf = CRFLayer(tag_size, tag2idx)

    self.reset_parameters()

  def reset_parameters(self):
    nn.init.xavier_normal_(self.embedding.weight)
    nn.init.xavier_normal_(self.hidden2tag.weight)

  def get_lstm_features(self, seq, mask):
    """
    :param seq: (seq_len, batch_size)
    :param mask: (seq_len, batch_size)
    """
    embed = self.embedding(seq) # (seq_len, batch_size, embedding_size)
    embed = self.dropout(embed)
    embed = nn.utils.rnn.pack_padded_sequence(embed, mask.sum(0).long())
    lstm_output, _ = self.bilstm(embed) # (seq_len, batch_size, hidden_size)
    lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output)
    lstm_output = lstm_output * mask.unsqueeze(-1)
    if self.with_ln:
      lstm_output = self.layer_norm(lstm_output)
    lstm_features = self.hidden2tag(lstm_output) * mask.unsqueeze(-1)  # (seq_len, batch_size, tag_size)
    return lstm_features

  def neg_log_likelihood(self, seq, tags, mask):
    """
    :param seq: (seq_len, batch_size)
    :param tags: (seq_len, batch_size)
    :param mask: (seq_len, batch_size)
    """
    lstm_features = self.get_lstm_features(seq, mask)
    forward_score = self.crf(lstm_features, mask)
    gold_score = self.crf.score_sentence(lstm_features, tags, mask)
    loss = (forward_score - gold_score).sum()

    return loss

  def predict(self, seq, mask):
    """
    :param seq: (seq_len, batch_size)
    :param mask: (seq_len, batch_size)
    """
    lstm_features = self.get_lstm_features(seq, mask)
    best_paths = self.crf.viterbi_decode(lstm_features, mask)

    return best_paths