import torch
import torch.nn as nn
import math

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.query = nn.Linear(in_features, out_features)
        self.key = nn.Linear(in_features, out_features)
        self.value = nn.Linear(in_features, out_features)

    def forward(self, h, adj):
        h_query = self.query(h)
        h_key = self.key(h)
        h_value = self.value(h)
        attention_scores = torch.matmul(h_query, h_key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.out_features)
        zero_vec = -9e15*torch.ones_like(attention_scores)
        attention_scores = torch.where(adj > 0, attention_scores, zero_vec)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, h_value)
        return context_layer

class self_attention_layer(nn.Module):
    def __init__(self, nfeat, nhid, nheads):
        """Dense version of GAT."""
        super(self_attention_layer, self).__init__()

        self.attentions = [GraphAttentionLayer(nfeat, nhid) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        return x
