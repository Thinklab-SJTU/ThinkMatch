import torch
import torch.nn as nn
from src.utils.pad_tensor import pad_tensor
from models.GCAN.cross_attention_layer import cross_attention_layer
from models.GCAN.self_attention_layer import self_attention_layer

def get_node_deep_feature(features_src, features_tgt, ns_src, ns_tgt):
    features_all_list = []
    ns_all_list = []
    for idx in range(features_src.shape[0]):
        n = ns_src[idx]
        features_all_list.append(features_src[idx, 0:n, :])
        ns_all_list.append(n)
        n = ns_tgt[idx]
        features_all_list.append(features_tgt[idx, 0:n, :])
        ns_all_list.append(n)
    node_deep_features = torch.cat(features_all_list,dim=0)
    ns = torch.stack(ns_all_list)

    return node_deep_features, ns

class GCA_module(nn.Module):
    def __init__(self, cross_parameters,self_parameters):
        super(GCA_module, self).__init__()
        self.cross_attention_layer = cross_attention_layer(input_dim=cross_parameters[0], output_dim=cross_parameters[1])
        self.self_attention_layer = self_attention_layer(nfeat=self_parameters[0],
                nhid=self_parameters[1], nheads=self_parameters[2])

    def forward(self, batch_feature_src, batch_feature_tgt, global_avg_weights, global_max_weights, ns_src, ns_tgt, adjacency_matrixs):

        ### cross_attention
        attention_src_list, attention_tgt_list, cross_attention = self.cross_attention_layer(batch_feature_src, batch_feature_tgt, global_avg_weights, global_max_weights)
        ### prepare for self_attention
        attention_src_input_list = [
            feature_src-attention_src for feature_src,attention_src in zip(batch_feature_src,attention_src_list)
        ]
        attention_tgt_input_list = [
            feature_src-attention_tgt for feature_src,attention_tgt in zip(batch_feature_tgt,attention_tgt_list)
        ]
        new_batch_feature_src = [
            torch.cat([attention_src_input, feature_src],dim=-1) for attention_src_input, feature_src in zip(attention_src_input_list,batch_feature_src)
        ]
        new_batch_feature_tgt = [
            torch.cat([attention_tgt_input, feature_tgt],dim=-1) for attention_tgt_input, feature_tgt in zip(attention_tgt_input_list,batch_feature_tgt)
        ]
        emb_src = torch.stack(pad_tensor(new_batch_feature_src), dim=0)
        emb_tgt = torch.stack(pad_tensor(new_batch_feature_tgt), dim=0)
        node_features, ns = get_node_deep_feature(emb_src, emb_tgt, ns_src, ns_tgt)
        ###self attention
        updated_node_features = self.self_attention_layer(node_features, adjacency_matrixs)
        node_features = node_features[:,updated_node_features.shape[1]:]-updated_node_features
        return cross_attention, node_features, ns