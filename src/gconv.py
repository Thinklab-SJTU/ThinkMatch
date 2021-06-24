import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Gconv(nn.Module):
    """
    Graph Convolutional Layer which is inspired and developed based on Graph Convolutional Network (GCN).
    Inspired by Kipf and Welling. Semi-Supervised Classification with Graph Convolutional Networks. ICLR 2017.
    """
    def __init__(self, in_features, out_features):
        super(Gconv, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features
        self.a_fc = nn.Linear(self.num_inputs, self.num_outputs)
        self.u_fc = nn.Linear(self.num_inputs, self.num_outputs)

    def forward(self, A, x, norm=True):
        """
        :param A: connectivity matrix {0,1}^(batch*n*n)
        :param x: node embedding batch*n*d
        :param norm: normalize connectivity matrix or not
        :return: new node embedding
        """
        if norm is True:
            A = F.normalize(A, p=1, dim=-2)
        ax = self.a_fc(x)
        ux = self.u_fc(x)
        x = torch.bmm(A, F.relu(ax)) + F.relu(ux) # has size (bs, N, num_outputs)
        return x


class ChannelIndependentConv(nn.Module):
    """
    Channel Independent Embedding Convolution
    Proposed by Yu et al. Learning deep graph matching with channel-independent embedding and Hungarian attention. ICLR 2020.
    """
    def __init__(self, in_features, out_features, in_edges, out_edges=None):
        super(ChannelIndependentConv, self).__init__()
        if out_edges is None:
            out_edges = out_features
        self.in_features = in_features
        self.out_features = out_features
        self.out_edges = out_edges
        # self.node_fc = nn.Linear(in_features, out_features // self.out_edges)
        self.node_fc = nn.Linear(in_features, out_features)
        self.node_sfc = nn.Linear(in_features, out_features)
        self.edge_fc = nn.Linear(in_edges, self.out_edges)

    def forward(self, A, emb_node, emb_edge, mode=1):
        """
        :param A: connectivity matrix {0,1}^(batch*n*n)
        :param emb_node: node embedding batch*n*d
        :param emb_edge: edge embedding batch*n*n*d
        :param mode: 1 or 2
        :return: new node embedding, new edge embedding
        """
        if mode == 1:
            node_x = self.node_fc(emb_node)
            node_sx = self.node_sfc(emb_node)
            edge_x = self.edge_fc(emb_edge)

            A = A.unsqueeze(-1)
            A = torch.mul(A.expand_as(edge_x), edge_x)

            node_x = torch.matmul(A.transpose(2, 3).transpose(1, 2),
                                  node_x.unsqueeze(2).transpose(2, 3).transpose(1, 2))
            node_x = node_x.squeeze(-1).transpose(1, 2)
            node_x = F.relu(node_x) + F.relu(node_sx)
            edge_x = F.relu(edge_x)

            return node_x, edge_x

        elif mode == 2:
            node_x = self.node_fc(emb_node)
            node_sx = self.node_sfc(emb_node)
            edge_x = self.edge_fc(emb_edge)

            d_x = node_x.unsqueeze(1) - node_x.unsqueeze(2)
            d_x = torch.sum(d_x ** 2, dim=3, keepdim=False)
            d_x = torch.exp(-d_x)

            A = A.unsqueeze(-1)
            A = torch.mul(A.expand_as(edge_x), edge_x)

            node_x = torch.matmul(A.transpose(2, 3).transpose(1, 2),
                                  node_x.unsqueeze(2).transpose(2, 3).transpose(1, 2))
            node_x = node_x.squeeze(-1).transpose(1, 2)
            node_x = F.relu(node_x) + F.relu(node_sx)
            edge_x = F.relu(edge_x)
            return node_x, edge_x, d_x


class Siamese_Gconv(nn.Module):
    def __init__(self, in_features, num_features):
        super(Siamese_Gconv, self).__init__()
        self.gconv = Gconv(in_features, num_features)

    def forward(self, g1, *args):
        # embx are tensors of size (bs, N, num_features)
        emb1 = self.gconv(*g1)
        if len(args) == 0:
            return emb1
        else:
            returns = [emb1]
            for g in args:
                returns.append(self.gconv(*g))
            return returns

class Siamese_ChannelIndependentConv(nn.Module):
    def __init__(self, in_features, num_features, in_edges, out_edges=None):
        super(Siamese_ChannelIndependentConv, self).__init__()
        self.in_feature = in_features
        self.gconv1 = ChannelIndependentConv(in_features, num_features, in_edges, out_edges)
        self.gconv2 = ChannelIndependentConv(in_features, num_features, in_edges, out_edges)

    def forward(self, g1, g2=None):
        emb1, emb_edge1 = self.gconv1(*g1)
        if g2 is None:
            return emb1, emb_edge1
        else:
            emb2, emb_edge2 = self.gconv2(*g2)
            # embx are tensors of size (bs, N, num_features)
            return emb1, emb2, emb_edge1, emb_edge2
