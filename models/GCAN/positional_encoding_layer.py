import torch.nn
from torch.nn import init
import torch
import torch.nn.functional as F
from torch_geometric.nn import SplineConv
from torch.nn.parameter import Parameter


class SConv(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(SConv, self).__init__()

        self.in_channels = input_features
        self.num_layers = 2
        self.splineCNN = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            conv = SplineConv(input_features, output_features, dim=2, kernel_size=5, aggr="max")
            self.splineCNN.append(conv)
            input_features = output_features

        input_features = output_features
        self.out_channels = input_features
        self.reset_parameters()

    def reset_parameters(self):
        for net in self.splineCNN:
            net.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        xs = [x]

        for net in self.splineCNN[:-1]:
            xs += [F.relu(net(xs[-1], edge_index, edge_attr))]

        xs += [self.splineCNN[-1](xs[-1], edge_index, edge_attr)]
        return xs[-1]


class positional_encoding_layer(torch.nn.Module):
    def __init__(self, input_node_dim):
        super(positional_encoding_layer, self).__init__()
        self.num_node_features = input_node_dim
        self.weight = Parameter(torch.Tensor(1))
        init.constant_(self.weight, 0.01)
        self.positional_encoding = SConv(input_features=self.num_node_features, output_features=self.num_node_features)

    def forward(self, graph):
        ori_features = graph.x
        result = self.positional_encoding(graph)
        graph.x = ori_features + 0.1 * result
        return graph

