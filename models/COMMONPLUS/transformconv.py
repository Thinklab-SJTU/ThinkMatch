import torch_geometric
import torch
from torch import nn
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import dense_to_sparse
import numpy as np
from src.lap_solvers.sinkhorn import Sinkhorn

# From "Gmtr: Graph matching transformers"
class TransformerConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, head, sk_channel=0, recurrence=3, depth=3, bias=False, sk_iter=20, sk_tau=0.05):
        super(TransformerConvLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.head = head
        self.depth = depth
        self.sk_channel = sk_channel
        self.recurrence = recurrence
        for i in range(depth):
            if sk_channel > 0:
                sk = Sinkhorn(sk_iter, sk_tau)
                classifier = nn.Linear(out_dim[i], sk_channel)
                self.add_module(f'sk_{i}', sk)
                self.add_module(f'classifier_{i}', classifier)
            if i != depth - 1:
                if i != 0 and sk_channel > 0:
                    attn = TransformerConv(in_dim[i] + sk_channel, out_dim[i], heads=head[i], edge_dim=in_dim[0], bias=False)
                else:
                    attn = TransformerConv(in_dim[i], out_dim[i], heads=head[i], edge_dim=in_dim[0], bias=False)
                if bias == False:
                    attn.apply(delete_bias)
                self.add_module(f'attn_{i}', attn)
                self.add_module(f'norm_{i}', nn.LayerNorm(out_dim[i], elementwise_affine=False))
            else:
                attn = TransformerConv(in_dim[i] + sk_channel, out_dim[i], heads=2, edge_dim=in_dim[0], bias=False, concat=False)
                if bias == False:
                    attn.apply(delete_bias)
                self.add_module(f'attn_{i}', attn)


    def forward(self, node, edge, A, n1=None, n2=None):
        """
        node: b n 1
        edge: b n n
        A: b n n
        """
        b, n, _ = node.shape
        node_attr = torch.zeros((b, n, self.out_dim[-1] + self.sk_channel))

        emb_ls = []
        for i in range(b):
            emb_ls.append(node[i])

        for r in range(self.recurrence):
            for l in range(self.depth):
                if r != 0 and l == 0:
                    continue
                attn = getattr(self, f'attn_{l}')
                for j in range(b):
                    val = dense_to_sparse(edge[j] * A[j])
                    emb = emb_ls[j]
                    emb = attn(emb, edge_index=val[0], edge_attr=val[1].unsqueeze(-1))
                    if l != self.depth - 1:
                        norm = getattr(self, f'norm_{l}')
                        emb = torch.relu(norm(emb))

                    if self.sk_channel > 0:
                        classifier = getattr(self, f'classifier_{l}')
                        sk_net = getattr(self, f'sk_{l}')
                        sk = classifier(emb.unsqueeze(0))
                        n1_rep = torch.repeat_interleave(n1, self.sk_channel, dim=0)
                        n2_rep = torch.repeat_interleave(n2, self.sk_channel, dim=0)
                        sk = sk.permute(0, 2, 1).reshape(sk.shape[0] * self.sk_channel, n2.max(), n1.max()).transpose(1, 2)
                        sk = sk_net(sk, n1_rep, n2_rep, dummy_row=True).transpose(2, 1).contiguous()
                        sk = sk.reshape(sk.shape[0], self.sk_channel, n1.max() * n2.max()).permute(0, 2, 1).squeeze(0)
                        emb = torch.cat((emb, sk), dim=-1)

                    emb_ls[j] = emb

        for i in range(b):
            node_attr[i] = emb_ls[i]

        return node_attr.to(node.device)


def delete_bias(m):
    if isinstance(m, nn.Linear):
        m.bias = None





