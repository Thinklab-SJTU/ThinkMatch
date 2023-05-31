import torch
import torch.nn as nn

def compute_cross_attention(Xs, Ys, cross_attention_list):
    attention_x_list = []
    attention_y_list = []
    for x,y,s in zip(Xs, Ys, cross_attention_list):
        a_x = torch.softmax(s, dim=1)  # i->j
        a_y = torch.softmax(s, dim=0)  # j->i
        attention_x = torch.mm(a_x, y)
        attention_x_list.append(attention_x)
        attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)
        attention_y_list.append(attention_y)
    return attention_x_list, attention_y_list

class cross_attention_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cross_attention_layer, self).__init__()
        self.d = output_dim
        self.A = torch.nn.Linear(input_dim, output_dim)
        self.B = torch.nn.Linear(input_dim, output_dim)


    def forward(self, Xs, Ys, Ws_avg,Ws_max):
        cross_attention_list = []
        for X, Y, W_avg, W_max in zip(Xs,Ys,Ws_avg,Ws_max):
            assert X.shape[1] == Y.shape[1] == self.d, (X.shape[1], Y.shape[1], self.d)
            attention_channel_A = torch.tanh(self.A(W_avg+0.1*W_max))
            attention_channel_B = torch.tanh(self.B(W_avg+0.1*W_max))
            Y = Y*attention_channel_B
            cross_attention = torch.matmul(X * attention_channel_A, Y.transpose(0, 1))
            cross_attention = torch.nn.functional.softplus(cross_attention) - 0.5
            cross_attention_list.append(cross_attention)
        attention_x_list, attention_y_list = compute_cross_attention(Xs, Ys, cross_attention_list)
        return attention_x_list, attention_y_list, cross_attention_list