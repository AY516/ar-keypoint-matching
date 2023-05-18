import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch



def masked_softmax(src, mask, dim=-1):
    out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)
    return out

def make_queries(h_s, h_t):
    n_s = h_s.size(dim=1)
    n_t = h_t.size(dim=1)

    queries = []

    for i in range(0,n_s):
        for j in range(0, n_t):
            query = torch.cat((h_s[:,i,:], h_t[:,j,:]), dim=-1)
            queries.append(query)
    queries = torch.stack(queries, dim=1)

    return queries
class Net(nn.Module):
    def __init__(self, psi, node_dim, hidden_size, hidden_out):
        # args for transformer to add: nhead, num_encoder_layers, num_decoder_layers
        # args for MLP to add: hidden_sizes
        super(Net, self).__init__()
        self.spline = psi
        self.mlp = nn.Sequential(
                                nn.Linear(2 * node_dim, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_out)
                                )
        self.transformer = nn.Transformer(d_model= hidden_out,
                                          nhead=4, 
                                          num_encoder_layers=8, 
                                          num_decoder_layers=6,
                                          batch_first=True)
        self.mlp_out = MLP([256, 256], 1)
       
        
    def forward(self, x_s, edge_index_s, edge_attr_s, batch_s,
                x_t, edge_index_t, edge_attr_t, batch_t):
        
        h_s = self.spline(x_s, edge_index_s, edge_attr_s)
        h_t = self.spline(x_t, edge_index_t, edge_attr_t)

        h_s, s_mask = to_dense_batch(h_s, batch_s, fill_value=0)
        h_t, t_mask = to_dense_batch(h_t, batch_t, fill_value=0)

        # print('batched h_s: ', h_s.size())
        # print('batched h_t: ', h_t.size())

        assert h_s.size(0) == h_t.size(0), 'batch-sizes are not equal'
        (B, N_s, D), N_t = h_s.size(), h_t.size(1)

        S_mask = ~torch.cat((s_mask, t_mask), dim=1)
        query_mask = ~(s_mask.view(B, N_s, 1) & t_mask.view(B, 1, N_t))
        query_mask = query_mask.view(B, -1)
        # print('src_mask size', S_mask.size())
        # print('tgt_mask size', query_mask.size())

        input = torch.cat((h_s, h_t), dim=1)
        # print('in_transformer: ', input.size())
        queries = make_queries(h_s, h_t)
        # print('queries: ', queries.size())
        queries = self.mlp(queries)
        # print('mlp out: ', queries.size())
        transformer_out = self.transformer(input, 
                                  queries, 
                                  src_key_padding_mask= S_mask, 
                                  tgt_key_padding_mask= query_mask)
        # print('out_transformer: ', transformer_out.size())
        output = self.mlp_out(transformer_out)
        # print('output: ', output.size())

        return output.squeeze(2)
        

    
    def loss(self, y, y_hat):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(y_hat,y)


class MLP(nn.Module):
        def __init__(self, h_sizes, out_size):
            super(MLP, self).__init__()
            self.hidden = nn.ModuleList()
            for k in range(len(h_sizes)-1):
                self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))

            self.out = nn.Linear(h_sizes[-1], out_size)
        
        def forward(self, x):

            # Feedforward
            for layer in self.hidden:
                x = nn.functional.relu(layer(x))
            output= nn.functional.softmax(self.out(x), dim=1)

            return output

             

