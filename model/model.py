import torch
import torch.nn as nn

def make_queries(h_s, h_t):
    n_s = h_s.size(dim=0)
    n_t = h_t.size(dim=0)

    queries = []

    for i in range(0,n_s):
        for j in range(0, n_t):
            query = torch.cat((h_s[i,:], h_t[j,:]))
            queries.append(query)
    queries = torch.stack(queries, dim=0)
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
                                          num_decoder_layers=6)
        self.mlp_out = MLP([256, 256], 256)
        
    def forward(self, x_s, edge_index_s, edge_attr_s,
                x_t, edge_index_t, x_t_edge_attr_t):
        
        h_s = self.spline(x_s, edge_index_s, edge_attr_s)
        print('h_s: ', h_s.size())
        h_t = self.spline(x_t, edge_index_t, x_t_edge_attr_t)
        print('h_t: ', h_t.size())

        input = torch.cat((h_s, h_t), dim=0)
        print('in_transformer: ', input.size())
        queries = make_queries(h_s, h_t)
        print('queries: ', queries.size())
        queries = self.mlp(queries)
        print('mlp out: ', queries.size())
        output = self.transformer(input, queries)
        print('out_transformer: ', output.size())
        output = self.mlp_out(output)
        out = torch.softmax(output, dim=1)

        return out


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

             

