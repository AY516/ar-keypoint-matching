import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
import lightning.pytorch as pl

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

class LitSimpleNet(pl.LightningModule):
    def __init__(self, SimpleNet, generate_y):
        super().__init__()
        self.SimpleNet = SimpleNet
        self.generate_y = generate_y

    
    def training_step(self, data):
        num_graphs = data.x_s_batch.max().item() + 1
        y = self.generate_y(num_nodes=data.x_s_batch.size()[0], batch_size= num_graphs)
        y_logits =  self.SimpleNet(data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch,
                    data.x_t, data.edge_index_t, data.edge_attr_t, data.x_t_batch)
        loss =  self.SimpleNet.loss(y_logits, y)
        self.log("loss:",  loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class SimpleNet(nn.Module):
    def __init__(self, psi, node_dim, hidden_size, hidden_out):
        super(SimpleNet, self).__init__()
        self.spline = psi
        self.mlp_proj = MLP([512, 256, 128], 256)
        self.mlp = MLP([256, 512, 1024, 512, 256], 1)

    def forward(self, x_s, edge_index_s, edge_attr_s, batch_s,
                x_t, edge_index_t, edge_attr_t, batch_t):
        
        h_s = self.spline(x_s, edge_index_s, edge_attr_s)
        h_t = self.spline(x_t, edge_index_t, edge_attr_t)
    
        h_s, s_mask = to_dense_batch(h_s, batch_s, fill_value=0)
        h_t, t_mask = to_dense_batch(h_t, batch_t, fill_value=0)

        (B, N_s, D), N_t = h_s.size(), h_t.size(1)

        #print('batched h_s: ', h_s.size())
        #print('batched h_t: ', h_t.size())

        queries = make_queries(h_s, h_t)
        queries = self.mlp_proj(queries)
        #print('queries: ', queries.size())

        output = self.mlp(queries).squeeze(2)
        #print('mlp in: ', input.size())
        #print('mlp out: ', output.size())
        return output
    
    def loss(self, y_hat, y):
        loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn(y_hat, y)

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
            # output= nn.functional.softmax(self.out(x), dim=1)
            output = self.out(x)

            return output