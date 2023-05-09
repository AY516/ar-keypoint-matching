import os.path as osp
import argparse
import torch
from torch_geometric.datasets import PascalVOCKeypoints as PascalVOC
from utils import ValidPairDataset
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

from model import SplineCNN, Net, SimpleNet
from scipy.optimize import linear_sum_assignment
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--isotropic', action='store_true')
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--rnd_dim', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--test_samples', type=int, default=1000)

parser.add_argument('--mlp_hidden_dim', type=int, default=1024)
parser.add_argument('--mlp_hidden_out', type=int, default=256)
args = parser.parse_args()


pre_filter = lambda data: data.pos.size(0) > 0  # noqa
transform = T.Compose([
    T.Delaunay(),
    T.FaceToEdge(),
    T.Distance() if args.isotropic else T.Cartesian(),
])
train_datasets = []
test_datasets = []
path = osp.join('.','data','PascalVOC')

for category in PascalVOC.categories:
    dataset = PascalVOC(path, category, train=True, transform=transform,
                        pre_filter=pre_filter)
    train_datasets += [ValidPairDataset(dataset, dataset, sample=True)]
    dataset = PascalVOC(path, category, train=False, transform=transform,
                        pre_filter=pre_filter)
    test_datasets += [ValidPairDataset(dataset, dataset, sample=True)]
 
# graph transformed VGG-16 keypoints
train_dataset = torch.utils.data.ConcatDataset(train_datasets)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                          follow_batch=['x_s', 'x_t'])


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.get_device_name())

# geometry-aware refinement
psi_1 = SplineCNN(dataset.num_node_features, args.dim,
                  dataset.num_edge_features, args.num_layers, cat=False,
                  dropout=0.5)

# Spline-CNN -> MLP -> transformer_enc -> queries
mlp_hidden_dim = 1024
model = SimpleNet(psi_1, args.dim, args.mlp_hidden_dim, args.mlp_hidden_out).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def generate_y(y_col):
    y_row = torch.arange(y_col.size(0), device=device)
    return torch.stack([y_row, y_col], dim=0)

def generate_y_cart(num_nodes, batch_size):
    node_idx_s = torch.arange(num_nodes/batch_size, dtype=int)
    node_idx_t = torch.arange(num_nodes/batch_size, dtype=int)
    y_cart_prod = torch.cartesian_prod(node_idx_s, node_idx_t)
    y = torch.zeros_like(y_cart_prod[:,0])
    y[y_cart_prod[:,0] == y_cart_prod[:,1]] = 1
    y = y.expand(batch_size, y.size()[0])
    return y.type(torch.DoubleTensor)

def acc(batch_size, num_nodes_s, y_pred):
    correct = total = 0
    y_pred = y_pred[:,1,:]
    y_gt = torch.arange(num_nodes_s/batch_size, dtype=int).to(device)
    y_gt = y_gt.expand(batch_size, y_gt.size()[0])
    num_samples = y_gt.size()[0] * y_gt.size()[1]
    return torch.eq(y_pred, y_gt).sum(), num_samples

def make_multi_label_tensor(y_gt_target, x_s_batch, x_t_batch):

    aranges = []
    for num in torch.unique(x_s_batch):
        count = (x_s_batch == num).sum().item()
        arange = torch.arange(count)
        aranges.append(arange)

    y_gt_source = torch.cat(aranges, dim=0).to(device=device)
    y_gt = torch.stack([y_gt_source, y_gt_target])

    num_nodes_s = torch.max(torch.bincount(x_s_batch))
    num_nodes_t = torch.max(torch.bincount(x_t_batch))
    num_graphs = len(torch.unique(x_s_batch))
    
    out_tensor = torch.zeros(num_graphs,num_nodes_s*num_nodes_t)

    for i in range(num_graphs):
        corr_tensor = torch.zeros(num_nodes_s, num_nodes_t)
        src_idx = torch.masked_select(y_gt_source, (x_s_batch == 0))
        tgt_idx = torch.masked_select(y_gt_target, (x_s_batch == 0))
        corr_tensor[src_idx, tgt_idx] = 1
        out_tensor[i,: ] = corr_tensor.flatten()

    return out_tensor.to(device=device)

# total_loss = 0
# for data in train_loader:
#     optimizer.zero_grad()
#     data = data.to(device)
#     y_gt = make_multi_label_tensor(data.y, data.x_s_batch, data.x_t_batch)
#     out = model(data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch,
#                 data.x_t, data.edge_index_t, data.edge_attr_t, data.x_t_batch)
 
#     loss = model.loss(out,y_gt)
#     loss.backward()
#     optimizer.step()
#     total_loss += loss.item()
#     break

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        y_gt = make_multi_label_tensor(data.y, data.x_s_batch, data.x_t_batch)
        out = model(data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch,
                    data.x_t, data.edge_index_t, data.edge_attr_t, data.x_t_batch)
    
        loss = model.loss(out,y_gt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(dataset):
    model.eval()

    loader = DataLoader(dataset, args.batch_size, shuffle=False,
                        follow_batch=['x_s', 'x_t'])

    correct = num_examples = 0
    while (num_examples < args.test_samples):
        for data in loader:
            num_nodes_s = torch.max(torch.bincount(data.x_s_batch))
            num_nodes_t = torch.max(torch.bincount(data.x_t_batch))
            num_graphs = len(torch.unique(data.x_s_batch))
            
            data = data.to(device)
            out = model(data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch,
                        data.x_t, data.edge_index_t, data.edge_attr_t, data.x_t_batch)
            y_gt = make_multi_label_tensor(data.y, data.x_s_batch, data.x_t_batch)

            C = -out.view(num_graphs, num_nodes_s, num_nodes_t)
            y_pred = torch.tensor(np.array([linear_sum_assignment(C[x,:,:].detach().cpu().numpy()) 
                                            for x in range(num_graphs)])).to(device)
            num_correct, num_samples += acc(num_graphs, num_nodes_s, y_pred)

            if num_examples >= args.test_samples:
                return num_correct / num_examples