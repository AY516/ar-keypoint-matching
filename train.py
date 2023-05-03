import os.path as osp
import argparse
import torch
from torch_geometric.datasets import PascalVOCKeypoints as PascalVOC
from utils import ValidPairDataset
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

from model import SplineCNN, Net


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
model = Net(psi_1, args.dim, args.mlp_hidden_dim, args.mlp_hidden_out).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def generate_y(y_col):
    y_row = torch.arange(y_col.size(0), device=device)
    return torch.stack([y_row, y_col], dim=0)

for data in train_loader:
    data = data.to(device)
    print('node_emb_s : ',data.x_s.size(), ' node_emb_t: ',data.x_t.size())
    print(data.y.size())
    ground_truth = generate_y(data.y)
    print(ground_truth.size())
    out = model(data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch,
                data.x_t, data.edge_index_t, data.edge_attr_t, data.x_t_batch)
    # flattened labels [n_nodes_in_batch x ] 
    ground_truth = generate_y(data.y)
    break

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        # out = model(data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch,
        #             data.x_t, data.edge_index_t, data.edge_attr_t, data.x_t_batch)

        y = generate_y(data.y)
        loss = model.loss(y, out)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * (data.x_s_batch.max().item() + 1)

    return total_loss / len(train_loader.dataset)