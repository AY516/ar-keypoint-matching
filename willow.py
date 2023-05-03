import os.path as osp
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import lightning.pytorch as pl
from torch_geometric.datasets import WILLOWObjectClass as WILLOW
from utils import PairDataset
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from model import SplineCNN, Net, SimpleNet, LitSimpleNet

parser = argparse.ArgumentParser()
parser.add_argument('--isotropic', action='store_true')
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--rnd_dim', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--test_samples', type=int, default=1000)
parser.add_argument('--mlp_hidden_dim', type=int, default=1024)
parser.add_argument('--mlp_hidden_out', type=int, default=256)
args = parser.parse_args()

pre_filter1 = lambda d: d.num_nodes > 0  # noqa
pre_filter2 = lambda d: d.num_nodes > 0 and d.name[:4] != '2007'  # noqa

transform = T.Compose([
    T.Delaunay(),
    T.FaceToEdge(),
    T.Distance() if args.isotropic else T.Cartesian(),
])

path = osp.join('.', 'data', 'WILLOW')
datasets = [WILLOW(path, cat, transform) for cat in WILLOW.categories]

datasets = [dataset.shuffle() for dataset in datasets]
train_datasets = [dataset[:20] for dataset in datasets]
test_datasets = [dataset[20:] for dataset in datasets]
train_datasets = [
    PairDataset(train_dataset, train_dataset, sample=False)
    for train_dataset in train_datasets
]
train_dataset = torch.utils.data.ConcatDataset(train_datasets)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                            follow_batch=['x_s', 'x_t'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(torch.cuda.get_device_name())

# geometry-aware refinement
psi_1 = SplineCNN(datasets[0].num_node_features, args.dim,
                  datasets[0].num_edge_features, args.num_layers, cat=False,
                  dropout=0.5)

mlp_hidden_dim = 1024
model = SimpleNet(psi_1, args.dim, args.mlp_hidden_dim, args.mlp_hidden_out).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def compute_grad_norm(model_params):
    return torch.norm(torch.stack([torch.norm(p.grad) for p in model_params]))

def generate_y(num_nodes, batch_size):
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

def get_pos_neg(pmat_pred, pmat_gt):
    """
    Calculates number of true positives, false positives and false negatives
    :param pmat_pred: predicted permutation matrix
    :param pmat_gt: ground truth permutation matrix
    :return: tp, fp, fn

    Modified from : BBGM 
    """
    device = pmat_pred.device
    pmat_gt = torch.eye(10).expand(batch_size, 10, 10)

    tp = torch.sum(pmat_pred * pmat_gt).float()
    fp = torch.sum(pmat_pred * (1 - pmat_gt)).float()
    fn = torch.sum((1 - pmat_pred) * pmat_gt).float()
    return tp, fp, fn


def train_mlp(train_loader, optimizer, epoch, writer):
    model.train()
    total_loss = 0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)

        num_graphs = data.x_s_batch.max().item() + 1
        y = generate_y(num_nodes=data.x_s_batch.size()[0], batch_size= num_graphs).to(device)
        y_logits = model(data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch,
                    data.x_t, data.edge_index_t, data.edge_attr_t, data.x_t_batch)
        # C = y_logits.view(args.batch_size, 10, 10)
        # matching = np.array([linear_sum_assignment(C[x,:,:].detach().cpu().numpy()) 
        #                      for x in range(num_graphs)])
        loss = model.loss(y_logits, y)
        loss.backward()
        
        grad_norm_model = compute_grad_norm(model.parameters())
        grad_norm_splineCNN = compute_grad_norm(model.spline.parameters())
        grad_mlp_proj = compute_grad_norm(model.mlp_proj.parameters())
        grad_mlp = compute_grad_norm(model.mlp.parameters())
        writer.add_scalars('grad norms',{'model': grad_norm_model,
                                        'grad norm Spline': grad_norm_splineCNN,
                                        'grad norm projMLP': grad_mlp_proj,
                                        'grad norm MLP': grad_mlp}, (epoch-1)*len(train_loader) + i)
        
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss

@torch.no_grad()
def test(test_dataset):
    model.load_state_dict(torch.load('model_more_hidden_layers_2.pt'))
    model.eval()

    test_loader1 = DataLoader(test_dataset, args.batch_size, shuffle=True)
    test_loader2 = DataLoader(test_dataset, args.batch_size, shuffle=True)

    correct = num_examples = 0
    while (num_examples < args.test_samples):
        for data_s, data_t in zip(test_loader1, test_loader2):
            data_s, data_t = data_s.to(device), data_t.to(device)

            n_nodes_s = data_s.x.size()[0]
            n_nodes_t = data_t.x.size()[0]
            batch_size = data_t.num_graphs

            y_logits = model(data_s.x, data_s.edge_index, data_s.edge_attr,
                           data_s.batch, data_t.x, data_t.edge_index,
                           data_t.edge_attr, data_t.batch)
            y = generate_y(num_nodes=50, batch_size=data_t.num_graphs)
            
            C = -y_logits.view(data_t.num_graphs, 10, 10)
            y_pred = torch.tensor(np.array([linear_sum_assignment(C[x,:,:].detach().cpu().numpy()) 
                                            for x in range(batch_size)])).to(device)
            
            num_correct, num_samples = acc(data_t.num_graphs, data_s.batch.size()[0], y_pred)
            
            # perm_mat_gt = torch.eye(10).expand(batch_size, C.size(0), C.size(1))
            perm_mat_pred = []
            correct += num_correct
            num_examples += num_samples

            if num_examples >= args.test_samples:
                return correct / num_examples

def train_tf(train_loader, optimizer):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        # print('node_emb_s : ',data.x_s.size(), ' node_emb_t: ',data.x_t.size())

        num_graphs = data.x_s_batch.max().item() + 1
        y = generate_y(num_nodes=data.x_s_batch.size()[0], batch_size= num_graphs).to(device)
        C = model(data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch,
                    data.x_t, data.edge_index_t, data.edge_attr_t, data.x_t_batch)

        o = np.array([linear_sum_assignment(C[x,:,:].detach().cpu().numpy()) for x in range(num_graphs)])
        o = torch.as_tensor(o,dtype=torch.int64)
        o_col = o[:,1,:]
    #     out = o_col.contiguous().view(50).to(device)
    #     loss = model.loss(y_logits, y)
    
    #     loss.backward()
    #     optimizer.step()
    #     total_loss += loss.item()
        
    # return total_loss

def run(train_loader, optimizer, test_datasets):
    writer = SummaryWriter()
    for epoch in tqdm(range(1, args.epochs+ 1)):
        total_loss = train_mlp(train_loader, optimizer, epoch, writer)
        writer.add_scalar('training loss', total_loss / args.batch_size, epoch)

    torch.save(model.state_dict(), 'model_more_hidden_layers_2.pt')

    accs = [100 * test(test_dataset) for test_dataset in test_datasets]
    print(' '.join([category.ljust(13) for category in WILLOW.categories]))
    print(' '.join([f'{acc:.2f}'.ljust(13) for acc in accs]))


# run(train_loader, optimizer, test_datasets)
accs = [100 * test(test_dataset) for test_dataset in test_datasets]
print(' '.join([category.ljust(13) for category in WILLOW.categories]))
print(' '.join([f'{acc:.2f}'.ljust(13) for acc in accs]))
