import os.path as osp
import argparse
from typing import Optional
from datetime import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import lightning.pytorch as pl
from torch_geometric.datasets import WILLOWObjectClass as WILLOW
from utils import PairDataset
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from scipy.optimize import linear_sum_assignment
import torchvision.transforms.functional as F
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
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--test_samples', type=int, default=1000)
parser.add_argument('--mlp_hidden_dim', type=int, default=1024)
parser.add_argument('--mlp_hidden_out', type=int, default=256)

parser.add_argument('--model_path', type=str, default='model_test.pt')
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

def f1_score(tp, fp, fn):

    device = tp.device

    const = torch.tensor(1e-7, device=device)
    precision = tp / (tp + fp + const)
    recall = tp / (tp + fn + const)
    f1 = 2 * precision * recall / (precision + recall + const)
    return f1

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

    tp = torch.sum(pmat_pred * pmat_gt).float()
    fp = torch.sum(pmat_pred * (1 - pmat_gt)).float()
    fn = torch.sum((1 - pmat_pred) * pmat_gt).float()
    return tp, fp, fn

def make_perm_mat_gt(batch_size, n_classes):
    perm_mat_gt = []
    for i in range(batch_size):
        perm_mat_gt.append(torch.eye(n_classes)[torch.arange(n_classes),:])
    return torch.stack(perm_mat_gt).to(device)

def make_perm_mat_pred(matching_vec):
    batch_size = matching_vec.size()[0]
    nodes = matching_vec.size()[1]
    perm_mat_pred = []
    for i in range(batch_size):
        index = matching_vec[i, :]
        one_hot_pred = torch.eye(nodes).to(device)[index,:]
        perm_mat_pred.append(one_hot_pred)
    
    return torch.stack(perm_mat_pred)


def train_mlp(train_loader, optimizer, epoch, writer):
    model.train()
    total_loss = 0
    total_correct = total_samples = 0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)

        num_graphs = data.x_s_batch.max().item() + 1
        y = generate_y(num_nodes=data.x_s_batch.size()[0], batch_size= num_graphs).to(device)
        y_logits = model(data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch,
                    data.x_t, data.edge_index_t, data.edge_attr_t, data.x_t_batch)
        loss = model.loss(y_logits, y)
        loss.backward()
        
        # grad norm
        grad_norm_model = compute_grad_norm(model.parameters())
        grad_norm_splineCNN = compute_grad_norm(model.spline.parameters())
        grad_mlp_proj = compute_grad_norm(model.mlp_proj.parameters())
        grad_mlp = compute_grad_norm(model.mlp.parameters())
        writer.add_scalars('grad norms',{'model': grad_norm_model,
                                        'grad norm Spline': grad_norm_splineCNN,
                                        'grad norm projMLP': grad_mlp_proj,
                                        'grad norm MLP': grad_mlp}, (epoch-1)*len(train_loader) + i)
        # train accuracy
        C = -y_logits.view(data.num_graphs, 10, 10)
        y_pred = torch.tensor(np.array([linear_sum_assignment(C[x,:,:].detach().cpu().numpy()) 
                                            for x in range(data.num_graphs)])).to(device)
            
        num_correct, num_samples = acc(data.num_graphs, num_graphs * 10, y_pred)
        optimizer.step()
        total_correct += num_correct.item()
        total_samples += num_samples
        total_loss += loss.item()
    train_acc = total_correct/total_samples
    train_loss = total_loss / len(train_loader)
    return train_loss, train_acc, model

@torch.no_grad()
def test(test_dataset, model: Optional[torch.nn.Module] = None):
    
    # This need to figure out for test loss
    if model is None:
        model = SimpleNet(psi_1, args.dim, args.mlp_hidden_dim, args.mlp_hidden_out).to(device)
        model.load_state_dict(torch.load(args.model_path))
    
    model.eval()

    test_loader1 = DataLoader(test_dataset, args.batch_size, shuffle=True)
    test_loader2 = DataLoader(test_dataset, args.batch_size, shuffle=True)

    correct = num_examples = 0
    tp = fp = fn = 0
    total_loss = 0
    while (num_examples < args.test_samples):
        for data_s, data_t in zip(test_loader1, test_loader2):
            data_s, data_t = data_s.to(device), data_t.to(device)
            batch_size = data_t.num_graphs

            y_logits = model(data_s.x, data_s.edge_index, data_s.edge_attr,
                           data_s.batch, data_t.x, data_t.edge_index,
                           data_t.edge_attr, data_t.batch)

            y = generate_y(num_nodes=batch_size * 10, batch_size=data_t.num_graphs).to(device)

            C = -y_logits.view(data_t.num_graphs, 10, 10)
            y_pred = torch.tensor(np.array([linear_sum_assignment(C[x,:,:].detach().cpu().numpy()) 
                                            for x in range(batch_size)])).to(device)
            loss = model.loss(y_logits, y)
            
            num_correct, num_samples = acc(data_t.num_graphs, data_s.batch.size()[0], y_pred)
            
            if num_examples == 0:
                # logits contain large -ve values, which gives incorrect heatmaps on normalising
                # abs and reciprocal circumvents this problem
                score_map = torch.nn.functional.softmax(-C[0,:,:], dim=0)
            
            perm_mat_gt = make_perm_mat_gt(batch_size, 10)
            perm_mat_pred = make_perm_mat_pred(y_pred[:,1,:])
            _tp, _fp, _fn = get_pos_neg(perm_mat_gt, perm_mat_pred)
            tp += _tp
            fp += _fp
            fn += _fn
            correct += num_correct
            num_examples += num_samples
            total_loss += loss

            if num_examples >= args.test_samples:
                f1 = f1_score(tp, fp, fn)
                accuracy = correct / num_examples
                test_loss = total_loss / num_examples
                # print('accuracy: ', accuracy)
                # print('tp: ', tp, 'fp: ', fp, 'fn: ', fn)
                # print('f1: ', f1)
                # print('--------------------------')
                return accuracy, f1, score_map, test_loss

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
    writer = SummaryWriter(log_dir=
                           "runs/{}".format(args.model_path + datetime.now().strftime("%Y%m%d-%H%M%S")))
    
    for epoch in tqdm(range(1, args.epochs+ 1)):
        train_loss, train_acc, model = train_mlp(train_loader, optimizer, epoch, writer)
        metrics = [test(test_dataset, model) for test_dataset in test_datasets]
        accs = torch.tensor([cat_metrics[0] for cat_metrics in metrics])
        test_loss = torch.tensor([cat_metrics[3] for cat_metrics in metrics])
        
        avg_test_loss = torch.sum(test_loss) / len(test_loss)
        avg_test_acc = torch.sum(accs) / len(accs)
        
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/test', avg_test_loss, epoch)
        writer.add_scalar('accuracy/train', train_acc, epoch)
        writer.add_scalar('accuracy/test', avg_test_acc, epoch)

    torch.save(model.state_dict(), args.model_path)

    accs = []
    f1_scores = []
    for i, test_dataset in enumerate(test_datasets):
        acc, f1, score_map, _ = test(test_dataset)
        accs.append(acc *100)
        f1_scores.append(f1)
        writer.add_image("heatmaps/{cat}".format(cat=WILLOW.categories[i]), 
                         score_map, dataformats='HW')
    
    # accs = [test(test_dataset) for test_dataset in test_datasets]
    print(' '.join([category.ljust(13) for category in WILLOW.categories]))
    print(' '.join([f'{acc:.2f}'.ljust(13) for acc in accs]))


run(train_loader, optimizer, test_datasets)

# metrics = [test(test_dataset) for test_dataset in test_datasets]
# print(' '.join([category.ljust(13) for category in WILLOW.categories]))
# print(' '.join([f'{acc:.2f}'.ljust(13) for acc in accs]))