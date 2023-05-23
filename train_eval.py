import argparse
from datetime import datetime
from tqdm import tqdm

import torch
from utils.evaluation import *
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import SplineCNN, Net, SimpleNet
from datasets.willow_obj import WillowObject
from datasets.pascal_obj import PascalObject

from scipy.optimize import linear_sum_assignment
import numpy as np

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
    perm_mat_gt = []

    for i in range(num_graphs):
        corr_tensor = torch.zeros(num_nodes_s, num_nodes_t)
        src_idx = torch.masked_select(y_gt_source, (x_s_batch == i))
        tgt_idx = torch.masked_select(y_gt_target, (x_s_batch == i))
        corr_tensor[src_idx, tgt_idx] = 1
        perm_mat_gt.append(corr_tensor)
        out_tensor[i,: ] = corr_tensor.flatten()
    perm_mat_gt = torch.stack(perm_mat_gt)
    
    return out_tensor.to(device=device), perm_mat_gt.to(device=device)

def train(epoch, writer, train_loader, val_loader):
    model.train()
    metrics={}
    total_loss = 0
    train_total_correct = train_total_samples = 0
    val_total_correct = val_total_samples = 0
    num_correct = num_samples = 0 
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)

        num_nodes_s = torch.max(torch.bincount(data.x_s_batch))
        num_nodes_t = torch.max(torch.bincount(data.x_t_batch))

        y_gt, perm_gt = make_multi_label_tensor(data.y, data.x_s_batch, data.x_t_batch)

        out = model(data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch,
                    data.x_t, data.edge_index_t, data.edge_attr_t, data.x_t_batch)
        
        loss = model.loss(out,y_gt)
        loss.backward()

        # grad norm
        # grad_norm_model = compute_grad_norm(model.parameters())
        # grad_norm_splineCNN = compute_grad_norm(model.spline.parameters())
        # grad_mlp_proj = compute_grad_norm(model.mlp_proj.parameters())
        # grad_mlp = compute_grad_norm(model.mlp.parameters())
        # writer.add_scalars('grad norms',{'model': grad_norm_model,
        #                                 'grad norm Spline': grad_norm_splineCNN,
        #                                 'grad norm tf': grad_mlp_proj,
        #                                 'grad norm mlp': grad_mlp}, (epoch-1)*len(train_loader) + i)
        
        # train accuracy
        C = -out.view(data.num_graphs, num_nodes_s, num_nodes_t)
        y_pred = torch.tensor(np.array([linear_sum_assignment(C[x,:,:].detach().cpu().numpy()) 
                                            for x in range(data.num_graphs)])).to(device)
        perm_pred = make_perm_mat_pred(y_pred[:,1,:], num_nodes_t).to(device)

        _, num_correct, num_samples = matching_accuracy(perm_pred, perm_gt)
        train_total_correct += num_correct
        train_total_samples += num_samples
        optimizer.step()
        total_loss += loss.item()
    
    metrics['train_loss'] = total_loss / len(train_loader)
    metrics['train_acc'] = train_total_correct / train_total_samples
    
    if val_loader != None:
        # model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                data = data.to(device)

                num_nodes_s = torch.max(torch.bincount(data.x_s_batch))
                num_nodes_t = torch.max(torch.bincount(data.x_t_batch))

                y_gt, perm_gt = make_multi_label_tensor(data.y, data.x_s_batch, data.x_t_batch)
                out = model(data.x_s, data.edge_index_s, data.edge_attr_s, data.x_s_batch,
                            data.x_t, data.edge_index_t, data.edge_attr_t, data.x_t_batch)
                
                C = -out.view(data.num_graphs, num_nodes_s, num_nodes_t)
                y_pred = torch.tensor(np.array([linear_sum_assignment(C[x,:,:].detach().cpu().numpy()) 
                                                    for x in range(data.num_graphs)])).to(device)
                perm_pred = make_perm_mat_pred(y_pred[:,1,:], num_nodes_t).to(device)

                _, num_correct, num_samples = matching_accuracy(perm_pred, perm_gt)
                val_total_correct += num_correct
                val_total_samples += num_samples

                loss = model.loss(out,y_gt)
                valid_loss += loss.item() 

        metrics['valid_loss'] = valid_loss / len(val_loader)
        metrics['valid_acc'] = val_total_correct / val_total_samples

    return metrics

@torch.no_grad()
def test(dataset):
    model.eval()

    loader = DataLoader(dataset, args.batch_size, shuffle=False,
                        follow_batch=['x_s', 'x_t'])
    
    test_samples = test_total_correct = 0
    tp = fp = fn = 0
    while (test_samples < args.test_samples):
        for data in loader:
            data = data.to(device)
            
            num_graphs = data.x_s_batch.max().item() + 1
            num_nodes_s = torch.max(torch.bincount(data.x_s_batch))
            num_nodes_t = torch.max(torch.bincount(data.x_t_batch))
            nodes_in_test_batch  = torch.bincount(data.x_s_batch)
            
            out = model(data.x_s, data.edge_index_s, data.edge_attr_s,
                             data.x_s_batch, data.x_t, data.edge_index_t,
                             data.edge_attr_t, data.x_t_batch)
            _, perm_gt = make_multi_label_tensor(data.y, data.x_s_batch, data.x_t_batch)
            
            C = -out.view(data.num_graphs, num_nodes_s, num_nodes_t)
            y_pred = torch.tensor(np.array([linear_sum_assignment(C[x,:,:].detach().cpu().numpy()) 
                                            for x in range(data.num_graphs)])).to(device)
            perm_pred = make_perm_mat_pred(y_pred[:,1,:], num_nodes_t).to(device)

            _, num_correct, num_samples = matching_accuracy(perm_pred, perm_gt)
            
            if num_samples == 0:
                # logits contain large -ve values, which gives incorrect heatmaps on normalising
                # applying softmax along the rows circumvents this problem
                score_map = torch.nn.functional.softmax(-C[0,:,:], dim=0)
            
            _tp, _fp, _fn = get_pos_neg(perm_gt, perm_pred)
            tp += _tp
            fp += _fp
            fn += _fn
            test_total_correct += num_correct
            test_samples += num_samples

            if test_samples >= args.test_samples:
                f1 = f1_score(tp, fp, fn)
                return test_total_correct / test_samples, f1

def run(dataset, train_loader, test_datasets, logdir, val_set):
    writer = SummaryWriter(logdir)

    for epoch in tqdm(range(1, args.epochs+ 1)):
        train_metrics = train(epoch, writer,train_loader, val_set)
        loss = train_metrics['train_loss']
        train_acc = train_metrics['train_acc']
        accs = [100 * test(test_dataset)[0] for test_dataset in test_datasets]
        accs += [sum(accs) / len(accs)]
        test_acc = accs[-1]
        
        writer.add_scalar('loss/train', train_metrics['train_loss'], epoch)
        writer.add_scalar('accuracy/train', train_metrics['train_acc'], epoch)
        writer.add_scalar('accuracy/test', test_acc, epoch)
        
        if val_set != None:
            writer.add_scalar('loss/val', train_metrics['valid_loss'], epoch)
            writer.add_scalar('accuracy/val', train_metrics['valid_acc'], epoch)
            val_acc = train_metrics['valid_acc']
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train_acc: {train_acc:.4f}, Val_acc: {val_acc:.4f}, Test_acc:{test_acc: .2f}')

        else:
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train_acc: {train_acc:.4f}, Test_acc:{test_acc: .2f}')


    
    # torch.save(model.state_dict(), args.model_path)
    # model.load_state_dict(torch.load(args.model_path))

    accs = [100 * test(test_dataset)[0] for test_dataset in test_datasets]
    accs += [sum(accs) / len(accs)]

    f1 = [100 * test(test_dataset)[1] for test_dataset in test_datasets]
    f1 += [sum(f1) / len(f1)]   

    # for i, test_dataset in enumerate(test_datasets):
    #     writer.add_image("heatmaps/{cat}".format(cat=PascalVOC.categories[i]), 
    #                      test(test_dataset)[2], dataformats='HW')

    print(' '.join([c[:5].ljust(5) for c in dataset.categories] + ['mean']))
    print(' '.join([f'{acc:.1f}'.ljust(5) for acc in accs]))
    print(' '.join([f'{f1:.1f}'.ljust(5) for f1 in f1]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--isotropic', action='store_true')
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--rnd_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--test_samples', type=int, default=1000)

    parser.add_argument('--mlp_hidden_dim', type=int, default=1024)
    parser.add_argument('--mlp_hidden_out', type=int, default=256)
    
    parser.add_argument('--model_path', type=str, default='willowOBJ.pt')
    
    # logdir parameters
    parser.add_argument('--model_name', type=str, default='MLP')
    parser.add_argument('--dataset', type=str, default='WillowOBJ')
    parser.add_argument('--exp_name', type=str, default='base_model')


    args = parser.parse_args()
    log_dir="{}_runs/{}/{}".format(args.dataset, args.model_name, args.exp_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    pre_filter = lambda data: data.pos.size(0) > 0  # noqa
    transform = T.Compose([
        T.Delaunay(),
        T.FaceToEdge(),
        T.Distance() if args.isotropic else T.Cartesian(),
    ])

    datasets = {'PascalVOC': PascalObject(transform, pre_filter),
                'WillowOBJ': WillowObject(transform),
                'Spair-71k': ""
                }
    dataset = datasets[args.dataset]
    train_dataset, test_datasets, num_node_features, num_edge_features = dataset.load_dataset()

    if dataset.have_validation_set():
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, 
                        follow_batch=['x_s', 'x_t'])
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, 
                        follow_batch=['x_s', 'x_t'])
    else:
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, 
                        follow_batch=['x_s', 'x_t'])
        val_loader = None

    # geometry-aware refinement
    psi_1 = SplineCNN(num_node_features, args.dim,
                      num_edge_features, args.num_layers, cat=False,
                      dropout=0.5)

    # Spline-CNN -> MLP -> transformer_enc -> queries
    mlp_hidden_dim = 1024
    model = SimpleNet(psi_1, args.dim, args.mlp_hidden_dim, args.mlp_hidden_out).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run(dataset, 
        train_loader, 
        test_datasets, 
        log_dir, 
        val_loader)