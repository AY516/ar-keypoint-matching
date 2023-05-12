import torch 

def compute_grad_norm(model_params):
    return torch.norm(torch.stack([torch.norm(p.grad) for p in model_params]))

def f1_score(tp, fp, fn):

    device = tp.device

    const = torch.tensor(1e-7, device=device)
    precision = tp / (tp + fp + const)
    recall = tp / (tp + fn + const)
    f1 = 2 * precision * recall / (precision + recall + const)
    return f1

def acc(batch_size, num_nodes_s, y_pred):

    device = y_pred.device

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


def make_perm_mat_pred(matching_vec, num_nodes_t):

    device = matching_vec.device

    batch_size = matching_vec.size()[0]
    nodes = matching_vec.size()[1]
    perm_mat_pred = []
    for i in range(batch_size):
        row_idx = torch.arange(nodes)
        one_hot_pred = torch.zeros(nodes, num_nodes_t)
        index = matching_vec[i, :]
        one_hot_pred[row_idx, index] = 1
        perm_mat_pred.append(one_hot_pred)
    
    return torch.stack(perm_mat_pred)

def matching_accuracy(pmat_pred, pmat_gt):
    """
    Matching Accuracy between predicted permutation matrix and ground truth permutation matrix.
    :param pmat_pred: predicted permutation matrix
    :param pmat_gt: ground truth permutation matrix
    :param ns: number of exact pairs
    :return: matching accuracy, matched num of pairs, total num of pairs
    """
    device = pmat_pred.device
    batch_num = pmat_pred.shape[0]

    pmat_gt = pmat_gt.to(device)

    assert torch.all((pmat_pred == 0) + (pmat_pred == 1)), "pmat_pred can noly contain 0/1 elements."
    assert torch.all((pmat_gt == 0) + (pmat_gt == 1)), "pmat_gt should noly contain 0/1 elements."
    assert torch.all(torch.sum(pmat_pred, dim=-1) <= 1) and torch.all(torch.sum(pmat_pred, dim=-2) <= 1)
    assert torch.all(torch.sum(pmat_gt, dim=-1) <= 1) and torch.all(torch.sum(pmat_gt, dim=-2) <= 1)

    match_num = 0
    total_num = 0

    for b in range(batch_num):
        match_num += torch.sum(pmat_pred[b] * pmat_gt[b])
        total_num += torch.sum(pmat_gt[b])

    return match_num / total_num, match_num, total_num