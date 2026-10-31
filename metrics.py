"""
Clustering evaluation metrics and inference utilities.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    accuracy_score,
    fowlkes_mallows_score,
    precision_score,
    recall_score,
    v_measure_score,
)
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def cluster_acc(y_true, y_pred):
    """Clustering accuracy via optimal label alignment (Hungarian algorithm)."""
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum(w[i, j] for i, j in zip(row_ind, col_ind)) / y_pred.size


def purity(y_true, y_pred):
    """Cluster purity: each cluster assigned to its majority class."""
    y_voted = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered = np.arange(len(labels))
    for k in range(len(labels)):
        y_true[y_true == labels[k]] = ordered[k]
    bins = np.concatenate([ordered, [ordered[-1] + 1]])
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        y_voted[y_pred == cluster] = np.argmax(hist)
    return accuracy_score(y_true, y_voted)


def evaluate(label, pred):
    """Compute all clustering metrics."""
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    f = fowlkes_mallows_score(label, pred)
    precision = precision_score(label, pred, average='macro',zero_division=0)
    recall = recall_score(label, pred, average='macro')
    return nmi, ari, acc, pur, f, precision, recall


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def inference(loader, model, view, data_size, class_num, device):
    """Run inference using model.forward_eval() (VACF + CIR handled internally).

    Args:
        loader: DataLoader
        model: DLC_MVC model
        view: number of views
        data_size: total number of samples
        class_num: number of clusters K
        device: torch device

    Returns:
        total_pred: consensus cluster predictions [N]
        fused_features: fused feature B [N, fuse_dim]
        labels_vector: ground truth labels [N]
        Hs_raw: raw consistency features (for diagnostic)
        Zs_raw: complementarity features (per-view, pre-VACF)
    """

    model.eval()
    fused_list = []
    Hs_raw = [[] for _ in range(view)]
    Zs_raw = [[] for _ in range(view)]
    labels_vector = []
    Qs_all = [[] for _ in range(view)]

    for xs, y, _ in loader:
        for v in range(view):
            xs[v] = xs[v].to(device)

        with torch.no_grad():
            Qs, B, Hs, Zs, _ = model.forward_eval(xs)

        fused_list.append(B.cpu().numpy())
        for v in range(view):
            Hs_raw[v].append(Hs[v].cpu().numpy())
            Zs_raw[v].append(Zs[v].cpu().numpy())
            Qs_all[v].append(Qs[v].cpu().numpy())
        labels_vector.extend(y.numpy())

    labels_vector = np.array(labels_vector).reshape(data_size)
    fused_features = np.concatenate(fused_list, axis=0)

    # Average Q across views for consensus prediction
    Qs_avg = np.zeros((data_size, class_num))
    for v in range(view):
        Qs_avg += np.concatenate(Qs_all[v], axis=0)
    Qs_avg /= view
    total_pred = np.argmax(Qs_avg, axis=1)

    for v in range(view):
        Hs_raw[v] = np.concatenate(Hs_raw[v], axis=0)
        Zs_raw[v] = np.concatenate(Zs_raw[v], axis=0)

    return total_pred, fused_features, labels_vector, Hs_raw, Zs_raw


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def valid(model, device, dataset, view, data_size, class_num):
    """Validate clustering performance.

    CIR is controlled internally by model.use_cir.

    Args:
        model: DLC_MVC model
        device: torch device
        dataset: PyTorch Dataset
        view: number of views
        data_size: dataset size
        class_num: number of clusters K

    Returns:
        acc, nmi, pur, ari
    """
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    total_pred, fused_features, labels_vector, Hs_raw, Zs_raw = inference(
        loader, model, view, data_size, class_num, device,
    )

    nmi, ari, acc, pur, f, precision, recall = evaluate(labels_vector, total_pred)
    # print(f'ACC = {acc:.4f}  NMI = {nmi:.4f}  ARI = {ari:.4f}  PUR = {pur:.4f}')
    return acc, nmi, pur, ari


def valid_detailed(model, device, dataset, view, data_size, class_num):
    """Detailed validation: consensus + per-view clustering quality.

    Also reports K-means clustering on H and Z features for diagnostic purposes.
    CIR is controlled internally by model.use_cir.
    """
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    total_pred, fused_features, labels_vector, Hs_raw, Zs_raw = inference(
        loader, model, view, data_size, class_num, device,
    )

    print('=' * 70)
    print('Consensus clustering (averaged Q across views):')
    nmi, ari, acc, pur = evaluate(labels_vector, total_pred)[:4]
    print(f'  ACC = {acc:.4f}  NMI = {nmi:.4f}  ARI = {ari:.4f}  PUR = {pur:.4f}')

    print('-' * 70)
    print('K-means on fused feature B:')
    km = KMeans(n_clusters=class_num, n_init=50, random_state=0)
    y_pred_b = km.fit_predict(fused_features)
    nmi_b, ari_b, acc_b, pur_b = evaluate(labels_vector, y_pred_b)[:4]
    print(f'  ACC = {acc_b:.4f}  NMI = {nmi_b:.4f}  ARI = {ari_b:.4f}  PUR = {pur_b:.4f}')

    print('-' * 70)
    print('K-means on H features (consistency space, per view):')
    for v in range(view):
        y_pred_h = KMeans(n_clusters=class_num, n_init=50, random_state=0).fit_predict(Hs_raw[v])
        nmi_h, ari_h, acc_h, pur_h = evaluate(labels_vector, y_pred_h)[:4]
        print(f'  View {v+1}: ACC = {acc_h:.4f}  NMI = {nmi_h:.4f}  ARI = {ari_h:.4f}  PUR = {pur_h:.4f}')

    print('-' * 70)
    print('K-means on Z features (complementarity space, per view):')
    for v in range(view):
        y_pred_z = KMeans(n_clusters=class_num, n_init=50, random_state=0).fit_predict(Zs_raw[v])
        nmi_z, ari_z, acc_z, pur_z = evaluate(labels_vector, y_pred_z)[:4]
        print(f'  View {v+1}: ACC = {acc_z:.4f}  NMI = {nmi_z:.4f}  ARI = {ari_z:.4f}  PUR = {pur_z:.4f}')

    return acc, nmi, pur, ari
