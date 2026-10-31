"""
DLC-MVC: Decoupled Feature Learning for Contrastive Multi-View Clustering
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datetime import datetime
from models import DLC_MVC
from losses import TotalLoss
from dataloader import load_data
from utils import (
    setup_seed,
    compute_target_distribution,
    save_model,
)
from metrics import valid, valid_detailed
import pandas as pd

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
DataSet = 'Handwritten'

def parse_args():
    parser = argparse.ArgumentParser(description='DLC-MVC Training')

    # Dataset
    parser.add_argument('--dataset', type=str, default=DataSet,
                        choices=['Handwritten', 'Synthetic3D'])
    parser.add_argument('--data_dir', type=str, default='./data/')

    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--pretrain_epochs', type=int, default=200)
    parser.add_argument('--fine_epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--T', type=int, default=1, help='Number of repeated runs')

    # Model architecture
    parser.add_argument('--enc_dim', type=int, default=256)
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--fuse_dim', type=int, default=128)

    # Loss weights
    parser.add_argument('--alpha', type=float, default=1,
                        help='Weight for contrastive loss L_con')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Weight for diversity loss L_div + L_orth')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Weight for clustering loss L_clc + L_clu')

    # Contrastive temperatures
    parser.add_argument('--temperature_f', type=float, default=0.5,
                        help='Instance-level contrastive temperature')
    parser.add_argument('--temperature_l', type=float, default=0.5,
                        help='Label-level contrastive temperature')


    # Misc
    parser.add_argument('--pretrain_path', type=str, default='./pretrain_models/')
    parser.add_argument('--k_means_round', type=int, default=10,
                        help='K-means target update interval (epochs)')
    parser.add_argument('--ema_momentum', type=float, default=0.1,
                        help='EMA momentum for target distribution smoothing (0=no EMA, 1=no smoothing)')

    parser.add_argument('--pretrain',default=True)

    args = parser.parse_args()
    return args

args = parse_args()


# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------

def pretrain_epoch(model, data_loader, view, optimizer, device):
    """One epoch of autoencoder pretraining (L_rec only)."""
    model.train()
    mse = torch.nn.MSELoss()
    tot_loss = 0.0

    for xs, _, _ in data_loader:
        for v in range(view):
            xs[v] = xs[v].to(device)

        optimizer.zero_grad()
        _, _, xrs = model(xs)

        loss = sum(mse(xs[v], xrs[v]) for v in range(view))
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    return tot_loss / len(data_loader)


def pretrain(model, data_loader, view, device, args):
    """Full pretraining phase."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(1, args.pretrain_epochs + 1):
        avg_loss = pretrain_epoch(model, data_loader, view, optimizer, device)
        if epoch % 50 == 0 or epoch == 1:
            print(f'  Pretrain Epoch {epoch:3d} | Loss = {avg_loss:.6f}')

    # Save pretrained model
    state = {
        'model_state_dict': model.state_dict(),
        'epoch': args.pretrain_epochs,
    }
    # torch.save(model.state_dict(), args.pretrain_path+args.dataset+'.pth')
    return model


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def finetune_epoch(model, criterion, data_loader, view, class_num, optimizer,
                   device, args, epoch, P_hat_global):
    """One epoch of joint fine-tuning with full objective.

    Args:
        P_hat_global: cached target distribution [N, K] for the full dataset
                      (updated periodically via K-means on B)
    """
    model.train()
    tot_loss = 0.0
    loss_log = {'rec': 0, 'con': 0, 'div': 0, 'orth': 0, 'clc': 0, 'clu': 0}

    for batch_idx, (xs, y, idx) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        batch_idx_np = idx.numpy()

        optimizer.zero_grad()

        # 1. Encode + decode
        Hs, Zs, xrs = model.forward(xs)

        # 2. Fuse + cluster assignments (VACF + CIR inside model)
        B, B_list, B_refined_list, Qs, Zs_out, weights,_ = model.forward_fuse(xs)

        # 4. Get target distribution for this batch
        if P_hat_global is not None:
            P_hat_batch = P_hat_global[batch_idx_np].to(device)
        else:
            P_hat_batch, _ = compute_target_distribution(B, class_num)

        # 5. Compute total loss
        total, loss_dict = criterion.forward(xs, xrs, Hs, Zs_out, Qs, P_hat_batch)
        total.backward()
        optimizer.step()

        tot_loss += total.item()
        for k in loss_log:
            loss_log[k] += loss_dict[k]

    N_batches = len(data_loader)
    return tot_loss / N_batches, {k: v / N_batches for k, v in loss_log.items()}


def finetune(model, data_loader, dataset, view, data_size, class_num, device, args):
    """Full fine-tuning phase.

    Updates target distribution every 10 epochs via K-means on fused features B.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = TotalLoss(
        class_num=class_num,
        batch_size=args.batch_size,
        temperature_f=args.temperature_f,
        temperature_l=args.temperature_l,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        device=device,
    ).to(device)

    best_acc = 0.0
    best_nmi = 0.0
    best_pur = 0.0
    best_ari = 0.0
    best_state = None
    best_epoch = 0

    history = {'loss': [], 'acc': [], 'nmi': [], 'ari': [],
               'con': [], 'div': [], 'orth': [], 'clc': [], 'clu': [],'pur': []}

    # Full-data loader (no drop_last) for computing target distribution over ALL samples
    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


    # Initial target distribution via K-means on fused features
    P_hat_global = None

    for epoch in range(1, args.fine_epochs + 1):
        # Update target distribution every k_means_round epochs, with EMA smoothing
        if epoch == 1 or epoch % args.k_means_round == 0:
            P_hat_new = _compute_global_target(
                model, full_loader, view, class_num, device, args
            )
            if P_hat_global is None:
                P_hat_global = P_hat_new
            else:
                P_hat_global = args.ema_momentum * P_hat_new + (1 - args.ema_momentum) * P_hat_global

        avg_loss, loss_log = finetune_epoch(
            model, criterion, data_loader, view, class_num,
            optimizer, device, args, epoch, P_hat_global
        )

        # Validation
        acc, nmi, pur, ari = valid(model, device, dataset, view, data_size, class_num)

        if epoch == args.fine_epochs:
            visualization_data = _collect_visualization_features(
                model, full_loader, view, device
            )
            feature_path = os.path.join(
                'pth', f'{args.dataset}_features_epoch{epoch}.pth'
            )
            os.makedirs(os.path.dirname(feature_path), exist_ok=True)
            # torch.save(visualization_data, feature_path)
            # print(f'Visualization features saved to: {feature_path}')

        # Record history
        history['loss'].append(avg_loss)
        history['acc'].append(acc)
        history['nmi'].append(nmi)
        history['ari'].append(ari)
        history['con'].append(loss_log['con'])
        history['div'].append(loss_log['div'])
        history['orth'].append(loss_log['orth'])
        history['clc'].append(loss_log['clc'])
        history['clu'].append(loss_log['clu'])
        history['pur'].append(pur)

        if acc > best_acc:
            best_acc = acc
            best_nmi = nmi
            best_pur = pur
            best_ari = ari
            best_epoch = epoch


            best_state = _collect_visualization_features(
                model, full_loader, view, device
            )
            best_state.update({
                'epoch': epoch,
                'acc': acc,
                'nmi': nmi,
                'pur': pur,
                'ari': ari,
            })
            checkpoint_path = os.path.join(
                'pth', f'{args.dataset}_best_model.pth'
            )
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            # torch.save(best_state, checkpoint_path)
            # print(f'Best-epoch features saved to: {checkpoint_path} '
            #       f'(epoch={epoch}, ACC={acc:.4f})')

        if epoch % args.k_means_round == 0 or epoch == 1:
            print(f'  Fine-tune Epoch {epoch:3d} | Loss = {avg_loss:.4f}')

        if epoch==45:
            continue
    # _save_history_to_excel(history, args.dataset)
    return best_acc, best_nmi, best_pur, best_ari, best_state, best_epoch, history


def _save_history_to_excel(history, dataset_name, save_dir='./result/'):
    """Save training history (loss, acc, nmi, pur) to Excel file.

    Args:
        history: Dictionary containing training history arrays
        dataset_name: Name of the dataset for filename generation
        save_dir: Directory to save the Excel file
    """
    os.makedirs(save_dir, exist_ok=True)

    # Prepare data for Excel
    epochs = list(range(1, len(history['loss']) + 1))
    data = {
        'Epoch': epochs,
        'Loss': history['loss'],
        'ACC': history['acc'],
        'NMI': history['nmi'],
        'PUR': history['pur']
    }

    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{dataset_name}_training_history_{timestamp}.xlsx'
    filepath = os.path.join(save_dir, filename)

    df = pd.DataFrame(data)
    df.to_excel(filepath, index=False, engine='openpyxl')
    print(f'Training history saved to: {filepath}')


def _compute_global_target(model, full_loader, view, class_num, device, args):
    """Compute target distribution P̂ over ALL dataset samples using K-means on B.

    Uses a full-data loader (no drop_last) so P_hat covers every sample index.
    """
    model.eval()
    all_B = []
    with torch.no_grad():
        for xs, _, _ in full_loader:
            for v in range(view):
                xs[v] = xs[v].to(device)
            B, _, _, _, _, _,_ = model.forward_fuse(xs)
            all_B.append(B.cpu().numpy())
    all_B = np.concatenate(all_B, axis=0)
    P_hat, _ = compute_target_distribution(all_B, class_num)
    return P_hat


def _collect_visualization_features(model, full_loader, view, device):
    """Collect all view-specific features and fused features in dataset order."""
    model.eval()
    all_raw = [[] for _ in range(view)]
    all_H = [[] for _ in range(view)]
    all_Bv = [[] for _ in range(view)]
    all_B, all_labels = [], []
    with torch.no_grad():
        for xs, labels, _ in full_loader:
            for v in range(view):
                xs[v] = xs[v].to(device)
            B, B_list, _, _, _, _, Hs = model.forward_fuse(xs)
            for v in range(view):
                all_raw[v].append(xs[v].cpu())
                all_H[v].append(Hs[v].cpu())
                all_Bv[v].append(B_list[v].cpu())
            all_B.append(B.cpu())
            all_labels.append(labels.cpu())
    return {
        'raw_views': [torch.cat(features, dim=0) for features in all_raw],
        'H_views': [torch.cat(features, dim=0) for features in all_H],
        'B_views': [torch.cat(features, dim=0) for features in all_Bv],
        'B_fused': torch.cat(all_B, dim=0),
        'labels': torch.cat(all_labels, dim=0),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():


    # Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #setup_seed(args.seed)

    # Load data
    dataset, dims, view, data_size, class_num = load_data(args.dataset, args.data_dir)
    args.batch_size = min(args.batch_size, len(dataset))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print('=' * 70)
    print(f'DLC-MVC  |  Dataset: {args.dataset}  |  Samples: {data_size}  '
          f'Views: {view}  |  Classes: {class_num}')
    print(f'Device: {device}  |  α={args.alpha}  β={args.beta}  γ={args.gamma}')
    print('=' * 70)

    # Multiple runs
    results = {'acc': [], 'nmi': [], 'pur': [], 'ari': []}
    best_overall = {'acc': 0.0, 'state': None}

    for run in range(args.T):
        if args.T > 1:
            print(f'\n{"="*50}\n  RUN {run + 1} / {args.T}\n{"="*50}')
            setup_seed(args.seed + run)

        t_start = time.time()

        # Initialize model
        model = DLC_MVC(
            view=view, dims=dims,
            enc_dim=args.enc_dim, proj_dim=args.proj_dim,
            fuse_dim=args.fuse_dim, class_num=class_num,
        ).to(device)

        # Stage 1: Pretraining
        print('\n[Stage 1] Pretraining (reconstruction only)...')
        t1 = time.time()
        if args.pretrain:
            model = pretrain(model, data_loader, view, device, args)
        else:
            model.load_state_dict(torch.load(args.pretrain_path+args.dataset+'.pth'))
        print(f'  Pretraining done in {time.time() - t1:.1f}s')

        # Stage 2: Fine-tuning
        print(f'\n[Stage 2] Fine-tuning (full objective)...')
        t2 = time.time()
        best_acc, best_nmi, best_pur, best_ari, best_state, best_epoch, history = finetune(
            model, data_loader, dataset, view, data_size, class_num, device, args
        )
        print(f'  Fine-tuning done in {time.time() - t2:.1f}s')


        # Final evaluation
        acc, nmi, pur, ari = valid(
            model, device, dataset, view, data_size, class_num,
        )


        results['acc'].append(acc)
        results['nmi'].append(nmi)
        results['pur'].append(pur)
        results['ari'].append(ari)

        if acc > best_overall['acc']:
            best_overall['acc'] = acc
            best_overall['state'] = best_state

        elapsed = time.time() - t_start
        print(f'\n  Run {run+1} finished in {elapsed:.1f}s | '
              f'Best ACC = {best_acc:.4f} (NMI={best_nmi:.4f} PUR={best_pur:.4f} ARI={best_ari:.4f})')



if __name__ == '__main__':
    main()
