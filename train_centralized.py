"""
Centralized baseline for NinaPro DB2 Exercise B with EMG_CNN / EMG_ATTN.
Usage:
    .venv/Scripts/python train_centralized.py --epochs 150 --batchsize 256 --lr 0.001 --cuda 0
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, '.')
from dataset import DatasetObject
from models import client_model

parser = argparse.ArgumentParser()
parser.add_argument('--epochs',     default=150,   type=int)
parser.add_argument('--batchsize',  default=256,   type=int)
parser.add_argument('--lr',         default=0.001, type=float)
parser.add_argument('--wd',         default=1e-4,  type=float)
parser.add_argument('--noise-std',  default=0.05,  type=float, help='Gaussian noise std (0=off)')
parser.add_argument('--amp-scale',  default=0.2,   type=float, help='amplitude jitter range ±amp_scale (0=off)')
parser.add_argument('--ch-drop',    default=0.1,   type=float, help='per-channel zero-out prob (0=off)')
parser.add_argument('--label-smooth', default=0.1, type=float, help='label smoothing epsilon (0=off)')
parser.add_argument('--n-subjects', default=40,    type=int,   help='informational only')
parser.add_argument('--model',      default='EMG_ATTN', type=str, choices=['EMG_CNN', 'EMG_ATTN', 'CNNATTEN'])
parser.add_argument('--seed',       default=42,    type=int)
parser.add_argument('--data-file',  default='./',  type=str)
parser.add_argument('--cuda',       default='0',   type=str,   help="GPU id or 'cpu'")
parser.add_argument('--vote-k',     default=1,     type=int,   help='majority-vote window count (1=off)')
parser.add_argument('--random-split', action='store_true', default=False,
                    help='use random 80/20 split instead of cross-repetition benchmark split')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device('cpu') if (args.cuda == 'cpu' or not torch.cuda.is_available()) \
         else torch.device('cuda:' + args.cuda)
print('Device:', device)

# ── Load data ──────────────────────────────────────────────────────────────────
# n_client=1, rule='iid' → all 40 subjects pooled, no subject-balancing data loss
print('Loading NinaPro data (%d subjects)...' % args.n_subjects)
data_obj = DatasetObject(
    dataset='ninapro', n_client=1, seed=args.seed,
    rule='iid', data_path=args.data_file,
    split='random' if args.random_split else 'cross_rep'
)

train_x = torch.tensor(data_obj.client_x.reshape(-1, 1, 12, data_obj.width)).float()
train_y = torch.tensor(data_obj.client_y.reshape(-1)).long()
test_x  = torch.tensor(data_obj.test_x).float()
test_y  = torch.tensor(data_obj.test_y.reshape(-1)).long()

print('Train samples: %d  |  Test samples: %d  |  Classes: %d'
      % (len(train_x), len(test_x), data_obj.n_cls))

train_loader = DataLoader(TensorDataset(train_x, train_y),
                          batch_size=args.batchsize, shuffle=True,  drop_last=False)
test_loader  = DataLoader(TensorDataset(test_x,  test_y),
                          batch_size=256,            shuffle=False, drop_last=False)

# ── Model ──────────────────────────────────────────────────────────────────────
model     = client_model(args.model, n_cls=data_obj.n_cls).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
# CosineAnnealingWarmRestarts: restarts every T_0 epochs, doubling period each time.
# Avoids the hard LR floor of StepLR; periodically escapes local minima.
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=30, T_mult=2, eta_min=1e-6
)
criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('%s parameters: %d\n' % (args.model, total_params))
print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Test Acc':>10} {'LR':>10}")
print('-' * 57)

best_test_acc = 0.0

# ── Training loop ──────────────────────────────────────────────────────────────
for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        # ── EMG augmentations ────────────────────────────────────────────────
        # 1. Gaussian noise (broadband — simulates sensor noise)
        if args.noise_std > 0:
            x = x + args.noise_std * torch.randn_like(x)

        # 2. Amplitude scaling per sample (simulates electrode impedance variation
        #    between repetitions — the primary source of cross-rep variability)
        if args.amp_scale > 0:
            scale = 1.0 + args.amp_scale * (2 * torch.rand(x.size(0), 1, 1, 1, device=device) - 1)
            x = x * scale

        # 3. Random channel dropout (simulates occasional electrode loss)
        if args.ch_drop > 0:
            mask = (torch.rand(x.size(0), 1, 12, 1, device=device) > args.ch_drop).float()
            x = x * mask
        # ────────────────────────────────────────────────────────────────────

        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (out.argmax(1) == y).sum().item()
        total      += len(y)
    scheduler.step()

    train_loss = total_loss / total
    train_acc  = correct / total * 100

    model.eval()
    correct_t, total_t = 0, 0
    with torch.no_grad():
        if args.vote_k <= 1:
            # Standard per-window evaluation
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out   = model(x)
                correct_t += (out.argmax(1) == y).sum().item()
                total_t   += len(y)
        else:
            # Majority voting: average softmax over K consecutive windows, then argmax.
            # Consecutive windows share the same gesture label (same 200ms segment).
            all_probs = []
            all_y     = []
            for x, y in test_loader:
                x = x.to(device)
                probs = torch.softmax(model(x), dim=1).cpu()
                all_probs.append(probs)
                all_y.append(y)
            all_probs = torch.cat(all_probs, dim=0)  # (N, n_cls)
            all_y     = torch.cat(all_y,     dim=0)  # (N,)
            K = args.vote_k
            n = len(all_y)
            for i in range(0, n - K + 1, K):
                avg_p = all_probs[i:i+K].mean(0)
                pred  = avg_p.argmax().item()
                true  = all_y[i].item()          # all K windows share the same label
                correct_t += int(pred == true)
                total_t   += 1
            # handle tail windows (< K remaining)
            tail = n - (n // K) * K
            if tail > 0:
                avg_p = all_probs[n - tail:].mean(0)
                pred  = avg_p.argmax().item()
                true  = all_y[n - tail].item()
                correct_t += int(pred == true)
                total_t   += 1
    test_acc = correct_t / total_t * 100

    if test_acc > best_test_acc:
        best_test_acc = test_acc

    lr_now = optimizer.param_groups[0]['lr']
    print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>9.2f}% {test_acc:>9.2f}% {lr_now:>10.6f}")

print(f'\nBest test accuracy: {best_test_acc:.2f}%')
print('Done.')
