import argparse, os, random, math, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

# -----------------------
# Utils
# -----------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def row_normalize(T: torch.Tensor, eps=1e-8):
    T = T.clamp_min(eps)
    return T / T.sum(dim=1, keepdim=True)  # 行和=1: P(ŷ=j|Y=i)

def to_tensor_images(x):
    x = x.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0

    # 扁平 [N, D] -> 还原为图片
    if x.ndim == 2:
        N, D = x.shape
        if D == 28*28:            # FashionMNIST
            x = x.reshape(N, 1, 28, 28)
            return torch.from_numpy(x)
        elif D == 32*32*3:        # CIFAR 扁平
            x = x.reshape(N, 3, 32, 32)
            return torch.from_numpy(x)
        elif D == 32*32:          # 以防灰度 32x32
            x = x.reshape(N, 1, 32, 32)
            return torch.from_numpy(x)
        else:
            raise ValueError(f"Unknown flattened image size D={D}")

    # 非扁平情况
    if x.ndim == 3:               # [N,H,W] 灰度
        x = x[:, None, :, :]
    elif x.ndim == 4:             # [N,H,W,C] 彩色
        x = np.transpose(x, (0, 3, 1, 2))
    else:
        raise ValueError(f"Unexpected image shape: {x.shape}")
    return torch.from_numpy(x)

def guess_keys(keys):
    keys = list(keys)

    def find_any(cands):
        # 先精确再包含（用于 Xte/Xts、Yte/Yts 等）
        for name in cands:
            for k in keys:
                if k.lower() == name.lower():
                    return k
        for name in cands:
            for k in keys:
                if name.lower() in k.lower():
                    return k
        return None

    def find_exact(cands):
        for name in cands:
            for k in keys:
                if k.lower() == name.lower():
                    return k
        return None

    kXtr = find_any(["Xtr","x_train","trainX","images_tr"])
    kStr = find_any(["Str","y_train","trainY","labels_tr"])
    kXte = find_any(["Xte","Xts","x_test","testX","images_te","images_ts","xts"])
    kYte = find_any(["Yte","Yts","Ste","y_test","testY","labels_te","labels_ts","yts"])
    kT   = find_exact(["T","transition","noise_T","trans"])  # 只精确匹配！

    print(f"[guess_keys] -> Xtr={kXtr}, Str={kStr}, Xte={kXte}, Yte={kYte}, T={kT}")
    return kXtr, kStr, kXte, kYte, kT

def load_npz_dataset(path):
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    kXtr, kStr, kXte, kYte, kT = guess_keys(keys)
    assert kXtr and kStr and kXte and kYte, f"NPZ keys not found. Found: {keys}"

    Xtr = to_tensor_images(data[kXtr])
    Str = torch.from_numpy(data[kStr].astype(np.int64))
    Xte = to_tensor_images(data[kXte])
    Yte = torch.from_numpy(data[kYte].astype(np.int64))

    T = None
    if kT is not None:
        T_np = data[kT].astype(np.float32)
        T = torch.from_numpy(T_np)

    num_classes = int(max(Str.max().item(), Yte.max().item()) + 1)
    return Xtr, Str, Xte, Yte, T, num_classes

# -----------------------
# Small CNN backbone
# -----------------------
class SmallCNN(nn.Module):
    def __init__(self, in_ch=1, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)

# -----------------------
# Losses
# -----------------------
def forward_ce_loss(logits, y_noisy, T_row_stoch):
    p = F.softmax(logits, dim=1)            # [B,C]
    p_tilde = torch.matmul(p, T_row_stoch)  # [B,C]
    log_p_tilde = torch.log(p_tilde.clamp_min(1e-12))
    return F.nll_loss(log_p_tilde, y_noisy)

def gce_loss(logits, y_noisy, q=0.7, eps=1e-12):
    p = F.softmax(logits, dim=1)
    p_y = p.gather(1, y_noisy.view(-1,1)).squeeze(1).clamp_min(eps)
    return ((1 - p_y.pow(q)) / q).mean()

# -----------------------
# Anchor-based T estimation
# -----------------------
@torch.no_grad()
def estimate_T_anchor(model, loader, num_classes, device, topk=0.02):
    model.eval()
    preds, probs, noisy = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = F.softmax(logits, dim=1)
        pr, pd = p.max(dim=1)
        preds.append(pd.cpu()); probs.append(pr.cpu()); noisy.append(yb)
    preds = torch.cat(preds); probs = torch.cat(probs); noisy = torch.cat(noisy)

    T = torch.zeros(num_classes, num_classes, dtype=torch.float64)
    for i in range(num_classes):
        idx = (preds == i).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            T[i] = torch.full((num_classes,), 1.0/num_classes, dtype=torch.float64)
            continue
        k = max(1, int(math.ceil(idx.numel() * topk)))
        top = idx[probs[idx].argsort(descending=True)[:k]]
        hist = torch.bincount(noisy[top], minlength=num_classes).double()
        T[i] = hist / hist.sum().clamp_min(1.0)
    return row_normalize(T.float())

# -----------------------
# Train / evaluate
# -----------------------
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            pred = out.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / total

def train_one(dataset_name, Xtr, Str, Xte, Yte, T_given, num_classes,
              epochs=50, batch_size=128, lr=1e-3, weight_decay=1e-4,
              estimate_T=False, warmup_epochs=8, topk=0.02, seed=0, device='cpu'):
    set_seed(seed)

    n = len(Xtr)
    n_val = int(0.2 * n)
    n_tr  = n - n_val
    ds_all = TensorDataset(Xtr, Str)
    tr_set, val_set = random_split(ds_all, [n_tr, n_val], generator=torch.Generator().manual_seed(seed))
    te_set = TensorDataset(Xte, Yte)

    # Windows 建议 num_workers=0
    tr_loader  = DataLoader(tr_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    te_loader  = DataLoader(te_set,  batch_size=batch_size, shuffle=False, num_workers=0)

    in_ch = Xtr.shape[1]
    model = SmallCNN(in_ch=in_ch, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state = -1.0, None

    # 先决定 T 的来源
    T_used = None
    if T_given is not None and not estimate_T:
        T_used = row_normalize(T_given.to(device))

    # 未知 T：warmup -> 估计 T̂
    if (T_used is None) and estimate_T:
        for _ in range(warmup_epochs):
            model.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = F.cross_entropy(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
        T_hat = estimate_T_anchor(model, val_loader, num_classes, device=device, topk=topk)
        T_used = T_hat.to(device)

    # 正式训练（Forward 或 CE）
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = forward_ce_loss(logits, yb, T_used) if T_used is not None else F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        val_acc = accuracy(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc = accuracy(model, te_loader, device)
    return test_acc, (T_used.detach().cpu().numpy() if T_used is not None else None)

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--estimate-T', action='store_true')
    parser.add_argument('--warmup-epochs', type=int, default=8)
    parser.add_argument('--topk', type=float, default=0.02)
    parser.add_argument('--device', type=str, default='cpu')  # 想用GPU就传 --device cuda
    args = parser.parse_args()

    assert os.path.exists(args.data), f"not found: {args.data}"
    Xtr, Str, Xte, Yte, T, C = load_npz_dataset(args.data)

    dataset_name = os.path.splitext(os.path.basename(args.data))[0]
    os.makedirs('results', exist_ok=True)
    out_csv = os.path.join('results', f'{dataset_name}_forward_runs.csv')
    out_Tnpz = os.path.join('results', f'{dataset_name}_T_used_runs.npz')

    all_acc, T_list = [], []
    for r in range(args.runs):
        seed = 2025_10_13 + r
        acc, T_used = train_one(
            dataset_name, Xtr, Str, Xte, Yte, T, C,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
            estimate_T=args.estimate_T or (T is None),
            warmup_epochs=args.warmup_epochs, topk=args.topk, seed=seed, device=args.device
        )
        all_acc.append(acc); T_list.append(T_used)
        print(f'Run {r+1}/{args.runs}: test acc = {acc:.4f}')

    mean = float(np.mean(all_acc)); std = float(np.std(all_acc))
    print(f'==> {dataset_name}: mean±std test acc = {mean:.4f} ± {std:.4f}')

    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'test_acc'])
        for i, a in enumerate(all_acc, 1): w.writerow([i, a])
        w.writerow(['mean', mean]); w.writerow(['std', std])

    np.savez(out_Tnpz, **{f'T_run{i+1}': T for i, T in enumerate(T_list) if T is not None})

if __name__ == '__main__':
    main()


