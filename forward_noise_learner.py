import argparse, os, random, math, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import os, csv, math, argparse, random
from typing import Tuple
from torchvision.models import resnet18
import torch.nn as nn

# -------------------------
#  Utils: seed & split
# -------------------------
def set_seed(seed: int = 20251013):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def stratified_split(y, val_ratio=0.2, seed=20251013):
    """stratified split on noisy labels y (1D numpy array)"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    idx = np.arange(len(y))
    tr_idx, val_idx = [], []
    for c in np.unique(y):
        c_idx = idx[y == c]
        rng.shuffle(c_idx)
        n_val = int(round(len(c_idx) * val_ratio))
        val_idx.append(c_idx[:n_val])
        tr_idx.append(c_idx[n_val:])
    return np.concatenate(tr_idx), np.concatenate(val_idx)

# -------------------------
#  Data helpers
# -------------------------
def infer_image_shape(x2d: np.ndarray) -> Tuple[int,int,int]:
    """Infer C,H,W from flattened vectors."""
    D = x2d.shape[1]
    if D == 28*28: return (1, 28, 28)
    if D == 32*32*3: return (3, 32, 32)
    # try square-ish gray
    s = int(round(math.sqrt(D)))
    if s*s == D: return (1, s, s)
    # fallback: 1xHxW where W=D
    return (1, 1, D)

def to_tensor_images(x: np.ndarray) -> torch.Tensor:
    x = x.astype(np.float32)
    # scale to [0,1] if looks like 0-255
    if x.max() > 1.5: x = x/255.0
    if x.ndim == 2:   # flattened
        C,H,W = infer_image_shape(x)
        x = x.reshape((-1, C, H, W))
    elif x.ndim == 3: # (N,H,W) -> (N,1,H,W)
        x = x[:, None, ...]
    elif x.ndim == 4: # (N,H,W,C) -> (N,C,H,W)
        if x.shape[-1] in (1,3):
            x = np.transpose(x, (0,3,1,2))
    return torch.from_numpy(x)

class NumpyTensorDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = to_tensor_images(X)
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return self.X.size(0)
    def __getitem__(self, i): return self.X[i], self.y[i]

# -------------------------
#  Models
# -------------------------
class SmallCNN(nn.Module):
    def __init__(self, in_ch=1, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),    nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),   nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.fc(x)
    
def make_resnet18_cifar(in_ch: int, num_classes: int):
    m = resnet18(weights=None)
    m.conv1  = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, num_classes)
    return m

# -------------------------
#  Losses (Forward correction)
# -------------------------
def row_normalize(T: torch.Tensor) -> torch.Tensor:
    return T / (T.sum(dim=1, keepdim=True) + 1e-12)

def forward_ce_loss(logits, y_noisy, T):
    probs = torch.softmax(logits, dim=1)   # [N,K]
    noisy_probs = probs @ T                # T: clean->noisy
    log_noisy = torch.log(noisy_probs + 1e-12)
    return F.nll_loss(log_noisy, y_noisy)
# -------------------------
#  Anchor-based T estimation
# -------------------------
@torch.no_grad()
def estimate_T_anchor(model, loader, num_classes, device, topk=0.02, min_prob=0.0):
    """Use validation set (noisy labels) to estimate T_hat.
       For each predicted clean class i, take high-confidence samples, tally observed noisy labels."""
    model.eval()
    probs_all, pred_all, noisy_all = [], [], []
    for x, s in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu()
        probs_all.append(probs)
        pred_all.append(probs.argmax(1))
        noisy_all.append(s)
    probs = torch.cat(probs_all)     # [N,K]
    preds = torch.cat(pred_all)      # [N]
    noisy = torch.cat(noisy_all)     # [N]
    T = torch.zeros((num_classes, num_classes), dtype=torch.float64)
    N = probs.size(0)
    for i in range(num_classes):
        idx = (preds == i).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            T[i] = torch.full((num_classes,), 1.0/num_classes, dtype=torch.float64)
            continue
       
        if min_prob > 0.0:
            keep = idx[probs[idx, i] >= min_prob]
        else:
            keep = torch.empty(0, dtype=torch.long)
        use = keep
        if use.numel() == 0:
            k = max(1, int(math.ceil(idx.numel()*topk)))
            order = probs[idx, i].argsort(descending=True)
            use = idx[order[:k]]
        hist = torch.bincount(noisy[use], minlength=num_classes).double()
        T[i] = hist / hist.sum().clamp_min(1.0)
    return T.float()

# -------------------------
#  Metrics: Acc, Macro-F1, NLL, ECE
# -------------------------
def macro_f1_from_preds(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> float:
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred): cm[t, p] += 1
    f1s = []
    for k in range(K):
        tp = cm[k,k]; fp = cm[:,k].sum()-tp; fn = cm[k,:].sum()-tp
        prec = tp / (tp+fp+1e-12); reca = tp / (tp+fn+1e-12)
        f1s.append(2*prec*reca/(prec+reca+1e-12))
    return float(np.mean(f1s))

def ece_score(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    bins = np.linspace(0.,1.,n_bins+1)
    ece = 0.0; N = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (conf > lo) & (conf <= hi)
        if not np.any(m): continue
        acc_b = (pred[m] == y_true[m]).mean()
        conf_b = conf[m].mean()
        ece += (m.mean())*abs(acc_b - conf_b)
    return float(ece)

@torch.no_grad()
def evaluate(model, loader, num_classes, device):
    model.eval()
    tot, correct, nll_sum = 0, 0, 0.0
    probs_all, y_all, p_all = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        nll_sum += F.nll_loss(torch.log(probs+1e-12), y, reduction='sum').item()
        pred = probs.argmax(1)
        correct += (pred==y).sum().item()
        tot += y.size(0)
        probs_all.append(probs.cpu()); y_all.append(y.cpu()); p_all.append(pred.cpu())
    probs = torch.cat(probs_all).numpy()
    y_true = torch.cat(y_all).numpy()
    y_pred = torch.cat(p_all).numpy()
    acc = correct / tot
    nll = nll_sum / tot
    macro_f1 = macro_f1_from_preds(y_true, y_pred, num_classes)
    ece = ece_score(probs, y_true, n_bins=15)
    return dict(acc=float(acc), macro_f1=float(macro_f1), nll=float(nll), ece=float(ece))

# -------------------------
#  Train loop (warm-up -> estimate T -> forward training)
# -------------------------
def train_one(dataset_name, Xtr, Str, Xval, Sval, Xte, Yte, T_given, C,
              epochs=50, batch_size=128, lr=1e-3, weight_decay=1e-4,
              warmup_epochs=8, topk=0.02, min_prob=0.0, mixT=0.0,
              seed=20251013, device='cpu', arch='cnn'):   

    set_seed(seed)
    num_classes = int(C)

    train_ds = NumpyTensorDataset(Xtr, Str)
    val_ds   = NumpyTensorDataset(Xval, Sval)
    test_ds  = NumpyTensorDataset(Xte, Yte)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    in_ch = train_ds.X.shape[1]
    use_resnet = (arch == 'resnet18') or (arch == 'auto' and in_ch == 3)


    print("[DEBUG] use_resnet =", use_resnet, "| arch passed =", arch)

    if use_resnet:
        m = resnet18(weights=None)            # 旧版可写 pretrained=False
        m.conv1  = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()             # CIFAR32 不需要 7x7+stride2 的池化
        m.fc = nn.Linear(512, num_classes)

        model = m.to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        model = SmallCNN(in_ch=in_ch, num_classes=num_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = None




    T_used = None
    if (T_given is not None) and (warmup_epochs == 0):
        T_used = row_normalize(T_given.clone().to(device))

    for ep in range(epochs):
        model.train()
        for x, y_noisy in train_loader:
            x, y_noisy = x.to(device), y_noisy.to(device)
            logits = model(x)
            if (T_used is None) or (ep < warmup_epochs):
                loss = F.cross_entropy(logits, y_noisy)         # warm-up
            else:
                loss = forward_ce_loss(logits, y_noisy, T_used) # forward correction
            opt.zero_grad(); loss.backward(); opt.step()

        # Switch to Forward: After the warmup ends, the first time
        if (ep+1) == warmup_epochs:
            if T_given is not None:
                T_used = row_normalize(T_given.clone().to(device))
            else:
                T_hat = estimate_T_anchor(model, val_loader, num_classes, device, topk=topk, min_prob=min_prob)
                T_hat = row_normalize(T_hat.to(device))
                if mixT > 0.0:
                    I = torch.eye(num_classes, device=device)
                    T_hat = row_normalize((1.0 - mixT)*T_hat + mixT*I)
                T_used = T_hat
        if scheduler is not None:
            scheduler.step()

    # Evaluation (clean test labels)
    metrics = evaluate(model, test_loader, num_classes, device)
    return metrics, (T_used.detach().cpu().numpy() if T_used is not None else None)

# -------------------------
#  NPZ loader & key guessing
# -------------------------
def load_npz_dataset(path: str):
    d = np.load(path, allow_pickle=True)
    keys_all = list(d.keys())

    def pick(*cands):
        for k in cands:
            if k in d: 
                return k
        raise KeyError(f"Missing any of {cands} in file. Found keys={keys_all}")

    k_Xtr = pick('Xtr','X_train','Xtrain','X_tr')
    k_Str = pick('Str','ytr','S','y_train','Ytr','Y_tr')
    k_Xte = pick('Xte','Xts','X_test','Xtest','X_te','X_ts')
    k_Yte = pick('Yte','Yts','yte','Y_test','Ytest','Y_te','Y_ts')
    k_T   = 'T' if 'T' in d else None

    Xtr, Str = d[k_Xtr], d[k_Str]
    Xte, Yte = d[k_Xte], d[k_Yte]
    T = d[k_T].astype(np.float32) if k_T else None
    C = int(max(Str.max(), Yte.max()) + 1)

    print(f"[keys] {keys_all}")
    print(f"[mapping] Xtr={k_Xtr}, Str={k_Str}, Xte={k_Xte}, Yte={k_Yte}, T={'T' if k_T else 'None'}")
    return Xtr, Str, Xte, Yte, T, C

# -------------------------
#  Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', choices=['auto','cnn','resnet18'], default='auto')
    parser.add_argument('--data', type=str, required=True, help='path to .npz')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=20251013)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--warmup-epochs', type=int, default=8)
    parser.add_argument('--topk', type=float, default=0.02, help='anchor top-k ratio if no min-prob')
    parser.add_argument('--min-prob', type=float, default=0.0, help='confidence threshold for anchors')
    parser.add_argument('--mixT', type=float, default=0.0, help='blend T_hat with identity: T <- (1-a)T + aI')
    args = parser.parse_args()

    Xtr, Str, Xte, Yte, T, C = load_npz_dataset(args.data)
    dataset_name = os.path.splitext(os.path.basename(args.data))[0]

    device = torch.device(args.device)

    os.makedirs('results', exist_ok=True)
    out_csv = os.path.join('results', f'{dataset_name}_forward_runs.csv')
    out_Tnpz = os.path.join('results', f'{dataset_name}_T_used_runs.npz')

    # CSV header
    with open(out_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['run', 'acc', 'macro_f1', 'nll', 'ece'])

    all_metrics = []
    T_collection = {}

    for r in range(args.runs):
        seed = args.seed + r
        # stratified split on noisy labels (consistent mechanism across train/val)
        tr_idx, val_idx = stratified_split(Str, val_ratio=args.val_ratio, seed=seed)
        X_tr, S_tr = Xtr[tr_idx], Str[tr_idx]
        X_val, S_val = Xtr[val_idx], Str[val_idx]

        metrics, T_used = train_one(
            dataset_name, X_tr, S_tr, X_val, S_val, Xte, Yte,
            None if T is None else torch.from_numpy(T),
            C=C,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs, topk=args.topk,
            min_prob=args.min_prob, mixT=args.mixT,
            seed=seed, device=device,
            arch=args.arch              
        )

        all_metrics.append(metrics)
        with open(out_csv, 'a', newline='') as f:
            csv.writer(f).writerow([r+1, metrics['acc'], metrics['macro_f1'], metrics['nll'], metrics['ece']])
        print(f"Run {r+1}/{args.runs}: acc={metrics['acc']:.4f}, macroF1={metrics['macro_f1']:.4f}, "
              f"NLL={metrics['nll']:.4f}, ECE={metrics['ece']:.4f}")

        if T_used is not None:
            T_collection[f'T_run{r+1}'] = T_used

    # aggregate
    accs  = np.array([m['acc'] for m in all_metrics], dtype=np.float64)
    f1s   = np.array([m['macro_f1'] for m in all_metrics], dtype=np.float64)
    nlls  = np.array([m['nll'] for m in all_metrics], dtype=np.float64)
    eces  = np.array([m['ece'] for m in all_metrics], dtype=np.float64)

    mean_acc, std_acc = accs.mean(), accs.std()
    mean_f1,  std_f1  = f1s.mean(),  f1s.std()
    mean_nll, std_nll = nlls.mean(), nlls.std()
    mean_ece, std_ece = eces.mean(), eces.std()

    print(f"==> {dataset_name}: mean±std acc = {mean_acc:.4f} ± {std_acc:.4f}")
    with open(out_csv, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mean', mean_acc, mean_f1, mean_nll, mean_ece])
        w.writerow(['std',  std_acc,  std_f1,  std_nll,  std_ece])

    if len(T_collection):
        np.savez(out_Tnpz, **T_collection)

if __name__ == '__main__':
    main()



















