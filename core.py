
# core.py (patched loader for flat arrays)
import random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import timm

# -------------------- Utils --------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _infer_hw_from_flat(D: int, channels: int):
    """Infer H=W from total dimension D and channels. Assumes square images."""
    if D % channels != 0:
        raise ValueError(f"Cannot infer HxW: D={D} is not divisible by C={channels}.")
    hw = D // channels
    side = int(round(np.sqrt(hw)))
    if side * side != hw:
        raise ValueError(f"Cannot infer square size from D/C={hw}. Not a perfect square.")
    return side, side

def to_chw_tensor(X: np.ndarray, grayscale: bool) -> torch.Tensor:
    """
    Accepts X with shape:
      - (N, H, W)                    -> adds channel to (N, H, W, 1)
      - (N, H, W, C)                 -> unchanged
      - (N, D) flattened             -> infers square H=W; C=1 if grayscale else 3
    Returns torch tensor shaped (N, C, H, W)
    """
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if X.ndim == 2:  # flattened
        N, D = X.shape
        C = 1 if grayscale else 3
        H, W = _infer_hw_from_flat(D, C)
        if grayscale:
            X = X.reshape(N, H, W, 1)
        else:
            X = X.reshape(N, H, W, 3)
    elif X.ndim == 3:  # (N, H, W) -> add channel
        X = X[..., None]
    elif X.ndim == 4:  # (N, H, W, C)
        pass
    else:
        raise ValueError(f"Unsupported X shape: {X.shape}")

    X_t = torch.from_numpy(X)
    X_t = X_t.permute(0, 3, 1, 2).contiguous()  # (N,C,H,W)
    return X_t

def make_transform(is_grayscale: bool):
    t = [transforms.ToPILImage(), transforms.Resize(224)]
    if is_grayscale:
        t += [transforms.Grayscale(num_output_channels=3)]
    t += [transforms.ToTensor(),
          transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])]
    return transforms.Compose(t)

class TfmSet(torch.utils.data.Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, tfm):
        self.X, self.y, self.tfm = X, y, tfm
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.tfm(self.X[i]), self.y[i]

def build_loaders_from_npz(path: str, batch_size=128, num_workers=4, grayscale=True):
    d = np.load(path, allow_pickle=True)
    Xtr, Str = d["Xtr"], d["Str"]
    Xts, Yts = d["Xts"], d["Yts"]

    Xtr_t = to_chw_tensor(Xtr, grayscale=grayscale)
    Xts_t = to_chw_tensor(Xts, grayscale=grayscale)
    ytr_t = torch.from_numpy(Str.astype(np.int64))
    yts_t = torch.from_numpy(Yts.astype(np.int64))

    tfm = make_transform(grayscale)
    N = len(ytr_t)
    idx = torch.randperm(N)
    n_train = int(0.8 * N)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    train_ds = TfmSet(Xtr_t[train_idx], ytr_t[train_idx], tfm)
    val_ds   = TfmSet(Xtr_t[val_idx],   ytr_t[val_idx],   tfm)
    test_ds  = TfmSet(Xts_t,            yts_t,            tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

# -------------------- Model --------------------
def make_vit(num_classes=3, model_name="vit_tiny_patch16_224", pretrained=False):
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

# -------------------- Losses --------------------
class ForwardCorrectCE(nn.Module):
    """Forward loss correction using known/estimated transition matrix T (clean->noisy)."""
    def __init__(self, T: torch.Tensor):
        super().__init__()
        self.register_buffer("T", T)  # [C,C], rows = clean i, cols = noisy j
    def forward(self, logits, y_noisy):
        p_clean = F.softmax(logits, dim=1)           # [B,C]
        p_noisy = torch.clamp(p_clean @ self.T, 1e-8, 1.0)
        return F.nll_loss(torch.log(p_noisy), y_noisy)

class BootstrappedCE(nn.Module):
    def __init__(self, beta=0.8):
        super().__init__()
        self.beta = beta
    def forward(self, logits, y):
        p = F.softmax(logits, dim=1)
        onehot = F.one_hot(y, num_classes=logits.size(1)).float()
        target = self.beta * onehot + (1 - self.beta) * p.detach()
        return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

class GCELoss(nn.Module):
    def __init__(self, q=0.7):
        super().__init__()
        assert 0 < q <= 1
        self.q = q
    def forward(self, logits, y):
        p = F.softmax(logits, dim=1)
        p_y = p.gather(1, y.view(-1,1)).clamp(1e-8, 1.0).squeeze(1)
        return ((1.0 - p_y.pow(self.q)) / self.q).mean()

# -------------------- T estimator --------------------
@torch.no_grad()
def estimate_T_confident(model, loader, device, top_k=200, num_classes=3):
    model.eval()
    probs, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        probs.append(F.softmax(model(xb), dim=1).cpu())
        ys.append(yb)
    P = torch.cat(probs, dim=0)  # [N,C]
    Y = torch.cat(ys, dim=0)     # [N]
    T = torch.zeros(num_classes, num_classes)
    for i in range(num_classes):
        conf_i = P[:, i]
        k = min(top_k, conf_i.numel())
        idx = torch.topk(conf_i, k=k).indices
        hist = torch.bincount(Y[idx], minlength=num_classes).float()
        T[i] = hist / max(hist.sum().item(), 1.0)
    return T

# -------------------- Train / Eval --------------------
def train_epochs(model, criterion, opt, train_loader, val_loader, device, epochs=15):
    model.to(device)
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

@torch.no_grad()
def top1_acc(model, loader, device):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / max(total, 1)

# -------------------- Runners --------------------
def run_known_T(npz_path: str, T: np.ndarray, trials=10, grayscale=True,
                model_name="vit_tiny_patch16_224", epochs=15, lr=3e-4, wd=0.05,
                batch_size=128, loss_name="forward", beta=0.8, q=0.7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = T.shape[0]
    T_t = torch.tensor(T, dtype=torch.float32, device=device)
    accs = []
    for t in range(trials):
        set_seed(1234 + t)
        train_loader, val_loader, test_loader = build_loaders_from_npz(
            npz_path, batch_size=batch_size, grayscale=grayscale
        )
        model = make_vit(num_classes=C, model_name=model_name).to(device)
        if loss_name == "forward":
            criterion = ForwardCorrectCE(T_t)
        elif loss_name == "bootstrap":
            criterion = BootstrappedCE(beta=beta)
        elif loss_name == "gce":
            criterion = GCELoss(q=q)
        else:
            raise ValueError(f"Unknown loss {loss_name}")
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        train_epochs(model, criterion, opt, train_loader, val_loader, device, epochs=epochs)
        accs.append(top1_acc(model, test_loader, device))
    return float(np.mean(accs)), float(np.std(accs))

def run_unknown_T(npz_path: str, trials=10, grayscale=False,
                  model_name="vit_tiny_patch16_224",
                  warmup_epochs=5, epochs=15, lr=3e-4, wd=0.05,
                  batch_size=128, top_k=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = 3
    ce = nn.CrossEntropyLoss()
    accs = []
    for t in range(trials):
        set_seed(2000 + t)
        train_loader, val_loader, test_loader = build_loaders_from_npz(
            npz_path, batch_size=batch_size, grayscale=grayscale
        )
        # warm-up
        warm = make_vit(num_classes=C, model_name=model_name).to(device)
        optw = torch.optim.AdamW(warm.parameters(), lr=lr, weight_decay=wd)
        train_epochs(warm, ce, optw, train_loader, val_loader, device, epochs=warmup_epochs)
        # estimate T
        T_hat = estimate_T_confident(warm, train_loader, device, top_k=top_k, num_classes=C).to(device)
        # retrain with forward correction
        model = make_vit(num_classes=C, model_name=model_name).to(device)
        criterion = ForwardCorrectCE(T_hat)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        train_epochs(model, criterion, opt, train_loader, val_loader, device, epochs=epochs)
        accs.append(top1_acc(model, test_loader, device))
    return float(np.mean(accs)), float(np.std(accs))

def parse_T_str(T_str: str) -> np.ndarray:
    """Parse 'a,b,c,d,e,f,g,h,i' into a 3x3 float32 matrix (row-major)."""
    arr = [float(x) for x in T_str.split(",")]
    if len(arr) != 9:
        raise ValueError("Provide 9 comma-separated numbers for a 3x3 T (row-major).")
    return np.array(arr, dtype=np.float32).reshape(3,3)
