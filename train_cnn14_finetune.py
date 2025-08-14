import os
import sys
import json
import time
import random
import argparse
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import soundfile as sf

# ========= OPTIONAL: stratified split via sklearn, with safe fallback =========
try:
    from sklearn.model_selection import StratifiedShuffleSplit
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False
# ============================================================================


# ========= ROBUST REPO RESOLUTION & IMPORT ==================================
def resolve_repo_path(cli_repo: Optional[str]) -> str:
    """
    Determine the audioset_tagging_cnn repo path via:
    1) --repo argument
    2) AUDIOSET_REPO environment variable
    3) Common fallbacks relative to this script
    """
    candidates: List[str] = []

    if cli_repo:
        candidates.append(cli_repo)

    env_repo = os.environ.get("AUDIOSET_REPO")
    if env_repo:
        candidates.append(env_repo)

    # Fallbacks: look around this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates += [
        os.path.join(script_dir, "audioset_tagging_cnn"),
        os.path.join(os.path.dirname(script_dir), "audioset_tagging_cnn"),
        os.path.join(os.path.expanduser("~"), "Sandbox", "audioset_tagging_cnn"),
    ]

    for path in candidates:
        if not path:
            continue
        pytorch_dir = os.path.join(path, "pytorch")
        models_py = os.path.join(pytorch_dir, "models.py")
        init_py = os.path.join(pytorch_dir, "__init__.py")
        if os.path.isdir(path) and os.path.isdir(pytorch_dir) and os.path.isfile(models_py):
            # ensure package import works
            if not os.path.isfile(init_py):
                try:
                    open(init_py, "a").close()
                except Exception:
                    pass
            return path

    raise FileNotFoundError(
        "Could not locate the 'audioset_tagging_cnn' repository. "
        "Pass --repo /path/to/audioset_tagging_cnn or set AUDIOSET_REPO env var. "
        "Expected structure: /path/to/audioset_tagging_cnn/pytorch/models.py"
    )


def import_cnn14(repo_path: str):
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    try:
        from pytorch.models import Cnn14  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Failed to import 'pytorch.models' from repo at: {repo_path}\n"
            f"sys.path[0]={sys.path[0]}\n"
            "Verify the repo contains 'pytorch/models.py'."
        ) from e
    return Cnn14
# ============================================================================


# We parse --repo early just for import; the full parser comes later.
_prelim = argparse.ArgumentParser(add_help=False)
_prelim.add_argument("--repo", type=str, default=None,
                     help="Path to your cloned audioset_tagging_cnn repository")
_pre_args, _ = _prelim.parse_known_args()
REPO_PATH = resolve_repo_path(_pre_args.repo)
Cnn14 = import_cnn14(REPO_PATH)


# ========= DATASET ===========================================================
class CNN14AudioDataset(Dataset):
    """
    Expects a DataFrame with columns: filepath,label  (plus optional segment_index)
    Loads mono 32kHz waveform segments saved by your preprocessor.
    """
    def __init__(self, df: pd.DataFrame, class_map: dict):
        self.df = df.reset_index(drop=True)
        self.class_map = class_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        label_str = row["label"]

        waveform, sr = sf.read(filepath)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform = waveform.astype(np.float32)

        x = torch.from_numpy(waveform)           # shape: (T,)
        y = self.class_map[label_str]            # integer class index
        return x, y


def collate_pad_same_length(batch: List[Tuple[torch.Tensor, int]]):
    # Your segments are fixed 10s @ 32k, so lengths should match.
    # Still pad safely if any mismatch occurs.
    xs, ys = zip(*batch)
    lens = [x.shape[0] for x in xs]
    max_len = max(lens)
    if all(l == max_len for l in lens):
        xb = torch.stack(xs, dim=0)
    else:
        xb = torch.stack([torch.nn.functional.pad(x, (0, max_len - x.shape[0])) for x in xs], dim=0)
    yb = torch.tensor(ys, dtype=torch.long)
    return xb, yb
# ============================================================================


# ========= UTILITIES =========================================================
def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()


def one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    y = torch.zeros((indices.shape[0], num_classes), device=indices.device)
    y.scatter_(1, indices.view(-1, 1), 1.0)
    return y
# ============================================================================


# ========= MODEL BUILD / HEAD SWAP ==========================================
def build_model(num_classes: int, pretrained_ckpt: str, device: torch.device, strict: bool = False) -> nn.Module:
    """
    Loads CNN14 with pretrained weights (527 classes), then replaces final layer
    with a new Linear(in_features -> num_classes).
    """
    # 1) Create base with 527 outputs to align with the checkpoint
    base = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
                 fmin=50, fmax=14000, classes_num=527)

    ckpt = torch.load(pretrained_ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    base.load_state_dict(state, strict=strict)  # strict=False tolerates head mismatch

    # 2) Replace final classifier layer to match your class count
    in_features = base.fc_audioset.in_features
    base.fc_audioset = nn.Linear(in_features, num_classes, bias=True)

    return base.to(device)
# ============================================================================


# ========= EVAL LOOP =========================================================
@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             loss_fn: nn.Module,
             multi_label: bool,
             num_classes: int,
             returns_sigmoid: bool) -> Tuple[float, float]:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        out_dict = model(xb)  # expect dict with 'clipwise_output'
        out = out_dict["clipwise_output"]

        if returns_sigmoid:
            # Convert probs to logits for BCEWithLogits if model already applied sigmoid
            out = out.clamp(1e-6, 1 - 1e-6)
            logits = torch.log(out) - torch.log(1 - out)
        else:
            logits = out

        if multi_label:
            targets = one_hot(yb, num_classes)
            loss = loss_fn(logits, targets)
            acc = accuracy_top1(logits, yb)  # single-label proxy
        else:
            loss = loss_fn(logits, yb)
            acc = accuracy_top1(logits, yb)

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc += acc * bs
        n += bs

    return total_loss / max(n, 1), total_acc / max(n, 1)
# ============================================================================


# ========= TRAIN =============================================================
def stratified_split(df: pd.DataFrame, labels: np.ndarray, val_size: float, seed: int):
    if HAVE_SKLEARN:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        train_idx, val_idx = next(splitter.split(df, labels))
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)
    # Fallback: simple per-class split
    rng = np.random.default_rng(seed)
    parts = []
    val_parts = []
    for cls, group in df.groupby("label", sort=False):
        g = group.sample(frac=1.0, random_state=seed)
        n_val = max(1, int(len(g) * val_size))
        val_parts.append(g.iloc[:n_val])
        parts.append(g.iloc[n_val:])
    df_train = pd.concat(parts, ignore_index=True)
    df_val = pd.concat(val_parts, ignore_index=True)
    return df_train, df_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default=None,
                        help="Path to your cloned audioset_tagging_cnn repository")
    parser.add_argument("--csv", default="./cnn14_ready_audio/metadata.csv")
    parser.add_argument("--outdir", default="./checkpoints_cnn14")
    parser.add_argument("--pretrained", type=str,
                        default=os.path.join(REPO_PATH, "pretrained_models", "Cnn14_mAP=0.431.pth"),
                        help="Path to pretrained CNN14 checkpoint (.pth)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs_head", type=int, default=4)
    parser.add_argument("--epochs_full", type=int, default=20)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--lr_full", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true")

    # Label space
    parser.add_argument("--classes", nargs="+", required=True,
                        help="Space-separated class names in order, e.g. --classes whale dolphin shrimp")

    # Keep flags for completeness (we'll auto-detect, but allow override)
    parser.add_argument("--model_returns_sigmoid", action="store_true",
                        help="If set, treat model outputs as post-sigmoid probabilities.")
    parser.add_argument("--use_ce", action="store_true",
                        help="Use CrossEntropyLoss (requires raw logits; no sigmoid in forward).")
    
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # === Build label maps
    class_names = args.classes
    num_classes = len(class_names)
    class_map = {c: i for i, c in enumerate(class_names)}

    with open(os.path.join(args.outdir, "classes.json"), "w") as f:
        json.dump({"classes": class_names}, f, indent=2)

    # === Load metadata and split train/val (stratified)
    df = pd.read_csv(args.csv)
    if "label" not in df.columns or "filepath" not in df.columns:
        raise ValueError("CSV must have at least columns: filepath,label")

    # Filter to your provided classes only
    df = df[df["label"].isin(class_map.keys())].copy()
    if len(df) == 0:
        raise ValueError("No rows left after filtering by provided classes.")

    # Print class counts for visibility
    print("Class counts in CSV after filter:")
    print(df["label"].value_counts())

    # === Split (stratified)
    y_indices = df["label"].map(class_map).values
    df_train, df_val = stratified_split(df, y_indices, val_size=args.val_size, seed=args.seed)

    # === Datasets / Loaders
    train_ds = CNN14AudioDataset(df_train, class_map)
    val_ds   = CNN14AudioDataset(df_val, class_map)

    persistent = True if args.num_workers > 0 else False
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_pad_same_length,
                              persistent_workers=persistent)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_pad_same_length,
                              persistent_workers=persistent)

    # === Model
    model = build_model(num_classes=num_classes, pretrained_ckpt=args.pretrained,
                        device=device, strict=False)

    # === Decide sigmoid behavior (auto-detect unless user forced a choice)
    if args.model_returns_sigmoid:
        returns_sigmoid = True
    else:
        # Probe with zeros: if outputs in [0,1], assume post-sigmoid probs
        with torch.no_grad():
            probe = torch.zeros(2, 320000, device=device)  # 10s @ 32k
            out = model(probe)["clipwise_output"]
            returns_sigmoid = bool((out.min() >= 0) and (out.max() <= 1))
    print(f"Model returns sigmoid probs: {returns_sigmoid}")
    
    # === Loss
    if args.use_ce:
        if returns_sigmoid:
            raise ValueError("CrossEntropyLoss requires raw logits; do not set --model_returns_sigmoid.")
        loss_fn = nn.CrossEntropyLoss()
        multi_label = False
    else:
        loss_fn = nn.BCEWithLogitsLoss()
        multi_label = True  # we train with one-hot even for single-label; accuracy still uses argmax

    # === Warm-up: freeze all but the new head
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc_audioset.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr_head, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    def run_epoch(train_mode: bool) -> Tuple[float, float]:
        if train_mode:
            model.train()
            loader = train_loader
        else:
            model.eval()
            loader = val_loader

        total_loss, total_acc, n = 0.0, 0.0, 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if args.use_ce:
                targets = yb
            else:
                targets = one_hot(yb, num_classes)

            if train_mode:
                opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                out_dict = model(xb)  # dict with 'clipwise_output'
                out = out_dict["clipwise_output"]

                if returns_sigmoid:
                    out = out.clamp(1e-6, 1 - 1e-6)
                    logits = torch.log(out) - torch.log(1 - out)
                else:
                    logits = out

                loss = loss_fn(logits, targets)
                acc = accuracy_top1(logits, yb)

            if train_mode:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_acc  += acc * bs
            n += bs

        return total_loss / max(n, 1), total_acc / max(n, 1)

    # === HEAD-ONLY WARMUP
    best_val = float("inf")
    for epoch in range(1, args.epochs_head + 1):
        tr_loss, tr_acc = run_epoch(train_mode=True)
        va_loss, va_acc = run_epoch(train_mode=False)
        print(f"[Head {epoch}/{args.epochs_head}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "model": model.state_dict(),
                "classes": class_names,
                "epoch": epoch,
                "phase": "head",
            }, os.path.join(args.outdir, "best_head.pt"))

    # === UNFREEZE & FULL FINE-TUNE
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_full, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs_full + 1):
        tr_loss, tr_acc = run_epoch(train_mode=True)
        va_loss, va_acc = run_epoch(train_mode=False)
        print(f"[Full {epoch}/{args.epochs_full}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f}")

        # Save checkpoint each epoch
        torch.save({
            "model": model.state_dict(),
            "classes": class_names,
            "epoch": epoch,
            "phase": "full",
        }, os.path.join(args.outdir, f"checkpoint_full_epoch{epoch:03d}.pt"))

        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "model": model.state_dict(),
                "classes": class_names,
                "epoch": epoch,
                "phase": "best_full",
            }, os.path.join(args.outdir, "best_full.pt"))

    print("Training complete. Best val loss:", best_val)


if __name__ == "__main__":
    main()