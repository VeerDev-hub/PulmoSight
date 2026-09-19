import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_CUDA = DEVICE.type == 'cuda'
AMP_DTYPE = torch.bfloat16 if USE_CUDA and torch.cuda.is_bf16_supported() else torch.float16
MEAN, STD = 0.449, 0.226


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_gray(path):
    # Top-level function (not a lambda) so it works with Windows worker processes.
    with Image.open(path) as im:
        return im.convert('L')


def create_transforms(size):
    normalize = transforms.Normalize([MEAN], [STD])
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomAffine(degrees=7, translate=(0.04, 0.04), scale=(0.93, 1.07)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, val_tf


def build_model(pretrained=True):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    # X-rays are grayscale: use a 1-channel stem (sum of the RGB filters) so we
    # move 3x less data to the GPU and do a little less work in the first conv.
    stem = model.features[0][0]
    gray = nn.Conv2d(1, stem.out_channels, stem.kernel_size, stem.stride, stem.padding, bias=False)
    with torch.no_grad():
        gray.weight.copy_(stem.weight.sum(dim=1, keepdim=True))
    model.features[0][0] = gray
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 2))
    return model


def make_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for p in model.parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)  # no decay on bias / norm
    return AdamW(
        [{'params': decay, 'weight_decay': weight_decay},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=lr,
    )


def autocast():
    return torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_CUDA)


def to_device(x, y):
    x = x.to(DEVICE, non_blocking=True).contiguous(memory_format=torch.channels_last)
    return x, y.to(DEVICE, non_blocking=True)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, accum_steps, epoch, epochs):
    model.train()
    loss_sum = torch.zeros((), device=DEVICE)
    correct = torch.zeros((), device=DEVICE)
    total = 0
    optimizer.zero_grad(set_to_none=True)

    for i, (x, y) in enumerate(tqdm(loader, desc=f'Epoch {epoch}/{epochs}')):
        x, y = to_device(x, y)
        with autocast():
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss / accum_steps).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # Accumulate on the GPU: calling .item() every step forces a sync.
        loss_sum += loss.detach() * y.size(0)
        correct += (out.argmax(1) == y).sum()
        total += y.size(0)

    return loss_sum.item() / total, correct.item() / total


@torch.no_grad()
def evaluate(model, loader, pos_idx):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x, y = to_device(x, y)
        with autocast():
            out = model(x)
        probs.append(out.float().softmax(1)[:, pos_idx].cpu())
        labels.append(y.cpu())
    probs = torch.cat(probs).numpy()
    is_pos = torch.cat(labels).numpy() == pos_idx
    pred = probs >= 0.5

    tp = int((pred & is_pos).sum())
    tn = int((~pred & ~is_pos).sum())
    fp = int((pred & ~is_pos).sum())
    fn = int((~pred & is_pos).sum())
    try:
        auroc = float(roc_auc_score(is_pos, probs))
    except ValueError:  # only one class present in val
        auroc = float('nan')
    return {
        'acc': (tp + tn) / max(len(is_pos), 1),
        'sensitivity': tp / max(tp + fn, 1),
        'specificity': tn / max(tn + fp, 1),
        'auroc': auroc,
    }


def parse_args():
    p = argparse.ArgumentParser(description='Train an EfficientNet-B0 TB classifier.')
    p.add_argument('--data-path', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--output-path', type=Path, default=Path(__file__).resolve().parent / 'models')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--accum-steps', type=int, default=1,
                   help='Gradient accumulation. If you hit CUDA OOM use --batch-size 16 --accum-steps 2')
    p.add_argument('--image-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--patience', type=int, default=6, help='Early stop after N epochs without AUROC gain')
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--positive-class', type=str, default='TB')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--compile', action='store_true', help='torch.compile (works best on Linux / WSL2)')
    p.add_argument('--no-pretrained', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True  # fixed input size -> autotune the fastest kernels

    print(f'Device: {DEVICE} | AMP dtype: {AMP_DTYPE if USE_CUDA else "off"}')
    train_tf, val_tf = create_transforms(args.image_size)
    train_data = datasets.ImageFolder(args.data_path / 'train', train_tf, loader=load_gray)
    val_data = datasets.ImageFolder(args.data_path / 'val', val_tf, loader=load_gray)
    if train_data.classes != val_data.classes:
        raise ValueError(f'Class folders differ: train={train_data.classes}, val={val_data.classes}')
    if len(train_data.classes) != 2:
        raise ValueError(f'Expected 2 classes, found {train_data.classes}')

    lowered = [c.lower() for c in train_data.classes]
    if args.positive_class.lower() not in lowered:
        raise ValueError(f'--positive-class {args.positive_class!r} not in {train_data.classes}')
    pos_idx = lowered.index(args.positive_class.lower())

    workers = args.workers if args.workers is not None else min(8, os.cpu_count() or 1)
    loader_kwargs = dict(num_workers=workers, pin_memory=USE_CUDA, persistent_workers=workers > 0)
    if workers > 0:
        loader_kwargs['prefetch_factor'] = 4
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_data, batch_size=args.batch_size * 2, shuffle=False, **loader_kwargs)

    counts = torch.bincount(torch.tensor(train_data.targets), minlength=2).float()
    class_weights = (counts.sum() / (2 * counts)).to(DEVICE)
    print(f'Train: {len(train_data)} | Val: {len(val_data)} | Classes: {train_data.classes} | '
          f'Counts: {counts.int().tolist()} | Workers: {workers}')

    model = build_model(pretrained=not args.no_pretrained).to(DEVICE, memory_format=torch.channels_last)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    if args.compile:
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    steps_per_epoch = -(-len(train_loader) // args.accum_steps)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                           steps_per_epoch=steps_per_epoch, pct_start=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_CUDA and AMP_DTYPE == torch.float16)

    args.output_path.mkdir(parents=True, exist_ok=True)
    best_auroc, bad_epochs, history = -1.0, 0, []

    def save(name, epoch, metrics):
        raw = getattr(model, '_orig_mod', model)  # unwrap torch.compile
        torch.save({
            'model_state_dict': raw.state_dict(),
            'classes': train_data.classes,
            'positive_class': train_data.classes[pos_idx],
            'image_size': args.image_size,
            'in_channels': 1,
            'mean': MEAN, 'std': STD,
            'epoch': epoch,
            'metrics': metrics,
        }, args.output_path / name)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            args.accum_steps, epoch, args.epochs)
        m = evaluate(model, val_loader, pos_idx)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, **m})
        print(f"Epoch {epoch}: loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val acc={m['acc']:.3f} sens={m['sensitivity']:.3f} "
              f"spec={m['specificity']:.3f} auroc={m['auroc']:.4f}")

        save('tb_last.pth', epoch, m)
        if m['auroc'] > best_auroc:
            best_auroc, bad_epochs = m['auroc'], 0
            save('tb_best_model.pth', epoch, m)
            print(f'   Saved best checkpoint (AUROC {best_auroc:.4f})')
        else:
            bad_epochs += 1

        (args.output_path / 'history.json').write_text(json.dumps(history, indent=2))
        if bad_epochs >= args.patience:
            print(f'No AUROC improvement for {args.patience} epochs, stopping early.')
            break

    print(f'\nTraining complete. Best val AUROC: {best_auroc:.4f}')


if __name__ == '__main__':
    main()