"""
train_model.py  (CNN model — PyTorch)
──────────────────────────────────────
Fine-tune MobileNetV2 on images collected by collect_data.py.
No landmarks — the full two-hand frame is the input.

Usage:
    python train_model.py --team 1
    python train_model.py --team 2 --epochs 20 --finetune-epochs 10

Outputs (saved to Teams/TeamN/models/):
    hand_sign_cnn.pth   — torch checkpoint (state_dict + class_names + img_size)

GPU notes:
    Training automatically uses the GPU if one is available.  Pass
    --no-gpu to force CPU, or --gpu-memory-mb N to cap memory usage.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SCRIPT_DIR     = pathlib.Path(__file__).parent
TEAMS_ROOT_DIR = SCRIPT_DIR.parent.parent.parent / "Teams"

IMG_SIZE   = (224, 224)
BATCH_SIZE = 16

# ImageNet normalization — required because we use pretrained MobileNetV2 weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _load_team_arch(team_dir: pathlib.Path):
    """Try to import Teams/TeamN/model_arch.py.  Returns its build_model
    callable if found and valid, else None.  See cnn/inference.py for the
    rationale — teams can ship a custom architecture alongside their weights.
    """
    arch_path = team_dir / "model_arch.py"
    if not arch_path.exists():
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        f"team_arch_{team_dir.name}", arch_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"[train_cnn] Failed to import {arch_path}: {e}  (using default)")
        return None
    if not hasattr(mod, "build_model") or not callable(mod.build_model):
        print(f"[train_cnn] {arch_path} has no build_model() — using default.")
        return None
    print(f"[train_cnn] Using team's custom architecture: {arch_path}")
    return mod.build_model


def build_model(num_classes: int, freeze_base: bool = True,
                team_dir: pathlib.Path | None = None):
    """MobileNetV2 with a custom 2-layer classifier head.

    If `team_dir/model_arch.py` exists, the team's `build_model()` is used
    instead; it must accept `num_classes`, `pretrained`, and `freeze_base`.
    """
    if team_dir is not None:
        custom = _load_team_arch(team_dir)
        if custom is not None:
            return custom(num_classes=num_classes, pretrained=True,
                          freeze_base=freeze_base)

    import torch.nn as nn
    from torchvision import models

    try:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        model = models.mobilenet_v2(weights=weights)
    except (AttributeError, TypeError):
        # torchvision < 0.13 fallback
        model = models.mobilenet_v2(pretrained=True)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the original classifier (Dropout + Linear(1280, 1000))
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )
    return model


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """Run one epoch of training or eval; returns (avg_loss, accuracy)."""
    import torch
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss   = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)
    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN hand-sign classifier (PyTorch)")
    parser.add_argument("--team",            type=int, required=True, choices=range(1, 7))
    parser.add_argument("--epochs",          type=int, default=15,
                        help="Head-training epochs (default: 15)")
    parser.add_argument("--finetune-epochs", type=int, default=5,
                        help="Fine-tune epochs with top base layers unfrozen (default: 5)")
    parser.add_argument("--batch-size",      type=int, default=BATCH_SIZE)
    parser.add_argument("--no-gpu",          action="store_true",
                        help="Force CPU even if a GPU is available")
    parser.add_argument("--gpu-memory-mb",   type=int, default=0,
                        help="Hard cap on GPU memory in MB (0 = grow as needed)")
    args = parser.parse_args()

    data_dir    = SCRIPT_DIR / "teams" / f"Team{args.team}"
    images_dir  = data_dir / "images"

    team_models = TEAMS_ROOT_DIR / f"Team{args.team}" / "models"
    team_models.mkdir(parents=True, exist_ok=True)
    model_path  = team_models / "hand_sign_cnn.pth"

    if not images_dir.exists():
        print(f"[train_cnn] Images directory not found: {images_dir}")
        print("  Run collect_data.py --team", args.team, "first.")
        sys.exit(1)

    sys.path.insert(0, str(SCRIPT_DIR))
    from gpu import configure_gpu
    device, device_status = configure_gpu(
        memory_limit_mb=args.gpu_memory_mb or None,
        prefer_gpu=not args.no_gpu,
    )

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, transforms
    except ImportError:
        print("[train_cnn] PyTorch / torchvision not found.")
        print("  Install with:  pip install torch torchvision")
        sys.exit(1)

    print("=" * 50)
    print(f"  CNN TRAINING — Team {args.team}")
    print(f"  Device: {device_status}")
    print("=" * 50)

    # ── Transforms ────────────────────────────────────────────────────────────
    train_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(
            degrees=12, translate=(0.08, 0.08)),
        transforms.ColorJitter(brightness=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # ── Dataset + 80/20 split (different transforms per split) ────────────────
    train_full = datasets.ImageFolder(str(images_dir), transform=train_tf)
    val_full   = datasets.ImageFolder(str(images_dir), transform=val_tf)

    if len(train_full) == 0 or len(train_full.classes) < 2:
        print("[train_cnn] Need at least 2 classes with images. Collect more data.")
        sys.exit(1)

    class_names = train_full.classes
    n        = len(train_full)
    val_size = max(1, int(0.2 * n))

    gen      = torch.Generator().manual_seed(42)
    indices  = torch.randperm(n, generator=gen).tolist()
    val_idx  = indices[:val_size]
    train_idx = indices[val_size:]

    train_set = Subset(train_full, train_idx)
    val_set   = Subset(val_full, val_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    print(f"\nClasses : {class_names}")
    print(f"Train   : {len(train_set)} samples")
    print(f"Val     : {len(val_set)} samples")

    # ── Model ─────────────────────────────────────────────────────────────────
    team_dir = TEAMS_ROOT_DIR / f"Team{args.team}"
    model = build_model(num_classes=len(class_names), freeze_base=True,
                        team_dir=team_dir).to(device)
    criterion = nn.CrossEntropyLoss()

    # ── Phase 1: train head ───────────────────────────────────────────────────
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = torch.optim.Adam(head_params, lr=1e-3)

    best_val_acc, best_state = 0.0, None
    print(f"\n[Phase 1] Training classifier head — {args.epochs} epochs ...")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        print(f"  Epoch {epoch:>2d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.1%}  |  "
              f"val_loss={va_loss:.4f} acc={va_acc:.1%}")
        if va_acc > best_val_acc:
            best_val_acc, best_state = va_acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Phase 2: fine-tune top features ───────────────────────────────────────
    if args.finetune_epochs > 0:
        # Unfreeze the last 4 inverted-residual blocks of MobileNetV2 features
        for layer in list(model.features.children())[-4:]:
            for param in layer.parameters():
                param.requires_grad = True

        ft_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(ft_params, lr=1e-5)

        print(f"\n[Phase 2] Fine-tuning top base layers — {args.finetune_epochs} epochs ...")
        for epoch in range(1, args.finetune_epochs + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
            print(f"  Epoch {epoch:>2d}/{args.finetune_epochs}  "
                  f"train_loss={tr_loss:.4f} acc={tr_acc:.1%}  |  "
                  f"val_loss={va_loss:.4f} acc={va_acc:.1%}")
            if va_acc > best_val_acc:
                best_val_acc, best_state = va_acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save({
        "state_dict":  {k: v.cpu() for k, v in model.state_dict().items()},
        "class_names": class_names,
        "img_size":    list(IMG_SIZE),
    }, str(model_path))

    print(f"\n  Best val accuracy: {best_val_acc:.1%}")
    print(f"  Checkpoint → {model_path}")
    print("\nDone. Set MODEL_TYPE=cnn in your team.env, then run inference.py --team", args.team)


if __name__ == "__main__":
    main()
