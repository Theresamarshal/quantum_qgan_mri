"""
=============================================================================
MS Lesion Segmentation - SwinUNETR with Overfitting Prevention
=============================================================================
Enhanced version with multiple strategies to prevent overfitting:
1. Data augmentation (even for preprocessed data)
2. Dropout regularization
3. L2 weight decay
4. Early stopping
5. Model capacity reduction options
6. Validation-based learning rate scheduling
=============================================================================
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # --- Paths -----------------------------------------------------------
    "flair_dir": "/content/drive/MyDrive/project/preprocessed_ms/ms/ms/flair",
    "mask_dir":  "/content/drive/MyDrive/project/preprocessed_ms/ms/ms/masks",
    "save_dir":  "/content/drive/MyDrive/project/segmentation_models_ms",

    # --- Model -----------------------------------------------------------
    "img_size":       (192, 192, 96),
    "feature_size":   48,               # Can reduce to 24 or 36 if overfitting
    "in_channels":    1,
    "out_channels":   1,
    
    "depths":         (2, 2, 2, 2),
    "num_heads":      (3, 6, 12, 24),
    "use_checkpoint": True,
    
    # NEW: Dropout for regularization
    "dropout_rate":   0.2,              # 20% dropout (0.0 = no dropout)

    # --- Training --------------------------------------------------------
    "batch_size":     1,
    "epochs":         300,
    "lr":             1e-4,
    "weight_decay":   1e-4,             # Increased from 1e-5 for stronger regularization
    "val_fraction":   0.15,
    "seed":           42,

    # --- Augmentation (IMPORTANT for preventing overfitting) ------------
    "use_augmentation": True,           # SET TO TRUE!
    "aug_flip_prob":    0.5,            # Probability of flipping
    "aug_rotate_prob":  0.3,            # Probability of rotation
    "aug_noise_prob":   0.15,           # Probability of adding noise
    "aug_brightness_prob": 0.15,        # Probability of brightness change
    
    # --- Loss ------------------------------------------------------------
    "ftl_alpha":      0.3,
    "ftl_beta":       0.7,
    "ftl_gamma":      1.5,

    # --- Inference -------------------------------------------------------
    "sw_roi_size":    (96, 96, 96),
    "sw_batch_size":  4,
    "sw_overlap":     0.5,
    "threshold":      0.5,

    # --- Optimization ----------------------------------------------------
    "use_amp":        True,
    "grad_clip":      1.0,
    
    "warmup_epochs":  10,
    "scheduler_type": "plateau",        # Changed to plateau - better for overfitting
    "plateau_patience": 15,             # Reduce LR if no improvement for 15 epochs
    "plateau_factor": 0.5,              # Reduce LR by half
    "min_lr":         1e-6,

    # --- Early Stopping (CRITICAL) ---------------------------------------
    "es_patience":    50,               # Stop if no improvement for 50 epochs
    "es_min_delta":   1e-4,

    # --- Checkpointing ---------------------------------------------------
    "save_interval":  10,               # Save less frequently to save space
    "keep_last_n":    2,

    # --- Resume ----------------------------------------------------------
    "resume_path":    None,

    # --- Device ----------------------------------------------------------
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers":    4,
    
    # --- Monitoring ------------------------------------------------------
    "monitor_overfitting": True,        # Track train/val gap
}

# =============================================================================
# LOGGER
# =============================================================================
class Logger:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_path     = self.save_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.metrics_path = self.save_dir / "metrics.json"
        self.metrics = {
            "epoch": [], "train_loss": [], "val_loss": [],
            "val_dice": [], "val_recall": [], "val_precision": [], "lr": [],
            "train_dice": []  # NEW: Track training Dice to detect overfitting
        }

    def log(self, msg: str):
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        out = f"[{ts}] {msg}"
        print(out)
        with open(self.log_path, "a") as f:
            f.write(out + "\n")

    def record(self, epoch, train_loss, val_loss, val_dice, val_recall, val_prec, lr, train_dice=None):
        for key, val in zip(
            ["epoch","train_loss","val_loss","val_dice","val_recall","val_precision","lr","train_dice"],
            [epoch, train_loss, val_loss, val_dice, val_recall, val_prec, lr, train_dice]
        ):
            if val is not None:
                self.metrics[key].append(float(val))
        with open(self.metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

# =============================================================================
# AUGMENTATION FUNCTIONS
# =============================================================================
class VolumeAugmentation:
    """
    Simple but effective augmentations for 3D medical images.
    Applied on-the-fly during training.
    """
    
    def __init__(self, config):
        self.flip_prob = config.get("aug_flip_prob", 0.5)
        self.rotate_prob = config.get("aug_rotate_prob", 0.3)
        self.noise_prob = config.get("aug_noise_prob", 0.15)
        self.brightness_prob = config.get("aug_brightness_prob", 0.15)
    
    def __call__(self, image, label):
        """
        Apply augmentations to image and label.
        
        Args:
            image: (1, H, W, D) tensor
            label: (1, H, W, D) tensor
        Returns:
            Augmented image and label
        """
        # Random flip along each axis
        if torch.rand(1).item() < self.flip_prob:
            # Flip left-right
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[2])
        
        if torch.rand(1).item() < self.flip_prob:
            # Flip anterior-posterior
            image = torch.flip(image, dims=[3])
            label = torch.flip(label, dims=[3])
        
        if torch.rand(1).item() < self.flip_prob:
            # Flip superior-inferior
            image = torch.flip(image, dims=[4])
            label = torch.flip(label, dims=[4])
        
        # Random 90° rotation
        if torch.rand(1).item() < self.rotate_prob:
            k = torch.randint(1, 4, (1,)).item()  # 90°, 180°, or 270°
            image = torch.rot90(image, k, dims=[2, 3])
            label = torch.rot90(label, k, dims=[2, 3])
        
        # Gaussian noise (only to image, not label)
        if torch.rand(1).item() < self.noise_prob:
            noise = torch.randn_like(image) * 0.02
            image = image + noise
        
        # Brightness adjustment (only to image)
        if torch.rand(1).item() < self.brightness_prob:
            factor = 0.9 + torch.rand(1).item() * 0.2  # [0.9, 1.1]
            image = image * factor
        
        return image, label

# =============================================================================
# DATASET
# =============================================================================
class MSFullVolumeDataset(Dataset):
    def __init__(self, flair_dir: str, mask_dir: str, augmentation=None):
        self.samples = []
        self.augmentation = augmentation

        flair_path = Path(flair_dir)
        mask_path  = Path(mask_dir)
        
        assert flair_path.exists(), f"FLAIR dir missing: {flair_dir}"
        assert mask_path.exists(),  f"Mask dir missing:  {mask_dir}"

        for flair_file in sorted(flair_path.glob("*_flair.npy")):
            mask_file = mask_path / flair_file.name.replace("_flair.npy", "_mask.npy")
            
            if mask_file.exists():
                self.samples.append({
                    "flair": str(flair_file),
                    "mask":  str(mask_file)
                })
            else:
                print(f"⚠️  Missing mask for {flair_file.name}")

        assert len(self.samples) > 0, f"No FLAIR-mask pairs found"
        
        first_flair = np.load(self.samples[0]["flair"])
        first_mask = np.load(self.samples[0]["mask"])
        
        print(f"  ✓ Loaded {len(self.samples)} FLAIR-mask pairs")
        print(f"  ✓ Sample FLAIR shape: {first_flair.shape}")
        print(f"  ✓ Sample MASK shape:  {first_mask.shape}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths = self.samples[idx]
        
        flair = np.load(paths["flair"]).astype(np.float32)
        mask  = np.load(paths["mask"]).astype(np.float32)

        mask = (mask > 0).astype(np.float32)

        # Handle different shapes
        if flair.ndim == 3:
            flair = flair[np.newaxis]
        elif flair.ndim == 4 and flair.shape[0] != 1:
            if flair.shape[-1] == 1:
                flair = np.transpose(flair, (3, 0, 1, 2))
        elif flair.ndim == 5:
            flair = flair.squeeze(0)
        
        if mask.ndim == 3:
            mask = mask[np.newaxis]
        elif mask.ndim == 4 and mask.shape[0] != 1:
            if mask.shape[-1] == 1:
                mask = np.transpose(mask, (3, 0, 1, 2))
        elif mask.ndim == 5:
            mask = mask.squeeze(0)

        assert flair.shape[0] == 1, f"Expected channel dim=1, got {flair.shape}"
        assert mask.shape[0] == 1, f"Expected channel dim=1, got {mask.shape}"

        image_tensor = torch.from_numpy(flair)
        label_tensor = torch.from_numpy(mask)
        
        # Apply augmentation if provided
        if self.augmentation is not None:
            # Add batch dimension for augmentation
            image_tensor = image_tensor.unsqueeze(0)  # (1, 1, H, W, D)
            label_tensor = label_tensor.unsqueeze(0)  # (1, 1, H, W, D)
            
            image_tensor, label_tensor = self.augmentation(image_tensor, label_tensor)
            
            # Remove batch dimension
            image_tensor = image_tensor.squeeze(0)
            label_tensor = label_tensor.squeeze(0)

        return {
            "image": image_tensor,
            "label": label_tensor
        }

# =============================================================================
# FOCAL TVERSKY LOSS
# =============================================================================
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 1.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs  = torch.sigmoid(logits).flatten()
        target = target.flatten()

        tp = (probs * target).sum()
        fp = (probs * (1 - target)).sum()
        fn = ((1 - probs) * target).sum()

        tversky_idx = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return torch.pow(1 - tversky_idx, self.gamma)

# =============================================================================
# METRICS
# =============================================================================
class SegMetrics:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.eps       = 1e-7

    def _bin(self, pred):
        return (pred > self.threshold).float()

    def dice(self, pred, target):
        p = self._bin(pred).flatten()
        t = target.flatten()
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        return 1.0 if denom == 0 else (2 * inter / (denom + self.eps)).item()

    def recall(self, pred, target):
        p = self._bin(pred).flatten()
        t = target.flatten()
        tp = (p * t).sum()
        fn = ((1 - p) * t).sum()
        d  = tp + fn
        return 1.0 if d == 0 else (tp / (d + self.eps)).item()

    def precision(self, pred, target):
        p = self._bin(pred).flatten()
        t = target.flatten()
        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        d  = tp + fp
        return 1.0 if d == 0 else (tp / (d + self.eps)).item()

# =============================================================================
# EARLY STOPPING
# =============================================================================
class EarlyStopping:
    def __init__(self, patience: int = 50, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = 0.0
        self.triggered = False

    def __call__(self, score: float) -> bool:
        if score > self.best + self.min_delta:
            self.best    = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered

# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================
class CheckpointManager:
    def __init__(self, save_dir: str, keep_last_n: int = 2, save_interval: int = 10):
        self.save_dir     = Path(save_dir)
        self.keep_last_n  = keep_last_n
        self.save_interval = save_interval
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_dice    = 0.0
        self.epoch_ckpts  = []

    def _create_checkpoint(self, model, optimizer, scheduler, scaler, epoch, dice, cfg):
        ckpt = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_dice":            self.best_dice,
            "current_dice":         dice,
            "config":               cfg,
        }
        if scaler is not None:
            ckpt["scaler_state_dict"] = scaler.state_dict()
        return ckpt

    def save(self, model, optimizer, scheduler, scaler, epoch, dice, cfg) -> bool:
        improved = False
        
        torch.save(
            self._create_checkpoint(model, optimizer, scheduler, scaler, epoch, dice, cfg),
            self.save_dir / "latest_checkpoint.pth"
        )
        
        if dice > self.best_dice:
            self.best_dice = dice
            improved = True
            torch.save(
                self._create_checkpoint(model, optimizer, scheduler, scaler, epoch, dice, cfg),
                self.save_dir / "best_model.pth"
            )
        
        if (epoch % self.save_interval) == 0:
            epoch_path = self.save_dir / f"checkpoint_epoch_{epoch:04d}.pth"
            torch.save(
                self._create_checkpoint(model, optimizer, scheduler, scaler, epoch, dice, cfg),
                epoch_path
            )
            self.epoch_ckpts.append(epoch_path)
            
            while len(self.epoch_ckpts) > self.keep_last_n:
                old_ckpt = self.epoch_ckpts.pop(0)
                if old_ckpt.exists():
                    old_ckpt.unlink()
        
        return improved

    def get_resume_path(self, resume_arg, save_dir):
        save_path = Path(save_dir)
        
        if resume_arg is None:
            return None
        
        if resume_arg.lower() == "latest":
            path = save_path / "latest_checkpoint.pth"
            return str(path) if path.exists() else None
        
        if resume_arg.lower() == "best":
            path = save_path / "best_model.pth"
            return str(path) if path.exists() else None
        
        path = Path(resume_arg)
        return str(path) if path.exists() else None

# =============================================================================
# RESUME
# =============================================================================
def load_checkpoint(path, model, optimizer, scheduler, scaler=None):
    print(f"\n{'='*70}")
    print(f"📂 RESUMING FROM CHECKPOINT")
    print(f"{'='*70}")
    print(f"Path: {path}")
    
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"✓ Model state loaded")
    
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"✓ Optimizer state loaded")
    
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"✓ Scheduler state loaded")
    
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        print(f"✓ AMP scaler state loaded")
    
    start_epoch = ckpt["epoch"] + 1
    best_dice   = ckpt.get("best_dice", 0.0)
    
    print(f"\nResuming from epoch {start_epoch}")
    print(f"Best Dice so far: {best_dice:.4f}")
    print(f"{'='*70}\n")
    
    return start_epoch, best_dice

# =============================================================================
# TRAIN ONE EPOCH
# =============================================================================
def train_one_epoch(model, loader, optimizer, loss_fn, scaler, metrics, device, use_amp, grad_clip):
    """
    Train with Dice monitoring to detect overfitting.
    """
    model.train()
    epoch_losses = []
    epoch_dices = []

    for batch in tqdm(loader, desc="  Train", leave=False):
        img = batch["image"].to(device)
        lbl = batch["label"].to(device)
        
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                out  = model(img)
                loss = loss_fn(out, lbl)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(img)
            loss = loss_fn(out, lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        epoch_losses.append(loss.item())
        
        # Track training Dice
        with torch.no_grad():
            probs = torch.sigmoid(out)
            epoch_dices.append(metrics.dice(probs, lbl))

    return np.mean(epoch_losses), np.mean(epoch_dices)

# =============================================================================
# VALIDATE ONE EPOCH
# =============================================================================
def validate_one_epoch(model, loader, loss_fn, metrics, device, sw_roi_size, sw_batch_size, sw_overlap):
    model.eval()
    losses, dices, recalls, precisions = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Val  ", leave=False):
            img = batch["image"].to(device)
            lbl = batch["label"].to(device)

            out = sliding_window_inference(
                inputs=img,
                roi_size=sw_roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=sw_overlap
            )
            
            loss  = loss_fn(out, lbl)
            probs = torch.sigmoid(out)

            losses.append(loss.item())
            dices.append(metrics.dice(probs, lbl))
            recalls.append(metrics.recall(probs, lbl))
            precisions.append(metrics.precision(probs, lbl))

    return {
        "loss":      np.mean(losses),
        "dice":      np.mean(dices),
        "recall":    np.mean(recalls),
        "precision": np.mean(precisions),
    }

# =============================================================================
# MAIN
# =============================================================================
def main():
    cfg    = CONFIG
    device = cfg["device"]
    logger = Logger(cfg["save_dir"])
    
    logger.log("=" * 70)
    logger.log("MS LESION SEGMENTATION — OVERFITTING PREVENTION MODE")
    logger.log("=" * 70)
    logger.log(f"Device: {device}")
    logger.log(f"Augmentation: {cfg['use_augmentation']}")
    logger.log(f"Dropout: {cfg['dropout_rate']}")
    logger.log(f"Weight Decay: {cfg['weight_decay']}")
    logger.log(f"Config:\n{json.dumps(cfg, indent=2)}")

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Dataset
    logger.log("\n" + "=" * 70)
    logger.log("BUILDING DATASET")
    logger.log("=" * 70)
    
    # Augmentation for training
    train_aug = VolumeAugmentation(cfg) if cfg["use_augmentation"] else None
    
    train_dataset_full = MSFullVolumeDataset(cfg["flair_dir"], cfg["mask_dir"], augmentation=train_aug)
    val_dataset_full = MSFullVolumeDataset(cfg["flair_dir"], cfg["mask_dir"], augmentation=None)  # No aug for val

    indices = list(range(len(train_dataset_full)))
    train_idx, val_idx = train_test_split(
        indices, test_size=cfg["val_fraction"], random_state=cfg["seed"]
    )
    
    train_ds = Subset(train_dataset_full, train_idx)
    val_ds   = Subset(val_dataset_full, val_idx)
    
    logger.log(f"  Train samples: {len(train_ds)}")
    logger.log(f"  Val samples:   {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Model with dropout
    logger.log("\n" + "=" * 70)
    logger.log("INITIALIZING MODEL")
    logger.log("=" * 70)
    
    model = SwinUNETR(
        img_size=cfg["img_size"],
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        feature_size=cfg["feature_size"],
        depths=cfg["depths"],
        num_heads=cfg["num_heads"],
        dropout_rate=cfg["dropout_rate"],  # ADDED DROPOUT
        use_checkpoint=cfg["use_checkpoint"],
        spatial_dims=3,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.log(f"  Model: SwinUNETR with {cfg['dropout_rate']:.1%} dropout")
    logger.log(f"  Total params:     {total_params:,}")
    logger.log(f"  Trainable params: {trainable_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )
    
    # Plateau scheduler - reduces LR when validation plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=cfg["plateau_factor"],
        patience=cfg["plateau_patience"],
        min_lr=cfg["min_lr"],
        verbose=True
    )
    
    loss_fn = FocalTverskyLoss(
        alpha=cfg["ftl_alpha"],
        beta=cfg["ftl_beta"],
        gamma=cfg["ftl_gamma"]
    )
    
    logger.log(f"\n  Loss: FocalTverskyLoss(α={cfg['ftl_alpha']}, β={cfg['ftl_beta']}, γ={cfg['ftl_gamma']})")
    logger.log(f"  Optimizer: AdamW(lr={cfg['lr']}, wd={cfg['weight_decay']})")
    logger.log(f"  Scheduler: ReduceLROnPlateau(patience={cfg['plateau_patience']})")

    scaler   = GradScaler() if cfg["use_amp"] else None
    metrics  = SegMetrics(threshold=cfg["threshold"])
    es       = EarlyStopping(patience=cfg["es_patience"], min_delta=cfg["es_min_delta"])
    ckpt_mgr = CheckpointManager(
        cfg["save_dir"],
        keep_last_n=cfg["keep_last_n"],
        save_interval=cfg["save_interval"]
    )

    # Resume
    start_epoch = 0
    resume_path = ckpt_mgr.get_resume_path(cfg["resume_path"], cfg["save_dir"])
    
    if resume_path:
        start_epoch, ckpt_mgr.best_dice = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler
        )

    # Training loop
    logger.log("\n" + "=" * 70)
    logger.log("TRAINING START WITH OVERFITTING MONITORING")
    logger.log("=" * 70)

    for epoch in range(start_epoch, cfg["epochs"]):
        logger.log(f"\nEpoch {epoch+1}/{cfg['epochs']}")
        logger.log("-" * 50)

        # Warmup
        if epoch < cfg["warmup_epochs"]:
            lr = cfg["lr"] * (epoch + 1) / cfg["warmup_epochs"]
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # Train with Dice tracking
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            scaler, metrics, device, cfg["use_amp"], cfg["grad_clip"]
        )

        # Validate
        val = validate_one_epoch(
            model, val_loader, loss_fn, metrics, device,
            cfg["sw_roi_size"], cfg["sw_batch_size"], cfg["sw_overlap"]
        )

        current_lr = optimizer.param_groups[0]["lr"]
        
        # Calculate overfitting gap
        dice_gap = train_dice - val["dice"]
        overfitting_warning = ""
        if cfg["monitor_overfitting"] and dice_gap > 0.1:
            overfitting_warning = f"  ⚠️  OVERFITTING WARNING: Train-Val gap = {dice_gap:.4f}"

        logger.log(
            f"  Train Loss={train_loss:.4f} Train Dice={train_dice:.4f} | "
            f"Val Loss={val['loss']:.4f} Val Dice={val['dice']:.4f} | "
            f"Recall={val['recall']:.4f} Precision={val['precision']:.4f} | LR={current_lr:.2e}"
        )
        if overfitting_warning:
            logger.log(overfitting_warning)
        
        logger.record(
            epoch+1, train_loss, val["loss"],
            val["dice"], val["recall"], val["precision"], current_lr, train_dice
        )

        # Save checkpoints
        is_best = ckpt_mgr.save(model, optimizer, scheduler, scaler, epoch+1, val["dice"], cfg)
        if is_best:
            logger.log(f"  ⭐ New best Dice: {val['dice']:.4f} — model saved")

        # Scheduler step (plateau based on validation)
        if epoch >= cfg["warmup_epochs"]:
            scheduler.step(val["dice"])

        # Early stopping
        if es(val["dice"]):
            logger.log(f"\n  🛑 Early stopping triggered at epoch {epoch+1}")
            logger.log(f"     Final Train Dice: {train_dice:.4f}, Val Dice: {val['dice']:.4f}")
            break

    logger.log("\n" + "=" * 70)
    logger.log("TRAINING COMPLETE")
    logger.log("=" * 70)
    logger.log(f"  Best Dice: {ckpt_mgr.best_dice:.4f}")
    logger.log(f"  Checkpoints: {cfg['save_dir']}")
    logger.log("=" * 70)

if __name__ == "__main__":
    main()
