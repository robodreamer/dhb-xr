"""
Training utilities for VFM token generators.

Provides:
- InvariantDataset: PyTorch dataset for DHB invariant sequences
- train_vfm_tokenizer: Training loop for joint tokenizer + flow matching
- Evaluation metrics and logging utilities
"""

from __future__ import annotations

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from typing import Optional, Dict, List, Callable, Any, Union
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = lambda x, **kwargs: x


class InvariantDataset(Dataset):
    """
    PyTorch Dataset for DHB invariant sequences.

    Supports loading from:
    - NumPy arrays
    - Lists of arrays
    - Directory of .npy files
    """

    def __init__(
        self,
        data: Union[np.ndarray, List[np.ndarray], str],
        seq_len: Optional[int] = None,
        normalize: bool = True,
        augment: bool = False,
    ):
        """
        Args:
            data: Invariant sequences as:
                - np.ndarray of shape (N, T, D)
                - List of (T_i, D) arrays (variable length)
                - Path to directory with .npy files
            seq_len: Fixed sequence length. If provided, sequences are
                padded/truncated to this length.
            normalize: Whether to normalize invariants.
            augment: Whether to apply data augmentation.
        """
        super().__init__()
        self.seq_len = seq_len
        self.normalize = normalize
        self.augment = augment

        # Load data
        if isinstance(data, str):
            self.sequences = self._load_from_dir(data)
        elif isinstance(data, np.ndarray):
            self.sequences = [data[i] for i in range(len(data))]
        else:
            self.sequences = list(data)

        # Compute normalization stats
        if normalize:
            all_data = np.concatenate(self.sequences, axis=0)
            self.mean = all_data.mean(axis=0)
            self.std = all_data.std(axis=0) + 1e-8
        else:
            self.mean = None
            self.std = None

    def _load_from_dir(self, path: str) -> List[np.ndarray]:
        """Load sequences from directory of .npy files."""
        sequences = []
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".npy"):
                arr = np.load(os.path.join(path, fname))
                sequences.append(arr)
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tensor:
        seq = self.sequences[idx].copy()

        # Normalize
        if self.normalize:
            seq = (seq - self.mean) / self.std

        # Pad/truncate to fixed length
        if self.seq_len is not None:
            if len(seq) < self.seq_len:
                # Pad with zeros
                pad = np.zeros((self.seq_len - len(seq), seq.shape[-1]))
                seq = np.concatenate([seq, pad], axis=0)
            elif len(seq) > self.seq_len:
                # Random crop
                if self.augment:
                    start = np.random.randint(0, len(seq) - self.seq_len + 1)
                else:
                    start = 0
                seq = seq[start:start + self.seq_len]

        # Augmentation
        if self.augment:
            seq = self._augment(seq)

        return torch.from_numpy(seq).float()

    def _augment(self, seq: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Random noise
        if np.random.random() < 0.3:
            noise = np.random.randn(*seq.shape) * 0.01
            seq = seq + noise

        # Random scale
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            seq = seq * scale

        return seq

    def denormalize(self, seq: Tensor) -> Tensor:
        """Convert normalized sequence back to original scale."""
        if not self.normalize:
            return seq
        mean = torch.from_numpy(self.mean).to(seq.device).float()
        std = torch.from_numpy(self.std).to(seq.device).float()
        return seq * std + mean


def train_vfm_tokenizer(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    tokenizer_weight: float = 1.0,
    flow_weight: float = 1.0,
    beta: float = 0.01,
    beta_schedule: Optional[Callable[[int], float]] = None,
    grad_clip: float = 1.0,
    log_every: int = 100,
    eval_every: int = 1000,
    save_every: int = 5000,
    save_dir: Optional[str] = None,
    device: str = "cpu",
    callback: Optional[Callable[[Dict], None]] = None,
) -> Dict[str, List[float]]:
    """
    Train VFMTokenGenerator with joint tokenizer + flow matching loss.

    Args:
        model: VFMTokenGenerator or similar model with loss() method.
        train_dataset: Training dataset of invariant sequences.
        val_dataset: Optional validation dataset.
        num_epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        tokenizer_weight: Weight for tokenizer reconstruction loss.
        flow_weight: Weight for flow matching loss.
        beta: KL divergence weight (for variational models).
        beta_schedule: Optional function epoch -> beta for KL annealing.
        grad_clip: Gradient clipping norm.
        log_every: Log frequency (steps).
        eval_every: Evaluation frequency (steps).
        save_every: Checkpoint save frequency (steps).
        save_dir: Directory for saving checkpoints.
        device: Device for training.
        callback: Optional callback function called with metrics dict.

    Returns:
        Dictionary of training history.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

    # Training history
    history = {
        "train_loss": [],
        "train_tokenizer_loss": [],
        "train_flow_loss": [],
        "train_kl_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }

    # Training loop
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {"total": 0, "tokenizer": 0, "flow": 0, "kl": 0}
        num_batches = 0

        # Get current beta
        current_beta = beta_schedule(epoch) if beta_schedule else beta

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            batch = batch.to(device)

            # Forward and loss
            optimizer.zero_grad()
            losses = model.loss(
                batch,
                tokenizer_weight=tokenizer_weight,
                flow_weight=flow_weight,
                beta=current_beta,
            )

            # Backward
            losses["total"].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Accumulate losses
            epoch_losses["total"] += losses["total"].item()
            epoch_losses["tokenizer"] += losses["tokenizer"].item()
            epoch_losses["flow"] += losses["flow"].item()
            epoch_losses["kl"] += losses["kl"].item()
            num_batches += 1
            global_step += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "tok": f"{losses['tokenizer'].item():.4f}",
                "flow": f"{losses['flow'].item():.4f}",
            })

            # Logging
            if global_step % log_every == 0:
                avg_loss = epoch_losses["total"] / num_batches
                history["train_loss"].append(avg_loss)
                history["learning_rate"].append(optimizer.param_groups[0]["lr"])

                if callback:
                    callback({
                        "step": global_step,
                        "epoch": epoch,
                        "train_loss": avg_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                    })

            # Evaluation
            if val_dataset is not None and global_step % eval_every == 0:
                val_loss = evaluate_model(model, val_loader, device, current_beta)
                history["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save_dir:
                        save_checkpoint(
                            model, optimizer, epoch, global_step,
                            os.path.join(save_dir, "best_model.pt")
                        )

            # Checkpointing
            if save_dir and global_step % save_every == 0:
                save_checkpoint(
                    model, optimizer, epoch, global_step,
                    os.path.join(save_dir, f"checkpoint_{global_step}.pt")
                )

        # End of epoch
        scheduler.step()

        # Record epoch averages
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        history["train_tokenizer_loss"].append(epoch_losses["tokenizer"])
        history["train_flow_loss"].append(epoch_losses["flow"])
        history["train_kl_loss"].append(epoch_losses["kl"])

        print(f"Epoch {epoch+1}: loss={epoch_losses['total']:.4f}, "
              f"tok={epoch_losses['tokenizer']:.4f}, "
              f"flow={epoch_losses['flow']:.4f}, "
              f"kl={epoch_losses['kl']:.4f}")

    return history


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    beta: float = 0.01,
) -> float:
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            losses = model.loss(batch, beta=beta)
            total_loss += losses["total"].item()
            num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    path: str,
):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
    }, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
) -> Dict:
    """Load training checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return {
        "epoch": checkpoint["epoch"],
        "step": checkpoint["step"],
    }


# ---- Evaluation metrics ----

def compute_reconstruction_error(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """
    Compute reconstruction error metrics.

    Returns MSE, MAE, and max error.
    """
    model.eval()
    all_mse = []
    all_mae = []
    all_max = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            outputs = model(batch)
            recon = outputs["invariants_reconstructed"]

            mse = ((batch - recon) ** 2).mean(dim=(1, 2))
            mae = (batch - recon).abs().mean(dim=(1, 2))
            max_err = (batch - recon).abs().max(dim=-1)[0].max(dim=-1)[0]

            all_mse.extend(mse.cpu().numpy())
            all_mae.extend(mae.cpu().numpy())
            all_max.extend(max_err.cpu().numpy())

    model.train()
    return {
        "mse": np.mean(all_mse),
        "mae": np.mean(all_mae),
        "max_error": np.mean(all_max),
    }


def compute_generation_diversity(
    model: nn.Module,
    num_samples: int = 100,
    seq_len: int = 50,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Compute diversity metrics for generated samples.

    Returns pairwise distance statistics.
    """
    model.eval()

    with torch.no_grad():
        # Generate samples
        if hasattr(model, "generate_multimodal"):
            samples = model.generate_multimodal(
                num_samples=num_samples // 4,
                seq_len=seq_len,
                num_modes=4,
                device=device,
            )
        else:
            samples = model.generate(
                num_samples=num_samples,
                seq_len=seq_len,
                device=device,
            )

        # Compute pairwise distances
        samples_flat = samples.reshape(samples.shape[0], -1)
        dists = torch.cdist(samples_flat, samples_flat)

        # Get upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()
        pairwise_dists = dists[mask]

    model.train()
    return {
        "mean_pairwise_dist": pairwise_dists.mean().item(),
        "std_pairwise_dist": pairwise_dists.std().item(),
        "min_pairwise_dist": pairwise_dists.min().item(),
        "max_pairwise_dist": pairwise_dists.max().item(),
    }


# ---- KL annealing schedules ----

def linear_kl_schedule(
    warmup_epochs: int,
    max_beta: float = 1.0,
) -> Callable[[int], float]:
    """Linear KL annealing from 0 to max_beta over warmup_epochs."""
    def schedule(epoch: int) -> float:
        if epoch < warmup_epochs:
            return max_beta * epoch / warmup_epochs
        return max_beta
    return schedule


def cyclical_kl_schedule(
    cycle_length: int = 10,
    num_cycles: int = 4,
    max_beta: float = 1.0,
) -> Callable[[int], float]:
    """Cyclical KL annealing."""
    def schedule(epoch: int) -> float:
        cycle = epoch // cycle_length
        if cycle >= num_cycles:
            return max_beta
        progress = (epoch % cycle_length) / cycle_length
        return max_beta * progress
    return schedule
