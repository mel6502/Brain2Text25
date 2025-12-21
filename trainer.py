import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple
import os
from tqdm import tqdm

from CTC_decode_acc_measurements import greedy_ctc_decode, sequence_accuracy


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    ctc_loss: nn.CTCLoss,
    decode: bool = True,
) -> Tuple[float, float, float]:
    """Evaluate model on a dataset.

    Calculates the average CTC loss and, optionally, decodes predictions and
    computes accuracy metrics.  When ``decode`` is ``False`` the
    Levenshtein and sequence accuracies are returned as zeros to avoid
    expensive decoding.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    dataloader : DataLoader
        DataLoader providing evaluation batches.
    ctc_loss : nn.CTCLoss
        Criterion for CTC loss.
    device : torch.device
        Device to perform computation on.
    decode : bool, optional
        Whether to compute accuracy metrics via greedy CTC decoding.

    Returns
    -------
    tuple
        ``(avg_loss, lev_acc, seq_acc)``.  When ``decode`` is ``False`` the
        accuracy metrics are zero.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0
    all_preds: List[List[int]] = []
    all_tgts: List[List[int]] = []
    with torch.no_grad():
        for xs, ys, input_lengths, target_lengths, day_idxs in dataloader:
            logits = model(xs, day_idxs)
            log_probs = logits.log_softmax(-1).permute(1, 0, 2)
            loss = ctc_loss(log_probs, ys, input_lengths, target_lengths)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            running_loss += loss.item()
            num_batches += 1
            if decode:
                # Greedy decode; logits shape (B, T, C)
                preds = greedy_ctc_decode(logits)
                # Recover targets per example
                # ys is concatenated, so slice according to target_lengths
                offset = 0
                for t_len in target_lengths:
                    all_tgts.append(ys[offset : offset + t_len].cpu().numpy().tolist())
                    offset += t_len
                all_preds.extend(preds)
    avg_loss = running_loss / max(1, num_batches)
    if decode and all_preds:
        print("Calculating sequence accuracy...")
        lev_acc, seq_acc = sequence_accuracy(all_preds, all_tgts)
    else:
        print("Skipping sequence accuracy calculation.")
        lev_acc, seq_acc = 0.0, 0.0
    return avg_loss, lev_acc, seq_acc


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        checkpoint_dir: str = "checkpoints"
    ):
        self.model = model
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize loss, optimizer, and scheduler
        self.criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )
        
        # Training history
        self.history = {
            "train_losses": [],
            "val_losses": [],
            "val_lev_accs": [],
            "val_seq_accs": [],
            "best_epoch": -1,
            "best_val_acc": -float("inf")
        }
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        show_progress: bool = True
    ) -> float:
        """Train for one epoch with optional progress bar."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Create progress bar
        if show_progress:
            pbar = tqdm(
                train_loader,
                desc=f"Training",
                leave=False,
                dynamic_ncols=True
            )
        else:
            pbar = train_loader
        
        for xs, labels, input_lengths, label_lengths, day_idxs in pbar:
            self.optimizer.zero_grad()
            
            logits = self.model(xs, day_idxs)
            loss = self.criterion(
                logits.log_softmax(dim=2).transpose(0, 1),
                labels,
                input_lengths,
                label_lengths
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar description
            if show_progress:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss / num_batches:.4f}"
                })
        
        if show_progress:
            pbar.close()
        
        return total_loss / max(num_batches, 1)
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        n_epochs: int = 1,
        decode_val: bool = True
    ) -> Dict[str, List]:
        """Main training loop."""
        for epoch in range(1, n_epochs + 1):
            # Training phase
            train_loss = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_lev_acc, val_seq_acc = evaluate(
                self.model, val_loader, self.criterion, decode=decode_val
            )
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Store metrics
            self.history["train_losses"].append(train_loss)
            self.history["val_losses"].append(val_loss)
            self.history["val_lev_accs"].append(val_lev_acc)
            self.history["val_seq_accs"].append(val_seq_acc)
            
            # Print progress
            print(
                f"Epoch {epoch}/{n_epochs} | "
                f"train loss: {train_loss:.4f} | "
                f"val loss: {val_loss:.4f} | "
                f"lev acc: {val_lev_acc:.4f} | "
                f"seq acc: {val_seq_acc:.4f}"
            )
            
            # Save best model
            if val_lev_acc > self.history["best_val_acc"]:
                self.history["best_val_acc"] = val_lev_acc
                self.history["best_epoch"] = epoch
                self.save_checkpoint(
                    filename="best_model.ckpt",
                    epoch=epoch,
                    val_loss=val_loss,
                    val_lev_acc=val_lev_acc
                )
            
            # Save latest model
            self.save_checkpoint(
                filename="last_model.pt",
                epoch=epoch,
                val_loss=val_loss,
                val_lev_acc=val_lev_acc
            )
        
        return self.history
    
    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        val_loss: float,
        val_lev_acc: float
    ) -> None:
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_lev_acc": val_lev_acc,
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "history": self.history
            },
            checkpoint_path
        )
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(
        self,
        filename: str = "best_model.ckpt",
        load_optimizer: bool = False
    ) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state"])
        
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.history = checkpoint["history"]
        
        print(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} "
            f"(val loss = {checkpoint['val_loss']:.4f}, "
            f"val lev acc = {checkpoint['val_lev_acc']:.4f})"
        )
        
        return checkpoint