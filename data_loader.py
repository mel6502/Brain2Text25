import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from utils import load_data, merge_sessions, prepare_split_dataset
from torch.nn.utils.rnn import pad_sequence
from models import GRUDecoder, CNNGRUDecoder
from torch.utils.data import Dataset



def add_white_noise(
    x: torch.Tensor,
    noise_level: float = 0.04,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Add white Gaussian noise scaled by signal RMS.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (T, D) or (B, T, D).
    noise_level : float
        Noise std as a fraction of signal RMS (typical: 0.01–0.08).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Noisy tensor with same shape as x.
    """
    # Compute RMS over time and features
    rms = x.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(eps)

    noise = torch.randn_like(x) * (noise_level * rms)
    return x + noise

def time_mask(
    x: torch.Tensor,
    max_width: int = 20,
    n_masks: int = 1,
) -> torch.Tensor:
    """
    Apply time masking along the temporal dimension.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (T, D).
    max_width : int
        Maximum width (in time steps) of each mask.
    n_masks : int
        Number of independent time masks.

    Returns
    -------
    torch.Tensor
        Time-masked tensor with same shape as x.
    """
    T = x.size(0)

    for _ in range(n_masks):
        if T <= 1:
            break

        w = torch.randint(0, max_width + 1, (1,)).item()
        if w == 0 or w >= T:
            continue

        t0 = torch.randint(0, T - w + 1, (1,)).item()
        x[t0 : t0 + w] = 0

    return x



def collate_fn(
    batch,
    model,
    device,
):
    xs, ys, day_idxs = zip(*batch)

    # Inputs
    # xs_tensors = [torch.tensor(x, dtype=torch.float32) for x in xs]
    raw_lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)

    # Compute input lengths after downsampling
    if isinstance(model, GRUDecoder) and model.patch_size > 0:
        input_lengths = ((raw_lengths - model.patch_size) // model.patch_stride) + 1
    elif isinstance(model, CNNGRUDecoder):
        kernel = model.conv1.kernel_size[0]
        stride = model.conv1.stride[0]
        padding = model.conv1.padding[0]
        input_lengths = ((raw_lengths + 2 * padding - (kernel - 1) - 1) // stride) + 1
    else:
        input_lengths = raw_lengths

    xs_padded = pad_sequence(xs_tensors, batch_first=True)
    day_idxs_tensor = torch.tensor(day_idxs, dtype=torch.long)

    # ✅ TEST SET (no labels)
    if ys[0] is None:
        return (
            xs_padded.to(device),
            None,
            input_lengths.to(device),
            None,
            day_idxs_tensor.to(device),
        )

    # ✅ TRAIN / VAL (with labels)
    target_tensors = [torch.tensor(y, dtype=torch.long) for y in ys]
    target_lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
    ys_concat = torch.cat(target_tensors)

    return (
        xs_padded.to(device),
        ys_concat.to(device),
        input_lengths.to(device),
        target_lengths.to(device),
        day_idxs_tensor.to(device),
    )



def load_and_prepare_data(
    max_train_trials: Optional[int] = None,
    max_val_trials: Optional[int] = None,
    max_test_trials: Optional[int] = None
) -> Tuple[List, List, List]:
    """Load and prepare data for all splits.
    
    Returns:
        Tuple of (train_examples, val_examples, test_examples)
    """
    # Load and merge data
    raw_data = load_data()
    merged = merge_sessions(raw_data)
    print("Data loaded and merged.")
    
    # Prepare examples for each split
    train_examples = prepare_split_dataset(
        merged["train"], drop_blank=True, max_trials=max_train_trials, has_labels=True
    )
    val_examples = prepare_split_dataset(
        merged.get("val", []), drop_blank=True, max_trials=max_val_trials, has_labels=True
    )
    test_examples = prepare_split_dataset(
        merged.get("test", []), drop_blank=True, max_trials=max_test_trials, has_labels=False
    )
    
    return train_examples, val_examples, test_examples


class BrainDataset(Dataset):
    def __init__(self, examples, augment: bool = False):
        self.examples = examples
        self.augment = augment

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x, y, day_idx = self.examples[idx]
        
        x = torch.as_tensor(x, dtype=torch.float32)

        if self.augment:
            x = add_white_noise(x, noise_level=0.04)
            x = time_mask(x, max_width=20)

        return x, y, day_idx


def create_dataloaders(
    train_examples: List,
    val_examples: List,
    test_examples: List,
    model: nn.Module,
    batch_size: int = 8,
    device: Optional[torch.device] = None
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader
]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Wrap raw examples in Datasets
    train_dataset = BrainDataset(train_examples, augment=True)
    val_dataset   = BrainDataset(val_examples, augment=False)
    test_dataset  = BrainDataset(test_examples, augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, model, device),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, model, device),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, model, device),
    )

    return train_loader, val_loader, test_loader