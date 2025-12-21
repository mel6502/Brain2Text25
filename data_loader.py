import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from utils import load_data, merge_sessions, prepare_split_dataset
from torch.nn.utils.rnn import pad_sequence
from models import GRUDecoder, CNNGRUDecoder

def collate_fn(
    batch,
    model,
    device,
):
    xs, ys, day_idxs = zip(*batch)

    # Inputs
    xs_tensors = [torch.tensor(x, dtype=torch.float32) for x in xs]
    raw_lengths = torch.tensor([x.shape[0] for x in xs_tensors], dtype=torch.long)

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


def create_dataloaders(
    train_examples: List,
    val_examples: List,
    test_examples: List,
    model: nn.Module,
    batch_size: int = 8,
    device: Optional[torch.device] = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create dataloaders for train, validation, and test splits."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = torch.utils.data.DataLoader(
        train_examples,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, model, device),
    )
    val_loader = torch.utils.data.DataLoader(
        val_examples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, model, device),
    )
    test_loader = torch.utils.data.DataLoader(
        test_examples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, model, device),
    )
    
    return train_loader, val_loader, test_loader