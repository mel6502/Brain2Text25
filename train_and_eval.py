"""
Complete training and evaluation script for phoneme recognition using GRU and
CNN‑GRU decoders.  This module defines two neural network architectures,
prepares the BCI speech dataset, implements training/validation/test loops,
computes edit‑distance based accuracy, runs inference on individual trials
and includes a simple hyperparameter tuning routine.  The aim of this file
is to provide a clean, end‑to‑end example that you can run locally to
reproduce the results and experiment with different model configurations.

Notes
-----
* The dataset is loaded via :func:`utils.load_data`.  That helper reads
  HDF5 files and returns a list of session dictionaries.  Each session
  dictionary contains keys ``"train"``, ``"val"`` and ``"test"`` with
  trial lists for the corresponding split.  When working on your own
  machine make sure the ``data/t15_copyTask_neuralData/hdf5_data_final``
  directory exists and contains the expected session files.
* The training loop uses the Connectionist Temporal Classification (CTC)
  objective.  During evaluation a greedy CTC decoder is applied to
  recover phoneme sequences; accuracy is measured using the normalised
  Levenshtein distance between predicted and target sequences.
* Hyperparameter tuning is implemented via a simple grid search.  You
  supply a dictionary of parameter lists and the code will evaluate
  models on a small number of epochs.  This is intended as a starting
  point; for more complex searches consider libraries such as
  Optuna or Ray Tune.

Example
-------
To train a GRU decoder on the provided dataset for two epochs and
evaluate it on the test set:

>>> from complete_model import train_and_evaluate, GRUDecoder
>>> params = {
...     "neural_dim": 512,
...     "n_units": 256,
...     "n_days": 2,
...     "n_classes": 41,
...     "rnn_dropout": 0.4,
...     "input_dropout": 0.2,
...     "n_layers": 2,
...     "patch_size": 14,
...     "patch_stride": 4
... }
>>> train_and_evaluate(GRUDecoder, params, n_epochs=2)

"""

from __future__ import annotations


from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence


from models import GRUDecoder, CNNGRUDecoder
from CTC_decode_acc_measurements import greedy_ctc_decode, sequence_accuracy
from tqdm import tqdm




try:
    from utils import load_data, merge_sessions, prepare_split_dataset
except ImportError as e:
    raise ImportError(
        "Failed to import utils. Make sure utils.py is in the working directory."
    ) from e





###############################################################################
#  Collate and length calculation functions
###############################################################################
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




###############################################################################
#  Training and evaluation
###############################################################################

def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    ctc_loss: nn.CTCLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    dataloader : DataLoader
        DataLoader providing training batches.
    ctc_loss : nn.CTCLoss
        Criterion for CTC loss.
    optimizer : Optimizer
        Optimiser to update model parameters.
    device : torch.device
        Device to perform computation on.

    Returns
    -------
    float
        Mean training loss over the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0
    for xs, ys, input_lengths, target_lengths, day_idxs in tqdm(dataloader):        
        optimizer.zero_grad()
        logits = model(xs, day_idxs)
        log_probs = logits.log_softmax(-1).permute(1, 0, 2)  # (T, B, C)
        loss = ctc_loss(log_probs, ys, input_lengths, target_lengths)
        # Skip NaN/infinite losses
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        running_loss += loss.item()
        num_batches += 1
    return running_loss / max(1, num_batches)


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


def train_and_evaluate(
    model_class: type[nn.Module],
    model_params: Dict[str, Any],
    n_epochs: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    batch_size: int = 8,
    device: torch.device | None = None,
    max_train_trials: int | None = None,
    max_val_trials: int | None = None,
    max_test_trials: int | None = None,
    decode_val: bool = True,
    ) -> nn.Module:
    """High‑level routine to train a model and report performance.

    Loads the dataset, prepares DataLoaders for train/val/test splits,
    instantiates the specified model class, trains for a given number of
    epochs and finally evaluates on the test set.  During evaluation
    greedy CTC decoding is used to compute normalised Levenshtein and
    exact sequence accuracies.

    Parameters
    ----------
    model_class : type
        Either :class:`GRUDecoder` or :class:`CNNGRUDecoder`.
    model_params : dict
        Hyperparameters passed to the model constructor.  Must include
        ``n_days`` equal to the number of sessions in the dataset and
        ``n_classes`` equal to the number of output classes.
    n_epochs : int, optional
        Number of training epochs.  Defaults to 1.
    learning_rate : float, optional
        Learning rate for AdamW.  Defaults to ``1e-3``.
    weight_decay : float, optional
        Weight decay for AdamW.  Defaults to ``1e-2``.
    batch_size : int, optional
        Mini‑batch size.  Defaults to 8.
    device : torch.device or None, optional
        Computation device.  When ``None`` selects GPU if available.
    max_train_trials, max_val_trials, max_test_trials : int or None, optional
        Limit the number of trials per session for training, validation and
        testing respectively.  Useful for debugging.
    decode_val, decode_test : bool, optional
        Whether to compute accuracy metrics during validation and test
        evaluation.  Decoding can be expensive so you may disable it on
        large datasets.

    Returns
    -------
    nn.Module
        The trained model.  The caller may save or further fine tune
        this model.
    """

    train_losses = []
    val_losses = []
    val_lev_accs = []
    val_seq_accs = []

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # Instantiate model
    model = model_class(**model_params)
    model.to(device)

    # Dataloaders
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
    # Optimiser and scheduler
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)
    optimiser = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=2
    )

    best_val_acc = -float("inf")
    best_epoch : int = -1

    # Training loop
    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimiser, device)
        val_loss, val_lev_acc, val_seq_acc = evaluate(
            model, val_loader, criterion, decode=decode_val
        )
        scheduler.step(val_loss)
        # ✅ store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_lev_accs.append(val_lev_acc)
        val_seq_accs.append(val_seq_acc)
        print(
            f"Epoch {epoch}/{n_epochs} | train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | lev acc: {val_lev_acc:.4f} | seq acc: {val_seq_acc:.4f}"
        )

        
        # ✅ SAVE BEST MODEL
        if val_lev_acc > best_val_acc: # can also monitor with val_lev_acc
            best_val_acc = val_lev_acc
            best_epoch = epoch

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_params": model_params,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_lev_acc": val_lev_acc,
                },
                "BaselineGRU/best_model.ckpt"
            )
        torch.save(model.state_dict(), "BaselineGRU/last_model.pt")

        
    ckpt = torch.load("BaselineGRU/best_model.ckpt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(
        f"Loaded best model from epoch {ckpt['epoch']} "
        f"(val loss = {ckpt['val_loss']:.4f})"
        f" with val lev acc = {ckpt['val_lev_acc']:.4f}"
    )

    model.eval()
    all_test_preds = []

    with torch.no_grad():
        for xs, _, _, _, day_idxs in test_loader:
            logits = model(xs, day_idxs)
            preds = greedy_ctc_decode(logits)
            all_test_preds.extend(preds)

    print(f"Generated predictions for {len(all_test_preds)} test samples")   

    losses_and_accs = {
        "train_losses": train_losses,
        "val_losses": val_losses, 
        "val_lev_accs": val_lev_accs,
        "val_seq_accs": val_seq_accs
    }

    return all_test_preds, model, losses_and_accs






# ###############################################################################
# #  Hyperparameter tuning
# ###############################################################################

# def tune_hyperparameters(
#     model_class: type[nn.Module],
#     param_grid: Dict[str, Sequence[Any]],
#     n_samples: int = 1,
#     n_epochs: int = 1,
#     learning_rate: float = 1e-3,
#     weight_decay: float = 1e-2,
#     batch_size: int = 8,
#     device: torch.device | None = None,
#     max_train_trials: int | None = None,
#     max_val_trials: int | None = None,
#     decode_val: bool = False,
# ) -> Dict[str, Any]:
#     """Simple hyperparameter tuning via grid or random search.

#     Generates combinations of hyperparameters from ``param_grid``, trains
#     each candidate model for a small number of epochs and selects the
#     configuration with the lowest validation loss.  To reduce training
#     time you may specify ``n_samples`` to randomly sample that many
#     combinations instead of evaluating the full grid.

#     Parameters
#     ----------
#     model_class : type
#         The model class to instantiate (``GRUDecoder`` or ``CNNGRUDecoder``).
#     param_grid : dict
#         Dictionary mapping parameter names to sequences of possible values.
#     n_samples : int, optional
#         Number of random parameter combinations to evaluate.  When equal to
#         the total number of combinations (default ``1``) the search is
#         exhaustive.
#     n_epochs : int, optional
#         Number of epochs to train each candidate model.  Defaults to 1.
#     learning_rate, weight_decay, batch_size, device, max_train_trials,
#     max_val_trials, decode_val : see :func:`train_and_evaluate`.

#     Returns
#     -------
#     dict
#         The hyperparameter setting achieving the lowest validation loss.
#     """
#     # Compute full list of parameter combinations
#     keys = list(param_grid.keys())
#     all_combinations = list(itertools.product(*(param_grid[k] for k in keys)))
#     if n_samples is None or n_samples <= 0 or n_samples >= len(all_combinations):
#         sampled = all_combinations
#     else:
#         sampled = random.sample(all_combinations, n_samples)
#     best_params: Dict[str, Any] | None = None
#     best_val_loss: float = float("inf")
#     # Load and merge data once outside the loop to avoid repeated I/O
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     raw_data = load_data()
#     merged = merge_sessions(raw_data)
#     train_examples = prepare_split_dataset(
#         merged["train"], drop_blank=True, max_trials=max_train_trials
#     )
#     val_examples = prepare_split_dataset(
#         merged.get("val", []), drop_blank=True, max_trials=max_val_trials
#     )
#     # Minimal dataloaders used for tuning
#     for combination in sampled:
#         params = {k: v for k, v in zip(keys, combination)}
#         # Add mandatory parameters if missing
#         params.setdefault("n_days", len(merged["train"]))
#         params.setdefault("n_classes", len(indexes_to_phonemes(list(range(41)))))
#         model = model_class(**params).to(device)
#         criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)
#         optimiser = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             optimiser, mode="min", factor=0.5, patience=1, verbose=False
#         )
#         # DataLoaders
#         train_loader = torch.utils.data.DataLoader(
#             train_examples,
#             batch_size=batch_size,
#             shuffle=True,
#             collate_fn=lambda b: collate_fn(b, model, device),
#         )
#         val_loader = torch.utils.data.DataLoader(
#             val_examples,
#             batch_size=batch_size,
#             shuffle=False,
#             collate_fn=lambda b: collate_fn(b, model, device),
#         )
#         # Train for a few epochs
#         for epoch in range(n_epochs):
#             train_epoch(model, train_loader, criterion, optimiser, device)
#             val_loss, _, _ = evaluate(model, val_loader, criterion, device, decode=decode_val)
#             scheduler.step(val_loss)
#         # Keep the best
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_params = params
#     assert best_params is not None, "Hyperparameter tuning failed to evaluate any models."
#     print(f"Best validation loss {best_val_loss:.4f} achieved with params: {best_params}")
#     return best_params


# ###############################################################################
# #  Inference utility
# ###############################################################################

# def run_inference(
#     model: nn.Module,
#     example: Tuple[np.ndarray, np.ndarray, int],
#     device: torch.device | None = None,
# ) -> Tuple[List[str], List[str]]:
#     """Perform inference on a single trial and decode to phonemes.

#     Takes a trained model and a single example (feature array, target
#     sequence, session index), runs a forward pass and greedy CTC decoding to
#     produce a predicted phoneme sequence.  Also converts the true target
#     sequence into phoneme symbols.

#     Parameters
#     ----------
#     model : nn.Module
#         Trained model.
#     example : tuple
#         A tuple ``(features, targets, session_idx)`` corresponding to one
#         trial.  ``features`` should be a 2‑D numpy array of shape
#         ``(time_steps, neural_dim)``, ``targets`` is a 1‑D numpy array of
#         class IDs and ``session_idx`` is the session index.
#     device : torch.device, optional
#         Device on which to perform inference.  Defaults to the model's
#         device.

#     Returns
#     -------
#     tuple
#         ``(pred_phonemes, target_phonemes)`` where each element is a list
#         of strings representing the phoneme sequence.
#     """
#     if device is None:
#         device = next(model.parameters()).device
#     features, targets, session_idx = example
#     model.eval()
#     with torch.no_grad():
#         x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
#         day_idx = torch.tensor([session_idx], dtype=torch.long).to(device)
#         logits = model(x, day_idx)
#         preds = greedy_ctc_decode(logits)[0]
#     # Convert to phoneme symbols (exclude blank index 0)
#     pred_phonemes = indexes_to_phonemes(preds)
#     target_phonemes = indexes_to_phonemes(targets.tolist())
#     return pred_phonemes, target_phonemes


# __all__ = [
#     "GRUDecoder",
#     "CNNGRUDecoder",
#     "merge_sessions",
#     "prepare_split_dataset",
#     "collate_fn",
#     "greedy_ctc_decode",
#     "sequence_accuracy",
#     "train_and_evaluate",
#     "tune_hyperparameters",
#     "run_inference",
# ]