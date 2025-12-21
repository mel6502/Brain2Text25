import torch
from typing import List, Sequence, Tuple

###############################################################################
#  CTC decoding and accuracy measurement
###############################################################################

def greedy_ctc_decode(logits: torch.Tensor) -> List[List[int]]:
    """Greedy CTC decode of batched logits.

    Given a tensor of unnormalised scores ``logits`` of shape ``(B, T, C)``
    where ``C`` is the number of classes (including the blank), this function
    performs a simple greedy decoding: for each time step it selects the
    class with the highest logit, collapses consecutive repeats and removes
    blank tokens (assumed to be index 0).

    Parameters
    ----------
    logits : torch.Tensor
        Logits output by the model with shape ``(B, T, C)``.

    Returns
    -------
    list of lists
        A list of ``B`` sequences where each sequence is a list of class
        indices after CTC post‑processing.
    """
    with torch.no_grad():
        max_indices = logits.argmax(dim=-1).cpu().numpy()  # shape (B, T)
    decoded: List[List[int]] = []
    for seq in max_indices:
        out_seq: List[int] = []
        prev = None
        for idx in seq:
            if idx != 0 and idx != prev:
                out_seq.append(int(idx))
            prev = idx
        decoded.append(out_seq)
    return decoded


def levenshtein_distance(s1: Sequence[int], s2: Sequence[int]) -> int:
    """Compute the Levenshtein (edit) distance between two sequences.

    This implementation uses a two‑row dynamic programming algorithm for
    efficiency.  It supports arbitrary sequences of integers.

    Parameters
    ----------
    s1, s2 : sequences
        The sequences to be compared.

    Returns
    -------
    int
        The edit distance between ``s1`` and ``s2``.
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, start=1):
        current_row = [i]
        for j, c2 in enumerate(s2, start=1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def sequence_accuracy(predictions: List[List[int]], targets: List[List[int]]) -> Tuple[float, float]:
    """Compute normalised edit distance and exact sequence accuracy.

    Two metrics are returned: ``lev_acc`` which is one minus the normalised
    Levenshtein distance (averaged over all examples) and ``seq_acc`` which
    is the proportion of predictions that exactly match the target sequence.

    Parameters
    ----------
    predictions : list of lists of ints
        The predicted sequences (after CTC decoding).
    targets : list of lists of ints
        The ground truth sequences.

    Returns
    -------
    tuple
        ``(lev_acc, seq_acc)`` where both metrics lie in ``[0, 1]``.  A
        higher value indicates better performance.
    """
    assert len(predictions) == len(targets), "Mismatched number of predictions and targets"
    total_normalised_distance = 0.0
    exact_matches = 0
    for pred, tgt in zip(predictions, targets):
        if len(tgt) == 0:
            # Avoid division by zero; consider empty target as correct if prediction is also empty
            norm_dist = 0.0 if len(pred) == 0 else 1.0
        else:
            dist = levenshtein_distance(pred, tgt)
            norm_dist = dist / len(tgt)
        total_normalised_distance += norm_dist
        if pred == tgt:
            exact_matches += 1
    lev_acc = 1.0 - total_normalised_distance / len(predictions)
    seq_acc = exact_matches / len(predictions)
    return lev_acc, seq_acc
