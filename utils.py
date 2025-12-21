import os
import h5py
from tqdm import tqdm
import numpy as np
from typing import Any, Dict, List, Tuple

LOGIT_TO_PHONEME = [
'BLANK',    # "BLANK" = CTC blank symbol
'AA', 'AE', 'AH', 'AO', 'AW',
'AY', 'B', 'CH', 'D', 'DH',
'EH', 'ER', 'EY', 'F', 'G',
'HH', 'IH', 'IY', 'JH', 'K',
'L', 'M', 'N', 'NG', 'OW',
'OY', 'P', 'R', 'S', 'SH',
'T', 'TH', 'UH', 'UW', 'V',
'W', 'Y', 'Z', 'ZH',
' | ',    # "|" = silence token
]


def load_h5py_file(file_path):
    data = {
        'neural_features': [],
        'n_time_steps': [],
        'seq_class_ids': [],
        'seq_len': [],
        'transcriptions': [],
        'sentence_label': [],
        'session': [],
        'block_num': [],
        'trial_num': [],
    }
    # Open the hdf5 file for that day
    with h5py.File(file_path, 'r') as f:

        keys = list(f.keys())

        # For each trial in the selected trials in that day
        for key in keys:
            g = f[key]

            neural_features = g['input_features'][:]
            n_time_steps = g.attrs['n_time_steps']
            seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
            seq_len = g.attrs['seq_len'] if 'seq_len' in g.attrs else None
            transcription = g['transcription'][:] if 'transcription' in g else None
            sentence_label = g.attrs['sentence_label'][:] if 'sentence_label' in g.attrs else None
            session = g.attrs['session']
            block_num = g.attrs['block_num']
            trial_num = g.attrs['trial_num']

            data['neural_features'].append(neural_features)
            data['n_time_steps'].append(n_time_steps)
            data['seq_class_ids'].append(seq_class_ids)
            data['seq_len'].append(seq_len)
            data['transcriptions'].append(transcription)
            data['sentence_label'].append(sentence_label)
            data['session'].append(session)
            data['block_num'].append(block_num)
            data['trial_num'].append(trial_num)
    return data


def load_data():
    data = []
    data_path = 'data/t15_copyTask_neuralData/hdf5_data_final'

    sessions = os.listdir(data_path)
    sessions.sort()

    early = 0
    for session in tqdm(sessions):
        data.append({})
        
        session_files = os.listdir(os.path.join(data_path, session))
        session_files.sort()
        if len(session_files)==1:
            for file in session_files:
                file_path = os.path.join(data_path, session, file)
                data[-1]["train"] = load_h5py_file(file_path)
        else:
            for file in session_files:
                file_path = os.path.join(data_path, session, file)
                if "train" in file:
                    data[-1]["train"] = load_h5py_file(file_path)
                elif "val" in file:
                    data[-1]["val"] = load_h5py_file(file_path)
                else:
                    data[-1]["test"] = load_h5py_file(file_path)
        early += 1
        if early>=2:
            break
    return data

def indexes_to_phonemes(indexes):
    phonemes = [LOGIT_TO_PHONEME[idx] for idx in indexes if idx in range(len(LOGIT_TO_PHONEME))]
    return phonemes



###############################################################################
#  Data preparation utilities
###############################################################################

def merge_sessions(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Merge individual session dictionaries into a single split dictionary.

    ``load_data`` returns a list of sessions.  Each element in the list is a
    dictionary mapping split names ("train", "val", "test") to a trial
    dictionary.  This function converts the list into a dictionary with
    keys ``train``, ``val``, ``test`` whose values are lists of session
    dictionaries.  The purpose is to make it easy to iterate over all
    sessions of a given split.

    Parameters
    ----------
    data : list of dicts
        Output of :func:`load_data`.

    Returns
    -------
    dict
        Dictionary with keys ``train``, ``val``, ``test`` and list values.
    """
    merged = {"train": [], "val": [], "test": []}
    for session in data:
        for split in ("train", "val", "test"):
            if split in session:
                merged[split].append(session[split])
    return merged


def prepare_split_dataset(
    split_data: List[Dict[str, Any]],
    drop_blank: bool = True,
    max_trials: int | None = None,
    has_labels: bool = True,
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Flatten a split into a list of examples with their session index.

    Parameters
    ----------
    split_data : list of dict
        A list where each element corresponds to a session.  Each session
        dictionary must contain keys ``neural_features``, ``seq_class_ids``
        and ``seq_len``.
    drop_blank : bool, optional
        If ``True``, examples whose target sequence contains the blank
        token (index 0) are discarded.  CTC models normally use the blank
        symbol internally, not as part of the label, so this ensures
        accuracy metrics are meaningful.  Defaults to ``True``.
    max_trials : int or None, optional
        Limit the number of trials returned from each session.  Useful for
        quick debugging.  When ``None`` (default) all trials are used.

    Returns
    -------
    list of tuples
        Each entry is a tuple ``(features, target, session_idx)`` where
        ``features`` is a numpy array of shape ``(time_steps, neural_dim)``,
        ``target`` is a 1‑D numpy array of class IDs and ``session_idx``
        identifies the session from which the trial originates.
    """
    examples: List[Tuple[np.ndarray, np.ndarray, int]] = []
    for session_idx, session in enumerate(split_data):
        num_trials = len(session["neural_features"])
        trial_indices = list(range(num_trials))
        if max_trials is not None:
            trial_indices = trial_indices[:max_trials]
        for i in trial_indices:
            features = session["neural_features"][i]
            if has_labels:
                targets = session.get("seq_class_ids", [None])[i]
                target_len = session.get("seq_len", [None])[i]

                if targets is None or target_len is None:
                    continue

                targets = targets[:target_len]

                # Drop sequences containing blank label if requested
                if drop_blank and (0 in targets):
                    continue

                examples.append(
                    (features, targets.astype(np.int64), session_idx)
                )
            else:
                # TEST SET: NO LABELS
                examples.append(
                    (features, None, session_idx)
                )
    return examples