import os
import h5py
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk, load_from_disk

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

PHONEME_TO_LOGIT = {p: i for i, p in enumerate(LOGIT_TO_PHONEME)}

def save_dataset_split_batched(split_name, save_path=None, data_path=None, session_ids=None):
    data_path = data_path or 'data/t15_copyTask_neuralData/hdf5_data_final'
    save_path = save_path or f'data/{split_name}_dataset'

    datasets_accum = []

    sessions = sorted(os.listdir(data_path))

    if session_ids is not None:
        sessions = [sessions[i] for i in session_ids]
    print(f"Processing {len(sessions)} sessions for split '{split_name}'")

    for session in tqdm(sessions):
        session_data = {}
        files = os.listdir(os.path.join(data_path, session))

        for file in files:
            subset = file.split('_')[-1].split('.')[0]
            if subset != split_name:
                continue

            file_data = load_h5py_file(os.path.join(data_path, session, file))
            for k, v in file_data.items():
                session_data.setdefault(k, []).extend(v)

        if not session_data:
            continue

        df = pd.DataFrame(session_data)
        df["neural_features"] = df["neural_features"].apply(lambda x: x.tolist())

        datasets_accum.append(Dataset.from_pandas(df, preserve_index=False))

        del df, session_data

    dataset = concatenate_datasets(datasets_accum)
    dataset.save_to_disk(save_path)
    return dataset

def save_large_dataset_split_batched(split_name, save_path=None, data_path=None, session_ids=None, batch_size=10):
    data_path = data_path or 'data/t15_copyTask_neuralData/hdf5_data_final'
    save_path = save_path or f'data/{split_name}_dataset'
    save_dir = os.path.dirname(save_path)

    sessions = sorted(os.listdir(data_path))

    for i in range(len(sessions)/batch_size):
        session_ids = range(i*batch_size, min((i+1)*batch_size, len(sessions)))
        batch_save_path = f"{save_path}_part{i}"
        save_dataset_split_batched(split_name, save_path=batch_save_path, data_path=data_path, session_ids=session_ids)

    datasets_accum = []
    
    for files in os.listdir(save_dir):
        if files.startswith(f"{save_path}_part"):
            dataset_part = load_from_disk(os.path.join(save_dir, files))
            datasets_accum.append(dataset_part)
    
    dataset = concatenate_datasets(datasets_accum)
    dataset.save_to_disk(save_path)
    

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


def load_data(path=None, session_numbers=None):
    data = []
    data_path = path if path is not None else 'data/t15_copyTask_neuralData/hdf5_data_final'

    sessions = os.listdir(data_path)
    if session_numbers is not None:
        sessions = [sessions[i] for i in session_numbers]
    sessions.sort()
    for session in tqdm(sessions):
        # Skip session when no test, train, val
        files = os.listdir(os.path.join(data_path, session))
        data.append({})
        
        for file in files:
            subset = file.split('_')[-1].split('.')[0]  # get subset from filename
            file_path = os.path.join(data_path, session, file)
            data[-1][subset] = load_h5py_file(file_path)
    return data

def indexes_to_phonemes(indexes):
    phonemes = [LOGIT_TO_PHONEME[idx] for idx in indexes if idx in range(len(LOGIT_TO_PHONEME))]
    return phonemes
