from datasets import load_dataset, Dataset
import pandas as pd
from g2p_en import G2p
import nltk
import sys
sys.path.append("..")
import utils

def g2p_to_40p(raw):
    """Converts g2p to 40 phonemes."""
    cleaned = []
    for p in raw:
        if p == ' ':
            cleaned.append('|')
        elif p.isalnum():
            cleaned.append(p.rstrip("012"))
    return cleaned

def phoneme_list2str(phoneme_list):
    return ' '.join(phoneme_list).replace(' | ', '|').replace('BLANK', '').strip('| ')

def save_dataset_from_data(data, split_name, save_path):
    all_phonemes = []
    all_texts = []

    for session_data in data:
        if split_name not in session_data:
            continue
        split = session_data[split_name]
        
        seq_class_ids_list = split['seq_class_ids']
        sentence_label_list = split['sentence_label']
        
        # Iterate through the trials in this session file
        for i in range(len(seq_class_ids_list)):
            seq_ids = seq_class_ids_list[i]
            text = sentence_label_list[i]
            
            if seq_ids is None or text is None:
                continue
                
            # Convert indices to phonemes
            phonemes = utils.indexes_to_phonemes(seq_ids)
            
            # Process phonemes using the function defined above
            processed_p = phoneme_list2str(phonemes)
            
            # Process text
            processed_t = text.decode('utf-8') if isinstance(text, bytes) else str(text)
            
            all_phonemes.append(processed_p)
            all_texts.append(processed_t)

    # Create a Hugging Face Dataset
    df = pd.DataFrame({'phonemes': all_phonemes, 'text': all_texts})
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(save_path)

    return dataset

def load_dataset_and_process(dataset_name='agentlans/high-quality-english-sentences', split='train', limit=None, save_path="data/english_sentences"):
    
    if limit: dataset = load_dataset(dataset_name, split=split).select(range(limit))
    else: dataset = load_dataset(dataset_name, split=split)

    g2p = G2p()

    def filter_and_convert(batch):

        results = {"phonemes": [], "text": []}
        
        for sentence in batch["text"]:
            results["text"].append(sentence)
            results["phonemes"].append(" ".join(g2p_to_40p(g2p(sentence))).strip('| '))
        
        return results

    processed_dataset = dataset.map(
        filter_and_convert, 
        batched=True, 
        remove_columns=dataset.column_names
    )

    processed_dataset.save_to_disk(save_path)

    return processed_dataset
