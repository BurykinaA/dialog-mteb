import os
import csv
import pandas as pd
import torch.utils.data as util_data
from torch.utils.data import Dataset, DataLoader
import torch
import json
from transformers import BertTokenizer
import random
import re
import logging

class PairSamples(Dataset):
    def __init__(self, train_x1, train_x2, pairsimi):
        assert len(pairsimi) == len(train_x1) == len(train_x2)
        self.train_x1 = train_x1
        self.train_x2 = train_x2
        self.pairsimi = pairsimi
        
        
    def __len__(self):
        return len(self.pairsimi)

    def __getitem__(self, idx):
        return {'text1': self.train_x1[idx], 'text2': self.train_x2[idx], 'pairsimi': self.pairsimi[idx]}


'''
Assumed data format:

sentence1, sentence2

'''
def pair_loader_csv(args):
    delimiter = "," if args.dataname.endswith(".csv") else "\t"
    file_path = os.path.join(args.datapath, args.dataname)

    # train_data = list(csv.reader(os.path.join(args.datapath, args.dataname), delimiter=delimiter))
    with open(file_path, mode='r', encoding='utf-8') as file:
        train_data = list(csv.reader(file, delimiter=delimiter))
    
    train_text1 = [d[0] for d in train_data]
    train_text2 = [d[1] for d in train_data]
    pairsimi = [1 for _ in train_data]

    train_dataset = PairSamples(train_text1, train_text2, pairsimi)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader


'''
Expect a txt file where each line contains a single sentence/paragraph.
'''
def pair_loader_txt(args):
    with open(os.path.join(args.datapath, args.dataname), "r") as f:
        texts = f.readlines()
        texts = [t.strip("\n") for t in texts]

    train_text1 = texts
    train_text2 = texts
    pairsimi = [1] * len(texts)

    train_dataset = PairSamples(train_text1, train_text2, pairsimi)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader

class FutureTODDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=512, mlm_probability=0.15, delimiter='\t'):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_probability = mlm_probability
        self.data = []

        ext = os.path.splitext(data_path)[1].lower()
        # Support both .txt (one dialogue per line) and CSV/TSV (dialogue possibly in a column)
        if ext == ".txt":
            with open(data_path, mode='r', encoding='utf-8') as f:
                for line in f:
                    dialogue = line.strip()
                    if not dialogue:
                        continue
                    # Keep only dialogues with at least two turns
                    turns = re.findall(r'(\[(?:USR|SYS)\].*?)(?=\[USR\]|\[SYS\]|$)', dialogue)
                    if len(turns) >= 2:
                        self.data.append(dialogue)
        else:
            delim = '\t' if ext in ['.tsv', '.tab'] else delimiter
            with open(data_path, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter=delim)
                for row in reader:
                    dialogue = None
                    # If there are 3+ columns, try the 3rd one (legacy format)
                    if len(row) >= 3:
                        dialogue = row[2]
                    elif len(row) == 1:
                        dialogue = row[0]
                    elif len(row) > 0:
                        dialogue = " ".join(row)
                    if dialogue:
                        turns = re.findall(r'(\[(?:USR|SYS)\].*?)(?=\[USR\]|\[SYS\]|$)', dialogue)
                        if len(turns) >= 2:
                            self.data.append(dialogue)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dialogue = self.data[idx]

        # Robust turn splitting: split on [USR] or [SYS] and keep delimiters
        parts = re.split(r'(\[(?:USR|SYS)\])', dialogue)
        turns = []
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                turn = parts[i] + parts[i + 1]
                turn = turn.strip()
                if turn:
                    turns.append(turn)

        if len(turns) < 2:
            # Fallback to original method if new method fails
            turns = re.findall(r'(\[(?:USR|SYS)\].*?)(?=\[USR\]|\[SYS\]|$)', dialogue)

        num_turns = len(turns)
        # Ensure at least one context turn and one future turn
        num_context_turns = random.randint(1, num_turns - 1)

        context_turns = turns[:num_context_turns]
        future_turns = turns[num_context_turns:]

        context_text = " ".join(context_turns).strip()

        # Randomly choose how many future turns to include
        P = random.choice([1, 3, 5, 'All'])
        if P == 'All':
            F = len(future_turns)
            L = random.randint(1, F)
            future_subset_turns = future_turns[:L]
        else:
            future_subset_turns = future_turns[:P]

        future_text = " ".join(future_subset_turns).strip()

        logging.info(f"\n--- Sample {idx} ---")
        logging.info(f"Context: {context_text}")
        logging.info(f"Future: {future_text}")

        # Tokenize context for MLM
        context_inputs = self.tokenizer(
            context_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # IMPORTANT: For contrastive learning, treat "full" as the FUTURE side so
        # the model can compute similarity/contrastive losses between context and future.
        future_inputs = self.tokenizer(
            future_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        context_input_ids = context_inputs['input_ids'].squeeze(0)
        labels = context_input_ids.clone()

        # Create MLM labels on the context only
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        context_input_ids[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        context_input_ids[indices_random] = random_words[indices_random]

        return {
            'context_input_ids': context_input_ids,
            'context_attention_mask': context_inputs['attention_mask'].squeeze(0),
            'context_mlm_labels': labels,
            # full_* now carry FUTURE tokens to be used for contrastive/similarity losses with context
            'full_input_ids': future_inputs['input_ids'].squeeze(0),
            'full_attention_mask': future_inputs['attention_mask'].squeeze(0),
        }

def get_dataloader(data_path, tokenizer, batch_size, max_len, shuffle=True):
    dataset = FutureTODDataset(data_path, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Assumes script is run from the project root.
    data_path = 'pretrain_jasper/processed_dialogues.txt'
    
    if not os.path.exists(data_path):
        logging.error(f"Data file not found at '{data_path}'. Please ensure the file exists.")
    else:
        logging.info(f"Loading data from: {data_path}")

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # You can adjust max_len if needed for your specific data.
        dataset = FutureTODDataset(data_path=data_path, tokenizer=tokenizer, max_len=512)
        
        logging.info(f"Dataset size: {len(dataset)}")

        if len(dataset) > 0:
            logging.info("\n--- Showing 3 samples from the dataset (logged from __getitem__) ---")
            for i in range(min(3, len(dataset))):
                _ = dataset[i] # This will trigger the logging in __getitem__

            # To check with DataLoader:
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
            
            try:
                logging.info("\n--- Checking a batch from DataLoader (items will be logged from __getitem__) ---")
                batch = next(iter(dataloader))
                logging.info(f"Batch keys: {list(batch.keys())}")
                logging.info("--- Batch check complete ---")
            except StopIteration:
                logging.warning("DataLoader is empty, cannot check batch.")