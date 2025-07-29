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


class FutureTODDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=512, mlm_probability=0.15, delimiter='\t'):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_probability = mlm_probability
        self.data = []

        with open(data_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=delimiter)
            for row in reader:
                if len(row) >= 3:
                    dialogue = row[2]
                    turns = re.findall(r'(\[(?:USR|SYS)\].*?)(?=\[USR\]|\[SYS\]|$)', dialogue)
                    if len(turns) >= 2:
                        self.data.append(dialogue)

    def __len__(self):
        return 2#len(self.data)

    def __getitem__(self, idx):
        dialogue = self.data[idx]

        turns = re.findall(r'(\[(?:USR|SYS)\].*?)(?=\[USR\]|\[SYS\]|$)', dialogue)
        num_turns = len(turns)

        num_context_turns = random.randint(1, num_turns - 1)
        
        context_turns = turns[:num_context_turns]
        future_turns = turns[num_context_turns:]

        context_text = "".join(context_turns).strip()

        P = random.choice([1, 3, 5, 'All'])

        if P == 'All':
            F = len(future_turns)
            L = random.randint(1, F)
            future_subset_turns = future_turns[:L]
        else:
            future_subset_turns = future_turns[:P]
        
        future_text = "".join(future_subset_turns).strip()

        full_text = context_text + " " + self.tokenizer.sep_token + " " + future_text

        context_inputs = self.tokenizer(context_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        
        full_inputs = self.tokenizer(full_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")

        context_input_ids = context_inputs['input_ids'].squeeze(0)
        labels = context_input_ids.clone()

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
            'full_input_ids': full_inputs['input_ids'].squeeze(0),
            'full_attention_mask': full_inputs['attention_mask'].squeeze(0),
        }

def get_dataloader(data_path, tokenizer, batch_size, max_len, shuffle=True):
    dataset = FutureTODDataset(data_path, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
