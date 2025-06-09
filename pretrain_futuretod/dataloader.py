import os
import csv
import pandas as pd
import torch.utils.data as util_data
from torch.utils.data import Dataset, DataLoader
import torch
import json
from transformers import BertTokenizer
import random

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
    def __init__(self, data_path, tokenizer, max_len=512, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_probability = mlm_probability
        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dialogue = self.data[idx]

        utterances = []
        for turn in dialogue:
            speaker_token = "[USR]" if turn['speaker'] == 'user' else "[SYS]"
            utterance = f"{speaker_token} {turn['utterance']}"
            utterances.append(utterance)

        num_turns = len(utterances)
        if num_turns < 2:
            split_turn_idx = num_turns - 1
        else:
            # Split after a random turn, ensuring at least one turn in context and one in future
            split_turn_idx = random.randint(0, num_turns - 2)

        context_utterances = utterances[:split_turn_idx + 1]
        future_utterances = utterances[split_turn_idx + 1:]
        
        selected_future_utterances = []
        if len(future_utterances) > 0:
            # Randomly select a number of future utterances to include
            num_future_to_include = random.randint(1, len(future_utterances))
            selected_future_utterances = future_utterances[:num_future_to_include]

        context_text = " ".join(context_utterances)
        full_text = " ".join(context_utterances + selected_future_utterances)

        context_inputs = self.tokenizer(context_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        full_inputs = self.tokenizer(full_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")

        # Prepare for Masked Language Modeling on context
        context_input_ids = context_inputs['input_ids'].squeeze(0)
        labels = context_input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        context_input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
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
    dataloader = DataLoader(FutureTODDataset(data_path, tokenizer, max_len), batch_size=batch_size, shuffle=shuffle)
    return dataloader
