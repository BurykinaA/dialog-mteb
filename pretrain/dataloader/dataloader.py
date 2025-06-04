import os
import csv
import random # Added for MLM
import torch # Added for MLM
import pandas as pd
import torch.utils.data as util_data
from torch.utils.data import Dataset

class PairSamples(Dataset):
    def __init__(self, train_x1, train_x2, pairsimi, tokenizer, max_length, mlm_probability):
        assert len(pairsimi) == len(train_x1) == len(train_x2)
        self.train_x1 = train_x1 # Context texts
        self.train_x2 = train_x2 # Future texts
        self.pairsimi = pairsimi
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size
        # Special tokens that should not be masked
        self.special_tokens_ids = {self.cls_token_id, self.sep_token_id, self.pad_token_id}


    def _apply_mlm_strategy(self, token_ids: torch.Tensor):
        """
        Apply MLM strategy to a sequence of token IDs.
        Returns:
            mlm_input_ids: Token IDs with MLM masking applied.
            mlm_labels: Labels for MLM (-100 for non-masked tokens).
        """
        mlm_input_ids = token_ids.clone()
        mlm_labels = torch.full_like(token_ids, -100)

        for i, token_id in enumerate(token_ids):
            if token_id.item() in self.special_tokens_ids:
                continue # Do not mask special tokens

            if random.random() < self.mlm_probability:
                mlm_labels[i] = token_id.item() # Original token is the label
                
                rand_val = random.random()
                if rand_val < 0.8: # 80% chance to replace with [MASK]
                    mlm_input_ids[i] = self.mask_token_id
                elif rand_val < 0.9: # 10% chance to replace with a random token
                    # Ensure random token is not a special token if possible,
                    # for simplicity, we sample from the whole vocab.
                    random_token_id = random.randint(0, self.vocab_size - 1)
                    mlm_input_ids[i] = random_token_id
                # else 10% chance to keep original token (mlm_input_ids[i] is already token_id)
        
        return mlm_input_ids, mlm_labels

    def __len__(self):
        return len(self.pairsimi)

    def __getitem__(self, idx):
        context_text = self.train_x1[idx]
        future_text = self.train_x2[idx]
        similarity_score = self.pairsimi[idx]

        # Tokenize context_text for MLM
        context_encoding = self.tokenizer.encode_plus(
            context_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt' # Return PyTorch tensors
        )
        
        original_context_input_ids = context_encoding['input_ids'].squeeze(0) # Remove batch dim
        context_attention_mask = context_encoding['attention_mask'].squeeze(0) # Remove batch dim

        # Apply MLM strategy to the tokenized context
        context_mlm_input_ids, context_mlm_labels = self._apply_mlm_strategy(original_context_input_ids)
        
        return {
            'text1': context_text, # Raw context text
            'text2': future_text,  # Raw future text
            'pairsimi': torch.tensor(similarity_score, dtype=torch.float), # Ensure pairsimi is a tensor
            'context_mlm_input_ids': context_mlm_input_ids,
            'context_mlm_attention_mask': context_attention_mask, # Mask for MLM inputs
            'context_mlm_labels': context_mlm_labels
        }


'''
Assumed data format:

sentence1, sentence2

'''
def pair_loader_csv(args, tokenizer): # Added tokenizer and args
    delimiter = "," if args.dataname.endswith(".csv") else "\t"
    file_path = os.path.join(args.datapath, args.dataname)

    with open(file_path, mode='r', encoding='utf-8') as file:
        train_data = list(csv.reader(file, delimiter=delimiter))
    
    train_text1 = [d[0] for d in train_data]
    train_text2 = [d[1] for d in train_data]
    pairsimi = [1.0 for _ in train_data] # Assuming 1.0 for positive pairs

    train_dataset = PairSamples(
        train_text1, 
        train_text2, 
        pairsimi,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mlm_probability=getattr(args, 'mlm_probability', 0.15) # Use arg or default
    )
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader


'''
Expect a txt file where each line contains a single sentence/paragraph.
'''
def pair_loader_txt(args, tokenizer): # Added tokenizer and args
    with open(os.path.join(args.datapath, args.dataname), "r") as f:
        texts = f.readlines()
        texts = [t.strip("\n") for t in texts]

    # For txt, text1 and text2 are the same, future is effectively the same as context.
    # This setup might need adjustment based on how FutureTOD expects context/future from single texts.
    # For now, following the existing logic:
    train_text1 = texts 
    train_text2 = texts # If future is different, this needs change
    pairsimi = [1.0] * len(texts)

    train_dataset = PairSamples(
        train_text1, 
        train_text2, 
        pairsimi,
        tokenizer=tokenizer,
        max_length=args.max_length,
        mlm_probability=getattr(args, 'mlm_probability', 0.15) # Use arg or default
    )
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader
