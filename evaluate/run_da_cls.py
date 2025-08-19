import argparse
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm, trange

from utils.data import get_dialogue_action_dataset


def set_seed(seed=24):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CLSMultiLabel(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.2, trust_remote_code=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        logits = self.classifier(cls)  # [B, L]
        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits, labels.float())
        return logits, loss


@torch.no_grad()
def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    all_labels = []
    all_preds = []
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['seq_labels']
        logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        prob = torch.sigmoid(logits).cpu()
        pred = (prob > threshold).long()
        all_labels.append(labels)
        all_preds.append(pred)
    y_true = torch.cat(all_labels, dim=0).numpy()
    y_pred = torch.cat(all_preds, dim=0).numpy()
    micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return micro, macro


def train(args):
    set_seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Reuse existing loader; it concatenates history; we'll use CLS pooling here.
    train_ds, test_ds, val_ds, num_labels = get_dialogue_action_dataset(
        BERT_MODEL=args.model_type,
        file_path=args.data_dir,
        max_seq_length=args.max_seq_length,
        concatenate=True,
        num_turn=1
    )
    # Optional few-shot by sampling a percent of train_ds
    if args.data_ratio > 0 and args.data_ratio < 100:
        n = len(train_ds)
        k = max(1, int(n * (args.data_ratio / 100.0)))
        idx = random.sample(range(n), k)
        # Subsample tensors in-place
        train_ds.encodings = {k_: v[idx] for k_, v in train_ds.encodings.items()}
        train_ds.seq_labels = [train_ds.seq_labels[i] for i in idx]

    model = CLSMultiLabel(args.model_type, num_labels, dropout=args.classifier_dropout)
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=args.per_gpu_batch_size, sampler=RandomSampler(train_ds))
    val_loader = DataLoader(val_ds, batch_size=args.per_gpu_batch_size, sampler=SequentialSampler(val_ds))
    test_loader = DataLoader(test_ds, batch_size=args.per_gpu_batch_size, sampler=SequentialSampler(test_ds))

    optim = torch.optim.AdamW(model.parameters(), lr=args.bert_lr, weight_decay=args.weight_decay)
    total_steps = max(1, int(len(train_ds) * args.epoch / args.per_gpu_batch_size))
    sched = get_linear_schedule_with_warmup(optim, int(0.05 * total_steps), total_steps)

    best_main = -1
    best_scores = (0.0, 0.0)
    early = 0

    for ep in trange(args.epoch, desc="Epoch"):
        model.train()
        for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['seq_labels'].to(device)
            logits, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()
            sched.step()
            optim.zero_grad()

            if (step + 1) % max(1, args.eval_steps) == 0:
                micro, macro = evaluate(model, val_loader, device)
                main = (micro + macro) / 2.0
                if main > best_main:
                    best_main = main
                    best_scores = evaluate(model, test_loader, device)
                    early = 0
                else:
                    early += 1
        if early > args.patience:
            break

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "result.txt"), "a") as f:
        f.write(f"Micro F1: {best_scores[0]:.4f}, Macro F1: {best_scores[1]:.4f}\n")
    with open(os.path.join(args.output_dir, "best_result.txt"), "a") as f:
        f.write(f"{best_scores[0]:.4f}\t{best_scores[1]:.4f}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--model_type", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--bert_lr", type=float, default=5e-5)
    p.add_argument("--epoch", type=int, default=20)
    p.add_argument("--per_gpu_batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=24)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--classifier_dropout", type=float, default=0.2)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--data_ratio", type=float, default=-1, help="Percent of training data (1, 10, ...). -1 full.")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()