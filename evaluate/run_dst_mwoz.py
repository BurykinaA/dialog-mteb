import argparse
import json
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from tqdm import tqdm, trange


def set_seed(seed=24):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DSTDataloader(torch.utils.data.Dataset):
    def __init__(self, encodings, labels_tensor):
        self.encodings = encodings
        self.labels_tensor = labels_tensor  # [N, num_slots]

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels_tensor[idx]
        return item

    def __len__(self):
        return self.encodings['input_ids'].shape[0]


class CLSDSTModel(nn.Module):
    def __init__(self, model_name, slot2num_labels, dropout=0.1, trust_remote_code=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.slot2head = nn.ModuleDict({slot: nn.Linear(hidden, num_cls) for slot, num_cls in slot2num_labels.items()})

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        logits = {slot: head(cls) for slot, head in self.slot2head.items()}

        loss = None
        if labels is not None:
            # labels: [B, num_slots] with per-slot class ids
            ce = nn.CrossEntropyLoss()
            loss_sum = 0.0
            for j, slot in enumerate(self.slot2head.keys()):
                loss_sum += ce(logits[slot], labels[:, j])
            loss = loss_sum

        return logits, loss


def load_mwoz_dst(data_dir, model_name, max_seq_length, ontology_path=None, percent=-1):
    # Expected files:
    #  - train.json, dev.json, test.json
    #  Each JSON is a list of dicts: {"text": "<dialogue history>", "state": {"domain-slot": "value", ...}}
    #  - ontology.json with: {"domain-slot": ["none", "dontcare", "value1", ...], ...}
    if ontology_path is None:
        ontology_path = os.path.join(data_dir, "ontology.json")
    with open(ontology_path, "r") as f:
        ontology = json.load(f)

    slots = sorted(list(ontology.keys()))
    slot2id = {s: i for i, s in enumerate(slots)}
    slot2num_labels = {s: len(ontology[s]) for s in slots}
    slot_value2id = {s: {v: i for i, v in enumerate(ontology[s])} for s in slots}

    def _load_split(name):
        with open(os.path.join(data_dir, f"{name}.json"), "r") as f:
            records = json.load(f)
        # Optional percent subsample for train split only
        if name == "train" and percent > 0 and percent < 100:
            k = max(1, int(len(records) * (percent / 100.0)))
            records = random.sample(records, k)
        return records

    train_records = _load_split("train")
    dev_records = _load_split("dev")
    test_records = _load_split("test")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    def _tensorize(records):
        texts = [r["text"] for r in records]
        enc = tokenizer(
            texts, return_tensors='pt', padding='longest', truncation=True, max_length=max_seq_length
        )
        labels = []
        for r in records:
            st = r["state"]
            y = []
            for s in slots:
                val = st.get(s, "none")
                val = val if val in slot_value2id[s] else "none"
                y.append(slot_value2id[s][val])
            labels.append(y)
        labels = torch.tensor(labels, dtype=torch.long)
        return enc, labels

    train_enc, train_y = _tensorize(train_records)
    dev_enc, dev_y = _tensorize(dev_records)
    test_enc, test_y = _tensorize(test_records)

    return (DSTDataloader(train_enc, train_y),
            DSTDataloader(dev_enc, dev_y),
            DSTDataloader(test_enc, test_y),
            slot2num_labels,
            slots)


@torch.no_grad()
def evaluate(model, dataloader, device, slots):
    model.eval()
    total_slots = 0
    correct_slots = 0
    joint_correct = 0
    total_examples = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = []
        for j, s in enumerate(slots):
            pred_j = torch.argmax(logits[s], dim=1)  # [B]
            preds.append(pred_j.unsqueeze(1))
        preds = torch.cat(preds, dim=1)  # [B, num_slots]
        correct = (preds == labels).long()
        correct_slots += correct.sum().item()
        total_slots += correct.numel()
        joint_correct += (correct.sum(dim=1) == correct.shape[1]).sum().item()
        total_examples += correct.shape[0]
    slot_acc = correct_slots / total_slots
    joint_acc = joint_correct / total_examples
    return joint_acc, slot_acc


def train(args):
    set_seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_ds, dev_ds, test_ds, slot2num_labels, slots = load_mwoz_dst(
        data_dir=args.data_dir,
        model_name=args.model_type,
        max_seq_length=args.max_seq_length,
        ontology_path=args.ontology,
        percent=args.data_ratio if args.data_ratio > 0 else -1
    )

    model = CLSDSTModel(args.model_type, slot2num_labels, dropout=args.classifier_dropout)
    model.to(device)

    train_sampler = RandomSampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.per_gpu_batch_size, sampler=train_sampler)
    dev_loader = DataLoader(dev_ds, batch_size=args.per_gpu_batch_size, sampler=SequentialSampler(dev_ds))
    test_loader = DataLoader(test_ds, batch_size=args.per_gpu_batch_size, sampler=SequentialSampler(test_ds))

    optim = torch.optim.AdamW(model.parameters(), lr=args.bert_lr, weight_decay=args.weight_decay)
    total_steps = max(1, int(len(train_ds) * args.epoch / (args.per_gpu_batch_size)))
    sched = get_linear_schedule_with_warmup(optim, int(0.05 * total_steps), total_steps)

    best_joint = -1.0
    best_test = (0.0, 0.0)
    early = 0

    for ep in trange(args.epoch, desc="Epoch"):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            _, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()
            sched.step()
            optim.zero_grad()
            epoch_loss += loss.item()

            if (step + 1) % max(1, args.eval_steps) == 0:
                joint, slot = evaluate(model, dev_loader, device, slots)
                if joint > best_joint:
                    early = 0
                    best_joint = joint
                    best_test = evaluate(model, test_loader, device, slots)
                else:
                    early += 1
        if early > args.patience:
            break

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "result.txt"), "a") as f:
        f.write(f"Joint Acc: {best_test[0]:.4f}, Slot Acc: {best_test[1]:.4f}\n")
    with open(os.path.join(args.output_dir, "best_result.txt"), "a") as f:
        f.write(f"{best_test[0]:.4f}\t{best_test[1]:.4f}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ontology", type=str, default=None)
    p.add_argument("--model_type", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--bert_lr", type=float, default=5e-5)
    p.add_argument("--epoch", type=int, default=20)
    p.add_argument("--per_gpu_batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=24)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--classifier_dropout", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--data_ratio", type=float, default=-1, help="Percent of training data to use. -1 means full.")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()