import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm, trange


def set_seed(seed=24):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_pairs_txt(path):
    # Each line: "<context>\t<response>"
    with open(path, "r") as f:
        rows = [l.rstrip("\n").split("\t") for l in f]
    ctx = [r[0] for r in rows]
    rsp = [r[1] for r in rows]
    return ctx, rsp


def load_rs_splits(data_dir):
    # Expect train.txt, dev.txt, test.txt with "context \t response" per line
    tr_ctx, tr_rsp = read_pairs_txt(os.path.join(data_dir, "train.txt"))
    dv_ctx, dv_rsp = read_pairs_txt(os.path.join(data_dir, "dev.txt"))
    te_ctx, te_rsp = read_pairs_txt(os.path.join(data_dir, "test.txt"))
    return (tr_ctx, tr_rsp), (dv_ctx, dv_rsp), (te_ctx, te_rsp)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, enc):
        self.enc = enc

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()}

    def __len__(self):
        return self.enc['input_ids'].shape[0]


class CLSEncoder(nn.Module):
    def __init__(self, model_name, trust_remote_code=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    def embed(self, input_ids, attention_mask, method="cls"):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if method == "cls":
            emb = out.last_hidden_state[:, 0, :]
        else:
            att = attention_mask.unsqueeze(-1)
            emb = (out.last_hidden_state * att).sum(dim=1) / att.sum(dim=1).clamp(min=1e-6)
        emb = emb.to(torch.float32)
        return emb

    def forward(self, input_ids, attention_mask):
        return self.embed(input_ids, attention_mask, method="cls")


def batchify_texts(tokenizer, texts, max_len):
    return tokenizer(texts, return_tensors='pt', padding='longest', truncation=True, max_length=max_len)


def embed_corpus(encoder, tokenizer, texts, device, batch_size, max_length):
    enc = batchify_texts(tokenizer, texts, max_length)
    ds = TextDataset(enc)
    dl = DataLoader(ds, batch_size=batch_size, sampler=SequentialSampler(ds))
    all_e = []
    with torch.no_grad():
        encoder.eval()
        encoder.to(device)
        for b in dl:
            e = encoder(b['input_ids'].to(device), b['attention_mask'].to(device))
            all_e.append(e.cpu())
    return torch.cat(all_e, dim=0)  # [N, H]


def train_inbatch(model, tokenizer, train_ctx, train_rsp, device, args):
    # In-batch negatives training
    bz = args.per_gpu_batch_size
    steps_per_epoch = max(1, len(train_ctx) // bz)
    optim = torch.optim.AdamW(model.parameters(), lr=args.bert_lr, weight_decay=args.weight_decay)
    total_steps = steps_per_epoch * args.epoch
    sched = get_linear_schedule_with_warmup(optim, int(0.05 * total_steps), total_steps)
    ce = nn.CrossEntropyLoss()

    # shard training data if percent subset requested
    if args.data_ratio > 0 and args.data_ratio < 100:
        k = max(1, int(len(train_ctx) * (args.data_ratio / 100.0)))
        idx = random.sample(range(len(train_ctx)), k)
        train_ctx = [train_ctx[i] for i in idx]
        train_rsp = [train_rsp[i] for i in idx]

    model.train()
    for ep in trange(args.epoch, desc="Epoch"):
        loader_iter = range(0, len(train_ctx), bz)
        running = 0.0
        for st in tqdm(loader_iter, desc="Iteration"):
            ctx_batch = train_ctx[st:st + bz]
            rsp_batch = train_rsp[st:st + bz]
            if len(ctx_batch) < 2:
                continue
            ctx_enc = batchify_texts(tokenizer, ctx_batch, args.max_seq_length)
            rsp_enc = batchify_texts(tokenizer, rsp_batch, args.max_resp_length)
            ctx_emb = model(ctx_enc['input_ids'].to(device), ctx_enc['attention_mask'].to(device))
            rsp_emb = model(rsp_enc['input_ids'].to(device), rsp_enc['attention_mask'].to(device))
            logits = torch.mm(ctx_emb, rsp_emb.t().contiguous())
            labels = torch.arange(logits.shape[0], device=logits.device)
            loss = ce(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()
            sched.step()
            optim.zero_grad()
            running += loss.item()


@torch.no_grad()
def eval_k_to_100(model, tokenizer, te_ctx, te_rsp, device, args):
    # Build candidate response pool from test responses
    rsp_pool = te_rsp
    # Pre-embed all test contexts and response pool
    ctx_emb = embed_corpus(model, tokenizer, te_ctx, device, args.eval_batch_size, args.max_seq_length)
    rsp_emb = embed_corpus(model, tokenizer, rsp_pool, device, args.eval_batch_size, args.max_resp_length)
    rsp_emb_t = rsp_emb.t().contiguous()

    acc1 = 0
    acc3 = 0
    for i in range(len(te_ctx)):
        # sample 99 negatives from pool
        all_idx = list(range(len(rsp_pool)))
        gt = te_rsp[i]
        # Find ground-truth index in pool; if multiple duplicates, choose one occurrence
        gt_indices = [j for j, r in enumerate(rsp_pool) if r == gt]
        if len(gt_indices) == 0:
            continue
        gt_idx = gt_indices[0]
        neg_idx = [j for j in all_idx if j != gt_idx]
        if len(neg_idx) < 99:
            cand_idx = neg_idx
        else:
            cand_idx = random.sample(neg_idx, 99)
        cand_idx.append(gt_idx)
        cand_mat = rsp_emb[cand_idx]  # [100, H]
        # score
        q = ctx_emb[i:i+1]  # [1, H]
        scores = torch.mm(q, cand_mat.t())
        rank = torch.argsort(scores, dim=1, descending=True)[0]
        top1 = cand_idx[rank[0].item()]
        top3 = [cand_idx[rank[j].item()] for j in range(min(3, len(cand_idx)))]
        acc1 += int(top1 == gt_idx)
        acc3 += int(gt_idx in top3)

    n = len(te_ctx)
    return acc1 / n, acc3 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--model_type", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--max_seq_length", type=int, default=128)
    ap.add_argument("--max_resp_length", type=int, default=32)
    ap.add_argument("--bert_lr", type=float, default=2e-5)
    ap.add_argument("--epoch", type=int, default=5)
    ap.add_argument("--per_gpu_batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=128)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=24)
    ap.add_argument("--data_ratio", type=float, default=-1, help="Percent of training data (1,10,...). -1 full.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    (tr_ctx, tr_rsp), (dv_ctx, dv_rsp), (te_ctx, te_rsp) = load_rs_splits(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, trust_remote_code=True)
    model = CLSEncoder(args.model_type)
    model.to(device)

    train_inbatch(model, tokenizer, tr_ctx, tr_rsp, device, args)
    acc1, acc3 = eval_k_to_100(model, tokenizer, te_ctx, te_rsp, device, args)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "result.txt"), "a") as f:
        f.write(f"1-to-100: {acc1:.4f}, 3-to-100: {acc3:.4f}\n")
    with open(os.path.join(args.output_dir, "best_result.txt"), "a") as f:
        f.write(f"{acc1:.4f}\t{acc3:.4f}\n")


if __name__ == "__main__":
    main()