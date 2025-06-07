import argparse
import os
import torch
import torch.nn.functional as F
import wandb
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer

from model import PSCBert
from dataloader import pair_loader_csv, pair_loader_txt
from utils import HardConLoss


def main(args):
    if args.wandb_project:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.loader_type == "txt":
        train_dataloader = pair_loader_txt(args)
    elif args.loader_type == "csv":
        train_dataloader = pair_loader_csv(args)
    else:
        raise ValueError("Invalid loader_type specified. Use 'txt' or 'csv'.")

    student_model = PSCBert.from_pretrained(args.model_name, output_hidden_states=True).to(device)
    teacher_model = PSCBert.from_pretrained(args.model_name, output_hidden_states=True).to(device)
    teacher_model.eval()
    teacher_model.load_state_dict(student_model.state_dict())

    contrastive_loss_fn = HardConLoss(temperature=args.temperature, contrast_type=args.contrast_type)

    optimizer = AdamW(student_model.parameters(), lr=args.learning_rate)
    num_training_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = int(num_training_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    for epoch in range(args.epochs):
        student_model.train()
        total_loss, total_distill_loss, total_contrastive_loss = 0, 0, 0
        progress_bar_train = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{args.epochs}", leave=False)
        for batch in progress_bar_train:
            optimizer.zero_grad()

            context_texts, future_texts = batch["text1"], batch["text2"]
            pairsimi = batch["pairsimi"].to(device)

            contrastive_inputs1 = tokenizer(context_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length)
            contrastive_inputs2 = tokenizer(future_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length)
            contrastive_input_ids = torch.stack([contrastive_inputs1["input_ids"], contrastive_inputs2["input_ids"]], dim=1).to(device)
            contrastive_attention_mask = torch.stack([contrastive_inputs1["attention_mask"], contrastive_inputs2["attention_mask"]], dim=1).to(device)
            cnst_feat1, cnst_feat2, _, _ = student_model(input_ids=contrastive_input_ids, attention_mask=contrastive_attention_mask, task_type="train")
            l_contrastive = contrastive_loss_fn(cnst_feat1, cnst_feat2, pairsimi)['instdisc_loss']

            student_inputs = tokenizer(context_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length).to(device)
            teacher_inputs = tokenizer(context_texts, future_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length).to(device)

            student_hidden_states = student_model(task_type="distill", **student_inputs)
            with torch.no_grad():
                teacher_hidden_states = teacher_model(task_type="distill", **teacher_inputs)

            l_distill = 0
            for s_state, t_state in zip(student_hidden_states[1:], teacher_hidden_states[1:]):
                l_distill += F.mse_loss(s_state[:, 0], t_state[:, 0])

            loss = l_contrastive + args.distill_weight * l_distill
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_distill_loss += l_distill.item()
            total_contrastive_loss += l_contrastive.item()
            if args.wandb_project:
                wandb.log({ "train_loss_step": loss.item(), "distill_loss_step": l_distill.item(), "contrastive_loss_step": l_contrastive.item(), "lr": scheduler.get_last_lr()[0] })
            progress_bar_train.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_dataloader)
        avg_distill_loss = total_distill_loss / len(train_dataloader)
        avg_contrastive_loss = total_contrastive_loss / len(train_dataloader)
        if args.wandb_project:
            wandb.log({ "train_loss_epoch": avg_train_loss, "distill_loss_epoch": avg_distill_loss, "contrastive_loss_epoch": avg_contrastive_loss, "epoch": epoch })
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Distill Loss = {avg_distill_loss:.4f}, Contrastive Loss = {avg_contrastive_loss:.4f}")

        if (epoch + 1) % args.teacher_update_epochs == 0:
            print(f"Updating teacher model at epoch {epoch+1}")
            teacher_model.load_state_dict(student_model.state_dict())

        if args.output_dir:
            output_path = os.path.join(args.output_dir, f"epoch_{epoch+1}")
            os.makedirs(output_path, exist_ok=True)
            student_model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            print(f"Model from epoch {epoch+1} saved to {output_path}")

    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain a sentence transformer model with self-distillation.")
    parser.add_argument("--model_name", type=str, default="cointegrated/rubert-tiny2", help="Model name from Hugging Face.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on ('cuda' or 'cpu').")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the trained model.")
    parser.add_argument("--wandb_project", type=str, default="dialog-mteb-pretrain", help="W&B project name. If not provided, W&B is disabled.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity name.")
    parser.add_argument("--datapath", type=str, default="./", help="Path to data directory.")
    parser.add_argument("--dataname", type=str, default="train.txt", help="Name of data file in datapath.")
    parser.add_argument("--loader_type", type=str, default="txt", choices=["txt", "csv"], help="Which data loader to use.")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for HardConLoss.")
    parser.add_argument("--contrast_type", type=str, default="HardNeg", choices=["Orig", "HardNeg"], help="Contrastive loss type.")
    parser.add_argument("--teacher_update_epochs", type=int, default=1, help="Frequency (in epochs) to update the teacher model.")
    parser.add_argument("--distill_weight", type=float, default=1.0, help="Weight for the distillation loss.")

    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    main(args)
