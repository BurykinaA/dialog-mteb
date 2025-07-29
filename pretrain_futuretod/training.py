import argparse
import os
import torch
import wandb
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer

from model import PSCBert
from dataloader import get_dataloader


def main(args):
    if args.wandb_project:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_path = args.load_from_checkpoint if args.load_from_checkpoint else args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    special_tokens = ['[USR]', '[SYS]'] #CLS
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    train_dataloader = get_dataloader(args.train_data_path, tokenizer, args.batch_size, args.max_len)

    model = PSCBert(model_path, num_special_tokens=len(special_tokens))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    for epoch in range(args.num_epochs):
        model.train()
        total_loss, total_mlm_loss, total_dist_loss = 0, 0, 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs['loss']

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_mlm_loss += outputs['mlm_loss'].item()
            total_dist_loss += outputs['distillation_loss'].item()
            if args.wandb_project:
                wandb.log({ "train_loss_step": loss.item(), "mlm_loss_step": outputs['mlm_loss'].item(), "distillation_loss_step": outputs['distillation_loss'].item(), "lr": scheduler.get_last_lr()[0] })
            progress_bar.set_postfix({
                'loss': loss.item(),
                'mlm': outputs['mlm_loss'].item(),
                'dist': outputs['distillation_loss'].item()
            })

        avg_loss = total_loss / len(train_dataloader)
        avg_mlm_loss = total_mlm_loss / len(train_dataloader)
        avg_dist_loss = total_dist_loss / len(train_dataloader)
        if args.wandb_project:
            wandb.log({ "train_loss_epoch": avg_loss, "mlm_loss_epoch": avg_mlm_loss, "distillation_loss_epoch": avg_dist_loss, "epoch": epoch })
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, MLM Loss = {avg_mlm_loss:.4f}, Distillation Loss = {avg_dist_loss:.4f}")

        if (epoch + 1) % args.teacher_update_every == 0:
            print(f"\nUpdating teacher model at end of epoch {epoch+1}")
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.update_teacher()

        if (epoch + 1) % args.save_every == 0:
            output_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.student.bert.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Saved model checkpoint to {output_dir}")

    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain a sentence transformer model with self-distillation.")
    parser.add_argument("--train_data_path", type=str, default="pretrain_futuretod/processed_dialogues.txt", help="Path to the training data.")
    parser.add_argument("--output_dir", type=str, default="./saved_model", help="Directory to save model checkpoints.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model name or path.")
    parser.add_argument("--load_from_checkpoint", type=str, default='./short_futuretod_2/checkpoint-epoch-15', help="Path to a checkpoint to load model and tokenizer from.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps.")
    parser.add_argument("--save_every", type=int, default=5, help="Save model every N epochs.")
    parser.add_argument("--teacher_update_every", type=int, default=10, help="Update teacher model every N epochs.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on ('cuda' or 'cpu').")
    parser.add_argument("--wandb_project", type=str, default="dialog-mteb-pretrain", help="W&B project name. If not provided, W&B is disabled.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity name.")

    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    main(args)
