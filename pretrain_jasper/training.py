import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer

from model import PSCBert
from dataloader import get_dataloader


def setup_distributed(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def train_worker(rank, world_size, args):
    """Main training function for each GPU worker."""
    setup_distributed(rank, world_size)
    
    # Only initialize wandb on the main process
    if rank == 0 and args.wandb_project:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)

    device = torch.device(f"cuda:{rank}")

    model_path = args.load_from_checkpoint if args.load_from_checkpoint else args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    special_tokens = ['[USR]', '[SYS]']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    # Get dataloader with distributed sampler
    train_dataloader = get_dataloader(
        args.train_data_path, 
        tokenizer, 
        args.batch_size, 
        args.max_len,
        distributed=True,
        world_size=world_size,
        rank=rank
    )

    model = PSCBert(
        model_path, 
        num_special_tokens=len(special_tokens),
        cosine_loss_weight=args.cosine_loss_weight,
        similarity_loss_weight=args.similarity_loss_weight,
        contrastive_loss_weight=args.contrastive_loss_weight,
        contrastive_margin=args.contrastive_margin
    )
    model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    for epoch in range(args.num_epochs):
        # Set epoch for distributed sampler
        train_dataloader.sampler.set_epoch(epoch)
        
        model.train()
        total_loss, total_mlm_loss, total_dist_loss = 0, 0, 0
        total_cosine_loss, total_similarity_loss, total_contrastive_loss = 0, 0, 0
        
        # Only show progress bar on main process
        if rank == 0:
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", leave=False)
        else:
            progress_bar = train_dataloader
            
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
            total_cosine_loss += outputs['cosine_loss'].item()
            total_similarity_loss += outputs['similarity_loss'].item()
            total_contrastive_loss += outputs['contrastive_loss'].item()

            # Only log to wandb on main process
            if rank == 0 and args.wandb_project:
                wandb.log({
                    "train_loss_step": loss.item(),
                    "mlm_loss_step": outputs['mlm_loss'].item(),
                    "distillation_loss_step": outputs['distillation_loss'].item(),
                    "cosine_loss_step": outputs['cosine_loss'].item(),
                    "similarity_loss_step": outputs['similarity_loss'].item(),
                    "contrastive_loss_step": outputs['contrastive_loss'].item(),
                    "lr": scheduler.get_last_lr()[0]
                })
                
            # Only update progress bar on main process
            if rank == 0:
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'dist': outputs['distillation_loss'].item(),
                    'cos': outputs['cosine_loss'].item(),
                    'sim': outputs['similarity_loss'].item(),
                    'cont': outputs['contrastive_loss'].item()
                })

        # Synchronize losses across all processes
        dist.all_reduce(torch.tensor(total_loss).cuda())
        dist.all_reduce(torch.tensor(total_mlm_loss).cuda())
        dist.all_reduce(torch.tensor(total_dist_loss).cuda())
        dist.all_reduce(torch.tensor(total_cosine_loss).cuda())
        dist.all_reduce(torch.tensor(total_similarity_loss).cuda())
        dist.all_reduce(torch.tensor(total_contrastive_loss).cuda())
        
        avg_loss = total_loss / (len(train_dataloader) * world_size)
        avg_mlm_loss = total_mlm_loss / (len(train_dataloader) * world_size)
        avg_dist_loss = total_dist_loss / (len(train_dataloader) * world_size)
        avg_cosine_loss = total_cosine_loss / (len(train_dataloader) * world_size)
        avg_similarity_loss = total_similarity_loss / (len(train_dataloader) * world_size)
        avg_contrastive_loss = total_contrastive_loss / (len(train_dataloader) * world_size)

        # Only log epoch metrics on main process
        if rank == 0:
            if args.wandb_project:
                wandb.log({
                    "train_loss_epoch": avg_loss,
                    "mlm_loss_epoch": avg_mlm_loss,
                    "distillation_loss_epoch": avg_dist_loss,
                    "cosine_loss_epoch": avg_cosine_loss,
                    "similarity_loss_epoch": avg_similarity_loss,
                    "contrastive_loss_epoch": avg_contrastive_loss,
                    "epoch": epoch
                })
            print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Dist Loss = {avg_dist_loss:.4f}, Cosine Loss = {avg_cosine_loss:.4f}, Sim Loss = {avg_similarity_loss:.4f}, Contrastive Loss = {avg_contrastive_loss:.4f}")

        # Update teacher model on all processes, but only print on main process
        if (epoch + 1) % args.teacher_update_every == 0:
            if rank == 0:
                print(f"\nUpdating teacher model at end of epoch {epoch+1}")
            model.module.update_teacher()

        # Only save on main process
        if rank == 0 and (epoch + 1) % args.save_every == 0:
            output_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model.module.student.bert.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Saved model checkpoint to {output_dir}")

    if rank == 0 and args.wandb_project:
        wandb.finish()
        
    cleanup_distributed()


def main(args):
    """Main function to launch distributed training."""
    world_size = args.world_size
    
    if world_size > 1:
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        # Single GPU training (fallback)
        train_worker(0, 1, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain a sentence transformer model with self-distillation.")
    parser.add_argument("--train_data_path", type=str, default="pretrain_futuretod/processed_dialogues.txt", help="Path to the training data.")
    parser.add_argument("--output_dir", type=str, default="./jasper_model_checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model name or path.")
    parser.add_argument("--load_from_checkpoint", type=str, default=None, help="Path to a checkpoint to load model and tokenizer from.")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size per GPU for training.")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps.")
    parser.add_argument("--save_every", type=int, default=5, help="Save model every N epochs.")
    parser.add_argument("--teacher_update_every", type=int, default=10, help="Update teacher model every N epochs.")
    parser.add_argument("--cosine_loss_weight", type=float, default=10.0, help="Weight for cosine similarity loss.")
    parser.add_argument("--similarity_loss_weight", type=float, default=200.0, help="Weight for similarity loss.")
    parser.add_argument("--contrastive_loss_weight", type=float, default=20.0, help="Weight for contrastive loss.")
    parser.add_argument("--contrastive_margin", type=float, default=0.5, help="Margin for contrastive loss.")
    parser.add_argument("--world_size", type=int, default=5, help="Number of GPUs to use for distributed training.")
    parser.add_argument("--wandb_project", type=str, default="dialog-mteb-pretrain", help="W&B project name. If not provided, W&B is disabled.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity name.")

    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    main(args)
