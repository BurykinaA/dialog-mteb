import os
import random
import torch
import numpy as np
# from tensorboardX import SummaryWriter # Removed TensorBoard
import wandb # Added wandb

def set_global_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def setup_path(args):
    resPath = args.mode
    resPath += f'.{args.contrast_type}'
    resPath += f'.epoch{args.epochs}'
    resPath += f'.{args.bert}'
    resPath += f'.{args.dataname}'
    resPath += f'.lr{args.lr}'
    resPath += f'.lrscale{args.lr_scale}'
    resPath += f'.bs{args.batch_size}'
    resPath += f'.tmp{args.temperature}'
    resPath += f'.decay{args.decay_rate}'
    resPath += f'.seed{args.seed}'
    resPath += f'.turn{args.num_turn}/'
    resPath = args.resdir + resPath
    print(f'results path: {resPath}')

    # tensorboard = SummaryWriter(resPath) # Removed TensorBoard
    return resPath # Only return resPath


def statistics_log(losses=None, global_step=0): # Removed tensorboard argument
    # print("[{}]-----".format(global_step))
    if losses is not None:
        log_dict = {}
        for key, val in losses.items():
            try:
                # Ensure value is a scalar float or int for wandb
                if hasattr(val, 'item'):
                    log_dict['train/'+key] = val.item()
                elif isinstance(val, (int, float)):
                    log_dict['train/'+key] = val
                else:
                    # Optionally handle or skip non-scalar/non-numeric types
                    print(f"Warning: Skipping logging for key '{key}' due to non-scalar type: {type(val)}")
                    continue 
            except Exception as e:
                print(f"Error processing loss item {key}: {val}, {e}")
                continue
            # print("{}:\t {:.3f}".format(key, val))
        
        if log_dict: # Log only if there are valid metrics
            wandb.log(log_dict, step=global_step)

            