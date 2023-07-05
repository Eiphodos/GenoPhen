import os
import argparse
import wandb
import torch
import numpy as np
import transformers
from tensorboardX import SummaryWriter

import misc.distributed_misc as dist_misc
from configs.config import build_config
from data.build_dataloader import build_pt_dataloaders
from models.tokenizer import build_tokenizer
from models.modeling import build_model


def main(args):
    ### Build config ###
    cfg = build_config(args.config)

    ### Setup distributed ###
    dist_misc.init_distributed_mode(args)

    ### Setup logging ###
    wandb.login()
    if dist_misc.is_main_process() and cfg.log_dir is not None:
        os.makedirs(cfg.log_dir, exist_ok=True)
        log_writer = SummaryWriter(logdir=cfg.log_dir)

    ### Setup torch ###
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Setup Determinism ###
    seed = cfg.seed + dist_misc.get_rank()
    transformers.set_seed(seed)

    ### Build tokenizer ###
    tokenizer = build_tokenizer(cfg)

    ### Build dataloaders ###
    val_dataloader, train_dataloader = build_pt_dataloaders(cfg, tokenizer)

    ### Build model ###
    model = build_model(cfg)
    model.to(device)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # distributed training parameters
    parser.add_argument("--distributed", type=bool, help="Enables distributed training")
    parser.add_argument('--world_size', default=1, type=int,  help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--dist_url', default='env://',  help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl',  help='Backend to use for distributed training')
    # Other settings
    parser.add_argument("--config", type=str, help="Name of config")
    args = parser.parse_args()
    main(args)