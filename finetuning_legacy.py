import os
import argparse
import wandb
import torch
import torch.distributed as dist
import transformers
from torcheval.metrics import Mean, MulticlassAccuracy
from torcheval.metrics.toolkit import sync_and_compute
#from tensorboardX import SummaryWriter

import misc.distributed_misc as dist_misc
#from misc.utils import resume_training
from misc.metrics import preprocess_mlm_acc
from configs.config import build_config, save_config
from data.build_dataloader import build_ft_legacy_dataloaders
from data.utils import compute_resistance_ratio_per_ab
from data.preprocess_data import preprocess_data
from models.tokenizer import build_tokenizer
from models.modeling import build_ft_legacy_model


def main(args):
    ### Build config ###
    cfg = build_config(finetuning_config=args.ft_config, model_config=args.model_config)

    ### Setup distributed ###
    dist_misc.init_distributed_mode(args)


    ### Setup logging ###
    if dist_misc.is_main_process():
        if args.wandb_logging:
            wandb.login()
            wandb.init(project="Finetuning", entity="eiphodos", config=cfg, dir=cfg['log_dir'])


    ### Setup torch ###
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg['device'] = device

    ### Setup Determinism ###
    seed = cfg['seed'] + dist_misc.get_rank()
    transformers.set_seed(seed)

    ### Preprocess data ###
    dataframe = preprocess_data(cfg)
    dataframe.to_csv(os.path.join(cfg['log_dir'], 'pp_data.tsv'), sep='\t')

    ### Build tokenizers ###
    tokenizer_geno = build_tokenizer(cfg, dataframe)
    print(tokenizer_geno)
    print("Built geno tokenizer successfully...")
    tokenizer_pheno = torch.load(cfg['tokenizer']['pheno']['pretrained_weights'])
    print("Built pheno tokenizer successfully...")
    cfg['vocab_len'] = len(tokenizer_pheno)

    ### Build dataloaders ###
    train_dataloader, val_dataloader = build_ft_legacy_dataloaders(cfg, dataframe, tokenizer_geno, tokenizer_pheno)
    print("Built train dataloader with {} items and val dataloader with {} items".format(len(train_dataloader),
                                                                                         len(val_dataloader)))


    ### Build model ###
    model = build_ft_legacy_model(cfg, tokenizer_geno, train_dataloader, val_dataloader)

    ### Save config ###
    save_config(cfg)

    print("Starting training...")
    model.train(cfg['training']['n_epochs'])

    if dist_misc.is_main_process():
        model.save_pretrained(cfg['log_dir'])
        tokenizer_geno.save_pretrained(cfg['log_dir'])
        tokenizer_pheno.save_pretrained(cfg['log_dir'])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # distributed training parameters
    parser.add_argument("--distributed", action='store_true', help="Enables distributed training")
    parser.add_argument('--world_size', default=1, type=int,  help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--dist_url', default='env://',  help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl',  help='Backend to use for distributed training')
    # Other settings
    parser.add_argument("--ft_config", type=str, help="Name of config")
    parser.add_argument("--model_config", type=str, help="Name of config")
    parser.add_argument('--no_wandb_logging', action='store_false', dest='wandb_logging', help='Disables wandb logging')
    parser.set_defaults(wandb_logging=True)
    args = parser.parse_args()
    main(args)
