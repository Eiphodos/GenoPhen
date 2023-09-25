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
from configs.config import build_config
from data.build_dataloader import build_ft_dataloaders
from data.preprocess_data import preprocess_data
from models.tokenizer import build_tokenizer
from models.modeling import build_model


def main(args):
    ### Build config ###
    cfg = build_config(finetuning_config=args.ft_config, model_config=args.model_config)

    ### Setup distributed ###
    dist_misc.init_distributed_mode(args)

    ### Setup logging ###
    if dist_misc.is_main_process():
        if cfg['log_dir'] is not None:
            os.makedirs(cfg['log_dir'], exist_ok=True)
            #log_writer = SummaryWriter(logdir=cfg['log_dir'])
        if args.wandb_logging:
            wandb.login()
            wandb.init(project="Finetuning", entity="eiphodos", config=cfg, dir=cfg['log_dir'])


    ### Setup torch ###
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Setup Determinism ###
    seed = cfg['seed'] + dist_misc.get_rank()
    transformers.set_seed(seed)

    ### Preprocess data ###
    dataframe = preprocess_data(cfg)
    dataframe.to_csv(os.path.join(cfg['log_dir'], 'pp_data.tsv'), sep='\t')

    ### Build tokenizer ###
    tokenizer = build_tokenizer(cfg, dataframe)
    print(tokenizer)
    print("Built tokenizer successfully...")


    ### Build dataloaders ###
    train_dataloader, val_dataloader = build_ft_dataloaders(cfg, dataframe, tokenizer)
    print("Build train dataloader with {} items and val dataloader with {} items".format(len(train_dataloader),
                                                                                         len(val_dataloader)))


    ### Build model ###
    model = build_model(cfg, tokenizer)
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    ### Build Optimizer ###
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg['optimizer']['lr'],
                                  weight_decay=cfg['optimizer']['weight_decay'])
    n_training_steps = cfg['training']['n_epochs'] * len(train_dataloader)
    scheduler = transformers.get_scheduler('linear',
                                           optimizer=optimizer,
                                           num_warmup_steps=0,
                                           num_training_steps=n_training_steps)

    ### Setup metrics ###
    val_acc = MulticlassAccuracy(device=device)
    train_loss = Mean(device=device)
    val_loss = Mean(device=device)

    ### Check if resuming ###
    # TODO: Implement resuming
    '''
    if cfg.resume:
        resume_training(cfg, model_without_ddp, optimizer, scheduler)
    '''

    print("Starting training...")

    for e in range(cfg['training']['n_epochs']):
        model.train()
        for j, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss.update(loss.detach())
        te_loss = sync_and_compute(train_loss) if args.distributed else train_loss.compute()
        wandb.log({'Training epoch avg loss': te_loss, "Epoch": e})
        print("Mean training loss for epoch {}/{} is: {}".format(e, cfg['training']['n_epochs'], te_loss))
        train_loss.reset()
        if ((e + 1) % cfg['training']['val_every_n_epochs']) == 0:
            if dist_misc.is_main_process():
                model.eval()
                with torch.no_grad():
                    for i, batch in enumerate(val_dataloader):
                        batch = {k: v.to(device) for k, v in batch.items()}
                        outputs = model(**batch)

                        loss = outputs.loss
                        val_loss.update(loss)
                        predictions = torch.argmax(outputs.logits, dim=-1)
                        pp_predictions, pp_labels = preprocess_mlm_acc(predictions, batch['labels'])
                        val_acc.update(pp_predictions, pp_labels)

                    ve_loss = val_loss.compute()
                    ve_acc = val_acc.compute()
                wandb.log({'Validation epoch loss': ve_loss,
                           "Val epoch accuracy": ve_acc, "Epoch": e})
                print("Mean validation loss for epoch {}/{} is: {}".format(e, cfg['training']['n_epochs'], ve_loss))
                print("Mean validation accuracy for epoch {}/{} is: {}".format(e, cfg['training']['n_epochs'], ve_acc))
                val_loss.reset()
                val_acc.reset()
            if args.distributed:
                dist.barrier()

    if dist_misc.is_main_process():
        model.save_pretrained(cfg['log_dir'])
        tokenizer.save_pretrained(cfg['log_dir'])

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
