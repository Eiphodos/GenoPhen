from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, SequentialSampler
from data.utils import cv_split_dataframe, combinatorial_data_generator
from data.datasets import GenoPhenoPTDataset, GenoPhenoFTDataset
import os

def build_pt_dataloaders(cfg, dataframe, tokenizer):
    train_dataframe, val_dataframe = cv_split_dataframe(cfg, dataframe)

    train_dataset = GenoPhenoPTDataset(train_dataframe, tokenizer)
    val_dataset = GenoPhenoPTDataset(val_dataframe, tokenizer)

    # Collator assumes data has been tokenized already, uses tokenizer to add padding to max batch len
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=cfg['training']['mlm_probability']
    )

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['data']['train_batch_size'], collate_fn=data_collator)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['data']['val_batch_size'], collate_fn=data_collator)

    return train_dataloader, val_dataloader


def build_ft_dataloaders(cfg, dataframe, tokenizer):
    train_dataframe, val_dataframe = cv_split_dataframe(cfg, dataframe)

    comb_train_dataframe = combinatorial_data_generator(cfg, train_dataframe)
    comb_val_dataframe = combinatorial_data_generator(cfg, val_dataframe)

    comb_train_dataframe.to_csv(os.path.join(cfg['log_dir'], 'comb_train_data.tsv'), sep='\t')
    comb_train_dataframe.to_csv(os.path.join(cfg['log_dir'], 'comb_val_data.tsv'), sep='\t')

    train_dataset = GenoPhenoFTDataset(cfg, comb_train_dataframe, tokenizer)
    val_dataset = GenoPhenoFTDataset(cfg, comb_val_dataframe, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=cfg['training']['mlm_probability']
    )

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['data']['train_batch_size'], collate_fn=data_collator)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['data']['val_batch_size'], collate_fn=data_collator)

    return train_dataloader, val_dataloader