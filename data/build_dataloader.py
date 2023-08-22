from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, SequentialSampler
from data.utils import cv_split_dataframe
from data.datasets import GenoPhenoPTDataSet, GenoPhenoFTDataset

def build_pt_dataloaders(cfg, dataframe, tokenizer):
    train_dataframe, val_dataframe = cv_split_dataframe(cfg, dataframe)

    train_dataset = GenoPhenoPTDataSet(train_dataframe, tokenizer)
    val_dataset = GenoPhenoPTDataSet(val_dataframe, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=cfg['training']['mlm_probability']
    )

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['data']['train_batch_size'], collate_fn=data_collator)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['data']['val_batch_size'], collate_fn=data_collator)

    return train_dataloader, val_dataloader


def build_ft_dataloaders(cfg, dataframe, tokenizer):
    train_dataframe, val_dataframe = cv_split_dataframe(cfg, dataframe)

    train_dataset = GenoPhenoFTDataSet(train_dataframe, tokenizer)
    val_dataset = GenoPhenoFTDataSet(val_dataframe, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=cfg['training']['mlm_probability']
    )

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['data']['train_batch_size'], collate_fn=data_collator)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['data']['val_batch_size'], collate_fn=data_collator)

    return train_dataloader, val_dataloader