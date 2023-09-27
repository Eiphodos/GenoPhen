from transformers import DataCollatorForLanguageModeling, DataCollator
from torch.utils.data import DataLoader, SequentialSampler
from data.utils import cv_split_dataframe, combinatorial_data_generator, weights_separated_by_label
from data.datasets import GenoPhenoPTDataset, GenoPhenoFTDataset, GenoPhenoFTDataset_legacy
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


def build_ft_legacy_dataloaders(cfg, dataframe, tokenizer_geno, tokenizer_pheno):
    train_dataframe, val_dataframe = cv_split_dataframe(cfg, dataframe)

    comb_train_dataframe = combinatorial_data_generator(cfg, train_dataframe)
    comb_val_dataframe = combinatorial_data_generator(cfg, val_dataframe)

    weights_train, weights_s_train, weights_r_train, res_ratio_train = weights_separated_by_label(cfg, comb_train_dataframe)
    weights_val, weights_s_val, weights_r_val, res_ratio_val = weights_separated_by_label(cfg, comb_val_dataframe)
    cfg['antibiotics']['train_ab_weights'] = {}
    cfg['antibiotics']['val_ab_weights'] = {}
    cfg['antibiotics']['res_ratio_train'] = res_ratio_train
    cfg['antibiotics']['res_ratio_val'] = res_ratio_val
    cfg['antibiotics']['train_ab_weights']['all'] = weights_train
    cfg['antibiotics']['train_ab_weights']['weights_s'] = weights_s_train
    cfg['antibiotics']['train_ab_weights']['weights_r'] = weights_r_train
    cfg['antibiotics']['val_ab_weights']['all'] = weights_val
    cfg['antibiotics']['val_ab_weights']['weights_s'] = weights_s_val
    cfg['antibiotics']['val_ab_weights']['weights_r'] = weights_r_val

    comb_train_dataframe.to_csv(os.path.join(cfg['log_dir'], 'comb_train_data.tsv'), sep='\t')
    comb_train_dataframe.to_csv(os.path.join(cfg['log_dir'], 'comb_val_data.tsv'), sep='\t')

    train_dataset = GenoPhenoFTDataset_legacy(cfg, comb_train_dataframe, tokenizer_geno, tokenizer_pheno)
    val_dataset = GenoPhenoFTDataset_legacy(cfg, comb_val_dataframe, tokenizer_geno, tokenizer_pheno)

    data_collator = DataCollator(tokenizer=tokenizer_geno)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['data']['train_batch_size'], collate_fn=data_collator)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['data']['val_batch_size'], collate_fn=data_collator)

    return train_dataloader, val_dataloader