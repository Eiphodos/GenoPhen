from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from data.hierarchical_datacollators import HierDataCollatorForLanguageModeling, HierDataCollatorWithPadding, HierSumDataCollatorForLanguageModeling, HierSumDataCollatorWithPadding
from torch.utils.data import DataLoader, SequentialSampler
from data.utils import cv_split_dataframe, combinatorial_data_generator, weights_separated_by_label
from data.datasets import GenoPTDataset, GenoPhenoFTDataset_legacy
import os


def build_pt_dataloaders(cfg, dataframe, tokenizer):
    train_dataframe, val_dataframe = cv_split_dataframe(cfg, dataframe)

    # Collator assumes data has been tokenized already, uses tokenizer to add padding to max batch len
    if cfg['data']['hierarchy']['use_hierarchy_data']:
        train_dataset = GenoPTDataset(train_dataframe, tokenizer,
                                      hierarchy_variant=cfg['data']['hierarchy']['hierarchy_variant'])
        val_dataset = GenoPTDataset(val_dataframe, tokenizer,
                                    hierarchy_variant=cfg['data']['hierarchy']['hierarchy_variant'])
        if cfg['data']['hierarchy']['hierarchy_variant'] == 'summed':
            data_collator = HierSumDataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=True, mlm_probability=cfg['training']['mlm_probability'])
        elif cfg['data']['hierarchy']['hierarchy_variant'] == 'separate':
            data_collator = HierDataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=True, mlm_probability=cfg['training']['mlm_probability'])
        else:
            raise NotImplementedError('Hierarchy variant {} is not implemented!'.format(cfg['data']['hierarchy']['hierarchy_variant']))
    else:
        train_dataset = GenoPTDataset(train_dataframe, tokenizer)
        val_dataset = GenoPTDataset(val_dataframe, tokenizer)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=cfg['training']['mlm_probability']
        )

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['data']['train_batch_size'], collate_fn=data_collator,
                                  num_workers=cfg['data']['train_n_workers'], pin_memory=cfg['data']['pin_memory'])
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['data']['val_batch_size'], collate_fn=data_collator,
                                num_workers=cfg['data']['val_n_workers'], pin_memory=cfg['data']['pin_memory'])

    return train_dataloader, val_dataloader


def build_ft_legacy_dataloaders(cfg, dataframe, tokenizer_geno, tokenizer_pheno):
    train_dataframe, val_dataframe = cv_split_dataframe(cfg, dataframe)
    print("Training set size after split: {}".format(len(train_dataframe)))
    print("Validation set size after split: {}".format(len(val_dataframe)))

    comb_train_dataframe = combinatorial_data_generator(cfg, train_dataframe)
    comb_val_dataframe = combinatorial_data_generator(cfg, val_dataframe)
    print("Training set size after combinatorics: {}".format(len(comb_train_dataframe)))
    print("Validation set size after combinatorics: {}".format(len(comb_val_dataframe)))

    weights_train, weights_s_train, weights_r_train, res_ratio_train = weights_separated_by_label(cfg, comb_train_dataframe)
    weights_val, weights_s_val, weights_r_val, res_ratio_val = weights_separated_by_label(cfg, comb_val_dataframe)
    cfg['antibiotics']['train_ab_weights'] = {}
    cfg['antibiotics']['val_ab_weights'] = {}
    cfg['antibiotics']['res_ratio_train'] = res_ratio_train
    cfg['antibiotics']['res_ratio_val'] = res_ratio_val
    cfg['antibiotics']['train_ab_weights']['all'] = weights_train.tolist()
    cfg['antibiotics']['train_ab_weights']['weights_s'] = weights_s_train.tolist()
    cfg['antibiotics']['train_ab_weights']['weights_r'] = weights_r_train.tolist()
    cfg['antibiotics']['val_ab_weights']['all'] = weights_val.tolist()
    cfg['antibiotics']['val_ab_weights']['weights_s'] = weights_s_val.tolist()
    cfg['antibiotics']['val_ab_weights']['weights_r'] = weights_r_val.tolist()

    comb_train_dataframe.to_csv(os.path.join(cfg['log_dir'], 'comb_train_data.tsv'), sep='\t')
    comb_train_dataframe.to_csv(os.path.join(cfg['log_dir'], 'comb_val_data.tsv'), sep='\t')

    if cfg['data']['hierarchy']['use_hierarchy_data']:
        train_dataset = GenoPhenoFTDataset_legacy(cfg, comb_train_dataframe, tokenizer_geno, tokenizer_pheno,
                                                  hierarchy_variant=cfg['data']['hierarchy']['hierarchy_variant'])
        val_dataset = GenoPhenoFTDataset_legacy(cfg, comb_val_dataframe, tokenizer_geno, tokenizer_pheno,
                                                hierarchy_variant=cfg['data']['hierarchy']['hierarchy_variant'])
        if cfg['data']['hierarchy']['hierarchy_variant'] == 'summed':
            data_collator = HierSumDataCollatorWithPadding(tokenizer=tokenizer_geno)
        elif cfg['data']['hierarchy']['hierarchy_variant'] == 'separate':
            data_collator = HierDataCollatorWithPadding(tokenizer=tokenizer_geno)
    else:
        train_dataset = GenoPhenoFTDataset_legacy(cfg, comb_train_dataframe, tokenizer_geno, tokenizer_pheno)
        val_dataset = GenoPhenoFTDataset_legacy(cfg, comb_val_dataframe, tokenizer_geno, tokenizer_pheno)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer_geno)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['data']['train_batch_size'], collate_fn=data_collator,
                                  num_workers=cfg['data']['train_n_workers'], pin_memory=cfg['data']['pin_memory'])
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['data']['val_batch_size'], collate_fn=data_collator,
                                num_workers=cfg['data']['val_n_workers'], pin_memory=cfg['data']['pin_memory'])

    return train_dataloader, val_dataloader