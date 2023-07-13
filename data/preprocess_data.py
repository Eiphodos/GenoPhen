import pandas as pd
import numpy as np
import os
import re

def preprocess_data(cfg):
    if len(cfg['data']['species']) > 1:
        multispecies = True
    df_list = []
    for species in cfg['data']['species']:
        temp_df = pd.read_csv(cfg['species'][species]["file_path"], sep='\t', low_memory=False )
        temp_df['species_name'] = species
        df_list.append(temp_df)
    df = pd.concat(df_list, ignore_index=True)
    columns = cfg['data']['columns']
    if multispecies:
        columns.append("species_name")
    df = df[columns]

    print("CONCAT: Dataframe with {} rows and {} columns".format(len(df), df.columns))

    if "AST_phenotypes" in columns:
        df = df[~df["AST_phenotypes"].isnull()]
        print("NULL PHENO: Dataframe with {} rows".format(len(df)))
        df = clean_pheno_data(cfg, df)
        print("CLEAN PHENO: Dataframe with {} rows".format(len(df)))
    if "AMR_genotypes_core" in columns:
        df = df[~df["AMR_genotypes_core"].isnull()]
        print("NULL GENO: Dataframe with {} rows".format(len(df)))
        df = clean_geno_data(cfg, df)
        print("CLEAN GENO: Dataframe with {} rows".format(len(df)))

    print("Example rows before filters: \n")
    for c in columns:
        print(df[c])

    if "geo_loc_name" in columns:
        df['geo_loc_name'] = df['geo_loc_name'].map(lambda x: x.split(":")[0] if type(x) is str else x) # Remove suffixes from location

    if cfg['data']['filter']['point']:
        df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(
            lambda x: ','.join([a for a in x.split(',') if '=POINT' not in a]))
        print("FILTER POINT: Dataframe with {} rows".format(len(df)))
    if cfg['data']['filter']['partial']:
        df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(
            lambda x: ','.join([a for a in x.split(',') if '=PARTIAL' not in a]))
        print("FILTER PARTIAL: Dataframe with {} rows".format(len(df)))
    if cfg['data']['filter']['partial_contig']:
        df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(
            lambda x: ','.join([a for a in x.split(',') if '=PARTIAL_END_OF_CONTIG' not in a]))
        print("FILTER PARTIAL_CONTIG: Dataframe with {} rows".format(len(df)))
    if cfg['data']['filter']['mistranslation']:
        df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(
            lambda x: ','.join([a for a in x.split(',') if '=MISTRANSLATION' not in a]))
        print("FILTER MISTRANSLATION: Dataframe with {} rows".format(len(df)))
    if cfg['data']['filter']['hmm']:
        df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(
            lambda x: ','.join([a for a in x.split(',') if '=HMM' not in a]))
        print("FILTER HMM: Dataframe with {} rows".format(len(df)))
    if cfg['data']['filter']['bla']:
        df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(
            lambda x: ','.join([remove_last_digits(a) if a[0:3] == 'bla' else a for a in x.split(',')]))
        print("FILTER BLA: Dataframe with {} rows".format(len(df)))
    if cfg['data']['filter']['aph_aac']:
        df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(filter_aap_aac)
        print("FILTER APH_AAC: Dataframe with {} rows".format(len(df)))

    if "AMR_genotypes_core" in columns and cfg['data']['filter']['min_geno'] > 0:
        df = df[df['AMR_genotypes_core'].apply(filter_len, min_length=cfg['data']['filter']['min_geno'])]
        print("FILTER GENO_LEN: Dataframe with {} rows".format(len(df)))
    if "AST_phenotypes" in columns and cfg['data']['filter']['min_pheno'] > 0:
        df = df[df['AST_phenotypes'].apply(filter_len, min_length=cfg['data']['filter']['min_pheno'])]
        print("FILTER PHENO_LEN: Dataframe with {} rows".format(len(df)))

    print("FINAL: Dataframe with {} rows and {} columns".format(len(df), df.columns))
    print("Example rows after filters: \n")
    for c in columns:
        print(df[c])
    return df


def filter_aap_aac(x):
    interesting_genes = ["aph", "aac"]
    roman_numbers = ["I", "V", "X"]

    all_items = x.split(',')
    all_items = [
        remove_last_digit(a) if a[0:3] in interesting_genes and any(r in a.split("-")[-1] for r in roman_numbers) else a
        for a in all_items]
    return ",".join(all_items)


def filter_len(x, min_length):
    if x == '':
        return False
    else:
        return x.count(',') + 1 >= min_length


def remove_last_digits(x):
    return re.sub(r"\d+$", "", x)


def remove_last_digit(x):
    return re.sub(r"\d$", "", x)


def remove_duplicates(x):
    all_items = pd.DataFrame([a.split('=') for a in x.split(',')]) # Splits sequence of AB resistances and creates dataframe
    all_items.drop_duplicates(subset=[0], inplace=True, keep='last') # Deletes duplicates based on AB name
    return ",".join(["=".join(a) for a in all_items.values.tolist()]) # Joins into list again

def convert_ab_to_abbrev(x, cfg):
    all_items = x.split(',')
    all_items = ["=".join([cfg['antibiotics'][a.split('=')[0]], a.split('=')[1]]) for a in all_items if a.split('=')[0] in cfg['antibiotics'].keys()]
    return ",".join(all_items)

def clean_pheno_data(cfg, df):
    df['AST_phenotypes'] = df['AST_phenotypes'].map(
        lambda x: ','.join([a.lower() for a in x.split(',') if '=ND' not in a])) #Remove ND resistance and lowercase everything
    df['AST_phenotypes'] = df['AST_phenotypes'].map(remove_duplicates)
    df['AST_phenotypes'] = df['AST_phenotypes'].apply(convert_ab_to_abbrev, cfg=cfg)
    return df

def clean_geno_data(cfg, df):
    df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(lambda x: x.replace("''", "b")) # Replace biss with b
    df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(lambda x: x.replace("'", "p")) # Replace prime with p
    return df