import pandas as pd
import numpy as np
import os
import re
from data.hierarchy import build_hierarchy_data

def preprocess_data(cfg):
    if cfg['data']['load_preprocessed_data']:
        df = pd.read_csv(cfg['data']['preprocessed_data_file_path'], sep='\t', low_memory=False, index_col=0)
        print("Loaded dataframe with {} rows and {} columns".format(len(df), df.columns))
    else:
        multispecies = len(cfg['data']['species']) > 1
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

        if "AMR_genotypes_core" in columns:
            df = df[~df["AMR_genotypes_core"].isnull()]
            print("NULL GENO: Dataframe with {} rows".format(len(df)))

        if "target_creation_date" in columns:
            df = df[~df["target_creation_date"].isnull()]
            print("NULL DATE: Dataframe with {} rows".format(len(df)))

        if cfg['data']['hierarchy']['use_hierarchy_data'] and "AMR_genotypes_core" in columns:
            df = build_hierarchy_data(cfg, df)
            columns.append("Hierarchy_data")
            print(df["Hierarchy_data"].iloc[0:5])

        if "AST_phenotypes" in columns:
            df = clean_pheno_data(cfg, df)
            print("CLEAN PHENO: Dataframe with {} rows".format(len(df)))
        if "AMR_genotypes_core" in columns:
            df = clean_geno_data(cfg, df)
            print("CLEAN GENO: Dataframe with {} rows".format(len(df)))
        if "target_creation_date" in columns:
            df = clean_date_data(cfg, df)
            print("CLEAN DATE: Dataframe with {} rows".format(len(df)))

        print("Example rows before filters: \n")
        for c in columns:
            print(df[c])

        if "geo_loc_name" in columns:
            df['geo_loc_name'] = df['geo_loc_name'].map(lambda x: x.split(":")[0] if type(x) is str else x) # Remove suffixes from location

        if "AMR_genotypes_core" in columns and cfg['data']['filter']['point']:
            df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(
                lambda x: ','.join([a for a in x.split(',') if '=POINT' not in a]))
            print("FILTER POINT: Dataframe with {} rows".format(len(df)))
        if "AMR_genotypes_core" in columns and cfg['data']['filter']['partial']:
            df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(
                lambda x: ','.join([a for a in x.split(',') if '=PARTIAL' not in a]))
            print("FILTER PARTIAL: Dataframe with {} rows".format(len(df)))
        if "AMR_genotypes_core" in columns and cfg['data']['filter']['partial_contig']:
            df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(
                lambda x: ','.join([a for a in x.split(',') if '=PARTIAL_END_OF_CONTIG' not in a]))
            print("FILTER PARTIAL_CONTIG: Dataframe with {} rows".format(len(df)))
        if "AMR_genotypes_core" in columns and cfg['data']['filter']['mistranslation']:
            df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(
                lambda x: ','.join([a for a in x.split(',') if '=MISTRANSLATION' not in a]))
            print("FILTER MISTRANSLATION: Dataframe with {} rows".format(len(df)))
        if "AMR_genotypes_core" in columns and cfg['data']['filter']['hmm']:
            df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(
                lambda x: ','.join([a for a in x.split(',') if '=HMM' not in a]))
            print("FILTER HMM: Dataframe with {} rows".format(len(df)))
        if "AMR_genotypes_core" in columns and cfg['data']['filter']['bla']:
            df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(
                lambda x: ','.join([remove_last_digits(a) if a[0:3] == 'bla' else a for a in x.split(',')]))
            print("FILTER BLA: Dataframe with {} rows".format(len(df)))
        if "AMR_genotypes_core" in columns and cfg['data']['filter']['aph_aac']:
            df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(filter_aap_aac)
            print("FILTER APH_AAC: Dataframe with {} rows".format(len(df)))
        if "AMR_genotypes_core" in columns and cfg['data']['filter']['min_geno'] > 0:
            df = df[df['AMR_genotypes_core'].apply(filter_len, min_length=cfg['data']['filter']['min_geno'])]
            print("FILTER GENO LEN: Dataframe with {} rows".format(len(df)))
        if "AST_phenotypes" in columns and cfg['data']['filter']['min_ab'] > 0:
            df = df[df['AST_phenotypes'].apply(filter_len, min_length=cfg['data']['filter']['min_ab'])]
            print("FILTER AB LEN: Dataframe with {} rows".format(len(df)))

        print("FINAL: Dataframe with {} rows and {} columns".format(len(df), df.columns))
        print("Example rows after filters: \n")
        for c in columns:
            print(df[c])

    if "AMR_genotypes_core" in cfg['data']['columns']:
        new_series = df['AMR_genotypes_core'].map(lambda x: len(x.split(',')))
        max_gene_len = new_series.max()
        print("Maximum number of genes in a isolate: {}".format(max_gene_len))
        cfg['data']['max_n_genes'] = max_gene_len
    return df


def filter_aap_aac(x):
    interesting_genes = ["aph", "aac"]
    roman_numbers = ["I", "V", "X"]

    all_items = x.split(',')
    all_items = [
        remove_last_non_digit(a) if a[0:3] in interesting_genes and any(r in a.split("-")[-1] for r in roman_numbers) else a
        for a in all_items]
    return ",".join(all_items)


def filter_doubles(x):
    amr_words = x.split(',')
    am_words = [i[:-1] for i in amr_words]
    all_items = [amr_words[i] for i in range(len(amr_words)) if am_words.count(am_words[i]) < 2]
    if len(all_items) < len(amr_words):
        print("Found doubles! \n Before filtering: {} \n After filtering: {}".format(amr_words, all_items))
    return ",".join(all_items)


def filter_non_rs(x):
    if len(x) == 0:
        return x
    try:
        amr_words = x.split(',')
        all_items = [w for w in amr_words if w[-1].lower() == "r" or w[-1].lower() == "s"]
        return ",".join(all_items)
    except IndexError as ie:
        print("Failed with word: {}".format(x))


def filter_len(x, min_length):
    if x == '':
        return False
    else:
        return x.count(',') + 1 >= min_length


def remove_last_digits(x):
    return re.sub(r"[\d-]+$", "", x)


def remove_last_digit(x):
    return re.sub(r"\d$", "", x)


def remove_last_non_digit(x):
    return re.sub(r"[^0-9IVX]$", "", x)


def remove_duplicates(x):
    all_items = pd.DataFrame([a.split('=') for a in x.split(',')]) # Splits sequence of AB resistances and creates dataframe
    all_items.drop_duplicates(subset=[0], inplace=True, keep=False) # Deletes duplicates based on AB name
    return ",".join(["=".join(a) for a in all_items.values.tolist()]) # Joins into list again


def convert_ab_to_abbrev(x, cfg):
    all_items = x.split(',')
    all_items = ["=".join([cfg['antibiotics'][a.split('=')[0]], a.split('=')[1]]) for a in all_items if a.split('=')[0] in cfg['antibiotics'].keys()]
    return ",".join(all_items)


def clean_pheno_data(cfg, df):
    df['AST_phenotypes'] = df['AST_phenotypes'].map(lambda x: ','.join([a.lower() for a in x.split(',') if a[-1] == "R" or a[-1] == "S"])) #Remove all resistances that are not R or S
    if cfg['data']['filter']['doubles']:
        df['AST_phenotypes'] = df['AST_phenotypes'].map(remove_duplicates)
    df['AST_phenotypes'] = df['AST_phenotypes'].apply(convert_ab_to_abbrev, cfg=cfg)
    return df


def clean_geno_data(cfg, df):
    df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(lambda x: x.replace("''", "b")) # Replace biss with b
    df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(lambda x: x.replace("'", "p")) # Replace prime with p
    return df


def clean_date_data(cfg, df):
    df['target_creation_date'] = df['target_creation_date'].map(lambda x: x[0:7]) # only keep YYYY-MM of the incoming YYYY-MM-DD
    return df


def clean_hierarchy_data(cfg, df):
    df['Hierarchy_data'] = df['Hierarchy_data'].map(lambda x: x.replace("''", "b")) # Replace biss with b
    df['Hierarchy_data'] = df['Hierarchy_data'].map(lambda x: x.replace("'", "p")) # Replace prime with p
    return df