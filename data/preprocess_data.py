import pandas as pd
import numpy as np
import os
from data.hierarchy import build_hierarchy_data
from data.preprocessing_utils import filter_len, remove_last_digits, clean_aap_aac, clean_pheno_data, clean_geno_data, clean_date_data

def preprocess_data(cfg):
    if cfg['data']['load_preprocessed_data']:
        df = pd.read_csv(cfg['data']['preprocessed_data_file_path'], sep='\t', low_memory=False, index_col=0)
        print("Loaded dataframe with {} rows and {} columns".format(len(df), df.columns))
        columns = df.columns
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

        print("Example rows before filters: \n")
        for c in columns:
            print(df[c])

        ### FILTERS ###
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
        if "AMR_genotypes_core" in columns and cfg['data']['filter']['min_geno'] > 0:
            df = df[df['AMR_genotypes_core'].apply(filter_len, min_length=cfg['data']['filter']['min_geno'])]
            print("FILTER GENO LEN: Dataframe with {} rows".format(len(df)))
        if "AST_phenotypes" in columns:
            df = clean_pheno_data(cfg, df)
            print("CLEAN PHENO: Dataframe with {} rows".format(len(df)))
        if "AST_phenotypes" in columns and cfg['data']['filter']['min_ab'] > 0:
            df = df[df['AST_phenotypes'].apply(filter_len, min_length=cfg['data']['filter']['min_ab'])]
            print("FILTER AB LEN: Dataframe with {} rows".format(len(df)))
        print("FILTERED: Dataframe with {} rows and {} columns".format(len(df), df.columns))

        ### BUILD HIERARCHY ###
        if cfg['data']['hierarchy']['use_hierarchy_data'] and "AMR_genotypes_core" in columns:
            df = build_hierarchy_data(cfg, df)
            columns.append("Hierarchy_data")
            print(df["Hierarchy_data"].iloc[0:5])

        ### CLEANING ###
        if "AMR_genotypes_core" in columns and cfg['data']['filter']['bla']:
            df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(
                lambda x: ','.join([remove_last_digits(a) if a[0:3] == 'bla' else a for a in x.split(',')]))
            print("FILTER BLA: Dataframe with {} rows".format(len(df)))
        if "AMR_genotypes_core" in columns and cfg['data']['filter']['aph_aac']:
            df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(clean_aap_aac)
            print("FILTER APH_AAC: Dataframe with {} rows".format(len(df)))
        if "AMR_genotypes_core" in columns:
            df = clean_geno_data(cfg, df)
            print("CLEAN GENO: Dataframe with {} rows".format(len(df)))
        if "geo_loc_name" in columns:
            df['geo_loc_name'] = df['geo_loc_name'].map(lambda x: x.split(":")[0] if type(x) is str else x) # Remove suffixes from location
        if "target_creation_date" in columns:
            df = clean_date_data(cfg, df)
            print("CLEAN DATE: Dataframe with {} rows".format(len(df)))

        print("FINAL: Dataframe with {} rows and {} columns".format(len(df), df.columns))
        print("Example rows after filters: \n")
        for c in columns:
            print(df[c])

    if "AMR_genotypes_core" in columns:
        new_series = df['AMR_genotypes_core'].map(lambda x: len(x.split(',')))
        max_gene_len = new_series.max()
        print("Maximum number of genes in a isolate: {}".format(max_gene_len))
        cfg['data']['max_n_genes'] = max_gene_len
        if cfg['data']['hierarchy']['use_hierarchy_data']:
            new_series = df['Hierarchy_data'].map(lambda x: max([len(a.split(';')) for a in x.split(',')]))
            max_hier_len = new_series.max()
            print("Maximum length of hierarchy for any trace: {}".format(max_hier_len))
            cfg['data']['max_n_hier'] = max_hier_len
    return df


