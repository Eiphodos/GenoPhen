import pandas as pd
import numpy as np


def build_gene_complement(cfg, df):
    unique_genes = []
    for j, row in df.iterrows():
        genes = row['AMR_genotypes_core'].split(',')
        unique_genes += genes
    unique_genes = list(set(unique_genes))
    cfg['genes']['unique_genes'] = unique_genes
    print("Number of unique genes found: {}".format(len(unique_genes)))

    df['existing_genes'] = df['AMR_genotypes_core'].apply(gene_exist, unique_genes=unique_genes)

    return df


def gene_exist(x, unique_genes):
    genes = x.split(',')
    gene_exist = []
    for g in genes:
        gi = unique_genes.index(g)
        gene_exist.append(gi)
    return gene_exist