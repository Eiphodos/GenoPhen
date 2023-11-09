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

    cfg['genes']['unique_genes_frequencies'] = {k: 0 for k in unique_genes}
    cfg['genes']['unique_genes_ratio'] = []
    total_genes = 0
    for j, row in df.iterrows():
        genes = row['AMR_genotypes_core'].split(',')
        for g in genes:
            cfg['genes']['unique_genes_frequencies'][g] += 1
            total_genes += 1

    print("Total genes found: {}".format(total_genes))
    mcg = max(cfg['genes']['unique_genes_frequencies'], key=cfg['genes']['unique_genes_frequencies'].get)
    print("Most common gene: {}".format(mcg))

    for k in cfg['genes']['unique_genes_frequencies'].keys():
        cfg['genes']['unique_genes_ratio'].append(cfg['genes']['unique_genes_frequencies'][k] / total_genes)

    fmcg = cfg['genes']['unique_genes_ratio'][list(cfg['genes']['unique_genes_frequencies'].keys()).index(mcg)]
    print("Frequency for most common gene: {}".format(fmcg))


    return df


def gene_exist(x, unique_genes):
    genes = x.split(',')
    gene_exist = []
    for g in genes:
        gi = unique_genes.index(g)
        gene_exist.append(gi)
    return gene_exist