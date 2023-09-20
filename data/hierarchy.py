import numpy as np
import pandas as pd
import re

def get_gene_trace(hdata, cdata, gene_word):
    trace = []
    # regex pattern for matching with pointwise mutations
    pattern = r'_[a-zA-Z0-9-]{3,}='
    if re.search(pattern, gene_word):
        gene_word = gene_word.split('=')[0]
        trace.append(gene_word)
        trace.append(gene_word.split('_')[0])
        cdata_gw = cdata[cdata['allele'] == gene_word]
        if len(cdata_gw) == 0:
            print("Could not find {} in cdata".format(gene_word))
        elif len(cdata_gw) > 1:
            #print("Multiple entries of {} in cdata, choosing first one".format(gene_word))
            trace.append(cdata_gw['subclass'].iloc[0])
            trace.append(cdata_gw['class'].iloc[0])
        else:
            trace.append(cdata_gw['subclass'].item())
            trace.append(cdata_gw['class'].item())
    else:
        if any([a in gene_word for a in ['=PARTIAL_END_OF_CONTIG', '=PARTIAL', '=HMM', '=MISTRANSLATION']]):
            gene_word = gene_word.split('=')[0]
        trace.append(gene_word)
        gene_data = hdata[hdata['symbol'] == gene_word]
        if len(gene_data) != 1:
            #print("Data for {} failed in hdata, trying with node-id".format(gene_word))
            gene_data = hdata[hdata['node_id'] == gene_word]
            if len(gene_data) != 1:
                #print("Trying with general family")
                gene_data = hdata[hdata['node_id'] == gene_word + '_gen']
                if len(gene_data) != 1:
                    print("Could not narrow down hierarchy for {} in symbol or node-id hdata, giving up".format(gene_word))
                    return ";".join(trace)
        sc = gene_data['subclass'].item()
        gc = gene_data['class'].item()
        parent = gene_data['parent_node_id'].item()
        while parent not in [np.nan, 'AMR', 'ALL']:
            trace.append(parent)
            parent = parent.replace("'", "\'")
            parent = hdata[hdata['node_id'] == parent]['parent_node_id'].item()
        trace.append(sc)
        trace.append(gc)

    return ";".join(trace)


def build_hierarchy_data(cfg, df):
    print("Reading hierarchy and catalog files...")
    hdata = pd.read_csv(cfg['data']['hierarchy']['hierarchy_file_path'], sep='\t')
    cdata = pd.read_csv(cfg['data']['hierarchy']['catalog_file_path'], sep='\t')

    print("Building hierarchy traces for all genes in dataframe...")
    df['Hierarchy_data'] = df['AMR_genotypes_core'].map(
        lambda x: ','.join([get_gene_trace(hdata, cdata, a) for a in x.split(',')]))

    return df