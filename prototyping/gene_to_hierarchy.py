import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from configs.config import build_config
#from data.preprocess_data import preprocess_data, filter_aap_aac
from data.utils import get_unique_word_list, create_corpus

def find_number_of_levels(data):
    max_levels = 0
    for index, row in data.iterrows():
        levels = 1
        parent = row['parent_node_id']
        leaf = row['node_id']
        trace = []
        trace.append(leaf)
        while parent not in [np.nan, 'AMR', 'ALL']:
            trace.append(parent)
            parent = parent.replace("'", "\'")
            levels += 1
            parent = data[data['node_id'] == parent]['parent_node_id'].item()
        if levels > max_levels:
            max_levels = levels
            print("max levels is now {} after processing {} with trace {}".format(max_levels, leaf, trace))
    return max_levels


def find_number_of_words(data):
    all_words = []
    for index, row in data.iterrows():
        parent = row['parent_node_id']
        leaf = row['node_id']
        all_words.append(leaf)
        while parent not in [np.nan, 'AMR', 'ALL']:
            all_words.append(parent)
            parent = parent.replace("'", "\'")
            parent = data[data['node_id'] == parent]['parent_node_id'].item()
    all_words = list(set(all_words))
    print("Total number of unique words: {}".format(len(all_words)))
    print("Examples: {}".format(all_words[0:6]))
    return all_words


def get_gene_trace(data, gene_word):
    trace = []
    trace.append(gene_word)
    gene_data = data[data['node_id'] == gene_word]
    if len(gene_data) == 0:
        print("Could not find {} in data".format(gene_word))
        return []
    parent = data[data['node_id'] == gene_word]['parent_node_id'].item()
    while parent not in [np.nan, 'AMR', 'ALL']:
        trace.append(parent)
        parent = parent.replace("'", "\'")
        parent = data[data['node_id'] == parent]['parent_node_id'].item()
    return trace

def get_gene_trace_symbol(data, gene_word):
    trace = []
    trace.append(gene_word)
    gene_data = data[data['symbol'] == gene_word]
    if len(gene_data) == 0:
        print("Could not find {} in data".format(gene_word))
        return []
    parent = gene_data['parent_node_id'].item()
    while parent not in [np.nan, 'AMR', 'ALL']:
        parent_data = data[data['node_id'] == parent]
        trace.append(parent_data['symbol'].item())
        parent = parent.replace("'", "\'")
        parent = data[data['node_id'] == parent]['parent_node_id'].item()
    return trace

#cfg = build_config(pt_config='pt')
#dataframe = preprocess_data(cfg)
#corpus = create_corpus(cfg, dataframe)
#vocab = get_unique_word_list(dataframe)

hdata = pd.read_csv(r'D:\Datasets\NCBI\ReferenceGeneHierarchy.txt', sep='\t')
ml = find_number_of_levels(hdata)
print("Max levels is {}".format(ml))
#aw = find_number_of_words(hdata)
all_traces = []
gw = "blaOXA-292"
gw2 = "aac(6')-IId"
trace = get_gene_trace(hdata, gw)
trace2 = get_gene_trace(hdata, gw2)
print("Trace for {} is {}".format(gw, trace))
print("Trace for {} is {}".format(gw2, trace2))
trace = get_gene_trace_symbol(hdata, gw)
trace2 = get_gene_trace_symbol(hdata, gw2)
print("Symbol Trace for {} is {}".format(gw, trace))
print("Symbol Trace for {} is {}".format(gw2, trace2))

#for w in vocab:
#    trace = get_gene_trace(hdata, w)
#    all_traces += trace
#all_traces = list(set(all_traces))
#print("Total number of unique words: {}".format(len(all_traces)))
