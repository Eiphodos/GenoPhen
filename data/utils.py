import pandas as pd
import os
import numpy as np
from sklearn.model_selection import KFold
import itertools


def get_unique_word_list(df):
    all_words = []
    for row in df.iterrows():
        for c in df.columns:
            if c == 'Hierarchy_data':
                    for g in row[1][c].split(','):
                        all_words += g.split(';')
            else:
                all_words += row[1][c].split(',')
    vocabulary = list(set(all_words))
    print("Size of vocabulary: {}".format(len(vocabulary)))
    return vocabulary


def cv_split_dataframe(cfg, dataframe):
    kf = KFold(n_splits=cfg['data']['cv_n_folds'], shuffle=True, random_state=cfg['seed'])
    train_folds = []
    val_folds = []
    for train_indexes, val_indexes in kf.split(dataframe):
            train_folds.append(train_indexes)
            val_folds.append(val_indexes)

    train_dataframe = dataframe.iloc[train_folds[cfg['data']['cv_fold']]]
    val_dataframe = dataframe.iloc[val_folds[cfg['data']['cv_fold']]]

    return train_dataframe, val_dataframe

def create_corpus(cfg, dataframe):
    df_list = dataframe.values.tolist()
    corpus = []
    for r in df_list:
        r = [a for a in r if not type(a) == float]
        r = ",".join(r)
        corpus.append(r)
    with open(os.path.join(cfg['log_dir'], 'corpus.txt'), 'w') as f:
        f.writelines(c + '\n' for c in corpus)
        f.close()
    return corpus


def combinatorial_data_generator(cfg, dataframe):

    new_dict = {}
    for c in dataframe.columns:
        new_dict[c] = []
    new_dict['AST_phenotypes_x'] = []
    new_dict['AST_phenotypes_y'] = []

    # The number of known antibiotics in input data must be smaller than
    # the minumum number of antibiotics in total as otherwise there is nothing to predict
    assert cfg['data']['known_ab'] < cfg['data']['filter']['min_ab']
    #assert cfg['data']['known_geno'] < cfg['data']['filter']['min_geno']

    for i, row in dataframe.iterrows():
        ab_list = row['AST_phenotypes'].split(',')
        # Create all possible permutations of antibiotics
        all_combs = list(itertools.combinations(ab_list, cfg['data']['known_ab']))
        for j in range(len(all_combs)):
            x_ab = list(all_combs[j])
            # Get the remaining ab as the unknown labels
            y_ab = (list(set(ab_list) ^ set(x_ab)))
            # Add known and unknown antibiotics to new dataframe
            new_dict['AST_phenotypes_x'].append(x_ab)
            new_dict['AST_phenotypes_y'].append(y_ab)
            # Add remaining genotype and phenotype data
            for c in dataframe.columns:
                    new_dict[c].append(row[c])

    new_dataframe = pd.DataFrame(new_dict)

    return new_dataframe


def weights_separated_by_label(cfg, dataframe):
    weights = np.zeros(16)
    weights_r = np.zeros(16)
    weights_s = np.zeros(16)
    for i, row in dataframe.iterrows():
        ab_list = row['AST_phenotypes'].split(',')
        for j in range(len(cfg['antibiotics']['index_list'])):
            if cfg['antibiotics']['index_list'][j]['abbrev'] + '=r' in ab_list:
                weights[j] += 1
                weights_r[j] += 1
            if cfg['antibiotics']['index_list'][j]['abbrev'] + '=s' in ab_list:
                weights[j] += 1
                weights_s[j] += 1
    weight_sum = sum(weights)
    weight_r_sum = sum(weights_r)
    weight_s_sum = sum(weights_s)
    ab_dict = {ab['abbrev']: [] for ab in cfg['antibiotics']['index_list']}
    for i, k in enumerate(ab_dict.keys()):
        ab_tot = weights_r[i] + weights_s[i]
        if ab_tot == 0:
            w1 = 0
            w2 = 0
        else:
            w1 = float(1 - (weights_s[i] / ab_tot)) # Inverse as they are used for loss weightings
            w2 = float(1 - (weights_r[i] / ab_tot))
        ab_dict[k].append(w1)
        ab_dict[k].append(w2)
    return weights/weight_sum, weights_s/weight_s_sum, weights_r/weight_r_sum, ab_dict

def compute_resistance_ratio_per_ab(cfg, dataframe):
    ab_dict = {ab['abbrev']: [] for ab in cfg['antibiotics']['index_list']}
    for i, row in dataframe.iterrows():
        ab_list = row['AST_phenotypes'].split(',')