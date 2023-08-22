import pandas as pd
from sklearn.model_selection import KFold

def get_unique_word_list(df):
    all_words = []
    for row in df.iterrows():
        for c in df.columns:
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