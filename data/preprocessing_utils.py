import re
from collections import OrderedDict
import pandas as pd


def clean_aap_aac(x):
    interesting_genes = ["aph", "aac"]
    roman_numbers = ["I", "V", "X"]

    all_items = x.split(',')
    all_items = [remove_last_non_digit(a) if a[0:3] in interesting_genes and any(
        r in a.split("-")[-1] for r in roman_numbers) else a for a in all_items]
    return ",".join(all_items)


def clean_bla_hier(x):
    return [remove_last_digits(a) if a[0:3] == 'bla' else a for a in x]


def clean_aap_aac_hier(x):
    interesting_genes = ["aph", "aac"]
    roman_numbers = ["I", "V", "X"]

    x = [remove_last_non_digit(a) if a[0:3] in interesting_genes and any(
        r in a.split("-")[-1] for r in roman_numbers) else a for a in x]
    return x


def clip_aac_hard(x):
    all_items = x.split(',')
    all_items = [y.split('-')[0] if 'aac' in y else y for y in all_items]
    x = ",".join(all_items)
    return x


def clean_biss_prime_hier(x):
    x = [a.replace("''", "b") for a in x]
    x = [a.replace("'", "p") for a in x]
    return x


def clean_doubles_hier(x):
    return list(OrderedDict.fromkeys(x))


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


def remove_point_mut_location(x):
    all_items = x.split(',')
    all_items = [y.split('_')[0] if '=POINT' in y else y for y in all_items]
    x = ",".join(all_items)
    return x


def remove_pheno_duplicates(x):
    all_items = pd.DataFrame([a.split('=') for a in x.split(',')]) # Splits sequence of AB resistances and creates dataframe
    all_items.drop_duplicates(subset=[0], inplace=True, keep=False) # Deletes duplicates based on AB name
    return ",".join(["=".join(a) for a in all_items.values.tolist()]) # Joins into list again


def remove_geno_duplicates(x):
    all_items = pd.DataFrame([a for a in x.split(',')]) # Splits sequence of genes and creates dataframe
    all_items.drop_duplicates(subset=[0], inplace=True, keep=False) # Deletes duplicates based on AB name
    return ",".join(["=".join(a) for a in all_items.values.tolist()]) # Joins into list again


def convert_ab_to_abbrev(x, cfg):
    all_items = x.split(',')
    all_items = ["=".join([cfg['antibiotics'][a.split('=')[0]], a.split('=')[1]]) for a in all_items if a.split('=')[0] in cfg['antibiotics'].keys()]
    return ",".join(all_items)


def clean_pheno_data(cfg, df):
    df['AST_phenotypes'] = df['AST_phenotypes'].map(lambda x: ','.join([a.lower() for a in x.split(',') if a[-1] == "R" or a[-1] == "S"])) #Remove all resistances that are not R or S
    if cfg['data']['filter']['doubles_pheno']:
        df['AST_phenotypes'] = df['AST_phenotypes'].map(remove_pheno_duplicates)
    df['AST_phenotypes'] = df['AST_phenotypes'].apply(convert_ab_to_abbrev, cfg=cfg)
    return df


def clean_geno_data(cfg, df):
    df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(lambda x: x.replace("''", "b")) # Replace biss with b
    df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(lambda x: x.replace("'", "p")) # Replace prime with p
    if cfg['data']['filter']['doubles_geno']:
        df['AMR_genotypes_core'] = df['AMR_genotypes_core'].map(remove_geno_duplicates)
    return df

def clean_hier_data(cfg, df):
    df['Hierarchy_data'] = df['Hierarchy_data'].map(lambda x: x.replace("''", "b")) # Replace biss with b
    df['Hierarchy_data'] = df['Hierarchy_data'].map(lambda x: x.replace("'", "p")) # Replace prime with p
    return df


def clean_date_data(cfg, df):
    df['target_creation_date'] = df['target_creation_date'].map(lambda x: x[0:7]) # only keep YYYY-MM of the incoming YYYY-MM-DD
    return df


def clean_hierarchy_data(cfg, df):
    df['Hierarchy_data'] = df['Hierarchy_data'].map(lambda x: x.replace("''", "b")) # Replace biss with b
    df['Hierarchy_data'] = df['Hierarchy_data'].map(lambda x: x.replace("'", "p")) # Replace prime with p
    return df