import os
import tokenizers
import transformers
import json
from models.roberta.tokenization_roberta import RobertaTokenizer
from data.utils import get_unique_word_list


def build_tokenizer(cfg, dataframe):
    if cfg['tokenizer']['use_pretrained']:
        if cfg['tokenizer']['class'] == "RobertaTokenizer":
            tokenizer_c = RobertaTokenizer
        tokenizer = tokenizer_c.from_pretrained(cfg['tokenizer']['geno']['pretrained_weights'])
    else:
        tokenizer = create_and_train_tokenizer(cfg, dataframe)
    return tokenizer


def create_and_train_tokenizer(cfg, dataframe):
    roberta_special_tokens = {"<s>": 0, "<pad>": 1, "</s>": 2, "<mask>": 3, "<unk>": 4}
    if cfg['data']['hierarchy']['use_hierarchy_data']:
        roberta_special_tokens["<gpsep>"] = 5

    vocab = get_unique_word_list(dataframe[cfg['tokenizer']['columns_for_vocab']])
    vocab_indexes = list(range(len(roberta_special_tokens), len(vocab) + len(roberta_special_tokens)))
    vvi = zip(vocab, vocab_indexes)
    other_tokens = {k: v for k, v in vvi}
    all_tokens = {**roberta_special_tokens, **other_tokens}
    vocab_path = os.path.join(cfg['log_dir'], "vocab.json")
    merges_path = os.path.join(cfg['log_dir'], "merges.txt")
    with open(vocab_path, "w") as vocabfile:
        json.dump(all_tokens, vocabfile)

    with open(merges_path, "w") as mergesfile:
        mergesfile.write("#version: 0.2")

    tokenizer = RobertaTokenizer(vocab_file=vocab_path, merges_file=merges_path)
    tokenizer.save_pretrained(save_directory=cfg['log_dir'])

    return tokenizer


