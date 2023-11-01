import os
import tokenizers
import transformers
from models.roberta.tokenization_roberta import RobertaTokenizer
from data.utils import get_unique_word_list, create_corpus


def build_tokenizer(cfg, dataframe):
    if cfg['tokenizer']['use_pretrained']:
        if cfg['tokenizer']['class'] == "RobertaTokenizer":
            tokenizer_c = RobertaTokenizer
        tokenizer = tokenizer_c.from_pretrained(cfg['tokenizer']['geno']['pretrained_weights'])
    else:
        tokenizer = create_and_train_tokenizer(cfg, dataframe)
    return tokenizer


def create_and_train_tokenizer(cfg, dataframe):
    base_tokenizer = tokenizers.ByteLevelBPETokenizer()

    corpus = create_corpus(cfg, dataframe)

    vocab = get_unique_word_list(dataframe[cfg['tokenizer']['columns_for_corpus']])
    print("Size of vocabulary before training: {}".format(len(vocab)))
    vocab = cfg['tokenizer']['special_token_list'] + vocab

    base_tokenizer.train(files=os.path.join(cfg['log_dir'], 'corpus.txt'), vocab_size=len(vocab), min_frequency=2,
                   special_tokens=vocab)
    print("Size of vocabulary after training: {}".format(len(vocab)))

    base_tokenizer.save_model(directory=cfg['log_dir'])

    if cfg['tokenizer']['class'] == "RobertaTokenizer":
        tokenizer_c = RobertaTokenizer
    tokenizer = tokenizer_c.from_pretrained(cfg['log_dir'], max_length=cfg['tokenizer']['max_len'])
    if cfg['data']['hierarchy']['use_hierarchy_data']:
        tokenizer.add_special_tokens({'additional_special_tokens': ["<gpsep>"]})
    tokenizer.save_pretrained(save_directory=cfg['log_dir'])

    return tokenizer


