import os
import tokenizers
import transformers
from data.utils import get_unique_word_list


def build_tokenizer(cfg, dataframe):
    if cfg['tokenizer']['use_pretrained']:
        tokenizer = tokenizers.cfg['tokenizer']['class'].from_pretrained(cfg['tokenizer']['model_path'])
    else:
        tokenizer = create_and_train_tokenizer(cfg, dataframe)
    return tokenizer


def create_and_train_tokenizer(cfg, dataframe):
    base_tokenizer = tokenizers.ByteLevelBPETokenizer()

    corpus = create_corpus(cfg, dataframe)

    vocab = get_unique_word_list(dataframe)
    vocab = cfg['tokenizer']['special_token_list'] + vocab

    base_tokenizer.train(files=os.path.join(cfg['log_dir'], 'corpus.txt'), vocab_size=len(vocab), min_frequency=2,
                   special_tokens=vocab)

    base_tokenizer.save_model(directory=cfg['log_dir'])

    if cfg['tokenizer']['class'] == "RobertaTokenizer":
        tokenizer_c = transformers.RobertaTokenizer
    tokenizer = tokenizer_c.from_pretrained(cfg['log_dir'], max_length=50)
    tokenizer.save_pretrained(save_directory=cfg['log_dir'])

    return tokenizer


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

