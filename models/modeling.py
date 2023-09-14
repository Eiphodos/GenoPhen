import transformers
from models.roberta.configuration_roberta import RobertaConfig
from models.roberta.modeling_roberta import RobertaForMaskedLM


def build_model(cfg, tokenizer):
    if cfg['model']['class'] == 'RobertaForMaskedLM':
        m_config = RobertaConfig(vocab_size=tokenizer.vocab_size,
                                              max_position_embeddings=50,
                                              num_attention_heads=cfg['model']['n_attention_heads'],
                                              num_hidden_layers=cfg['model']['n_hidden_layers'],
                                              type_vocab_size=1,
                                              hidden_size=cfg['model']['hidden_size'])
        model = RobertaForMaskedLM(m_config)

    return model
