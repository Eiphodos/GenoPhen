import transformers
import torch
from models.roberta.configuration_roberta import RobertaConfig
from models.roberta.modeling_roberta import RobertaForMaskedLM, RobertaModel
from models.legacy.antibiotic_model import AntibioticModelTrain
from models.legacy.d2l_bert import BERTModel
from models.legacy.integrated_model import IntegratedModel


def build_pt_model(cfg, tokenizer):
    if cfg['model']['class'] == 'RobertaForMaskedLM':
        m_config = RobertaConfig(vocab_size=tokenizer.vocab_size,
                                              max_position_embeddings=50,
                                              num_attention_heads=cfg['model']['n_attention_heads'],
                                              num_hidden_layers=cfg['model']['n_hidden_layers'],
                                              type_vocab_size=1,
                                              hidden_size=cfg['model']['hidden_size'])
        model = RobertaForMaskedLM(m_config)

    return model


def build_ft_legacy_model(cfg, tokenizer_geno, tokenizer_pheno, train_dataloader, val_dataloader, losses):
    if cfg['model']['class'] == 'IntegratedModel':
        if cfg['model']['geno']['class'] == 'RobertaModel':
            geno_m_config = RobertaConfig(vocab_size=tokenizer_geno.vocab_size,
                                                  max_position_embeddings=50,
                                                  num_attention_heads=cfg['model']['n_attention_heads'],
                                                  num_hidden_layers=cfg['model']['n_hidden_layers'],
                                                  type_vocab_size=1,
                                                  hidden_size=cfg['model']['hidden_size'])
            if cfg['model']['geno']['use_pretrained']:
                geno_model = RobertaModel.from_pretrained(cfg['model']['geno']['pretrained_weights'], geno_m_config)
            else:
                geno_model = RobertaModel(geno_m_config)

        if cfg['model']['pheno']['class'] == 'AntibioticModelTrain':
            pheno_encoder = BERTModel(vocab_size=cfg['vocab_len'], 
                                      num_hiddens=cfg['model']['pheno']['num_hiddens'],
                                      norm_shape=cfg['model']['pheno']['norm_shape'],
                                      ffn_num_input=cfg['model']['pheno']['ffn_num_input'],
                                      ffn_num_hiddens=cfg['model']['pheno']['ffn_num_hiddens'],
                                      attention_heads=cfg['model']['pheno']['attention_heads'],
                                      attention_layers=cfg['model']['pheno']['attention_layers'],
                                      dropout=cfg['model']['pheno']['dropout'],
                                      key_size=cfg['model']['pheno']['key_size'],
                                      query_size=cfg['model']['pheno']['query_size'],
                                      value_size=cfg['model']['pheno']['value_size'],
                                      number_ab=len(cfg['antibiotics']['antibiotics_in_use']))
            optimizer = torch.optim.Adam(
                list(geno_model.parameters()) + list(pheno_encoder.parameters()),
                lr=cfg['optimizer']['lr'],
                weight_decay=cfg['optimizer']['weight_decay'])

            pheno_model = AntibioticModelTrain(train_loader=train_dataloader,
                                         val_loader=val_dataloader,
                                         net=pheno_encoder,
                                         losses=losses,
                                         device=cfg['device'],
                                         optimizer=optimizer,
                                         batch_size=cfg['data']['train_batch_size'],
                                         number_ab=len(cfg['antibiotics']['antibiotics_in_use']),
                                         saved_model_name='Legacy AB model')

            if cfg['model']['pheno']['use_pretrained']:
                pheno_model.load_model(cfg['model']['pheno']['pretrained_weights'])

            hidden_dim = 1000

            model = IntegratedModel(cfg, pheno_model, geno_model, train_dataloader, val_dataloader)

    return model
