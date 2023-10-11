import transformers
import torch
import torch.nn as nn
from models.roberta.configuration_roberta import RobertaConfig
from models.roberta.modeling_roberta import RobertaForMaskedLM, RobertaModel
from models.roberta.hierarchical_roberta import RobertaHierForMaskedLM, RobertaHierModel
from models.roberta.hierarchical_embeddings import RobertaHierarchicalEmbeddingsV1
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

    if cfg['model']['class'] == 'RobertaHierForMaskedLM':
        m_config = RobertaConfig(vocab_size=tokenizer.vocab_size,
                                              max_position_embeddings=50,
                                              max_genes=cfg['data']['max_n_genes'] + 1,
                                              num_attention_heads=cfg['model']['n_attention_heads'],
                                              num_hidden_layers=cfg['model']['n_hidden_layers'],
                                              type_vocab_size=1,
                                              hidden_size=cfg['model']['hidden_size'])
        emb = RobertaHierarchicalEmbeddingsV1(config=m_config)
        model = RobertaHierForMaskedLM(m_config, emb)

    return model


def build_ft_legacy_model(cfg, tokenizer_geno, train_dataloader, val_dataloader):
    if cfg['model']['class'] == 'IntegratedModel':
        if cfg['model']['geno']['class'] == 'RobertaModel':
            geno_m_config = RobertaConfig(vocab_size=tokenizer_geno.vocab_size,
                                                  max_position_embeddings=50,
                                                  num_attention_heads=cfg['model']['geno']['n_attention_heads'],
                                                  num_hidden_layers=cfg['model']['geno']['n_hidden_layers'],
                                                  type_vocab_size=1,
                                                  hidden_size=cfg['model']['geno']['hidden_size'])
            if cfg['model']['geno']['use_pretrained']:
                if cfg['model']['geno']['pretrained_weights'][-4:] == '.pth':
                    sd = torch.load(cfg['model']['geno']['pretrained_weights'], map_location='cpu')
                    model_sd = sd['model']
                    if "module." in list(model_sd.keys())[0]:
                        print("Tag 'module.' found in state dict - fixing!")
                        for key in list(model_sd.keys()):
                            model_sd[key.replace("module.", "")] = model_sd.pop(key)
                    if "roberta." in list(model_sd.keys())[0]:
                        print("Tag 'roberta.' found in state dict - fixing!")
                        for key in list(model_sd.keys()):
                            model_sd[key.replace("roberta.", "")] = model_sd.pop(key)
                    geno_model = RobertaModel(geno_m_config)
                    geno_model.load_state_dict(model_sd, strict=False)
                else:
                    geno_model = RobertaModel.from_pretrained(cfg['model']['geno']['pretrained_weights'], geno_m_config)
            else:
                geno_model = RobertaModel(geno_m_config)
        elif cfg['model']['geno']['class'] == 'RobertaHierModel':
            geno_m_config = RobertaConfig(vocab_size=tokenizer_geno.vocab_size,
                                     max_position_embeddings=50,
                                     max_genes=cfg['data']['max_n_genes'] + 1,
                                     num_attention_heads=cfg['model']['geno']['n_attention_heads'],
                                     num_hidden_layers=cfg['model']['geno']['n_hidden_layers'],
                                     type_vocab_size=1,
                                     hidden_size=cfg['model']['geno']['hidden_size'])
            emb = RobertaHierarchicalEmbeddingsV1(config=geno_m_config)
            if cfg['model']['geno']['use_pretrained']:
                if cfg['model']['geno']['pretrained_weights'][-4:] == '.pth':
                    sd = torch.load(cfg['model']['geno']['pretrained_weights'], map_location='cpu')
                    model_sd = sd['model']
                    if "module." in list(model_sd.keys())[0]:
                        print("Tag 'module.' found in state dict - fixing!")
                        for key in list(model_sd.keys()):
                            model_sd[key.replace("module.", "")] = model_sd.pop(key)
                    if "roberta." in list(model_sd.keys())[0]:
                        print("Tag 'roberta.' found in state dict - fixing!")
                        for key in list(model_sd.keys()):
                            model_sd[key.replace("roberta.", "")] = model_sd.pop(key)
                    geno_model = RobertaHierModel(geno_m_config, emb)
                    geno_model.load_state_dict(model_sd, strict=False)
                else:
                    geno_model = RobertaHierModel.from_pretrained(cfg['model']['geno']['pretrained_weights'], geno_m_config)
            else:
                geno_model = RobertaHierModel(geno_m_config, emb)

        geno_model.to(cfg['device'])

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
            pheno_encoder.to(cfg['device'])

            optimizer = torch.optim.Adam(
                list(geno_model.parameters()) + list(pheno_encoder.parameters()),
                lr=cfg['optimizer']['lr'],
                weight_decay=cfg['optimizer']['weight_decay'])

            losses = [nn.CrossEntropyLoss(weight=torch.tensor(v, requires_grad=False, dtype=torch.float).to(cfg['device']),
                                          reduction='mean') for v in cfg['antibiotics']['res_ratio_train'].values()]

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

            model = IntegratedModel(cfg, pheno_model, geno_model, train_dataloader, val_dataloader)

    return model
