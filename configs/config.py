import yaml
import os
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def build_config(args):
    with open(os.path.join(ROOT_DIR, 'config.yaml')) as f:
        try:
            config = yaml.safe_load(f)
            test_ab_dict(config)
        except yaml.YAMLError as exc:
            print(exc)
    if bool(args.model_config):
        with open(os.path.join(ROOT_DIR, 'model_configs', args.model_config + '.yaml')) as f:
            try:
                m_config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

        config = data_merge(config, m_config)
    if bool(args.pt_config):
        with open(os.path.join(ROOT_DIR, 'pretraining_configs', args.pt_config + '.yaml')) as f:
            try:
                pt_config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

        config = data_merge(config, pt_config)
    if bool(args.ft_config):
        with open(os.path.join(ROOT_DIR, 'finetuning_configs', args.ft_config + '.yaml')) as f:
            try:
                ft_config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

        config = data_merge(config, ft_config)

    config['log_dir'] = args.log_dir
    config['model']['geno']['pretrained_weights'] = args.geno_model_weights
    config['model']['pheno']['pretrained_weights'] = args.pheno_model_weights
    config['tokenizer']['geno']['pretrained_weights'] = args.geno_tokenizer_weights
    config['tokenizer']['pheno']['pretrained_weights'] = args.pheno_tokenizer_weights
    for s in config['data']['species']:
        if s == 'E_Coli':
            config['species'][s]["file_path"] = args.ecoli_file
        elif s == 'Kleb':
            config['species'][s]["file_path"] = args.kleb_file

    if not config['log_dir'] is None:
        config['log_dir'] = os.path.join(config['log_dir'], config['model']['class'] + '_' + time.strftime("%Y-%m-%d-%H_%M"))
    os.makedirs(config['log_dir'], exist_ok=True)

    return config


def test_ab_dict(config):
    for k, v in config['antibiotics'].items():
        if k not in ["index_list", "antibiotics_in_use", 'antibiotics_weights']:
            assert k == config['antibiotics'][v]

    for i in config['antibiotics']["index_list"]:
        assert i['name'] == config['antibiotics'][i['abbrev']]
        assert i['abbrev'] == config['antibiotics'][i['name']]
    print("All AB dict tests passed successfully!")


class YamlReaderError(Exception):
    pass


def data_merge(a, b):
    """merges b into a and return merged result

    NOTE: tuples and arbitrary objects are not handled as it is totally ambiguous what should happen"""
    key = None
    # ## debug output
    # sys.stderr.write("DEBUG: %s to %s\n" %(b,a))
    try:
        if a is None or isinstance(a, str) or isinstance(a, int) or isinstance(a, float) or isinstance(a, list):
            # border case for first run or if a is a primitive
            a = b
        elif isinstance(a, dict):
            # dicts must be merged
            if isinstance(b, dict):
                for key in b:
                    if key in a:
                        a[key] = data_merge(a[key], b[key])
                    else:
                        a[key] = b[key]
            else:
                raise YamlReaderError('Cannot merge non-dict "%s" into dict "%s"' % (b, a))
        else:
            raise YamlReaderError('NOT IMPLEMENTED "%s" into "%s"' % (b, a))
    except TypeError as e:
        raise YamlReaderError('TypeError "%s" in key "%s" when merging "%s" into "%s"' % (e, key, b, a))
    return a


def save_config(cfg):
    with open(os.path.join(cfg['log_dir'], "full_config.yaml"), "w") as f:
        yaml.dump(cfg, f)
