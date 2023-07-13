import yaml
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def build_config(pt_config=None, finetuning_config=None, model_config=None):
    with open(os.path.join(ROOT_DIR, 'config.yaml')) as f:
        try:
            config = yaml.safe_load(f)
            test_ab_dict(config)
        except yaml.YAMLError as exc:
            print(exc)
    if model_config is not None:
        with open(os.path.join(ROOT_DIR, 'model_configs', model_config + '.yaml')) as f:
            try:
                m_config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

        config = data_merge(config, m_config)
    if pt_config is not None:
        with open(os.path.join(ROOT_DIR, 'pretraining_configs', pt_config + '.yaml')) as f:
            try:
                pt_config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

        config = data_merge(config, pt_config)
    if finetuning_config is not None:
        with open(os.path.join(ROOT_DIR, 'finetuning_configs', finetuning_config + '.yaml')) as f:
            try:
                ft_config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

        config = data_merge(config, ft_config)

    return config


def test_ab_dict(config):
    for k, v in config['antibiotics'].items():
        if k != "index_list":
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

