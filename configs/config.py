import yaml
import os


def build_config(model_config):
    with open('config.yaml') as f:
        try:
            g_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    with open(os.path.join('model_configs', model_config + '.yaml' )) as f:
        try:
            m_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    config = {**g_config, **m_config}

    return config