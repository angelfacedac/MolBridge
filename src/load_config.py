import yaml


def load_config(config_path='./config.yml'):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


CONFIG = load_config()

