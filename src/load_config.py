import yaml


def load_config(config_path='./config.yml'):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # 自动根据num_classes配置模式
    if config['data']['num_classes'] == 1:
        # 二分类模式
        config['train']['loss_fn'] = 'BCEWithLogitsLoss'
        config['data']['actual_output_dim'] = 1
        config['data']['is_binary'] = True
        print(f"检测到二分类模式 (num_classes=1)，自动设置为 BCEWithLogitsLoss")
    else:
        # 多分类模式
        config['train']['loss_fn'] = 'CrossEntropyLoss'
        config['data']['actual_output_dim'] = config['data']['num_classes']
        config['data']['is_binary'] = False
        print(f"检测到多分类模式 (num_classes={config['data']['num_classes']})，保持 CrossEntropyLoss")

    return config


CONFIG = load_config()

