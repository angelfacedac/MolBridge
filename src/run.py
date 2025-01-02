import os

import torch
from torch import optim

from src.experiments.test import test
from src.experiments.valida import valid
from src.load_config import CONFIG
from src.datasets.get_dataloader import get_dataloader
from src.experiments.train import train
from src.utils import backup_, add_graph_, set_seed_
from src.utils.manager import Manager
from src.models.mymodel import MyModel

DATA_SOURCE = CONFIG['data']['source']
MODEL_NAME = CONFIG['model_name']
EXPERIMENT_NAME = CONFIG['experiment_name']
FOLDS = CONFIG['folds']
OPTIMIZER = CONFIG['train']['optimizer']
LR = CONFIG['train']['lr']
EPOCHS = CONFIG['train']['epochs']
DEVICE = torch.device(CONFIG['device'])


@set_seed_(CONFIG['train']['seed'])
@add_graph_(os.path.join('logs', DATA_SOURCE, MODEL_NAME, EXPERIMENT_NAME))
@backup_(os.path.join('logs', DATA_SOURCE, MODEL_NAME, EXPERIMENT_NAME, 'project_backup.zip'))
def run():
    for fold_id in FOLDS:

        dataloader_train = get_dataloader(fold_id, 'train')
        dataloader_valid = get_dataloader(fold_id, 'val')
        dataloader_test = get_dataloader(fold_id, 'test')

        model = MyModel()
        model.to(DEVICE)

        opt = [
            getattr(optim, OPTIMIZER['name'][0])(
                model.parameters(),
                lr=LR
            )
            for _ in range(len(OPTIMIZER))
        ]

        manager = Manager(fold_id, model)

        for epoch in range(EPOCHS):

            if epoch + 1 > EPOCHS * OPTIMIZER['threshold']:
                opt_id = 1
            else:
                opt_id = 0

            manager.manage_train(epoch + 1, train(model, dataloader_train, opt[opt_id]))
            manager.manage_valid(epoch + 1, *valid(model, dataloader_valid))
            manager.manage_test(epoch + 1, *test(model, dataloader_test))

        manager.add_embedding('test')

        manager.close()
