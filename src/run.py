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

        opt = getattr(optim, OPTIMIZER['name'])(
            model.parameters(),
            lr=LR,
            weight_decay=OPTIMIZER['weight_decay']
        )
        print("model:", model)
        print("opt:", opt)
        # exit()

        manager = Manager(fold_id, model)
        manager.start()
        manager.writer.add_text("model", str(model))
        manager.writer.add_text("opt", str(opt))

        for epoch in range(EPOCHS):

            manager.manage_train(epoch + 1, train(model, dataloader_train, opt))
            manager.manage_valid(epoch + 1, *valid(model, dataloader_valid))
            manager.manage_test(epoch + 1, *test(model, dataloader_test))

        manager.add_embedding('test')

        manager.close()
