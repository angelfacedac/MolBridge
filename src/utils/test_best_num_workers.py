import os
from time import time
import logging
import multiprocessing as mp
import torch
from torch import optim
from torch.utils.data import DataLoader

from src.datasets.dataloader.collate_fn import collate_fn
from src.datasets.dataset import Mydata
from src.experiments.test import test
from src.experiments.train import train
from src.experiments.valida import valid
from src.load_config import CONFIG
from src.models.mymodel import MyModel
from src.utils.manager import Manager

logging.basicConfig(level=logging.DEBUG)
DATA_SOURCE = CONFIG['data']['source']
BATCH_SIZE = CONFIG['train']['batch_size']
FOLDS = CONFIG['folds']
OPTIMIZER = CONFIG['train']['optimizer']
LR = CONFIG['train']['lr']
EPOCHS = CONFIG['train']['epochs']
DEVICE = torch.device(CONFIG['device'])


def test_best_num_workers():
    logging.debug(f"num of CPU: {mp.cpu_count()}")
    for num_workers in range(2, mp.cpu_count(), 2):
        manager = Manager(0)
        train_dataloader = get_dataloader(num_workers, 'train')
        val_dataloader = get_dataloader(num_workers, 'val')
        test_dataloader = get_dataloader(num_workers, 'test')
        model = MyModel()
        model.to(DEVICE)
        opt = [
            getattr(optim, OPTIMIZER['name'][0])(
                model.parameters(),
                lr=LR
            )
            for _ in range(len(OPTIMIZER))
        ]
        start = time()
        for epoch in range(10):
            if epoch + 1 > EPOCHS * OPTIMIZER['threshold']:
                opt_id = 1
            else:
                opt_id = 0

            manager.manage_train(epoch + 1, train(model, train_dataloader, opt[opt_id]))
            manager.manage_valid(epoch + 1, *valid(model, val_dataloader))
            manager.manage_test(epoch + 1, *test(model, test_dataloader))
        end = time()
        logging.debug("Finish with:{} second, num_workers={}".format(end - start, num_workers))


def get_dataloader(num_workers, stage):
    dataset = Mydata(
        root_path=os.path.join(
            'src', 'data', DATA_SOURCE, str(0)
        ),
        kind_path=stage + '.csv'
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(stage == 'train'),
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    return dataloader

