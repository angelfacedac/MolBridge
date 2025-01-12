import logging
import multiprocessing as mp
import os
from time import time

import torch
from torch import optim
from torch.utils.data import DataLoader

from src.datasets.dataloader.collate_fn import collate_fn_pyg, collate_fn
from src.datasets.dataset import Mydata
from src.experiments.test import test_pyg, test
from src.experiments.train import train_pyg, train
from src.experiments.valida import valid_pyg, valid
from src.load_config import CONFIG
from src.models.mymodel import MyModelPYG, MyModel
from src.utils import set_seed_
from src.utils.manager import Manager

logging.basicConfig(level=logging.DEBUG)
DATA_SOURCE = CONFIG['data']['source']
BATCH_SIZE = CONFIG['train']['batch_size']
FOLDS = CONFIG['folds']
OPTIMIZER = CONFIG['train']['optimizer']
LR = CONFIG['train']['lr']
EPOCHS = CONFIG['train']['epochs']
DEVICE = torch.device(CONFIG['device'])


@set_seed_(CONFIG['train']['seed'])
def test_best_num_workers():
    logging.debug(f"num of CPU: {mp.cpu_count()}")
    for num_workers in range(0, mp.cpu_count()):

        train_dataloader = get_dataloader(num_workers, 'train')
        val_dataloader = get_dataloader(num_workers, 'val')
        test_dataloader = get_dataloader(num_workers, 'test')

        model = MyModelPYG()
        model.to(DEVICE)

        opt = getattr(optim, OPTIMIZER['name'])(
            model.parameters(),
            lr=LR,
            weight_decay=OPTIMIZER['weight_decay']
        )

        manager = Manager(0, model)
        manager.start()
        manager.writer.add_text("model", str(model))
        manager.writer.add_text("opt", str(opt))

        start = time()
        for epoch in range(10):
            manager.manage_train(epoch + 1, train_pyg(model, train_dataloader, opt))
            manager.manage_valid(epoch + 1, *valid_pyg(model, val_dataloader))
            manager.manage_test(epoch + 1, *test_pyg(model, test_dataloader))

        # manager.add_embedding('test', is_pyg=True)

        manager.close()

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
        collate_fn=collate_fn_pyg,
        num_workers=num_workers
    )
    return dataloader
