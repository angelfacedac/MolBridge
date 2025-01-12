import os.path

from torch.utils.data import DataLoader

from src.datasets.dataloader.collate_fn import collate_fn, collate_fn_pyg
from src.datasets.dataset import Mydata
from src.load_config import CONFIG

DATA_SOURCE = CONFIG['data']['source']
BATCH_SIZE = CONFIG['train']['batch_size']
NUM_WORKERS = CONFIG['train']['num_workers']


def get_dataloader(fold_id, stage):
    dataset = Mydata(
        root_path=os.path.join(
            'src', 'data', DATA_SOURCE, str(fold_id)
        ),
        kind_path=stage + '.csv'
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(stage == 'train'),
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS
    )
    return dataloader


def get_dataloader_pyg(fold_id, stage):
    dataset = Mydata(
        root_path=os.path.join(
            'src', 'data', DATA_SOURCE, str(fold_id)
        ),
        kind_path=stage + '.csv'
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(stage == 'train'),
        collate_fn=collate_fn_pyg,
        num_workers=NUM_WORKERS
    )
    return dataloader

