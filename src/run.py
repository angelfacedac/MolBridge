import os

import torch
from torch import optim
from tqdm import tqdm

from src.datasets.get_dataloader import get_dataloader, get_dataloader_pyg
from src.experiments.test import test, test_pyg
from src.experiments.train import train, train_pyg
from src.experiments.valida import valid, valid_pyg
from src.load_config import CONFIG
from src.models.mymodel import MyModel, MyModelPYG
from src.utils import backup_, add_graph_, set_seed_
from src.utils.manager import Manager
from src.utils.early_stopping import EarlyStopping

DATA_SOURCE = CONFIG['data']['source']
MODEL_NAME = CONFIG['model_name']
EXPERIMENT_NAME = CONFIG['experiment_name']
FOLDS = CONFIG['folds']
OPTIMIZER = CONFIG['train']['optimizer']
LR = CONFIG['train']['lr']
EPOCHS = CONFIG['train']['epochs']
DEVICE = torch.device(CONFIG['device'])


@set_seed_(CONFIG['train']['seed'])
# @add_graph_(os.path.join('logs', DATA_SOURCE, MODEL_NAME, EXPERIMENT_NAME))
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
        # print("model:", model)
        # print("opt:", opt)
        # exit()

        manager = Manager(fold_id, model)
        manager.start()
        manager.writer.add_text("model", str(model))
        manager.writer.add_text("opt", str(opt))

        # 配置manager的最佳模型追踪（可以与early_stopping监控不同的指标）
        manager_monitor = CONFIG.get('best_model_tracking', {}).get('monitor', 'combined')
        manager_mode = CONFIG.get('best_model_tracking', {}).get('mode', 'max')
        manager.set_best_model_tracking(manager_monitor, manager_mode)

        # 初始化早停
        early_stopping = EarlyStopping(
            patience=CONFIG.get('early_stopping', {}).get('patience', 10),
            min_delta=CONFIG.get('early_stopping', {}).get('min_delta', 0),
            monitor=CONFIG.get('early_stopping', {}).get('monitor', 'val_loss'),
            mode=CONFIG.get('early_stopping', {}).get('mode', 'min'),
            verbose=CONFIG.get('early_stopping', {}).get('verbose', True)
        )

        for epoch in range(EPOCHS):

            manager.manage_train(epoch + 1, train(model, dataloader_train, opt))
            val_loss, y_true, y_pred, y_scores = valid(model, dataloader_valid)
            val_loss, val_acc, val_f1, val_pre, val_rec = manager.manage_valid(epoch + 1, val_loss, y_true, y_pred, y_scores)
            # manager.manage_test(epoch + 1, *test(model, dataloader_test))

            # 早停检查
            if CONFIG.get('early_stopping', {}).get('enabled', False):
                monitor_value = val_loss  # 默认监控验证损失
                if early_stopping.monitor == 'val_acc':
                    monitor_value = val_acc
                elif early_stopping.monitor == 'val_f1':
                    monitor_value = val_f1
                elif early_stopping.monitor == 'combined':
                    monitor_value = val_f1 + val_acc

                if early_stopping(monitor_value):
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    print(f"Best {early_stopping.monitor}: {early_stopping.get_best_score():.6f}")
                    break

        # manager.add_embedding('test')
        manager.wait_all_tasks()  # 先等待
        acc, f1, pre, rec = manager.test(dataloader_test, model=model)
        print("="*50 + "test" + "="*50)
        print("acc | macro_f1 | macro_pre | macro_rec")
        print(f"{acc:.04f}|{f1:.04f}|{pre:.04f}|{rec:.04f}")
        print("="*100)

        manager.close()


@set_seed_(CONFIG['train']['seed'])
# @add_graph_(os.path.join('logs', DATA_SOURCE, MODEL_NAME, EXPERIMENT_NAME))
@backup_(os.path.join('logs', DATA_SOURCE, MODEL_NAME, EXPERIMENT_NAME, 'project_backup.zip'))
def run_pyg():
    for fold_id in FOLDS:

        dataloader_train = get_dataloader_pyg(fold_id, 'train')
        dataloader_valid = get_dataloader_pyg(fold_id, 'val')
        dataloader_test = get_dataloader_pyg(fold_id, 'test')

        model = MyModelPYG()
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

            manager.manage_train(epoch + 1, train_pyg(model, dataloader_train, opt))
            manager.manage_valid(epoch + 1, *valid_pyg(model, dataloader_valid))
            manager.manage_test(epoch + 1, *test_pyg(model, dataloader_test))

        manager.add_embedding('test', is_pyg=True)

        manager.close()

