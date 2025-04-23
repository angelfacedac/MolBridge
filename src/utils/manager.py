import concurrent.futures
import os
import signal

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from src.datasets.dataloader.collate_fn import collate_fn, collate_fn_pyg
from src.experiments.move_data_to_device import move_data_to_device
from src.experiments.test import test
from src.load_config import CONFIG
from src.models import mymodel
from src.utils.metrics import metrics

DATA_SOURCE = CONFIG['data']['source']
MODEL_NAME = CONFIG['model_name']
EXPERIMENT_NAME = CONFIG['experiment_name']
MULTIPROC_MAX_WORKERS = CONFIG['train']['multiproc_max_workers']
EPOCHS = CONFIG['train']['epochs']
DEVICE = torch.device(CONFIG['device'])


class Manager:
    def __init__(self, fold_id=None, model=None):
        self.activations = None
        self.model = model
        self.my_metrics_for_model = {
            'valid': np.zeros(EPOCHS + 1),
            'test': np.zeros(EPOCHS + 1)
        }
        self.max_id = {
            'valid': 0,
            'test': 0
        }
        self.fold_path = os.path.join('logs', DATA_SOURCE, MODEL_NAME, EXPERIMENT_NAME, str(fold_id))
        self.data_fold_path = os.path.join('src', 'data', DATA_SOURCE, str(fold_id))

        if fold_id is not None:
            self.writer = SummaryWriter(self.fold_path)
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=MULTIPROC_MAX_WORKERS)
            self.add_config()

    def manage_train(self, epoch, loss):
        self.add_loss(epoch, 'train', loss)
        self.add_model_parameters_histogram(epoch)

    def manage_valid(self, epoch, loss, y_true, y_pred, y_scores):
        y_true, y_pred = self.move_tensor2numpy(y_true, y_pred)
        self.add_loss(epoch, 'valid', loss)
        self.add_metrics_by_y(epoch, 'valid', y_true, y_pred)

    def manage_test(self, epoch, loss, y_true, y_pred, y_scores):
        y_true, y_pred = self.move_tensor2numpy(y_true, y_pred)
        self.add_loss(epoch, 'test', loss)
        self.add_metrics_by_y(epoch, 'test', y_true, y_pred)

    def test(self, dataloader, model_params_path=None):
        if model_params_path is not None:
            path = model_params_path
        else:
            path = os.path.join(self.fold_path, f'{self.max_id["valid"]:03d}_valid.pth')
        model_params = torch.load(
            path,
            weights_only=True
        )
        model = mymodel.MyModel()
        model.load_state_dict(model_params)
        model.to(DEVICE)
        loss, y_true, y_pred, y_scores = test(model, dataloader)
        y_true, y_pred = self.move_tensor2numpy(y_true, y_pred)
        macro_precision, macro_recall, macro_f1, accuracy = metrics(y_true, y_pred)
        return accuracy, macro_f1, macro_precision, macro_recall

    def add_metrics(self, epoch, stage, **metrics):
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'{metric_name}/{stage}', metric_value, epoch)

    def add_loss(self, epoch, stage, loss):
        self.writer.add_scalar(f'loss/{stage}', loss, epoch)

    def add_metrics_by_y(self, epoch, stage, y_true, y_pred):
        def callback(future):
            macro_precision, macro_recall, macro_f1, accuracy = future.result()
            self.add_metrics(
                epoch,
                stage,
                accuracy=accuracy,
                macro_f1=macro_f1,
                macro_precision=macro_precision,
                macro_recall=macro_recall,
            )
            self.my_metrics_for_model[stage][epoch] = macro_f1 + accuracy
            if self.my_metrics_for_model[stage][epoch] > self.my_metrics_for_model[stage][self.max_id[stage]]:
                if self.max_id[stage] != 0:
                    os.remove(os.path.join(self.fold_path, f'{self.max_id[stage]:03d}_{stage}.pth'))
                self.max_id[stage] = epoch
                torch.save(self.model.state_dict(), os.path.join(self.fold_path, f'{epoch:03d}_{stage}.pth'))
                self.writer.add_text(
                    f"metrics/{stage}",
                    f"|{accuracy:.4f}|{macro_f1:.4f}|{macro_precision:.4f}|{macro_recall:.4f}|",
                    epoch
                )

        self.executor.submit(metrics, y_true, y_pred).add_done_callback(callback)

    def wait_all_tasks(self):
        self.executor.shutdown(wait=True)  # 等待所有任务完成
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=MULTIPROC_MAX_WORKERS)  # 重建执行器

    def graceful_exit(self, signum, frame):
        print("Received termination signal. Shutting down gracefully...")
        self.executor.shutdown(wait=True)
        exit(0)

    def start(self):
        # 注册信号处理函数
        signal.signal(signal.SIGTERM, self.graceful_exit)
        signal.signal(signal.SIGINT, self.graceful_exit)

    def add_embedding(self, stage, is_pyg=False):
        data_path = None
        if stage == 'valid':
            data_path = os.path.join(self.data_fold_path, "val.csv")
        elif stage == 'test':
            data_path = os.path.join(self.data_fold_path, "test.csv")
        df = pd.read_csv(data_path)
        df = df[df["label"].between(0, 3)]
        sampled_df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), 400)))
        sampled_df = [sampled_df.iloc[i] for i in range(len(sampled_df))]

        def forward_hook(module, input, output):
            self.activations = input[0].detach()

        if is_pyg:
            model = mymodel.MyModelPYG()
            model.to(DEVICE)
            model.load_state_dict(
                torch.load(
                    os.path.join(self.fold_path, f'{self.max_id[stage]:03d}_{stage}.pth'),
                    weights_only=True
                )
            )
            handle = model.mlp.register_forward_hook(forward_hook)

            data = collate_fn_pyg(sampled_df)
            data = data.to(DEVICE)
            labels = torch.LongTensor(data.y)
            model(data)
        else:
            model = mymodel.MyModel()
            model.to(DEVICE)
            model.load_state_dict(
                torch.load(
                    os.path.join(self.fold_path, f'{self.max_id[stage]:03d}_{stage}.pth'),
                    weights_only=True
                )
            )
            handle = model.mlp.register_forward_hook(forward_hook)

            embeds, adjs, masks, cnn_masks, labels = collate_fn(sampled_df)
            embeds, adjs, masks, cnn_masks, labels = move_data_to_device(
                (embeds, adjs, masks, cnn_masks, labels),
                DEVICE
            )
            model(embeds, adjs, masks, cnn_masks, labels)

        self.writer.add_embedding(
            mat=self.activations,
            metadata=labels,
            global_step=self.max_id[stage],
            tag=f'{stage}_embedding'
        )

        handle.remove()

    def add_model_parameters_histogram(self, epoch):
        for name, param in self.model.named_parameters():
            # print(f"\nParameter: {name}")
            # print(f"Shape: {param.shape}")
            # print(f"Min: {param.min().item():.6f}")
            # print(f"Max: {param.max().item():.6f}")
            # print(f"Mean: {param.mean().item():.6f}")
            # print(f"Contains NaN: {torch.isnan(param).any()}")
            # print(f"Contains Inf: {torch.isinf(param).any()}")
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        # exit()

    def add_config(self):
        config_str = '\n'.join([f'{k}: {v}' for k, v in CONFIG.items()])
        self.writer.add_text('config', config_str, global_step=0)

    def move_tensor2numpy(self, y_true, y_pred):
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        return y_true, y_pred

    def close(self):
        self.executor.shutdown(wait=True)
        self.writer.close()
