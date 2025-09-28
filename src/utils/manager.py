import os
import signal

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from src.datasets.dataloader.collate_fn import collate_fn, collate_fn_pyg
from src.experiments.move_data_to_device import move_data_to_device
from src.load_config import CONFIG
from src.models import mymodel
from src.utils.metrics import metrics
from src.utils.prediction_saver import PredictionSaver

DATA_SOURCE = CONFIG['data']['source']
MODEL_NAME = CONFIG['model_name']
EXPERIMENT_NAME = CONFIG['experiment_name']
EPOCHS = CONFIG['train']['epochs']
DEVICE = torch.device(CONFIG['device'])


class Manager:
    def __init__(self, fold_id=None, model=None):
        self.activations = None
        self.model = model
        self.fold_id = fold_id
        self.fold_path = os.path.join('logs', DATA_SOURCE, MODEL_NAME, EXPERIMENT_NAME, str(fold_id))
        self.data_fold_path = os.path.join('src', 'data', DATA_SOURCE, str(fold_id))

        # 最佳模型追踪
        self.best_score = None
        self.best_epoch = 0
        self.best_model_path = None
        self.monitor_metric = 'combined'  # 可以是 'val_loss', 'val_acc', 'val_f1', 'combined'
        self.monitor_mode = 'max'  # 'min' for loss, 'max' for accuracy/f1

        if fold_id is not None:
            self.writer = SummaryWriter(self.fold_path)
            self.add_config()
            self.best_model_path = os.path.join(self.fold_path, 'best_model.pth')

            # 初始化预测结果储存器
            self.prediction_saver = PredictionSaver(fold_id)

    def set_best_model_tracking(self, monitor_metric='combined', monitor_mode='max'):
        """
        设置最佳模型追踪参数

        Args:
            monitor_metric (str): 监控的指标 ('val_loss', 'val_acc', 'val_f1', 'combined')
            monitor_mode (str): 'min' 表示指标越小越好，'max' 表示指标越大越好
        """
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode

        # 根据模式初始化最佳分数
        if monitor_mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')

    def manage_train(self, epoch, loss):
        self.add_loss(epoch, 'train', loss)
        self.add_model_parameters_histogram(epoch)

    def manage_valid(self, epoch, loss, y_true, y_pred, y_scores):
        y_true_np, y_pred_np = self.move_tensor2numpy(y_true, y_pred)
        self.add_loss(epoch, 'valid', loss)
        macro_precision, macro_recall, macro_f1, accuracy = self.add_metrics_by_y(epoch, 'valid', y_true_np, y_pred_np)

        # 检查是否需要保存最佳模型
        self._check_and_save_best_model(epoch, loss, accuracy, macro_f1)

        return loss, accuracy, macro_f1, macro_precision, macro_recall

    def _check_and_save_best_model(self, epoch, loss, accuracy, macro_f1):
        """
        检查当前模型是否是最佳模型，如果是则保存

        Args:
            epoch: 当前epoch
            loss: 验证损失
            accuracy: 验证准确率
            macro_f1: 验证macro F1
        """
        if self.best_model_path is None:
            return

        # 计算当前监控的指标值
        if self.monitor_metric == 'val_loss':
            current_score = loss
        elif self.monitor_metric == 'val_acc':
            current_score = accuracy
        elif self.monitor_metric == 'val_f1':
            current_score = macro_f1
        elif self.monitor_metric == 'combined':
            current_score = macro_f1 + accuracy
        else:
            current_score = macro_f1 + accuracy

        # 判断是否是最佳模型
        is_best = False
        if self.best_score is None:
            is_best = True
        elif self.monitor_mode == 'min' and current_score < self.best_score:
            is_best = True
        elif self.monitor_mode == 'max' and current_score > self.best_score:
            is_best = True

        if is_best:
            self.best_score = current_score
            self.best_epoch = epoch

            # 保存最佳模型
            if self.model is not None:
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"Best model saved at epoch {epoch}, {self.monitor_metric}: {current_score:.6f}")

    def manage_test(self, epoch, loss, y_true, y_pred, y_scores):
        y_true, y_pred = self.move_tensor2numpy(y_true, y_pred)
        self.add_loss(epoch, 'test', loss)
        self.add_metrics_by_y(epoch, 'test', y_true, y_pred)

    def test(self, dataloader, model=None, model_params_path=None, stage='test', save_predictions=True):
        from src.experiments.test import test

        if model is not None:
            # 直接使用传入的模型
            test_model = model
        elif model_params_path is not None:
            # 从指定路径加载模型
            model_params = torch.load(model_params_path, weights_only=True)
            test_model = mymodel.MyModel()
            test_model.load_state_dict(model_params)
            test_model.to(DEVICE)
        else:
            # 尝试加载最佳模型
            best_model_path = os.path.join(self.fold_path, 'best_model.pth')
            if os.path.exists(best_model_path):
                model_params = torch.load(best_model_path, weights_only=True)
                test_model = mymodel.MyModel()
                test_model.load_state_dict(model_params)
                test_model.to(DEVICE)
                model_params_path = best_model_path
            else:
                raise ValueError("No model provided and no best_model.pth found")

        loss, y_true, y_pred, y_scores = test(test_model, dataloader)

        # 保存原始张量用于储存
        y_true_tensor, y_pred_tensor, y_scores_tensor = y_true.clone(), y_pred.clone(), y_scores.clone()

        y_true, y_pred = self.move_tensor2numpy(y_true, y_pred)
        macro_precision, macro_recall, macro_f1, accuracy = metrics(y_true, y_pred)

        # 储存预测结果
        if save_predictions and hasattr(self, 'prediction_saver'):
            metrics_dict = {
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'loss': loss
            }

            # 尝试获取药物对信息
            drug_pairs = self._get_drug_pairs_from_dataloader(dataloader)

            additional_info = {
                'model_path': model_params_path if model_params_path else 'direct_model',
                'fold_id': self.fold_id
            }

            saved_file = self.prediction_saver.save_predictions(
                y_true_tensor, y_pred_tensor, y_scores_tensor,
                metrics_dict, drug_pairs=drug_pairs,
                additional_info=additional_info
            )

            if saved_file:
                print(f"预测结果已保存到: {saved_file}")

        return accuracy, macro_f1, macro_precision, macro_recall

    def add_metrics(self, epoch, stage, **metrics):
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'{metric_name}/{stage}', metric_value, epoch)

    def add_loss(self, epoch, stage, loss):
        self.writer.add_scalar(f'loss/{stage}', loss, epoch)

    def add_metrics_by_y(self, epoch, stage, y_true, y_pred):
        # 直接计算指标，移除多进程
        macro_precision, macro_recall, macro_f1, accuracy = metrics(y_true, y_pred)
        self.add_metrics(
            epoch,
            stage,
            accuracy=accuracy,
            macro_f1=macro_f1,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
        )
        # 只记录指标到TensorBoard，不再保存模型
        self.writer.add_text(
            f"metrics/{stage}",
            f"|{accuracy:.4f}|{macro_f1:.4f}|{macro_precision:.4f}|{macro_recall:.4f}|",
            epoch
        )

        return macro_precision, macro_recall, macro_f1, accuracy

    def wait_all_tasks(self):
        # 移除多进程，无需等待任务
        pass

    def graceful_exit(self, signum, frame):
        print("Received termination signal. Shutting down gracefully...")
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

        # 使用最佳模型
        best_model_path = os.path.join(self.fold_path, 'best_model.pth')
        if not os.path.exists(best_model_path):
            print(f"Warning: {best_model_path} not found, skipping embedding")
            return

        if is_pyg:
            model = mymodel.MyModelPYG()
            model.to(DEVICE)
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
            handle = model.mlp.register_forward_hook(forward_hook)

            data = collate_fn_pyg(sampled_df)
            data = data.to(DEVICE)
            labels = torch.LongTensor(data.y)
            model(data)
        else:
            model = mymodel.MyModel()
            model.to(DEVICE)
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
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
            global_step=0,  # 不再使用max_id
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

    def _get_drug_pairs_from_dataloader(self, dataloader):
        """尝试从dataloader中获取药物对信息"""
        try:
            # 获取dataset中的原始数据
            if hasattr(dataloader.dataset, 'data'):
                df = dataloader.dataset.data
                drug_pairs = [(row['smiles1'], row['smiles2']) for _, row in df.iterrows()]
                return drug_pairs
            elif hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'df'):
                df = dataloader.dataset.df
                drug_pairs = [(row['smiles1'], row['smiles2']) for _, row in df.iterrows()]
                return drug_pairs
        except Exception:
            pass
        return None

    def close(self):
        self.writer.close()
