import numpy as np


class EarlyStopping:
    """早停工具类，监控验证指标并在性能停止改善时停止训练"""

    def __init__(self, patience=7, min_delta=0, monitor='val_loss', mode='min', verbose=True):
        """
        Args:
            patience (int): 在没有改善的情况下等待的epoch数
            min_delta (float): 被认为是改善的最小变化量
            monitor (str): 监控的指标名称 ('val_loss', 'val_acc', 'val_f1')
            mode (str): 'min' 表示指标越小越好，'max' 表示指标越大越好
            verbose (bool): 是否打印早停信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose

        self.wait = 0
        self.stopped_epoch = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            self.best = -np.inf
            self.min_delta *= 1

    def __call__(self, current_value):
        """
        检查是否应该早停

        Args:
            current_value: 当前监控指标的值

        Returns:
            bool: True表示应该停止训练，False表示继续训练
        """
        if self.monitor_op(current_value - self.min_delta, self.best):
            self.best = current_value
            self.wait = 0

            if self.verbose:
                print(f'EarlyStopping: {self.monitor} improved to {current_value:.6f}')
        else:
            self.wait += 1
            if self.verbose:
                print(f'EarlyStopping: {self.monitor} did not improve from {self.best:.6f}, patience: {self.wait}/{self.patience}')

            if self.wait >= self.patience:
                self.stopped_epoch = True
                if self.verbose:
                    print(f'EarlyStopping: Stopping training. Best {self.monitor}: {self.best:.6f}')
                return True

        return False

    def get_best_score(self):
        """获取最佳分数"""
        return self.best