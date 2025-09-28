import os
import json
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from typing import Optional, Dict, Any, Union, Tuple

from src.load_config import CONFIG


class PredictionSaver:
    """
    预测结果储存工具类
    用于保存DDI预测的详细结果，包括预测标签、概率分布和性能指标
    """

    def __init__(self, fold_id: int, stage: str = 'test'):
        """
        初始化预测结果储存器

        Args:
            fold_id: 交叉验证折数ID
            stage: 数据集阶段 ('train', 'val', 'test')
        """
        self.fold_id = fold_id
        self.stage = stage
        self.data_source = CONFIG['data']['source']
        self.model_name = CONFIG['model_name']
        self.experiment_name = CONFIG['experiment_name']
        self.num_classes = CONFIG['data']['num_classes']

        # 创建储存目录
        self.base_path = os.path.join(
            'logs', self.data_source, self.model_name,
            self.experiment_name, str(fold_id), 'predictions'
        )
        os.makedirs(self.base_path, exist_ok=True)

        # 获取储存配置
        self.save_config = CONFIG.get('save_predictions', {})
        self.save_enabled = self.save_config.get('enabled', True)
        self.save_detailed = self.save_config.get('save_detailed', True)
        self.save_summary = self.save_config.get('save_summary', True)

    def save_predictions(
        self,
        y_true: Union[torch.Tensor, np.ndarray],
        y_pred: Union[torch.Tensor, np.ndarray],
        y_scores: Union[torch.Tensor, np.ndarray],
        metrics: Dict[str, float],
        sample_ids: Optional[list] = None,
        drug_pairs: Optional[list] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        保存预测结果到文件

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_scores: 预测概率分布
            metrics: 性能指标字典
            sample_ids: 样本ID列表
            drug_pairs: 药物对信息列表
            additional_info: 额外信息字典

        Returns:
            保存的文件路径，如果未启用储存则返回None
        """
        if not self.save_enabled:
            return None

        # 转换tensor到numpy
        y_true = self._tensor_to_numpy(y_true)
        y_pred = self._tensor_to_numpy(y_pred)
        y_scores = self._tensor_to_numpy(y_scores)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存详细预测结果
        predictions_file = None
        if self.save_detailed:
            predictions_file = self._save_detailed_predictions(
                y_true, y_pred, y_scores, timestamp,
                sample_ids, drug_pairs
            )

        # 保存预测摘要
        if self.save_summary:
            self._save_prediction_summary(
                y_true, y_pred, y_scores, metrics,
                timestamp, additional_info
            )

        return predictions_file

    def _save_detailed_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        timestamp: str,
        sample_ids: Optional[list],
        drug_pairs: Optional[list]
    ) -> str:
        """保存详细的预测结果到CSV文件"""

        # 确保所有数组都是1维的
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # 构建基础数据
        data = {
            'sample_id': sample_ids if sample_ids is not None else list(range(len(y_true))),
            'true_label': y_true.astype(int),
            'predicted_label': y_pred.astype(int),
            'prediction_correct': (y_true == y_pred).astype(int)
        }

        # 处理二分类和多分类的confidence计算
        if self.num_classes == 1:
            # 二分类模式：y_scores是1D数组
            y_scores = y_scores.flatten()
            sigmoid_probs = 1.0 / (1.0 + np.exp(-y_scores))  # sigmoid
            data['confidence'] = np.maximum(sigmoid_probs, 1.0 - sigmoid_probs)  # 最大概率作为confidence
            data['prob_class_0'] = 1.0 - sigmoid_probs
            data['prob_class_1'] = sigmoid_probs
        else:
            # 多分类模式：y_scores是2D数组
            if y_scores.ndim == 1:
                y_scores = y_scores.reshape(-1, 1)
            data['confidence'] = np.max(y_scores, axis=1)
            # 添加每个类别的概率
            for i in range(self.num_classes):
                data[f'prob_class_{i}'] = y_scores[:, i]

        # 添加药物对信息
        if drug_pairs is not None:
            if isinstance(drug_pairs[0], (list, tuple)) and len(drug_pairs[0]) >= 2:
                data['drug1'] = [pair[0] for pair in drug_pairs]
                data['drug2'] = [pair[1] for pair in drug_pairs]
            else:
                data['drug_pair'] = drug_pairs

        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        filename = f'{self.stage}_predictions_{timestamp}.csv'
        filepath = os.path.join(self.base_path, filename)
        df.to_csv(filepath, index=False)

        return filepath

    def _save_prediction_summary(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        metrics: Dict[str, float],
        timestamp: str,
        additional_info: Optional[Dict[str, Any]]
    ):
        """保存预测摘要到JSON文件"""

        # 确保所有数组都是1维的
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        y_scores = y_scores.flatten()

        # 计算类别分布
        if self.num_classes == 1:
            # 二分类模式：扩展到2类进行统计
            minlength = 2
        else:
            # 多分类模式
            minlength = self.num_classes

        true_distribution = {int(i): int(count) for i, count in enumerate(np.bincount(y_true.astype(int), minlength=minlength))}
        pred_distribution = {int(i): int(count) for i, count in enumerate(np.bincount(y_pred.astype(int), minlength=minlength))}

        # 构建摘要数据
        summary = {
            'metadata': {
                'timestamp': timestamp,
                'fold_id': self.fold_id,
                'stage': self.stage,
                'data_source': self.data_source,
                'model_name': self.model_name,
                'experiment_name': self.experiment_name,
                'num_samples': int(len(y_true)),
                'num_classes': self.num_classes
            },
            'metrics': metrics,
            'class_distribution': {
                'true_labels': true_distribution,
                'predicted_labels': pred_distribution
            },
            'statistics': {
                'correct_predictions': int(np.sum(y_true == y_pred)),
                'accuracy': float(np.mean(y_true == y_pred)),
                'mean_confidence': float(np.mean(np.abs(y_scores))),
                'std_confidence': float(np.std(np.abs(y_scores)))
            },
            'config': {
                'model_config': CONFIG.get('model', {}),
                'train_config': CONFIG.get('train', {}),
                'data_config': CONFIG.get('data', {})
            }
        }

        # 添加额外信息
        if additional_info:
            summary['additional_info'] = additional_info

        # 保存到JSON文件
        filename = f'{self.stage}_summary_{timestamp}.json'
        filepath = os.path.join(self.base_path, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def _tensor_to_numpy(self, data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """将tensor转换为numpy数组"""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data

    def load_predictions(self, filename: str) -> pd.DataFrame:
        """
        加载之前保存的预测结果

        Args:
            filename: 预测结果文件名

        Returns:
            预测结果DataFrame
        """
        filepath = os.path.join(self.base_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"预测结果文件不存在: {filepath}")

        return pd.read_csv(filepath)

    def load_summary(self, filename: str) -> Dict[str, Any]:
        """
        加载预测摘要

        Args:
            filename: 摘要文件名

        Returns:
            摘要信息字典
        """
        filepath = os.path.join(self.base_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"摘要文件不存在: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_saved_predictions(self) -> Tuple[list, list]:
        """
        列出所有已保存的预测文件

        Returns:
            (预测结果文件列表, 摘要文件列表)
        """
        if not os.path.exists(self.base_path):
            return [], []

        files = os.listdir(self.base_path)
        prediction_files = [f for f in files if f.endswith('.csv')]
        summary_files = [f for f in files if f.endswith('.json')]

        return sorted(prediction_files), sorted(summary_files)