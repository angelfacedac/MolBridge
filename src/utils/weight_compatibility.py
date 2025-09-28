"""
权重兼容性检查和加载工具
用于确保重构过程中不破坏已训练模型的权重加载
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from datetime import datetime


class WeightCompatibilityChecker:
    """权重兼容性检查器"""

    def __init__(self):
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'checks': []
        }

    def check_compatibility(self,
                          model_weights_path: str,
                          reference_model: nn.Module,
                          model_name: str = "unknown") -> Dict[str, Any]:
        """
        检查权重文件与参考模型的兼容性

        Args:
            model_weights_path: 权重文件路径
            reference_model: 参考模型实例
            model_name: 模型名称（用于报告）

        Returns:
            兼容性检查报告
        """
        check_result = {
            'model_name': model_name,
            'weights_path': model_weights_path,
            'compatible': True,
            'issues': [],
            'details': {}
        }

        try:
            # 加载权重文件
            if not os.path.exists(model_weights_path):
                check_result['compatible'] = False
                check_result['issues'].append(f"权重文件不存在: {model_weights_path}")
                return check_result

            saved_state = torch.load(model_weights_path, map_location='cpu', weights_only=True)
            model_state = reference_model.state_dict()

            # 检查参数名称
            saved_keys = set(saved_state.keys())
            model_keys = set(model_state.keys())

            missing_keys = saved_keys - model_keys
            extra_keys = model_keys - saved_keys

            if missing_keys:
                check_result['compatible'] = False
                check_result['issues'].append(f"模型中缺少参数: {list(missing_keys)}")
                check_result['details']['missing_keys'] = list(missing_keys)

            if extra_keys:
                check_result['compatible'] = False
                check_result['issues'].append(f"模型中多余参数: {list(extra_keys)}")
                check_result['details']['extra_keys'] = list(extra_keys)

            # 检查参数形状
            shape_mismatches = []
            for key in saved_keys.intersection(model_keys):
                saved_shape = saved_state[key].shape
                model_shape = model_state[key].shape

                if saved_shape != model_shape:
                    shape_mismatches.append({
                        'parameter': key,
                        'saved_shape': list(saved_shape),
                        'model_shape': list(model_shape)
                    })

            if shape_mismatches:
                check_result['compatible'] = False
                check_result['issues'].append("参数形状不匹配")
                check_result['details']['shape_mismatches'] = shape_mismatches

            # 统计信息
            check_result['details']['total_parameters'] = len(model_keys)
            check_result['details']['matched_parameters'] = len(saved_keys.intersection(model_keys))

        except Exception as e:
            check_result['compatible'] = False
            check_result['issues'].append(f"检查过程中出错: {str(e)}")

        self.report['checks'].append(check_result)
        return check_result

    def save_report(self, output_path: str = "weight_compatibility_report.json"):
        """保存兼容性检查报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        print(f"兼容性检查报告已保存到: {output_path}")

    def print_summary(self):
        """打印检查摘要"""
        total_checks = len(self.report['checks'])
        compatible_count = sum(1 for check in self.report['checks'] if check['compatible'])

        print(f"\n{'='*60}")
        print(f"权重兼容性检查摘要")
        print(f"{'='*60}")
        print(f"总检查数: {total_checks}")
        print(f"兼容数量: {compatible_count}")
        print(f"不兼容数量: {total_checks - compatible_count}")
        print(f"兼容率: {compatible_count/total_checks*100:.1f}%" if total_checks > 0 else "无检查项目")

        for check in self.report['checks']:
            status = "✅ 兼容" if check['compatible'] else "❌ 不兼容"
            print(f"\n{check['model_name']}: {status}")
            if not check['compatible']:
                for issue in check['issues']:
                    print(f"  - {issue}")


class WeightCompatibilityLoader:
    """权重兼容性加载器"""

    @staticmethod
    def safe_load_weights(model: nn.Module,
                         weights_path: str,
                         strict: bool = True,
                         device: str = 'cpu') -> Tuple[bool, List[str]]:
        """
        安全加载权重，提供详细的错误信息

        Args:
            model: 目标模型
            weights_path: 权重文件路径
            strict: 是否严格匹配
            device: 加载设备

        Returns:
            (成功标志, 错误信息列表)
        """
        errors = []

        try:
            if not os.path.exists(weights_path):
                errors.append(f"权重文件不存在: {weights_path}")
                return False, errors

            # 加载权重
            saved_state = torch.load(weights_path, map_location=device, weights_only=True)

            # 尝试加载到模型
            missing_keys, unexpected_keys = model.load_state_dict(saved_state, strict=strict)

            if missing_keys:
                errors.append(f"缺少键: {missing_keys}")

            if unexpected_keys:
                errors.append(f"意外键: {unexpected_keys}")

            if strict and (missing_keys or unexpected_keys):
                return False, errors

            return True, errors

        except Exception as e:
            errors.append(f"加载过程中出错: {str(e)}")
            return False, errors

    @staticmethod
    def create_parameter_mapping(old_model_state: Dict[str, torch.Tensor],
                               new_model_state: Dict[str, torch.Tensor]) -> Dict[str, str]:
        """
        创建参数名称映射表（用于参数重命名的情况）

        Args:
            old_model_state: 旧模型状态字典
            new_model_state: 新模型状态字典

        Returns:
            参数名称映射字典 {old_name: new_name}
        """
        mapping = {}

        old_keys = list(old_model_state.keys())
        new_keys = list(new_model_state.keys())

        # 简单的形状匹配策略
        for old_key in old_keys:
            old_shape = old_model_state[old_key].shape

            # 寻找形状匹配的新参数
            candidates = [new_key for new_key in new_keys
                         if new_model_state[new_key].shape == old_shape]

            if len(candidates) == 1:
                mapping[old_key] = candidates[0]
                new_keys.remove(candidates[0])  # 避免重复映射

        return mapping

    @staticmethod
    def apply_parameter_mapping(state_dict: Dict[str, torch.Tensor],
                              mapping: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """
        应用参数名称映射

        Args:
            state_dict: 原始状态字典
            mapping: 参数名称映射

        Returns:
            重命名后的状态字典
        """
        new_state_dict = {}

        for old_name, tensor in state_dict.items():
            new_name = mapping.get(old_name, old_name)
            new_state_dict[new_name] = tensor

        return new_state_dict


def scan_for_model_weights(directory: str, pattern: str = "*.pth") -> List[str]:
    """
    扫描目录中的模型权重文件

    Args:
        directory: 搜索目录
        pattern: 文件模式

    Returns:
        权重文件路径列表
    """
    weight_files = []

    for path in Path(directory).rglob(pattern):
        if path.is_file():
            weight_files.append(str(path))

    return sorted(weight_files)


def check_all_saved_models(model_factory_func, base_directory: str = "logs") -> WeightCompatibilityChecker:
    """
    检查所有已保存模型的兼容性

    Args:
        model_factory_func: 创建模型的工厂函数
        base_directory: 日志基础目录

    Returns:
        兼容性检查器实例
    """
    checker = WeightCompatibilityChecker()

    # 创建参考模型
    reference_model = model_factory_func()

    # 扫描权重文件
    weight_files = scan_for_model_weights(base_directory)

    print(f"找到 {len(weight_files)} 个权重文件，开始兼容性检查...")

    for weight_file in weight_files:
        model_name = os.path.basename(weight_file)
        print(f"检查: {model_name}")
        checker.check_compatibility(weight_file, reference_model, model_name)

    return checker


if __name__ == "__main__":
    # 示例用法
    print("权重兼容性检查工具 - 使用示例")
    print("请在主程序中调用相关函数进行检查")