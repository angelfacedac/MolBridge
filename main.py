"""MolBridge 训练及独立评估入口。"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from method.train import evaluate_checkpoint, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="配置文件；重评时可用它指定迁移后的数据目录")
    parser.add_argument("--dataset", help="训练数据集，如 Deng、Ryu 或 inductive")
    parser.add_argument("--fold", nargs="+", type=int, help="只训练指定折，如 --fold 0 1")
    parser.add_argument("--device", help="auto、cpu 或 cuda:0")
    parser.add_argument("--evaluate", type=Path, metavar="BEST_PT", help="从权重重算测试指标")
    args = parser.parse_args()
    config_path = args.config or Path(__file__).with_name("config.yml")
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    data_root = (config_path.parent / config["data_root"]).resolve()
    device_name = args.device or config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if args.evaluate:
        if args.dataset or args.fold is not None:
            parser.error("评估的数据集和折号由权重决定，请去掉 --dataset 和 --fold。")
        # 原运行记录保留了训练时使用的绝对数据路径。显式配置优先，用于跨机迁移。
        manifest = args.evaluate.resolve().parent.parent / "run.json"
        if args.config is None and manifest.is_file():
            data_root = Path(json.loads(manifest.read_text(encoding="utf-8"))["data_root"])
            if not data_root.is_dir():
                parser.error(f"原运行的数据目录不存在：{data_root}，请用 --config 指定新位置。")
        evaluate_checkpoint(args.evaluate, data_root, device)
        return

    dataset = args.dataset or config["dataset"]
    if dataset not in config["datasets"]:
        parser.error(f"未知数据集 {dataset}，可选 {list(config['datasets'])}")
    folds = args.fold if args.fold is not None else config["datasets"][dataset]["folds"]
    if len(set(folds)) != len(folds) or any(f not in config["datasets"][dataset]["folds"] for f in folds):
        parser.error("折号重复或超出当前数据集的范围。")
    output_root = (config_path.parent / config["output_root"]).resolve()
    train(config, dataset, folds, data_root, output_root, device)


if __name__ == "__main__":
    main()
