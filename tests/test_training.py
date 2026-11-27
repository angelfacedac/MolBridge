"""回归检查：验证集选模、测试权重和独立重评。"""

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from main import main
from method.train import train_epoch


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class TrainingPipelineTest(unittest.TestCase):
    def test_best_checkpoint_and_independent_evaluation(self):
        """两轮产生相反预测，验证集选中第一轮；所有测试划分都应使用它。"""
        pairs = [("CO", "N", 0), ("C", "CC", 1), ("CN", "O", 0)]
        splits = {
            "train": pairs,
            "val": [("O", "CCC", 0)],
            "test": pairs,
            "s1": pairs[::-1],
            "s2": pairs[1:] + pairs[:1],
        }
        test_splits = ["test", "s1", "s2"]
        config = {
            "data_root": "data", "output_root": "runs", "device": "cpu", "dataset": "tiny",
            "datasets": {"tiny": {"num_classes": 2, "folds": [0], "test_splits": test_splits}},
            "model": {"hidden_dim": 8, "classifier_dim": 8, "ffn_dim": 16,
                      "num_layers": 1, "num_heads": 2, "dropout": 0},
            "training": {"seed": 42, "batch_size": 2, "num_workers": 0, "lr": 0.001,
                         "weight_decay": 0, "epochs": 2, "patience": 0},
        }
        epoch_classes = iter([0, 1])

        def train_with_distinct_predictions(model, loader, optimizer, device):
            loss = train_epoch(model, loader, optimizer, device)
            # 仅固定这两轮的输出类别，真实的验证指标、选模、保存和重载全部照常执行。
            head = model.classifier[-1]
            with torch.no_grad():
                head.weight.zero_()
                head.bias.zero_()
                head.bias[next(epoch_classes)] = 1
            return loss

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            fold = root / "data" / "tiny" / "0"
            fold.mkdir(parents=True)
            for name, rows in splits.items():
                with (fold / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["smile_1", "smile_2", "label"])
                    writer.writerows(rows)
            config_path = root / "config.yml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            command = ["main.py", "--config", str(config_path), "--device", "cpu"]

            with patch.object(sys, "argv", [*command, "--fold", "0"]), redirect_stdout(io.StringIO()):
                with patch("method.train.train_epoch", side_effect=train_with_distinct_predictions):
                    main()

            output, = (root / "runs").glob("*/fold-0")
            history = read_rows(output / "history.csv")
            self.assertEqual([float(row["val_macro_f1"]) for row in history], [1.0, 0.0])
            self.assertEqual(json.loads((output / "metrics.json").read_text(encoding="utf-8"))["epoch"], 1)
            with patch.object(sys, "argv", [*command, "--evaluate", str(output / "best.pt")]), redirect_stdout(io.StringIO()):
                main()

            replay, = output.glob("eval-*")
            self.assertEqual((replay / "metrics.json").read_bytes(), (output / "metrics.json").read_bytes())
            for name in test_splits:
                with self.subTest(split=name):
                    filename = f"{name}_predictions.csv"
                    rows = read_rows(output / filename)
                    observed = [(row["smile_1"], row["smile_2"], int(row["label"])) for row in rows]
                    self.assertEqual(observed, splits[name])
                    self.assertEqual([int(row["prediction"]) for row in rows], [0] * len(observed))
                    self.assertEqual((replay / filename).read_bytes(), (output / filename).read_bytes())
