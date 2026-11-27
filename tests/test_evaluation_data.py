"""重评应使用原运行的测试集，迁移到新位置时仍需校验其内容。"""

import hashlib
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

import main
from method.model import MolBridge


class EvaluationDataTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "runs" / "example"
        self.fold = self.run / "fold-0"
        self.fold.mkdir(parents=True)
        self.checkpoint = self.fold / "best.pt"
        args = dict(num_classes=2, hidden_dim=4, ffn_dim=8, num_layers=1, num_heads=1,
                    dropout=0.0, classifier_dim=4)
        torch.save(dict(weights=MolBridge(**args).state_dict(), model_args=args, epoch=2,
                        dataset="Fixture", fold=0, test_splits=["s1", "s2"], batch_size=4,
                        num_workers=0), self.checkpoint)
        self.data_root = self.root / "original"
        self.folder = self.data_root / "Fixture" / "0"
        self.folder.mkdir(parents=True)
        self.hashes = {}
        for split, label in [("s1", 0), ("s2", 1)]:
            content = f"smile_1,smile_2,label\nCO,N,{label}\n".encode()
            (self.folder / f"{split}.csv").write_bytes(content)
            self.hashes[split] = hashlib.sha256(content).hexdigest()
        (self.fold / "split_hashes.json").write_text(json.dumps(self.hashes))
        (self.run / "run.json").write_text(json.dumps({"data_root": str(self.data_root)}))

    def run_cli(self, *extra):
        args = ["main.py", "--evaluate", str(self.checkpoint), "--device", "cpu", *extra]
        with patch.object(sys, "argv", args), redirect_stdout(StringIO()):
            main.main()

    def test_default_uses_original_data_root(self):
        self.run_cli()
        output = next(self.fold.glob("eval-*/metrics.json"))
        self.assertEqual(json.loads(output.read_text())["split_sha256"], self.hashes)

    def test_changed_later_split_fails_before_writing_outputs(self):
        with (self.folder / "s2.csv").open("a") as stream:
            stream.write("N,O,0\n")
        with self.assertRaisesRegex(ValueError, "s2.csv.*SHA-256"):
            self.run_cli()
        self.assertEqual(list(self.fold.glob("eval-*")), [])

    def test_explicit_config_relocates_data_and_supports_detached_checkpoint(self):
        self.data_root.rename(self.root / "moved")
        config = self.root / "moved.yml"
        config.write_text("data_root: moved\ndevice: cpu\n")
        error = StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit):
            self.run_cli()
        self.assertIn("--config", error.getvalue())
        self.run_cli("--config", str(config))
        self.assertEqual(json.loads(next(self.fold.glob("eval-*/metrics.json")).read_text())
                         ["split_sha256"], self.hashes)

        # 单独拷走的权重也能按指定目录评估，并记录这次实际读取的文件哈希。
        detached = self.root / "detached" / "best.pt"
        detached.parent.mkdir()
        self.checkpoint.rename(detached)
        self.checkpoint = detached
        self.run_cli("--config", str(config))
        self.assertEqual(json.loads(next(detached.parent.glob("eval-*/metrics.json")).read_text())
                         ["split_sha256"], self.hashes)
