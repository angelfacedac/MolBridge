"""训练时只按验证集选权重，测试时从磁盘加载所选权重。"""

import csv
import hashlib
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import DrugPairDataset, collate_pairs
from .metrics import classification_metrics, write_predictions
from .model import MolBridge


def write_json(path, values):
    path.write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def split_hashes(folder, splits):
    hashes = {}
    for split in splits:
        path = folder / f"{split}.csv"
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def make_loader(path, num_classes, cache, settings, *, shuffle=False):
    dataset = DrugPairDataset(path, cache, num_classes=num_classes)
    return DataLoader(dataset, batch_size=settings["batch_size"], shuffle=shuffle,
                      num_workers=settings["num_workers"], collate_fn=collate_pairs)


def batch_loss(scores, labels):
    if scores.shape[1] == 1:
        return F.binary_cross_entropy_with_logits(scores[:, 0], labels.float())
    return F.cross_entropy(scores, labels)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        features, bonds, padding, labels = (tensor.to(device) for tensor in batch)
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(model(features, bonds, padding), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    predictions, labels = [], []
    total_loss = 0.0
    for batch in loader:
        features, bonds, padding, target = (tensor.to(device) for tensor in batch)
        scores = model(features, bonds, padding)
        total_loss += batch_loss(scores, target).item() * len(target)
        predicted = (scores[:, 0] > 0).long() if scores.shape[1] == 1 else scores.argmax(dim=1)
        predictions.extend(predicted.cpu().tolist())
        labels.extend(target.cpu().tolist())
    metrics = classification_metrics(labels, predictions)
    metrics["loss"] = total_loss / len(labels)
    return metrics, predictions


def evaluate_checkpoint(path, data_root, device, *, output_dir=None, cache=None):
    # 同一个入口用于训练结束时测试和日后重新评分，输出行序与原 CSV 对齐。
    saved = torch.load(path, map_location="cpu", weights_only=True)
    folder = data_root / saved["dataset"] / str(saved["fold"])
    hashes = split_hashes(folder, saved["test_splits"])
    # 先核验全部测试文件，避免较晚的划分核验失败时留下前一划分的输出。
    hash_file = path.parent / "split_hashes.json"
    if hash_file.is_file():
        expected = json.loads(hash_file.read_text(encoding="utf-8"))
        for split, digest in hashes.items():
            if digest != expected.get(split):
                raise ValueError(f"{folder / (split + '.csv')} 的 SHA-256 与 {hash_file} 记录不符。")
    model = MolBridge(**saved["model_args"])
    model.load_state_dict(saved["weights"])
    model.to(device)
    if output_dir is None:
        output_dir = Path(mkdtemp(prefix="eval-", dir=path.resolve().parent))
    if cache is None:
        cache = {}
    results = {}
    for split in saved["test_splits"]:
        data_path = folder / f"{split}.csv"
        loader = make_loader(data_path, saved["model_args"]["num_classes"], cache, saved)
        results[split], predictions = evaluate(model, loader, device)
        write_predictions(output_dir / f"{split}_predictions.csv", loader.dataset.records, predictions)
    write_json(output_dir / "metrics.json", {"epoch": saved["epoch"], "scores": results, "split_sha256": hashes})
    print(f"评估目录 {output_dir}\n{results}", flush=True)
    return results


def train_fold(config, dataset, fold, data_root, directory, device, cache):
    spec = config["datasets"][dataset]
    settings = config["training"]
    model_args = {**config["model"], "num_classes": spec["num_classes"]}
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    folder = data_root / dataset / str(fold)
    train_loader = make_loader(folder / "train.csv", spec["num_classes"], cache, settings, shuffle=True)
    valid_loader = make_loader(folder / "val.csv", spec["num_classes"], cache, settings)
    model = MolBridge(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    best_f1, best_epoch = -1.0, 0
    directory.mkdir()
    hashes = split_hashes(folder, ["train", "val", *spec["test_splits"]])
    write_json(directory / "split_hashes.json", hashes)
    with (directory / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1"])
        for epoch in range(1, settings["epochs"] + 1):
            loss = train_epoch(model, train_loader, optimizer, device)
            metrics, _ = evaluate(model, valid_loader, device)
            score = metrics["macro_f1"]
            writer.writerow([epoch, loss, metrics["loss"], metrics["accuracy"], score])
            stream.flush()
            print(f"fold={fold} epoch={epoch} train_loss={loss:.4f} val_macro_f1={score:.4f}", flush=True)
            if score > best_f1:
                best_f1, best_epoch = score, epoch
                saved = {
                    "weights": model.state_dict(), "model_args": model_args, "epoch": epoch,
                    "val_macro_f1": score, "dataset": dataset, "fold": fold,
                    "test_splits": spec["test_splits"], "batch_size": settings["batch_size"],
                    "num_workers": settings["num_workers"],
                }
                temporary = directory / "best.tmp"
                torch.save(saved, temporary)
                temporary.replace(directory / "best.pt")
            if settings["patience"] and epoch - best_epoch >= settings["patience"]:
                break
    # 释放训练模型和优化器占用的显存，再加载验证集选中的权重。
    del model, optimizer
    return evaluate_checkpoint(directory / "best.pt", data_root, device, output_dir=directory, cache=cache)


def code_revision():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(
            ["git", "--no-optional-locks", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def train(config, dataset, folds, data_root, output_root, device):
    settings = config["training"]
    if settings["epochs"] < 1 or settings["batch_size"] < 1 or settings["patience"] < 0:
        raise ValueError("epochs 和 batch_size 必须为正数，patience 必须为非负数。")
    output_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{dataset}-{datetime.now():%Y%m%d-%H%M%S}-"
    directory = Path(mkdtemp(prefix=prefix, dir=output_root))
    print(f"训练目录 {directory}", flush=True)
    write_json(directory / "run.json", {
        "config": config, "dataset": dataset, "folds": folds, "device": str(device),
        "data_root": str(data_root), "torch_version": str(torch.__version__), "code": code_revision(),
    })
    results, cache = {}, {}
    for fold in folds:
        results[fold] = train_fold(config, dataset, fold, data_root, directory / f"fold-{fold}", device, cache)
    means = {
        split: {metric: sum(row[split][metric] for row in results.values()) / len(results)
                for metric in first_fold}
        for split, first_fold in results[folds[0]].items()
    }
    write_json(directory / "summary.json", {"folds": results, "mean": means})
