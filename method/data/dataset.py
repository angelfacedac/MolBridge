"""读取药对 CSV，缓存单分子图，并将两列药物分别打包。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import NamedTuple

import torch
from torch.utils.data import DataLoader, Dataset

from .molecule import MoleculeBatch, MoleculeGraph, molecule_graph


class DrugPair(NamedTuple):
    smile_1: str
    smile_2: str
    label: int


class DrugPairDataset(Dataset):
    """records 与 CSV 行序一致。同一数据集的各个划分可以共享分子缓存。"""

    def __init__(
        self,
        path: str | Path,
        molecule_cache: dict[str, MoleculeGraph] | None = None,
        *,
        num_classes: int | None = None,
    ):
        if num_classes is not None and num_classes < 1:
            raise ValueError("num_classes 必须大于零")
        # 单输出二分类同样接受 0 和 1 两个标签。
        label_count = 2 if num_classes == 1 else num_classes
        self.records: list[DrugPair] = []
        self.molecule_cache = {} if molecule_cache is None else molecule_cache
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"smile_1", "smile_2", "label"}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} 缺少列：{', '.join(sorted(missing))}")

            for row in reader:
                location = f"{path}:{reader.line_num}"
                if None in row:
                    raise ValueError(f"{location} 列数超过表头，请检查 CSV 格式")
                try:
                    label = int(row["label"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{location} 标签必须是非负整数，实际为 {row['label']!r}") from exc
                if label < 0 or (label_count is not None and label >= label_count):
                    raise ValueError(f"{location} 标签 {label} 超出有效类别范围")
                first = (row["smile_1"] or "").strip()
                second = (row["smile_2"] or "").strip()
                for smiles in (first, second):
                    if smiles not in self.molecule_cache:
                        try:
                            self.molecule_cache[smiles] = molecule_graph(smiles)
                        except (ValueError, RuntimeError) as exc:
                            raise ValueError(f"{location} 无效 SMILES {smiles!r}：{exc}") from exc
                self.records.append(DrugPair(first, second, label))
        if not self.records:
            raise ValueError(f"{path} 没有药对记录")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[MoleculeGraph, MoleculeGraph, int]:
        first, second, label = self.records[index]
        return self.molecule_cache[first], self.molecule_cache[second], label


def collate_pairs(samples: list[tuple[MoleculeGraph, MoleculeGraph, int]]) -> tuple[
    MoleculeBatch, MoleculeBatch, torch.Tensor
]:
    """两列分子分别打包。两列的同一行组成一个药对，对应同一行标签。"""
    first, second, labels = zip(*samples)
    return MoleculeBatch.from_graphs(first), MoleculeBatch.from_graphs(second), torch.tensor(labels)


def make_loader(path, num_classes, cache, settings, *, shuffle=False):
    dataset = DrugPairDataset(path, cache, num_classes=num_classes)
    return DataLoader(dataset, batch_size=settings["batch_size"], shuffle=shuffle,
                      num_workers=settings["num_workers"], collate_fn=collate_pairs)
