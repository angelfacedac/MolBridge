"""读入药对 CSV，把两个完整分子组装成一张联合图。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import NamedTuple, Sequence

import torch
from rdkit import Chem
from torch.utils.data import Dataset


# 顺序是论文所用 75 维原子特征的一部分，改变顺序会改变模型输入。
ELEMENTS = (
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca",
    "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag",
    "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd",
    "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown",
)
HYBRIDIZATIONS = (
    Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2,
)
ATOM_DIM = 75


def _one_hot(value, choices: Sequence) -> list[bool]:
    """沿用原编码表约定，未列出的取值落入最后一栏。"""
    if value not in choices:
        value = choices[-1]
    return [value == choice for choice in choices]


def _atom_features(atom: Chem.Atom) -> list[int]:
    degree = atom.GetDegree()
    if degree > 10:
        raise ValueError(f"原子序号 {atom.GetIdx()} 的邻居数 {degree} 超出编码范围 0–10")
    if hasattr(Chem, "ValenceType"):
        valence = atom.GetValence(Chem.ValenceType.IMPLICIT)
    else:
        valence = atom.GetImplicitValence()

    return [
        *_one_hot(atom.GetSymbol(), ELEMENTS),
        *_one_hot(degree, range(11)),
        *_one_hot(valence, range(7)),
        atom.GetFormalCharge(), atom.GetNumRadicalElectrons(),
        *_one_hot(atom.GetHybridization(), HYBRIDIZATIONS),
        atom.GetIsAromatic(),
        *_one_hot(atom.GetTotalNumHs(), range(5)),
    ]


class MoleculeGraph(NamedTuple):
    features: torch.Tensor
    bonds: torch.Tensor


def molecule_graph(smiles: str) -> MoleculeGraph:
    """生成全部原子的特征和带自环的化学键图。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("无法解析为含原子的分子")
    # 与公开代码保持相同的规范化、编号顺序，便于用旧权重核对编码。
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
    if mol is None:
        raise ValueError("规范化后的分子无法解析")
    mol = Chem.RenumberAtoms(mol, list(Chem.CanonicalRankAtoms(mol)))
    features = torch.tensor([_atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    bonds = torch.tensor(Chem.GetAdjacencyMatrix(mol), dtype=torch.float32)
    bonds.fill_diagonal_(1)
    return MoleculeGraph(features, bonds)


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
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """A、B 的原子紧邻排列，在每个药对末尾补齐。padding_mask 的 True 表示填充位。"""
    width = max(len(first.features) + len(second.features) for first, second, _ in samples)
    batch_size = len(samples)
    features = torch.zeros(batch_size, width, ATOM_DIM, dtype=torch.float32)
    bonds = torch.zeros(batch_size, width, width, dtype=torch.float32)
    padding_mask = torch.ones(batch_size, width, dtype=torch.bool)
    targets = torch.empty(batch_size, dtype=torch.long)
    for index, (first, second, label) in enumerate(samples):
        start = 0
        for graph in (first, second):
            end = start + len(graph.features)
            features[index, start:end] = graph.features
            # 两块对角子矩阵各自包含化学键和自环，跨分子权重由模型学习。
            bonds[index, start:end, start:end] = graph.bonds
            start = end
        padding_mask[index, :start] = False
        targets[index] = label
    return features, bonds, padding_mask, targets
