"""按论文使用的字段顺序提取 75 维原子特征。"""

from typing import Sequence

from rdkit import Chem


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


def _one_hot(value, choices: Sequence) -> list[bool]:
    """沿用原编码表约定，未列出的取值落入最后一栏。"""
    if value not in choices:
        value = choices[-1]
    return [value == choice for choice in choices]


def atom_features(atom: Chem.Atom) -> list[int]:
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
