from rdkit import Chem


def norm_smile(smile):
    molecule = Chem.MolFromSmiles(smile)
    # 规范化 SMILES
    canonical_smile = Chem.MolToSmiles(molecule, canonical=True)
    return canonical_smile