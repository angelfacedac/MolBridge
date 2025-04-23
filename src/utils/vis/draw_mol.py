from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D


def draw_molecule_with_atom_indices(smiles, atom_indices):
    molecule = Chem.MolFromSmiles(smiles)
    # 规范化 SMILES
    canonical_smile = Chem.MolToSmiles(molecule, canonical=True)
    # print(smile, canonical_smile)
    molecule = Chem.MolFromSmiles(canonical_smile)
    # 固定原子顺序
    atom_order = list(Chem.CanonicalRankAtoms(molecule))
    mol = Chem.RenumberAtoms(molecule, atom_order)

    # 创建一个绘图对象
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)
    drawer.DrawMolecule(mol, highlightAtoms=atom_indices)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:', '')

    # 保存SVG文件
    with open(f"{smiles}_molecule.svg", "w") as f:
        f.write(svg)


def draw_combined_molecule_with_atom_indices(smiles1, smiles2, atoms1, atoms2, edge,
                                             color=(0.9, 0, 0),
                                             offset=10.0,
                                             output_file="combined_molecule.svg"):
    molecule = Chem.MolFromSmiles(smiles1)
    # 规范化 SMILES
    canonical_smile = Chem.MolToSmiles(molecule, canonical=True)
    # print(smile, canonical_smile)
    molecule = Chem.MolFromSmiles(canonical_smile)
    # 固定原子顺序
    atom_order = list(Chem.CanonicalRankAtoms(molecule))
    mol1 = Chem.RenumberAtoms(molecule, atom_order)

    molecule = Chem.MolFromSmiles(smiles2)
    # 规范化 SMILES
    canonical_smile = Chem.MolToSmiles(molecule, canonical=True)
    # print(smile, canonical_smile)
    molecule = Chem.MolFromSmiles(canonical_smile)
    # 固定原子顺序
    atom_order = list(Chem.CanonicalRankAtoms(molecule))
    mol2 = Chem.RenumberAtoms(molecule, atom_order)

    # 计算2D坐标
    AllChem.Compute2DCoords(mol1)
    AllChem.Compute2DCoords(mol2)

    combined_mol = Chem.CombineMols(mol1, mol2)

    # 计算原子索引偏移量
    num_atoms_mol1 = mol1.GetNumAtoms()

    # 调整drug2的原子索引
    adjusted_drug2_atoms = [atom_index + num_atoms_mol1 for atom_index in atoms2]

    # 合并原子索引
    combined_atoms = atoms1 + adjusted_drug2_atoms

    highlight_edges = [(i, j) for _, i, j in edge]
    for k, (i, j) in enumerate(highlight_edges):
        if i >= 50 :
            i -= 50 - num_atoms_mol1
        if j >= 50 :
            j -= 50 - num_atoms_mol1
        highlight_edges[k] = (i, j)

    edge_color = {}
    for k, (i, j) in enumerate(highlight_edges):
        w = edge[k][0]
        color_k = (w*color[0], w*color[1], w*color[2])
        edge_color[(i, j)] = color_k
        edge_color[(j, i)] = color_k

    # 动态添加边 todo
    editable_mol = Chem.EditableMol(combined_mol)
    add_bond = []
    for i, j in highlight_edges:
        if i==j: continue
        if (i, j) in add_bond or (j, i) in add_bond: continue
        if not combined_mol.GetBondBetweenAtoms(i, j):
            editable_mol.AddBond(i, j, order=Chem.BondType.SINGLE)
            add_bond.append((i, j))
    combined_mol_with_edges = editable_mol.GetMol()

    # 计算边的索引
    highlight_bonds = []
    for i, j in highlight_edges:
        bond = combined_mol_with_edges.GetBondBetweenAtoms(i, j)
        if bond:
            highlight_bonds.append(bond.GetIdx())

    for k, (i, j) in enumerate(add_bond):
        add_bond[k] = combined_mol_with_edges.GetBondBetweenAtoms(i, j).GetIdx()

    # 计算合并后分子的2D坐标 todo: combined_mol_with_edges
    AllChem.Compute2DCoords(combined_mol_with_edges)

    # 调整第二个分子的位置 todo: combined_mol_with_edges
    conf = combined_mol_with_edges.GetConformer()
    conf1 = mol1.GetConformer()
    conf2 = mol2.GetConformer()
    offset = offset  # 调整偏移量以避免重叠
    for atom_idx in range(num_atoms_mol1):
        pos = conf1.GetAtomPosition(atom_idx)
        conf.SetAtomPosition(atom_idx, (pos.x, pos.y, pos.z))
    for atom_idx in range(num_atoms_mol1, combined_mol_with_edges.GetNumAtoms()): # todo: combined_mol_with_edges
        pos = conf2.GetAtomPosition(atom_idx - num_atoms_mol1)
        conf.SetAtomPosition(atom_idx, (pos.x, pos.y + offset, pos.z))

    # 创建一个绘图对象
    drawer = rdMolDraw2D.MolDraw2DSVG(800, 400)  # 增加宽度以适应两个分子
    drawer.SetLineWidth(0.5)  # 设置边的粗细为2.0
    # 设置高亮原子和边
    highlight_atom_color = {atom_idx: color for atom_idx in combined_atoms}  # 红色
    highlight_bond_color = {bond_idx: edge_color[
        (combined_mol_with_edges.GetBondWithIdx(bond_idx).GetBeginAtomIdx(),
         combined_mol_with_edges.GetBondWithIdx(bond_idx).GetEndAtomIdx())
    ] for bond_idx in highlight_bonds}  # 红色
    drawer.DrawMolecule(combined_mol_with_edges, highlightAtoms=combined_atoms, highlightAtomColors=highlight_atom_color, highlightBonds=highlight_bonds, highlightBondColors=highlight_bond_color)
    # drawer.DrawMolecule(combined_mol, highlightAtoms=combined_atoms, highlightAtomColors=highlight_atom_color)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:', '')

    # 添加晕染效果
    # def add_glow_effect(svg, atom_indices, color, conf, radius=1, num_circles=5, opacity=0.5):
    #     import xml.etree.ElementTree as ET
    #     from xml.dom import minidom
    #
    #     # 解析SVG
    #     root = ET.fromstring(svg)
    #     ns = {'svg': 'http://www.w3.org/2000/svg'}
    #
    #     # 获取原子位置
    #     atom_positions = {atom.GetIdx(): conf.GetAtomPosition(atom.GetIdx()) for atom in
    #                       combined_mol_with_edges.GetAtoms()}
    #
    #     # 添加晕染效果
    #     for atom_idx in atom_indices:
    #         pos = atom_positions[atom_idx]
    #         x, y = pos.x, pos.y
    #
    #         for i in range(num_circles):
    #             circle_radius = radius * (i + 1)
    #             circle_opacity = opacity * (1 - i / num_circles)
    #             circle_color = f"rgba({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)}, {circle_opacity})"
    #             circle = ET.Element('circle')
    #             circle.set('cx', str(x))
    #             circle.set('cy', str(y))
    #             circle.set('r', str(circle_radius))
    #             circle.set('fill', circle_color)
    #             root.append(circle)
    #
    #     # 格式化SVG
    #     rough_string = ET.tostring(root, 'utf-8')
    #     reparsed = minidom.parseString(rough_string)
    #     pretty_svg = reparsed.toprettyxml(indent="  ")
    #
    #     return pretty_svg
    #
    #     # 修改SVG内容以设置虚线边
    # def set_dashed_bonds(svg, highlight_bonds, conf):
    #     import xml.etree.ElementTree as ET
    #     from xml.dom import minidom
    #
    #     root = ET.fromstring(svg)
    #     ns = {'svg': 'http://www.w3.org/2000/svg'}
    #
    #     # 获取所有路径元素
    #     paths = root.findall('.//svg:path', ns)
    #
    #     # 找到高亮边的路径并设置为虚线
    #     for path in paths:
    #         d = path.get('d')
    #         if d:
    #             # 假设路径的d属性包含边的信息
    #             # 这里假设路径的d属性中包含高亮边的信息
    #             # 你可以根据实际情况调整这个逻辑
    #             for bond_idx in highlight_bonds:
    #                 bond = combined_mol_with_edges.GetBondWithIdx(bond_idx)
    #                 start_atom = bond.GetBeginAtomIdx()
    #                 end_atom = bond.GetEndAtomIdx()
    #                 start_pos = conf.GetAtomPosition(start_atom)
    #                 end_pos = conf.GetAtomPosition(end_atom)
    #                 start_x, start_y = start_pos.x, start_pos.y
    #                 end_x, end_y = end_pos.x, end_pos.y
    #
    #                 # 检查路径是否匹配边的起点和终点
    #                 if f"M {start_x} {start_y} L {end_x} {end_y}" in d or f"M {end_x} {end_y} L {start_x} {start_y}" in d:
    #                     path.set('stroke-dasharray', '4,2')  # 设置虚线样式
    #                     path.set('stroke', 'black')  # 设置边的颜色
    #
    #     # 格式化SVG
    #     rough_string = ET.tostring(root, 'utf-8')
    #     reparsed = minidom.parseString(rough_string)
    #     pretty_svg = reparsed.toprettyxml(indent="  ")
    #
    #     return pretty_svg
    #
    # svg = add_glow_effect(svg, combined_atoms, color, conf)
    # svg = set_dashed_bonds(svg, add_bond, conf)

    # 保存SVG文件
    with open(output_file, "w") as f:
        f.write(svg)


def draw_combined_molecule(smiles1, smiles2, atoms1, atoms2, edge,
                                             color=(0.9, 0, 0),
                                             offset=10.0,
                                             output_file="combined_molecule_atom.svg"):
    molecule = Chem.MolFromSmiles(smiles1)
    # 规范化 SMILES
    canonical_smile = Chem.MolToSmiles(molecule, canonical=True)
    # print(smile, canonical_smile)
    molecule = Chem.MolFromSmiles(canonical_smile)
    # 固定原子顺序
    atom_order = list(Chem.CanonicalRankAtoms(molecule))
    mol1 = Chem.RenumberAtoms(molecule, atom_order)

    molecule = Chem.MolFromSmiles(smiles2)
    # 规范化 SMILES
    canonical_smile = Chem.MolToSmiles(molecule, canonical=True)
    # print(smile, canonical_smile)
    molecule = Chem.MolFromSmiles(canonical_smile)
    # 固定原子顺序
    atom_order = list(Chem.CanonicalRankAtoms(molecule))
    mol2 = Chem.RenumberAtoms(molecule, atom_order)

    # 计算2D坐标
    AllChem.Compute2DCoords(mol1)
    AllChem.Compute2DCoords(mol2)

    combined_mol = Chem.CombineMols(mol1, mol2)

    # 计算原子索引偏移量
    num_atoms_mol1 = mol1.GetNumAtoms()

    # 调整drug2的原子索引
    adjusted_drug2_atoms = [atom_index + num_atoms_mol1 for atom_index in atoms2]

    # 合并原子索引
    combined_atoms = atoms1 + adjusted_drug2_atoms

    # 计算合并后分子的2D坐标 todo: combined_mol_with_edges
    AllChem.Compute2DCoords(combined_mol)

    # 调整第二个分子的位置 todo: combined_mol_with_edges
    conf = combined_mol.GetConformer()
    conf1 = mol1.GetConformer()
    conf2 = mol2.GetConformer()
    offset = offset  # 调整偏移量以避免重叠
    for atom_idx in range(num_atoms_mol1):
        pos = conf1.GetAtomPosition(atom_idx)
        conf.SetAtomPosition(atom_idx, (pos.x, pos.y, pos.z))
    for atom_idx in range(num_atoms_mol1, combined_mol.GetNumAtoms()): # todo: combined_mol_with_edges
        pos = conf2.GetAtomPosition(atom_idx - num_atoms_mol1)
        conf.SetAtomPosition(atom_idx, (pos.x, pos.y + offset, pos.z))

    # 创建一个绘图对象
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)  # 增加宽度以适应两个分子
    drawer.SetLineWidth(0.5)  # 设置边的粗细为2.0
    # 设置高亮原子和边
    highlight_atom_color = {atom_idx: color for atom_idx in combined_atoms}  # 红色
    highlight_bond_color = {}  # 不高亮任何边
    drawer.DrawMolecule(combined_mol, highlightAtoms=combined_atoms, highlightAtomColors=highlight_atom_color,
                        highlightBonds=[], highlightBondColors=highlight_bond_color)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:', '')

    # 保存SVG文件
    with open(output_file, "w") as f:
        f.write(svg)

    import cairosvg
    cairosvg.svg2eps(url='combined_molecule_atom.svg', write_to='combined_molecule_atom.eps')


def draw_molecule_by_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    # 规范化 SMILES
    canonical_smile = Chem.MolToSmiles(molecule, canonical=True)
    # print(smile, canonical_smile)
    molecule = Chem.MolFromSmiles(canonical_smile)
    # 固定原子顺序
    atom_order = list(Chem.CanonicalRankAtoms(molecule))
    mol = Chem.RenumberAtoms(molecule, atom_order)

    # 创建一个绘图对象
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:', '')

    # 保存SVG文件
    with open(f"molecule.svg", "w") as f:
        f.write(svg)

    import cairosvg
    cairosvg.svg2eps(url='molecule.svg', write_to='molecule.eps')



