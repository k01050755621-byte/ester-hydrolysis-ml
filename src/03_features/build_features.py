# -*- coding: utf-8 -*-
"""ML 补强 · 第 1 步：统一特征工厂（v4）

为 vlib_final.csv（12,380）+ 文献分子（dataset_v1.csv 中带标签者）构建完整特征矩阵。

特征分四组（便于消融）：
  A 基础块   16 个整分子描述符（与原 v1/v2/v3 一致）—— 用于复现"位置异构体失明"
  M 机理块   酯键微环境 / 取代基电子效应 / 空间位阻（本项目新增，可解释性主来源）
  F 指纹块   Morgan r=2, 1024 bit（全位，不裁剪）
  K 键块     MACCS keys, 167 bit

输出：
  feat_vlib.csv        12,380 × 特征
  feat_lit.csv         文献分子 × 特征
  labels_452.csv       合并标签（编号/SMILES/来源/标签）
  feature_manifest.json 特征分组与列名
"""
import os, json
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, Crippen, rdMolDescriptors, MACCSkeys
RDLogger.DisableLog('rdApp.*')

HOME = os.path.expanduser('~/polywork'); os.chdir(HOME)
OUT = 'fix_20260928/ml_final'
NBITS = 1024

# ---------------- SMARTS ----------------
P = {
    'ester':        Chem.MolFromSmarts('[#6][CX3](=O)[OX2][#6]'),          # C-C(=O)-O-C
    'acylAr':       Chem.MolFromSmarts('[c][CX3](=O)[OX2][#6]'),           # 苯甲酸型（酰基连芳环）
    'alkAr':        Chem.MolFromSmarts('[#6][CX3](=O)[OX2][c]'),           # 酚酯型（烷氧基连芳环）
    'arF':          Chem.MolFromSmarts('[c][F]'),
    'arCl':         Chem.MolFromSmarts('[c][Cl]'),
    'arBr':         Chem.MolFromSmarts('[c][Br]'),
    'arI':          Chem.MolFromSmarts('[c][I]'),
    'arMe':         Chem.MolFromSmarts('[c][CH3]'),
    'arOMe':        Chem.MolFromSmarts('[c][OX2][CH3]'),
    'arCF3':        Chem.MolFromSmarts('[c][CX4](F)(F)F'),
    'arNO2':        Chem.MolFromSmarts('[c][NX3+](=O)[O-]'),
    'arNO2b':       Chem.MolFromSmarts('[c][N+](=O)[O-]'),
    'arOH':         Chem.MolFromSmarts('[c][OX2H]'),
    'arNH2':        Chem.MolFromSmarts('[c][NX3;H2;!$(N[#6]=[O,N])]'),
    'arylC':        Chem.MolFromSmarts('[c]'),
    'conjCO':       Chem.MolFromSmarts('[c][CX3]=O'),                       # 羰基与芳环共轭
    'tert_C':       Chem.MolFromSmarts('[CX4]([#6])([#6])([#6])[#6]'),
    'ss':           Chem.MolFromSmarts('[#16X2][#16X2]'),                   # S-S
    'ss_ring':      Chem.MolFromSmarts('[#16X2;R][#16X2;R]'),
}

BASE_COLS = ['分子量','重原子数','可旋转键数','氢键供体','氢键受体','极性表面积','LogP',
             '芳环数','总环数','杂原子数','S原子数','N原子数','O原子数','C原子数',
             '分子里自由度','环状二硫']
MECH_COLS = ['n_ester','n_acylAr','n_alkAr','n_carbonyl','n_arylRing','n_aliRing',
             'n_arF','n_arCl','n_arBr','n_arI','n_arMe','n_arOMe','n_arCF3','n_arNO2',
             'n_arOH','n_arNH2','n_EWG','n_EDG','n_halogen','ar_subst_total','ar_subst_frac',
             'n_conjCO','n_tertC','avg_acyl_deg','avg_alkoxy_deg','longest_chain',
             'ss_flag','ss_ring_flag','ester_per_MW','halogen_per_ring']


def _longest_aliphatic_chain(m):
    """最长脂肪链（sp3 碳之间单键的最长路径，忽略环内）。"""
    adj = {}
    for b in m.GetBonds():
        if b.IsInRing() or b.GetBondTypeAsDouble() != 1.0:
            continue
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ai, aj = m.GetAtomWithIdx(i), m.GetAtomWithIdx(j)
        if ai.GetSymbol() == 'C' and aj.GetSymbol() == 'C' and not ai.GetIsAromatic() and not aj.GetIsAromatic():
            adj.setdefault(i, []).append(j); adj.setdefault(j, []).append(i)
    if not adj:
        return 0
    best = 1
    def dfs(u, seen):
        nonlocal best
        best = max(best, len(seen))
        for w in adj.get(u, []):
            if w not in seen:
                dfs(w, seen | {w})
    for s in adj:
        dfs(s, {s})
    return best


def feats(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    d = {}
    # --- A 基础块 ---
    d['分子量'] = Descriptors.MolWt(m)
    d['重原子数'] = m.GetNumHeavyAtoms()
    d['可旋转键数'] = Lipinski.NumRotatableBonds(m)
    d['氢键供体'] = Lipinski.NumHDonors(m)
    d['氢键受体'] = Lipinski.NumHAcceptors(m)
    d['极性表面积'] = rdMolDescriptors.CalcTPSA(m)
    d['LogP'] = Crippen.MolLogP(m)
    d['芳环数'] = rdMolDescriptors.CalcNumAromaticRings(m)
    d['总环数'] = rdMolDescriptors.CalcNumRings(m)
    d['杂原子数'] = rdMolDescriptors.CalcNumHeteroatoms(m)
    d['S原子数'] = sum(1 for a in m.GetAtoms() if a.GetSymbol() == 'S')
    d['N原子数'] = sum(1 for a in m.GetAtoms() if a.GetSymbol() == 'N')
    d['O原子数'] = sum(1 for a in m.GetAtoms() if a.GetSymbol() == 'O')
    d['C原子数'] = sum(1 for a in m.GetAtoms() if a.GetSymbol() == 'C')
    d['分子里自由度'] = rdMolDescriptors.CalcNumSpiroAtoms(m) + rdMolDescriptors.CalcNumBridgeheadAtoms(m)
    d['环状二硫'] = int(any(b.IsInRing() and
                      m.GetAtomWithIdx(b.GetBeginAtomIdx()).GetSymbol() == 'S' and
                      m.GetAtomWithIdx(b.GetEndAtomIdx()).GetSymbol() == 'S' for b in m.GetBonds()))

    # --- M 机理块 ---
    cnt = lambda k: len(m.GetSubstructMatches(P[k]))
    n_est = cnt('ester')
    d['n_ester'] = n_est
    d['n_acylAr'] = cnt('acylAr')
    d['n_alkAr'] = cnt('alkAr')
    d['n_carbonyl'] = len(m.GetSubstructMatches(Chem.MolFromSmarts('[CX3]=O')))
    d['n_arylRing'] = rdMolDescriptors.CalcNumAromaticRings(m)
    d['n_aliRing'] = rdMolDescriptors.CalcNumRings(m) - d['n_arylRing']
    d['n_arF'] = cnt('arF'); d['n_arCl'] = cnt('arCl'); d['n_arBr'] = cnt('arBr'); d['n_arI'] = cnt('arI')
    d['n_arMe'] = cnt('arMe'); d['n_arOMe'] = cnt('arOMe'); d['n_arCF3'] = cnt('arCF3')
    d['n_arNO2'] = max(cnt('arNO2'), cnt('arNO2b'))
    d['n_arOH'] = cnt('arOH'); d['n_arNH2'] = cnt('arNH2')
    d['n_EWG'] = d['n_arF'] + d['n_arCl'] + d['n_arBr'] + d['n_arI'] + d['n_arCF3'] + d['n_arNO2']
    d['n_EDG'] = d['n_arMe'] + d['n_arOMe'] + d['n_arOH'] + d['n_arNH2']
    d['n_halogen'] = d['n_arF'] + d['n_arCl'] + d['n_arBr'] + d['n_arI']
    d['ar_subst_total'] = d['n_EWG'] + d['n_EDG']
    n_ar_c = cnt('arylC')
    d['ar_subst_frac'] = d['ar_subst_total'] / n_ar_c if n_ar_c else 0.0
    d['n_conjCO'] = cnt('conjCO')
    d['n_tertC'] = cnt('tert_C')
    d['ss_flag'] = int(cnt('ss') > 0)
    d['ss_ring_flag'] = int(cnt('ss_ring') > 0)

    # 空间位阻：酰基碳上非羰基碳取代数 / 烷氧基碳的度
    acyl_deg, alk_deg = [], []
    for match in m.GetSubstructMatches(P['ester']):
        acyl = match[1]      # [CX3]
        oc = match[3]        # 酯氧
        # 酰基侧：酰基碳的碳邻居数（排除羰基氧）
        nac = sum(1 for nb in m.GetAtomWithIdx(acyl).GetNeighbors()
                  if nb.GetSymbol() == 'C' and nb.GetIdx() != oc)
        acyl_deg.append(nac)
        # 烷氧基侧：O 相连碳的度（是否芳碳）
        o_atom = m.GetAtomWithIdx(oc)
        for nb in o_atom.GetNeighbors():
            if nb.GetIdx() != acyl and nb.GetSymbol() == 'C':
                alk_deg.append(nb.GetDegree())
                break
    d['avg_acyl_deg'] = float(np.mean(acyl_deg)) if acyl_deg else 0.0
    d['avg_alkoxy_deg'] = float(np.mean(alk_deg)) if alk_deg else 0.0
    d['longest_chain'] = _longest_aliphatic_chain(m)
    mw = d['分子量'] if d['分子量'] else 1.0
    d['ester_per_MW'] = n_est / mw * 100
    d['halogen_per_ring'] = d['n_halogen'] / d['n_arylRing'] if d['n_arylRing'] else 0.0

    # --- F 指纹块（全 1024 位）---
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=NBITS)
    arr = np.zeros((NBITS,), dtype=np.int8); DataStructs.ConvertToNumpyArray(fp, arr)
    for i in range(NBITS):
        d[f'fp{i}'] = int(arr[i])

    # --- K MACCS 块 ---
    mk = MACCSkeys.GenMACCSKeys(m)
    marr = np.zeros((167,), dtype=np.int8); DataStructs.ConvertToNumpyArray(mk, marr)
    for i in range(167):
        d[f'mk{i}'] = int(marr[i])
    return d


def main():
    v = pd.read_csv('vlib_final.csv')
    d1 = pd.read_csv('dataset_v1.csv')
    e = pd.read_csv('ester_hyd.csv')
    a3 = pd.read_csv('al_round3_labels.csv')

    smi_v = dict(zip(v['编号'], v['SMILES']))
    smi_d = dict(zip(d1['编号'], d1['SMILES']))

    # 标签合并（与 fixF 一致）
    a3 = a3[a3['dE水解_kJmol'].notna()].copy()
    lab = pd.concat([e[['编号', 'dE水解_kJmol']].assign(来源='302'),
                     a3[['编号', 'dE水解_kJmol']].assign(来源='AL3')], ignore_index=True)
    lab = lab.drop_duplicates('编号', keep='first')
    lab['SMILES'] = lab['编号'].map(lambda i: smi_v.get(i, smi_d.get(i)))
    lab = lab[lab['SMILES'].notna()].copy()
    print(f'标签总数 {len(lab)} | 库内 {(lab["编号"].isin(smi_v)).sum()} | 文献 {(~lab["编号"].isin(smi_v)).sum()}')

    print('1/3 构建全库特征（12,380）...')
    rows = []
    for i, s in zip(v['编号'], v['SMILES']):
        f = feats(s)
        f['编号'] = i
        rows.append(f)
    FV = pd.DataFrame(rows).fillna(0)
    print('   列数', FV.shape)

    print('2/3 构建标签分子特征...')
    lit_ids = [i for i in lab['编号'] if i not in smi_v]
    rows = []
    for i in lit_ids:
        f = feats(smi_d[i]); f['编号'] = i; rows.append(f)
    FL = pd.DataFrame(rows).fillna(0) if rows else pd.DataFrame(columns=FV.columns)

    print('3/3 落盘...')
    FV.to_csv(f'{OUT}/feat_vlib.csv', index=False)
    FL.to_csv(f'{OUT}/feat_lit.csv', index=False)

    fp_cols = [c for c in FV.columns if c.startswith('fp')]
    mk_cols = [c for c in FV.columns if c.startswith('mk')]
    manifest = {
        'n_vlib': len(FV), 'n_lit': len(FL),
        'n_base': len(BASE_COLS), 'n_mech': len(MECH_COLS),
        'n_fp': len(fp_cols), 'n_maccs': len(mk_cols),
        'base_cols': BASE_COLS, 'mech_cols': MECH_COLS,
        'fp_bits_nonzero_vlib': int((FV[fp_cols].sum(axis=0) > 0).sum()) if len(fp_cols) else 0,
        'mk_bits_nonzero_vlib': int((FV[mk_cols].sum(axis=0) > 0).sum()) if len(mk_cols) else 0,
    }
    json.dump(manifest, open(f'{OUT}/feature_manifest.json', 'w'), ensure_ascii=False, indent=2)
    lab.to_csv(f'{OUT}/labels_452.csv', index=False)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('完成。')


if __name__ == '__main__':
    main()
