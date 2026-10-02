# -*- coding: utf-8 -*-
"""AL Round 4 xTB 批量标注（与 AL3 协议完全一致：GFN2-xTB 气相优化 + 酯水解 SMARTS 单位点）

输入：ml_final/al4_query.csv
输出：polywork/xtb_vlib4/（几何与日志，可复现）、ml_final/al4_labels.csv
断点续跑：已存在的 .log 直接复用。
"""
import os, re, subprocess, time
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdChemReactions import ReactionFromSmarts
RDLogger.DisableLog('rdApp.*')

HOME = os.path.expanduser('~/polywork')
KJ = 2625.5
XTB = os.path.expanduser('~/miniconda3/envs/poly/bin/xtb')
OUT = os.path.join(HOME, 'fix_20260928', 'ml_final')
sub = pd.read_csv(os.path.join(OUT, 'al4_query.csv'))
WORK = os.path.join(HOME, 'xtb_vlib4')
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)

rxn_ester = ReactionFromSmarts("[#6:1](=[#8:2])-[#8;H0:3]-[#6:4]>>[#6:1](=[#8:2])-[OH].[#6:4]-[OH]")

def embed(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    m = Chem.AddHs(m)
    if AllChem.EmbedMolecule(m, AllChem.ETKDGv3()) != 0: return None
    AllChem.MMFFOptimizeMolecule(m)
    return m

def opt(m, tag):
    log = f'{tag}.log'
    if not os.path.exists(log):
        Chem.MolToXYZFile(m, f'{tag}.xyz')
        with open(log, 'w') as f:
            subprocess.run([XTB, f'{tag}.xyz', '--opt', '--gfn', '2'], stdout=f, stderr=subprocess.STDOUT)
    txt = open(log).read()
    es = re.findall(r'TOTAL ENERGY\s+(-?\d+\.\d+)', txt)
    return float(es[-1]) if es else None

def products_of(m):
    ps = rxn_ester.RunReactants((m,))
    if not ps: return None
    out = []
    for p in ps[0]:
        try:
            p.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(p, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception:
            pass
        out.append(p)
    return out

E_water = opt(embed('O'), 'water')
print(f'water = {E_water}', flush=True)
rows = []; t0 = time.time()
for k, (_, r) in enumerate(sub.iterrows(), 1):
    tag = str(r['编号'])
    try:
        m0 = embed(r['SMILES']); e0 = opt(m0, tag)
        ps = products_of(Chem.MolFromSmiles(r['SMILES'])); esum = 0.0; ok = True
        for i, mp in enumerate(ps):
            ep = opt(embed(Chem.MolToSmiles(mp)), f'{tag}_p{i}')
            if ep is None: ok = False; break
            esum += ep
        dE = (esum - e0 - E_water) * KJ if (ok and e0 is not None) else None
        rows.append({'编号': tag, '名称': r['名称'], 'SMILES': r['SMILES'], 'dE水解_kJmol': dE,
                     'query类型': r['query类型'], '预测_v4': r['预测_v4']})
        print(f'[{k}/{len(sub)}] {tag} dE={None if dE is None else round(dE,2)}', flush=True)
    except Exception as ex:
        rows.append({'编号': tag, '名称': r['名称'], 'SMILES': r['SMILES'], 'dE水解_kJmol': None,
                     'query类型': r['query类型'], '预测_v4': r['预测_v4']})
        print(f'[{k}/{len(sub)}] {tag} FAILED: {ex}', flush=True)
    if k % 20 == 0:
        pd.DataFrame(rows).to_csv(os.path.join(OUT, 'al4_labels.csv'), index=False)
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'al4_labels.csv'), index=False)
print(f'==== AL4 标注结束: {len(rows)} 个, 用时 {time.time()-t0:.0f}s ====', flush=True)
