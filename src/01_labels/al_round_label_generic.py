# -*- coding: utf-8 -*-
"""通用 AL 轮次 xTB 标注：python al_round_label_generic.py <round>
输出：polywork/xtb_vlib<round>/ 与 ml_final/al<round>_labels.csv
"""
import os, re, sys, subprocess, time
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdChemReactions import ReactionFromSmarts
RDLogger.DisableLog('rdApp.*')

RN = int(sys.argv[1])
HOME = os.path.expanduser('~/polywork'); KJ = 2625.5
XTB = os.path.expanduser('~/miniconda3/envs/poly/bin/xtb')
OUT = os.path.join(HOME, 'fix_20260928', 'ml_final')
sub = pd.read_csv(os.path.join(OUT, f'al{RN}_query.csv'))
WORK = os.path.join(HOME, f'xtb_vlib{RN}')
os.makedirs(WORK, exist_ok=True); os.chdir(WORK)

rxn = ReactionFromSmarts("[#6:1](=[#8:2])-[#8;H0:3]-[#6:4]>>[#6:1](=[#8:2])-[OH].[#6:4]-[OH]")

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
    txt = open(log).read(); es = re.findall(r'TOTAL ENERGY\s+(-?\d+\.\d+)', txt)
    return float(es[-1]) if es else None

def products_of(m):
    ps = rxn.RunReactants((m,))
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

Ew = opt(embed('O'), 'water'); print('water =', Ew, flush=True)
rows = []; t0 = time.time()
for k, (_, r) in enumerate(sub.iterrows(), 1):
    tag = str(r['编号'])
    try:
        e0 = opt(embed(r['SMILES']), tag)
        ps = products_of(Chem.MolFromSmiles(r['SMILES'])); esum = 0.0; ok = True
        for i, mp in enumerate(ps):
            ep = opt(embed(Chem.MolToSmiles(mp)), f'{tag}_p{i}')
            if ep is None: ok = False; break
            esum += ep
        dE = (esum - e0 - Ew) * KJ if (ok and e0 is not None) else None
        rows.append({'编号': tag, '名称': r['名称'], 'SMILES': r['SMILES'], 'dE水解_kJmol': dE,
                     'query类型': r['query类型'], '模型预测': r['模型预测']})
        print(f'[{k}/{len(sub)}] {tag} dE={None if dE is None else round(dE,2)}', flush=True)
    except Exception as ex:
        rows.append({'编号': tag, '名称': r['名称'], 'SMILES': r['SMILES'], 'dE水解_kJmol': None,
                     'query类型': r['query类型'], '模型预测': r['模型预测']})
        print(f'[{k}/{len(sub)}] {tag} FAILED: {ex}', flush=True)
    if k % 20 == 0:
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f'al{RN}_labels.csv'), index=False)
pd.DataFrame(rows).to_csv(os.path.join(OUT, f'al{RN}_labels.csv'), index=False)
print(f'==== AL{RN} 标注结束: {len(rows)} 个, 用时 {time.time()-t0:.0f}s ====', flush=True)
