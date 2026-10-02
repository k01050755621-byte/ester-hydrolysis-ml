# -*- coding: utf-8 -*-
"""诊断：xTB 水解标签的构象噪声有多大？

动机：三个甲基位置异构体 C1074(对) / C6533(间) / C1223(邻) 的标签是
-30.6 / -33.7 / +9.7 kJ/mol，而同一批分子的 DFT 校准目录重算是
-30.6 / -10.0 / -30.6 —— 同分异构体的甲基位置远程于反应中心，
真值差不应有 20-40 kJ/mol。怀疑是"单构象 GFN2 气相优化"的构象噪声。

做法：对 parent / P1 / P2 各生成 K 个独立 ETKDG 构象，逐个 GFN2 优化，
取每个物种的**最低能量**做 ΔE（多构象最低值口径），并统计：
  - 每个物种的构象能差 (max-min)
  - 用"每个物种随机取 1 个构象"重复 M 次，得到 ΔE 的分布（模拟单构象口径的抖动）
输出：fix_20260928/calib_expand/conformer_noise.csv + 屏幕汇总
"""
import os, re, subprocess, itertools, random
import numpy as np, pandas as pd
from multiprocessing import Pool
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdChemReactions import ReactionFromSmarts
RDLogger.DisableLog('rdApp.*')

HOME = os.path.expanduser('~/polywork')
OUT = os.path.join(HOME, 'fix_20260928', 'calib_expand', 'conf_noise')
os.makedirs(OUT, exist_ok=True)
XTB = os.path.expanduser('~/miniconda3/envs/poly/bin/xtb')
KJ = 2625.5
K = int(os.environ.get('KCONF', '6'))      # 每个物种的构象数
M = 300                                    # 随机单构象抽样次数

RXN = ReactionFromSmarts('[#6:1](=[#8:2])-[#8;H0:3]-[#6:4]>>[#6:1](=[#8:2])-[OH].[#6:4]-[OH]')
v = pd.read_csv(os.path.join(HOME, 'vlib_final.csv')).set_index('编号')
lab = pd.read_csv(os.path.join(HOME, 'ester_hyd.csv')).set_index('编号')
cal = pd.read_csv(os.path.join(HOME, 'calib_ester_results.csv')).set_index('编号')


def make_conf(args):
    """生成第 i 个构象的 xyz 并做 GFN2 优化；返回 (tag, conf_idx, energy)"""
    tag, smi, i = args
    d = os.path.join(OUT, tag); os.makedirs(d, exist_ok=True)
    log = os.path.join(d, f'c{i}.log')
    if not os.path.exists(log):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        ps = AllChem.ETKDGv3(); ps.randomSeed = 1000 + i
        if AllChem.EmbedMolecule(m, ps) != 0:
            return (tag, i, None)
        AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
        xyz = os.path.join(d, f'c{i}.xyz'); Chem.MolToXYZFile(m, xyz)
        with open(log, 'w') as f:
            subprocess.run([XTB, xyz, '--opt', '--gfn', '2'], cwd=d,
                           stdout=f, stderr=subprocess.STDOUT)
    t = open(log).read()
    e = re.findall(r'TOTAL ENERGY\s+(-?\d+\.\d+)', t)
    return (tag, i, float(e[-1]) if e else None)


def main():
    ids = ['C1074', 'C6533', 'C1223']
    jobs = [('water', 'O', i) for i in range(3)]
    spec = {}
    for cid in ids:
        smi = v.loc[cid, 'SMILES']
        ps = RXN.RunReactants((Chem.MolFromSmiles(smi),))[0]
        prods = []
        for p in ps:
            p.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(p, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            prods.append(Chem.MolToSmiles(p))
        spec[cid] = {'parent': smi, 'P1': prods[0], 'P2': prods[1]}
        for role, s in spec[cid].items():
            jobs += [(f'{cid}_{role}', s, i) for i in range(K)]

    print(f'共 {len(jobs)} 个 GFN2 优化任务（K={K}）', flush=True)
    with Pool(5) as pool:
        res = pool.map(make_conf, jobs, chunksize=1)
    E = {}
    for tag, i, e in res:
        E.setdefault(tag, {})[i] = e

    rows = []
    for cid in ids:
        parts = {}
        for role in ['parent', 'P1', 'P2']:
            es = [e for e in E[f'{cid}_{role}'].values() if e is not None]
            parts[role] = sorted(es)
        ew = [e for e in E['water'].values() if e is not None]
        w_min, w_max = min(ew), max(ew)
        dE_min = (parts['P1'][0] + parts['P2'][0] - parts['parent'][0] - w_min) * KJ
        dE_max = (parts['P1'][-1] + parts['P2'][-1] - parts['parent'][-1] - w_min) * KJ
        # 随机单构象抽样
        rng = np.random.RandomState(0); sims = []
        for _ in range(M):
            e = (rng.choice(parts['P1']) + rng.choice(parts['P2'])
                 - rng.choice(parts['parent']) - rng.choice(ew)) * KJ
            sims.append(e)
        sims = np.array(sims)
        rows.append({'编号': cid,
                     '标签库': round(float(lab.loc[cid, 'dE水解_kJmol']), 1),
                     '校准文件': round(float(cal.loc[cid, '水解_xTB_kJmol']), 1),
                     '多构象最低_ΔE': round(dE_min, 1),
                     '最差构象_ΔE': round(dE_max, 1),
                     '单构象抽样_均值': round(float(sims.mean()), 1),
                     '单构象抽样_标准差': round(float(sims.std()), 1),
                     '单构象抽样_范围': f'{sims.min():.1f} ~ {sims.max():.1f}',
                     'parent构象跨度': round((parts['parent'][-1] - parts['parent'][0]) * KJ, 1),
                     'P1构象跨度': round((parts['P1'][-1] - parts['P1'][0]) * KJ, 1),
                     'P2构象跨度': round((parts['P2'][-1] - parts['P2'][0]) * KJ, 1)})
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(HOME, 'fix_20260928', 'calib_expand', 'conformer_noise.csv'), index=False)
    print('\n==== 构象噪声诊断 ====')
    print(d.to_string(index=False))
    print(f'\n（K={K}，随机单构象抽样 M={M}）')


if __name__ == '__main__':
    main()
