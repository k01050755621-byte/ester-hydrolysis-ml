# -*- coding: utf-8 -*-
"""v7 重标注：修掉两条缺陷，让标签可复现且不受单构象噪声支配。

缺陷（已实测）
 1) 可复现性：旧流程用 `AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())`，
    ETKDGv3 的 randomSeed 默认 -1（随机）→ 同一分子两次嵌入最大原子位移 7.4 Å，
    再经 GFN2 优化后 ΔE 抖动 σ≈12 kJ/mol、范围 −30~+15 kJ/mol。
    实测同一分子同一反应：标签库 −33.7 vs 独立重算 −10.0 vs 多构象最低 −1.6。
 2) 单构象：只取一个构象，parent 的构象能差可达 23–31 kJ/mol，
    直接进了 ΔE。

v7 协议（本文档版本，写进 SI）
 - 每个物种（parent / P1 / P2）用 ETKDGv3 + **固定种子** 生成 K 个构象（默认 6）；
 - 每个构象 GFN2 优化（`xtb --opt --gfn 2`），**取该物种的最低能量**；
 - ΔE = E(P1)min + E(P2)min − E(parent)min − E(H2O)，与旧口径同一热化学式；
 - 水解位点：仍取 SMARTS 的第一个匹配（与旧标签同一位点，保证可比）；
 - 水：单独取多构象最低值，全程共用。
另存每个物种的**最低构象几何**（best_geom/），供 DFT 校准从最低构象起步。

输出：fix_20260928/relabel_v7/labels_v7.csv（含旧值对照 + 构象跨度）
      fix_20260928/relabel_v7/best_geom/<编号>_<role>.xyz
可中断续跑（每个构象一个缓存日志）。
"""
import os, re, sys, time, zlib, subprocess, datetime
import numpy as np, pandas as pd
from multiprocessing import Pool
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdChemReactions import ReactionFromSmarts
RDLogger.DisableLog('rdApp.*')

HOME = os.path.expanduser('~/polywork')
OUT = os.path.join(HOME, 'fix_20260928', 'relabel_v7')
GEOM = os.path.join(OUT, 'best_geom')
os.makedirs(GEOM, exist_ok=True)
XTB = os.path.expanduser('~/miniconda3/envs/poly/bin/xtb')
KJ = 2625.5
K = int(os.environ.get('KCONF', '6'))
NW = int(os.environ.get('NWORK', '8'))

RXN = ReactionFromSmarts('[#6:1](=[#8:2])-[#8;H0:3]-[#6:4]>>[#6:1](=[#8:2])-[OH].[#6:4]-[OH]')


def dseed(*parts):
    """跨进程稳定的确定性种子（不能用 python hash）"""
    return zlib.crc32('|'.join(map(str, parts)).encode()) % 100000


def opt_one(job):
    """生成第 k 个构象并 GFN2 优化。返回 (编号, role, k, E, xyz_path)"""
    cid, role, k, smi = job
    d = os.path.join(OUT, cid, role); os.makedirs(d, exist_ok=True)
    log = os.path.join(d, f'c{k}.log'); xyz = os.path.join(d, f'c{k}.xyz')
    if not os.path.exists(log):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        ps = AllChem.ETKDGv3(); ps.randomSeed = dseed(cid, role, k)
        if AllChem.EmbedMolecule(m, ps) != 0:
            return (cid, role, k, None, None)
        AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
        Chem.MolToXYZFile(m, xyz)
        for att in range(2):
            with open(log, 'w') as f:
                subprocess.run([XTB, xyz, '--opt', '--gfn', '2'], cwd=d,
                               stdout=f, stderr=subprocess.STDOUT)
            t = open(log).read()
            e = re.findall(r'TOTAL ENERGY\s+(-?\d+\.\d+)', t)
            if e:
                return (cid, role, k, float(e[-1]), os.path.join(d, 'xtbopt.xyz'))
    t = open(log).read()
    e = re.findall(r'TOTAL ENERGY\s+(-?\d+\.\d+)', t)
    if not e:
        return (cid, role, k, None, None)
    return (cid, role, k, float(e[-1]), os.path.join(d, 'xtbopt.xyz'))


def species_of(cid, smi):
    mol = Chem.MolFromSmiles(smi)
    ps = RXN.RunReactants((mol,))
    if not ps or len(ps[0]) != 2:
        raise RuntimeError(f'{cid} 产物枚举失败')
    out = []
    for p in ps[0]:
        p.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(p, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        out.append(Chem.MolToSmiles(p))
    return {'parent': smi, 'P1': out[0], 'P2': out[1]}


def main():
    ids_env = os.environ.get('IDS')
    lab = pd.read_csv(os.path.join(HOME, 'fix_20260928', 'ml_final', 'labels_final.csv'))
    # ★ 修 bug（2026-09-28 20:20）：原先只从 vlib_final.csv 取 SMILES 并据此过滤，
    #   导致 dataset_v1.csv 里的 100 个基础酯（C131–C134、C264–C359，来源=302）
    #   被静默丢弃（1099 → 999）。改为多源合并后再过滤。
    _src = []
    for _f in ['vlib_final.csv', 'dataset_v1.csv']:
        _p = os.path.join(HOME, _f)
        if os.path.exists(_p):
            _d = pd.read_csv(_p)
            if '编号' in _d.columns and 'SMILES' in _d.columns:
                _src.append(_d[['编号', 'SMILES']])
    v = (pd.concat(_src, ignore_index=True)
           .dropna(subset=['SMILES'])
           .drop_duplicates('编号')
           .set_index('编号'))
    _miss = [i for i in lab['编号'] if i not in v.index]
    if _miss:
        print(f'[warn] {len(_miss)} 个标签分子在所有 SMILES 源中都找不到，将跳过：{_miss[:10]}')
    ids = [i for i in (ids_env.split(',') if ids_env else lab['编号'].tolist()) if i in v.index]
    P = lambda m: print(f'[{datetime.datetime.now():%H:%M:%S}] {m}', flush=True)
    P(f'v7 重标注：{len(ids)} 个分子 × 3 物种 × {K} 构象 = {len(ids)*3*K} 个 GFN2 任务，{NW} 进程')

    jobs = []
    for cid in ids:
        for role, s in species_of(cid, v.loc[cid, 'SMILES']).items():
            jobs += [(cid, role, k, s) for k in range(K)]

    # 水先算完（保证 summarize 时参考值一定存在）
    E = {}
    with Pool(min(NW, K)) as pool:
        wres = pool.map(opt_one, [('water', 'w', k, 'O') for k in range(K)], chunksize=1)
    for cid, role, k, e, xyzp in wres:
        E[(cid, role, k)] = e
    ew = [e for _, _, _, e, _ in wres if e is not None]
    P(f'水（{K} 构象）最低 = {min(ew):.6f}，跨度 {(max(ew)-min(ew))*KJ:.2f} kJ/mol')

    t0 = time.time()
    resf = os.path.join(OUT, 'labels_v7.csv')
    done_ids = set(pd.read_csv(resf)['编号']) if os.path.exists(resf) else set()
    P(f'分子任务 {len(jobs)} 个；已完成 {len(done_ids)} 个分子（跳过）')
    with Pool(NW) as pool:
        for n, (cid, role, k, e, xyzp) in enumerate(pool.imap_unordered(opt_one, jobs, chunksize=4), 1):
            E[(cid, role, k)] = e
            if n % 200 == 0:
                el = time.time() - t0
                P(f'  进度 {n}/{len(jobs)}  {el/60:.1f} min  预计剩余 {el/n*(len(jobs)-n)/60:.0f} min')
            if cid not in done_ids and all(
                    (cid, r, kk) in E for r in ['parent', 'P1', 'P2'] for kk in range(K)):
                row = summarize(cid, E, lab, v)
                pd.DataFrame([row]).to_csv(resf, mode='a', header=not os.path.exists(resf), index=False)
                done_ids.add(cid)
                P(f'  [saved] {cid}  v7={row["v7_kJmol"]}  旧={row["旧标签_kJmol"]}  差={row["差_新旧"]}')

    # 收尾统计
    P(f'水最低能量 = {min(ew):.6f}（跨度 {(max(ew)-min(ew))*KJ:.2f} kJ/mol）')
    json_meta = {'K': K, 'n_mol': len(ids), 'water_min_Hartree': min(ew),
                 'water_spread_kJmol': round((max(ew) - min(ew)) * KJ, 2),
                 'protocol': 'ETKDGv3 fixed-seed K-conformer + GFN2 opt + per-species global minimum'}
    import json; json.dump(json_meta, open(os.path.join(OUT, 'relabel_meta.json'), 'w'),
                           ensure_ascii=False, indent=2)
    P('完成。')


def summarize(cid, E, lab, v):
    """取每个物种的最低构象，算 v7 ΔE，并写 best_geom"""
    pick, span = {}, {}
    for role in ['parent', 'P1', 'P2']:
        es = [(E[(cid, role, k)], k) for k in range(K) if E.get((cid, role, k)) is not None]
        es.sort()
        pick[role], span[role] = es[0], round((es[-1][0] - es[0][0]) * KJ, 1)
    ew = [E[('water', 'w', k)] for k in range(K) if E.get(('water', 'w', k)) is not None]
    ew_min = min(ew) if ew else None
    de = (pick['P1'][0] + pick['P2'][0] - pick['parent'][0] - ew_min) * KJ
    for role, (e, k) in pick.items():
        src = os.path.join(OUT, cid, role, 'xtbopt.xyz')
        if os.path.exists(src):
            import shutil; shutil.copyfile(src, os.path.join(GEOM, f'{cid}_{role}.xyz'))
    old = lab.set_index('编号')
    return {'编号': cid, 'v7_kJmol': round(float(de), 2),
            '旧标签_kJmol': (round(float(old.loc[cid, 'dE水解_kJmol']), 2) if cid in old.index else None),
            '差_新旧': (round(float(de) - float(old.loc[cid, 'dE水解_kJmol']), 2) if cid in old.index else None),
            'parent构象跨度': span['parent'], 'P1构象跨度': span['P1'], 'P2构象跨度': span['P2'],
            'best_conf_idx': f"{pick['parent'][1]}/{pick['P1'][1]}/{pick['P2'][1]}",
            '来源': (old.loc[cid, '来源'] if cid in old.index else 'new')}


if __name__ == '__main__':
    main()
