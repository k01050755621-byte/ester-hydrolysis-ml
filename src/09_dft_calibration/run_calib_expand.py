# -*- coding: utf-8 -*-
"""DFT 校准扩容 · 执行器（协议与 calib_ester / fixC_C2606_redo 完全一致）

每个分子做三个物种（parent / hydP1 / hydP2）：
  (1) wB97X-D3/def2-SVP 几何优化（从 xTB GFN2 几何起步，最多续跑 3 次）
  (2) wB97X-D3/def2-TZVP 气相单点
  (3) IEFPCM(water, GePol Area=0.3) wB97X-D3/def2-TZVP 单点
水解能 = E(P1) + E(P2) - E(parent) - E(water)，气相与 PCM 各一套。

用法：
  python run_calib_expand.py C7712 C12225 ...      # 指定分子
环境变量：PSI4_MEM（默认 5GB）、PSI4_THREADS（默认 4）、WORKER（写日志/结果后缀）
所有中间量有缓存，可随时中断续跑。
"""
import os, re, sys, subprocess, datetime, json
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdChemReactions import ReactionFromSmarts
RDLogger.DisableLog('rdApp.*')
import psi4

HOME = os.path.expanduser('~/polywork')
OUT = os.path.join(HOME, 'fix_20260928', 'calib_expand')
os.makedirs(OUT, exist_ok=True)
KJ = 2625.5
WORKER = os.environ.get('WORKER', 'w1')
MEM = os.environ.get('PSI4_MEM', '5 GB')
THR = int(os.environ.get('PSI4_THREADS', '4'))

XTB = os.path.expanduser('~/miniconda3/envs/poly/bin/xtb')

psi4.set_memory(MEM)
psi4.set_num_threads(THR)
psi4.core.set_output_file(os.path.join(OUT, f'psi4_{WORKER}.out'), False)

LOG = open(os.path.join(OUT, f'log_{WORKER}.log'), 'a', buffering=1)
def P(m):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] {m}', flush=True); LOG.write(f'[{ts}] {m}\n')

RXN = ReactionFromSmarts('[#6:1](=[#8:2])-[#8;H0:3]-[#6:4]>>[#6:1](=[#8:2])-[OH].[#6:4]-[OH]')

# ★ 关掉 PCM 分支（2026-09-28 22:20，实测判定为废数）
#   对照实验：拿水自己的 opt.xyz、用本脚本的 pcm_sp 协议重算，得 E = -1.057090 Eh；
#   而水的参考 PCM 值（smd_water.log）是 -76.448318 Eh，气相是 -76.438771 Eh。
#   即 psi4 在该写法下返回的是**溶剂化交互能**而非溶液总能量；对本项目分子表现为在气相能量上
#   加一个 ≈ +38.5 Eh 的常数偏移（C642 三个物种实测 +38.46/+38.55/+38.61），非物理。
#   决定：液相列停用，校准统一用**气相** DFT —— 这与已有 6 分子校准（calib_ester_results_fixed_v2.csv
#   只有气相列）口径完全一致；xTB 标签本身也是气相。副作用：机时约降为 1/3。
PCM_ON = os.environ.get('PCM_ON', '0') == '1'

E_W_GAS = float(re.search(r'SP_ENERGY\s+(-?\d+\.\d+)',
                          open(os.path.join(HOME, 'calib_ester', 'water', 'dft_opt.log')).read()).group(1))
_m = re.findall(r'SP_ENERGY\s*=\s*(-?\d+\.\d+)', open(os.path.join(HOME, 'smd_water.log')).read())
E_W_PCM = float(_m[-1]) if _m else None

# ★ 修 bug（2026-09-28 22:10）：xtb_e('water') 原先去读 $OUT/water/xtb.log，而该文件是
#   走"v7 最低构象"分支时**不会生成**的副产物 → FileNotFoundError → 整个分子一行都不落盘
#   （C642 三个物种的 DFT 全部白算）。水的 GFN2 参考能量与 E_W_GAS/E_W_PCM 同源，直接读旧目录。
WATER_XTB_LOG = os.path.join(HOME, 'calib_ester', 'water', 'xtb.log')
def _read_xtb_energy(p):
    if not os.path.exists(p):
        return None
    hit = re.findall(r'TOTAL ENERGY\s+(-?\d+\.\d+)', open(p).read())
    return float(hit[-1]) if hit else None
E_W_XTB = _read_xtb_energy(WATER_XTB_LOG)
if E_W_XTB is None:      # 兜底：旧目录不可用时，就地算一个落到 $OUT/water
    wd = os.path.join(OUT, 'water'); os.makedirs(wd, exist_ok=True)
    if not os.path.exists(os.path.join(wd, 'xtb.log')):
        m = Chem.AddHs(Chem.MolFromSmiles('O'))
        assert AllChem.EmbedMolecule(m, AllChem.ETKDGv3()) == 0
        AllChem.MMFFOptimizeMolecule(m)
        Chem.MolToXYZFile(m, os.path.join(wd, 'start.xyz'))
        with open(os.path.join(wd, 'xtb.log'), 'w') as f:
            subprocess.run([XTB, 'start.xyz', '--opt', '--gfn', '2'], cwd=wd,
                           stdout=f, stderr=subprocess.STDOUT)
    E_W_XTB = _read_xtb_energy(os.path.join(wd, 'xtb.log'))
assert E_W_XTB is not None, '水的 GFN2 参考能量取不到：xTB 参考不可用，拒绝继续（避免整轮结果无效）'
P(f'water ref: gas={E_W_GAS:.6f} pcm={E_W_PCM} xtb={E_W_XTB:.6f}')


def embed(smi):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    assert AllChem.EmbedMolecule(m, AllChem.ETKDGv3()) == 0, f'embed fail {smi}'
    AllChem.MMFFOptimizeMolecule(m)
    return m


def sanitize_loose(p):
    try:
        p.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(p, Chem.SanitizeFlags.SANITIZE_ALL
                         ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception:
        pass
    return p


# ★ 修 bug（2026-09-28 21:20）：relabel_v7 的 best_geom 用 P1/P2 命名，
#   本脚本用 hydP1/hydP2，原先直接拼 f'{name}.xyz' 导致**只有 parent 命中**，
#   水解产物退回单构象 GFN2 流程，破坏"全部物种用 v7 最低构象起步"的协议统一。
ROLE2GEOM = {'parent': 'parent', 'hydP1': 'P1', 'hydP2': 'P2'}


def xtb_opt(name, mol):
    """优先使用 v7 重标注产出的**最低能量构象**作为 DFT 起点（协议一致）；
    没有则退回旧的单构象 GFN2 流程。"""
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    cid, role = name.rsplit('_', 1)
    best = os.path.join(HOME, 'fix_20260928', 'relabel_v7', 'best_geom',
                        f'{cid}_{ROLE2GEOM.get(role, role)}.xyz')
    xyz = os.path.join(d, 'xtbopt.xyz')
    if os.path.exists(best) and not os.path.exists(xyz):
        import shutil
        shutil.copyfile(best, xyz)
        P(f'  [geom] {name} 用 v7 最低构象作起点')
    if not os.path.exists(xyz):
        Chem.MolToXYZFile(mol, os.path.join(d, 'start.xyz'))
        P(f'  [geom] {name} 无 v7 几何，退回单构象 GFN2 优化')
        with open(os.path.join(d, 'xtb.log'), 'w') as f:
            subprocess.run([XTB, 'start.xyz', '--opt', '--gfn', '2'], cwd=d,
                           stdout=f, stderr=subprocess.STDOUT)
    txt = open(xyz).read().strip().splitlines()
    return '\n'.join(txt[2:])


def read_xyz(p):
    L = [l.strip() for l in open(p) if l.strip()]
    return '\n'.join(L[2:2 + int(L[0])])


def dft_opt_sp(name, xyz_block, charge=0, mult=1):
    """DFT 优化(SVP) + 气相单点(TZVP)，带缓存与最多 3 次续跑。返回 (E_sp, E_opt, n_atoms)"""
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    flog = os.path.join(d, 'dft_opt.log')
    if os.path.exists(flog):
        t = open(flog).read()
        ms, mo = re.search(r'SP_ENERGY\s+(-?\d+\.\d+)', t), re.search(r'OPT_ENERGY\s+(-?\d+\.\d+)', t)
        if ms:
            P(f'  [cache] {name} SP={float(ms.group(1)):.6f}')
            return float(ms.group(1)), (float(mo.group(1)) if mo else None), len(xyz_block.splitlines())
    mol = psi4.geometry(f'{charge} {mult}\nsymmetry c1\n{xyz_block}')
    psi4.set_options({'basis': 'def2-SVP', 'scf_type': 'df',
                      'reference': 'uks' if mult > 1 else 'rks',
                      'g_convergence': 'gau', 'opt_coordinates': 'cartesian',
                      'geom_maxiter': 300, 'full_hess_every': -1})
    e_opt = None
    for att in range(3):
        try:
            e_opt = psi4.optimize('wB97X-D3', molecule=mol); break
        except psi4.driver.p4util.exceptions.OptimizationConvergenceError as ex:
            P(f'  [retry] {name} 第{att+1}次未收敛，从最近几何续跑')
            mol = ex.wfn.molecule(); psi4.core.clean()
    if e_opt is None:
        raise RuntimeError(f'{name} 优化三次仍未收敛')
    xyz_opt = mol.to_string(dtype='xyz').strip().splitlines()[2:]
    with open(os.path.join(d, 'opt.xyz'), 'w') as f:
        f.write(f'{len(xyz_opt)}\n\n' + '\n'.join(xyz_opt) + '\n')
    psi4.core.clean()
    mol2 = psi4.geometry(f'{charge} {mult}\nsymmetry c1\n' + '\n'.join(xyz_opt))
    psi4.set_options({'basis': 'def2-TZVP', 'scf_type': 'df'})
    e_sp = psi4.energy('wB97X-D3', molecule=mol2)
    with open(flog, 'w') as f:
        f.write(f'OPT_ENERGY {e_opt}\nSP_ENERGY {e_sp}\n')
    psi4.core.clean()
    P(f'  [done] {name} n={len(xyz_opt)} SP={e_sp:.6f}')
    return e_sp, e_opt, len(xyz_opt)


def pcm_sp(name, charge=0, mult=1):
    d = os.path.join(OUT, name)
    flog = os.path.join(d, 'pcm.log')
    if os.path.exists(flog):
        m = re.search(r'SP_ENERGY\s+(-?\d+\.\d+)', open(flog).read())
        if m:
            P(f'  [cache] {name} PCM={float(m.group(1)):.6f}')
            return float(m.group(1))
    xyz = read_xyz(os.path.join(d, 'opt.xyz'))
    mol = psi4.geometry(f'{charge} {mult}\nsymmetry c1\n{xyz}')
    psi4.set_options({'basis': 'def2-TZVP', 'scf_type': 'df',
                      'reference': 'uks' if mult > 1 else 'rks',
                      'pcm': True, 'pcm_scf_type': 'total'})
    psi4.pcm_helper("""Units = Angstrom
Medium {
SolverType = IEFPCM
Solvent = Water
}
Cavity {
Type = GePol
Area = 0.3
}""")
    e = psi4.energy('wB97X-D3', molecule=mol)
    with open(flog, 'w') as f:
        f.write(f'SP_ENERGY {e}\n')
    psi4.core.clean()
    P(f'  [done] {name} PCM={e:.6f}')
    return e


def xtb_e(name):
    """物种的 GFN2 能量。
    ★ 修 bug（2026-09-28 21:20）：走"v7 最低构象"分支的物种**不产生 xtb.log**
    （直接复制几何，未跑 xtb），原先直接 open(xtb.log) → FileNotFoundError →
    整个分子行不落盘（DFT 白算）。改为缺日志时对 xtbopt.xyz 补一个 GFN2 单点并落盘。"""
    d = os.path.join(OUT, name)
    log = os.path.join(d, 'xtb.log')
    if not os.path.exists(log):
        with open(log, 'w') as f:
            subprocess.run([XTB, 'xtbopt.xyz', '--gfn', '2'], cwd=d,
                           stdout=f, stderr=subprocess.STDOUT)
    txt = open(log).read()
    hit = re.findall(r'TOTAL ENERGY\s+(-?\d+\.\d+)', txt)
    if not hit:
        raise RuntimeError(f'{name}: xtb.log 中未找到 TOTAL ENERGY')
    return float(hit[-1])


def run_one(cid, smi, name):
    P(f'=== {cid} {name} ===')
    mol = Chem.MolFromSmiles(smi)
    ps = RXN.RunReactants((mol,))
    assert ps and len(ps[0]) == 2, f'{cid} 产物数异常'
    roles = {'parent': smi,
             'hydP1': Chem.MolToSmiles(sanitize_loose(ps[0][0])),
             'hydP2': Chem.MolToSmiles(sanitize_loose(ps[0][1]))}
    E = {}; NAT = {}
    for role, s in roles.items():
        xyz = xtb_opt(f'{cid}_{role}', embed(s))
        E[role], _, NAT[role] = dft_opt_sp(f'{cid}_{role}', xyz)
    Ep = {}
    if PCM_ON:
        for role in roles:
            Ep[role] = pcm_sp(f'{cid}_{role}')
    hyd_gas = (E['hydP1'] + E['hydP2'] - E['parent'] - E_W_GAS) * KJ
    hyd_pcm = ((Ep['hydP1'] + Ep['hydP2'] - Ep['parent'] - E_W_PCM) * KJ
               if (PCM_ON and E_W_PCM is not None) else None)
    xg = (xtb_e(f'{cid}_hydP1') + xtb_e(f'{cid}_hydP2') - xtb_e(f'{cid}_parent') - E_W_XTB) * KJ
    P(f'>>> {cid} DFT气相={hyd_gas:.1f}  DFT_PCM={hyd_pcm if hyd_pcm is None else round(hyd_pcm,1)}  xTB={xg:.1f}')
    return {'编号': cid, '名称': name,
            '水解_DFT_气相_kJmol': round(hyd_gas, 1),
            '水解_DFT_PCM_kJmol': (round(hyd_pcm, 1) if hyd_pcm is not None else None),
            '水解_xTB_kJmol_重算': round(xg, 1),
            'n原子_parent': NAT['parent'], 'n原子_P1': NAT['hydP1'], 'n原子_P2': NAT['hydP2'],
            'SMILES': smi}


if __name__ == '__main__':
    ids = sys.argv[1:]
    sel = pd.read_csv(os.path.join(OUT, 'selection.csv'))
    sel = sel.set_index('编号')
    resf = os.path.join(OUT, f'results_{WORKER}.csv')
    done = set()
    if os.path.exists(resf):
        done = set(pd.read_csv(resf)['编号'])
    P(f'worker={WORKER} mem={MEM} threads={THR} 目标 {len(ids)} 个，已完成 {len(done)}')
    n_try = n_ok = n_fail = 0
    for cid in ids:
        if cid in done:
            P(f'  [skip] {cid} 已在结果中'); continue
        n_try += 1
        try:
            row = run_one(cid, sel.loc[cid, 'SMILES'], sel.loc[cid, '名称'])
        except Exception as ex:
            n_fail += 1
            P(f'!! {cid} 失败: {type(ex).__name__}: {ex}'); continue
        pd.DataFrame([row]).to_csv(resf, mode='a', header=not os.path.exists(resf), index=False)
        n_ok += 1
        P(f'   [saved] {resf}')
    # ★ 静默丢数据自查：尝试数必须等于落盘数，否则显式告警（本文档 SKILL 的强制纪律）
    P(f'---- 汇总：尝试 {n_try} | 落盘 {n_ok} | 失败 {n_fail} ----')
    if n_try != n_ok:
        P(f'!!! 告警：有 {n_try - n_ok} 个分子未能落盘，请检查上方的 !! 行（这部分 DFT 机时已浪费）')
    if os.path.exists(resf):
        got = pd.read_csv(resf)['编号'].tolist()
        P(f'结果文件现有 {len(got)} 行；本次目标中缺失: {[i for i in ids if i not in set(got)]}')
    P(f'==== worker {WORKER} 结束 ====')
