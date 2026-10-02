# -*- coding: utf-8 -*-
"""协议等价性验证：把几何优化从笛卡尔坐标换成内坐标，
用已有分子（C1074_hydP2，旧协议仅 20 步、4 分钟即收敛）重算，
比较 opt 能量与 TZVP 气相单点能量是否一致（判据：|ΔE| < 2 kJ/mol 即等价）。

若等价 → 19 分子统一用内坐标重跑（预计提速 3-5 倍）。
"""
import os, re, time
import psi4
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

H = os.path.expanduser('~/polywork')
W = os.path.join(H, 'fix_20260928', 'calib_expand', 'val_internal')
os.makedirs(W, exist_ok=True)
KJ = 2625.5
psi4.set_memory(os.environ.get('PSI4_MEM', '4 GB'))
psi4.set_num_threads(int(os.environ.get('PSI4_THREADS', '3')))
psi4.core.set_output_file(os.path.join(W, 'val.out'), False)


def xyz_block(p):
    L = [l.strip() for l in open(p) if l.strip()]
    return '\n'.join(L[2:2 + int(L[0])])


def step_count(p):
    return len(re.findall(r'Step    Total Energy', open(p).read())) if os.path.exists(p) else 0


CASES = env_ids = os.environ.get('CASES', 'C1074_hydP2').split(',')
print(f'=== 协议等价性验证（内坐标 vs 笛卡尔）===  用例: {CASES}', flush=True)

for name in CASES:
    o = f'{H}/calib_ester/{name}'
    start = xyz_block(f'{o}/start.xyz')
    old_sp = float(re.search(r'SP_ENERGY\s+(-?\d+\.\d+)',
                             open(f'{o}/dft_opt.log').read()).group(1))
    old_steps = step_count(f'{o}/psi4.out')

    t0 = time.time()
    mol = psi4.geometry(f'0 1\nsymmetry c1\n{start}')
    psi4.set_options({'basis': 'def2-SVP', 'scf_type': 'df', 'reference': 'rks',
                      'g_convergence': 'gau', 'opt_coordinates': 'internal',
                      'geom_maxiter': 300, 'full_hess_every': -1})
    try:
        e_opt = psi4.optimize('wB97X-D3', molecule=mol)
    except Exception as ex:
        print(f'  [internal 失败] {name}: {ex} → 退化为 both', flush=True)
        mol = psi4.geometry(f'0 1\nsymmetry c1\n{start}')
        psi4.set_options({'opt_coordinates': 'both'})
        e_opt = psi4.optimize('wB97X-D3', molecule=mol)
    t_opt = (time.time() - t0) / 60
    xyz = mol.to_string(dtype='xyz').strip().splitlines()[2:]
    psi4.core.clean()

    m2 = psi4.geometry('0 1\nsymmetry c1\n' + '\n'.join(xyz))
    psi4.set_options({'basis': 'def2-TZVP', 'scf_type': 'df'})
    e_sp = psi4.energy('wB97X-D3', molecule=m2)
    psi4.core.clean()

    with open(os.path.join(W, f'{name}_internal.xyz'), 'w') as f:
        f.write(f'{len(xyz)}\n\n' + '\n'.join(xyz) + '\n')

    d_sp = (e_sp - old_sp) * KJ
    print(f'\n--- {name} ---', flush=True)
    print(f'  内坐标: SP={e_sp:.6f}  opt={e_opt:.6f}  opt 耗时 {t_opt:.1f} min', flush=True)
    print(f'  笛卡尔: SP={old_sp:.6f}  几何步数 {old_steps} 步', flush=True)
    print(f'  ΔE(单点) = {d_sp:+.3f} kJ/mol  →  ' +
          ('等价（可换协议）' if abs(d_sp) < 2 else '不一致（保留笛卡尔）'), flush=True)
    with open(os.path.join(W, f'{name}_compare.txt'), 'w') as f:
        f.write(f'{name}\ninternal_SP {e_sp}\ncartesian_SP {old_sp}\n'
                f'dE_kJmol {d_sp}\ninternal_opt_min {t_opt}\n')

print('\n验证完成。', flush=True)
