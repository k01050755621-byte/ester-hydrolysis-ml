# -*- coding: utf-8 -*-
"""只重画 fig7（conformal/Mondrian）与 fig9（逐轮前瞻验证）到 v7 口径。

原因：ml6 里这两张图读的是 v6_summary.json（`_sf` 优先取 v6_summary.json，
而 v6_summary.json 里含 'mondrian' 与 'prospective' 两个 key），所以它们停在 v6 口径。
本脚本改读 v7 产物：mondrian_v7.json + v7_final_summary.json，其余版式与 ml6 保持一致。
"""
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HOME = os.path.expanduser('~/polywork'); os.chdir(HOME)
OUT = 'fix_20260928/ml_final'; FIG = f'{OUT}/figs'
plt.rcParams.update({'font.size': 8, 'axes.linewidth': .8, 'font.family': 'DejaVu Sans'})

# ---------- Fig 7: conformal（Mondrian 分类别 + 逐划分）----------
md = json.load(open(f'{OUT}/mondrian_v7.json'))
mond, cov = md['mondrian'], md['mondrian_coverage']
classes = [c for c in ['Al-Al', 'Al-Ar', 'Ar-Al', 'Ar-Ar'] if c in mond]
fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.8))
q = [mond[c]['q90'] for c in classes]
ax[0].bar(classes, q, color='#4C72B0', width=.55)
ax[0].set_ylabel(r'$q_{90}$ (kJ mol$^{-1}$)')
ax[0].set_title(r'(a) Class-conditional $q_{90}$', fontsize=9.5)
for i, v in enumerate(q):
    ax[0].text(i, v + .2, f'{v:.1f}', ha='center', fontsize=7.5)
ax[0].tick_params(axis='x', labelsize=7.5)

cv = [0 if cov[c] is None else cov[c] for c in classes]
ax[1].bar(classes, cv, color='#55A868', width=.55)
ax[1].axhline(.9, color='k', ls='--', lw=1, label='nominal 90%')
ax[1].set_ylim(0, 1.08); ax[1].set_ylabel('Empirical coverage')
ax[1].set_title('(b) Coverage on held-out test', fontsize=9.5)
ax[1].legend(frameon=False, fontsize=7); ax[1].tick_params(axis='x', labelsize=7.5)
for i, v in enumerate(cv):
    ax[1].text(i, v + .015, f'{v:.3f}', ha='center', fontsize=7.5)

cs = md['splits']
ax[2].bar([f'split {i+1}' for i in range(len(cs))], [c['q90'] for c in cs], color='#8172B2', width=.5)
ax[2].set_ylabel(r'$q_{90}$ (kJ mol$^{-1}$)'); ax[2].set_title('(c) Split-to-split (global)', fontsize=9.5)
for i, c in enumerate(cs):
    ax[2].text(i, c['q90'] + .2, f"{c['q90']:.1f}\n{c['coverage_test']:.1%}", ha='center', fontsize=7)
ax[2].set_ylim(0, max(c['q90'] for c in cs) * 1.35)
fig.tight_layout(); fig.savefig(f'{FIG}/fig7_conformal.png', bbox_inches='tight'); plt.close(fig)
print('saved -> figs/fig7_conformal.png  (v7 口径, 3 panels)')

# ---------- Fig 9: 逐轮前瞻验证（v7）----------
pr = json.load(open(f'{OUT}/v7_final_summary.json'))['prospective']
ks = [k for k in ['AL3', 'AL4', 'AL5'] if k in pr]
fig, ax = plt.subplots(1, 3, figsize=(8.2, 2.7))
ax[0].bar(ks, [pr[k]['Spearman'] for k in ks], color='#4C72B0', width=.5)
ax[0].set_ylabel(r'Spearman $\rho$'); ax[0].set_ylim(0, 1)
ax[0].set_title('(a) Prospective ranking', fontsize=9)
for i, k in enumerate(ks):
    ax[0].text(i, pr[k]['Spearman'] + .02, f"{pr[k]['Spearman']:.3f}", ha='center', fontsize=7.5)
ax[1].bar(ks, [pr[k]['MAE'] for k in ks], color='#C44E52', width=.5)
ax[1].set_ylabel(r'MAE (kJ mol$^{-1}$)'); ax[1].set_title('(b) Prospective MAE', fontsize=9)
for i, k in enumerate(ks):
    ax[1].text(i, pr[k]['MAE'] + .1, f"{pr[k]['MAE']:.2f}", ha='center', fontsize=7.5)
ax[2].bar(ks, [pr[k]['R2'] for k in ks], color='#55A868', width=.5)
ax[2].set_ylabel(r'R$^2$'); ax[2].axhline(0, color='k', lw=.8)
ax[2].set_title('(c) Prospective R$^2$', fontsize=9)
for i, k in enumerate(ks):
    ax[2].text(i, pr[k]['R2'] + .02, f"{pr[k]['R2']:.3f}", ha='center', fontsize=7.5)
fig.tight_layout(); fig.savefig(f'{FIG}/fig9_prospective.png', bbox_inches='tight'); plt.close(fig)
print('saved -> figs/fig9_prospective.png  (v7 口径)')
print('前瞻 v7:', {k: (pr[k]['Spearman'], pr[k]['MAE'], pr[k]['R2']) for k in ks})
