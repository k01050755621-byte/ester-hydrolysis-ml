# -*- coding: utf-8 -*-
"""只重画 fig2（学习曲线）到 v7 口径。
原因：fig2 原先读 learning_curve_v4.csv + label_ladder.csv，都是旧标签口径。
本脚本只读 ml13 产出的 learning_curve_v7.csv（池内子采样 + 逐轮 AL，同一 RF/骨架分组协议）。
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HOME = os.path.expanduser('~/polywork'); os.chdir(HOME)
OUT = 'fix_20260928/ml_final'; FIG = f'{OUT}/figs'
plt.rcParams.update({'font.size': 8, 'axes.linewidth': .8, 'font.family': 'DejaVu Sans'})

lc = pd.read_csv(f'{OUT}/learning_curve_v7.csv')
subs = lc[lc['来源'] == '池内子采样'].sort_values('n_labels')
al = lc[lc['来源'].str.startswith('逐轮')].sort_values('n_labels')

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
ax[0].plot(subs['n_labels'], subs['MAE'], 'o-', color='#9ecae1', ms=3.5, lw=1,
           label='subsampled from 452-pool')
ax[0].plot(al['n_labels'], al['MAE'], 's-', color='#4C72B0', ms=5, label='successive AL rounds')
ax[0].set_xlabel('Number of labels'); ax[0].set_ylabel(r'MAE (kJ mol$^{-1}$)')
ax[0].set_title('(a) Learning curve', fontsize=9.5); ax[0].legend(frameon=False, fontsize=6.5)
ax[1].plot(subs['n_labels'], subs['R2'], 'o-', color='#9ecae1', ms=3.5, lw=1, label='subsampled')
ax[1].plot(al['n_labels'], al['R2'], 's-', color='#C44E52', ms=5, label='AL rounds')
ax[1].axhline(0, color='gray', ls='--', lw=.9)
ax[1].set_xlabel('Number of labels'); ax[1].set_ylabel(r'R$^2$ (scaffold-grouped CV)')
ax[1].set_title('(b) Explained variance', fontsize=9.5); ax[1].legend(frameon=False, fontsize=6.5)
fig.tight_layout(); fig.savefig(f'{FIG}/fig2_learning_curve.png', bbox_inches='tight'); plt.close(fig)
print('saved -> figs/fig2_learning_curve.png   (v7 口径)')
print('池内子采样 R2:', subs['R2'].tolist())
print('逐轮 AL    R2:', al['R2'].tolist())
