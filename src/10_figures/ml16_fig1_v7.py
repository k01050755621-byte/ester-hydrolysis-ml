# -*- coding: utf-8 -*-
"""fig1 重画（v7 + 19 分子 DFT 校准终版）：
(a) 标签分布 — labels_final_v7.csv
(b) xTB vs DFT 校准 — calib_expand/calib_19_merged.csv（气相，19 分子），
    附线性修正线 DFT = a + b·xTB。
只重画这一张图，其他图不动。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.expanduser('~/polywork/fix_20260928/calib_expand/calib_19_merged.csv')
FIG = os.path.join(OUT, 'figs')

LABF = pd.read_csv(os.path.join(OUT, 'labels_final_v7.csv'))
yy = LABF['dE水解_kJmol'].values

c = pd.read_csv(CAL)
xt = c['水解_xTB_kJmol_重算'].values.astype(float)
dt = c['水解_DFT_气相_kJmol'].values.astype(float)

# 线性修正
b, a = np.polyfit(xt, dt, 1)
res = dt - (a + b * xt)
r = np.corrcoef(xt, dt)[0, 1]
mae_raw = np.mean(np.abs(dt - xt))
mae_cor = np.mean(np.abs(res))

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax[0].hist(yy, bins=40, color='#4C72B0', edgecolor='white', linewidth=.5)
ax[0].set_xlabel(r'Hydrolysis energy $\Delta E$ (kJ mol$^{-1}$)')
ax[0].set_ylabel('Count'); ax[0].set_title(f'(a) Label distribution (n={len(yy)})')
ax[0].axvline(0, color='k', lw=.8, ls='--')

ax[1].scatter(xt, dt, s=38, color='#C44E52', zorder=3, edgecolor='white', linewidth=.4)
lim = [min(xt.min(), dt.min()) - 6, max(xt.max(), dt.max()) + 6]
xs = np.array(lim)
ax[1].plot(xs, xs, 'k--', lw=.8, label='y = x', zorder=2)
ax[1].plot(xs, a + b * xs, color='#4C72B0', lw=1.2,
           label=f'fit: {a:+.1f} {b:.3f}x', zorder=2)
ax[1].set_xlim(lim); ax[1].set_ylim(lim)
ax[1].set_xlabel(r'GFN2-xTB $\Delta E$ (kJ mol$^{-1}$)')
ax[1].set_ylabel(r'DFT $\Delta E$ (kJ mol$^{-1}$)')
ax[1].set_title(f'(b) xTB–DFT calibration (n={len(dt)}, r={r:.3f})')
ax[1].text(.03, .04, f'raw MAE = {mae_raw:.1f}\ncorrected MAE = {mae_cor:.1f}',
           transform=ax[1].transAxes, fontsize=7.5, va='bottom',
           bbox=dict(fc='white', ec='.7', alpha=.9, pad=2))
ax[1].legend(fontsize=7, loc='upper left', frameon=False)

fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig1_labels_calibration.png'), bbox_inches='tight', dpi=200)
plt.close(fig)
print(f'fig1 重画完成: n={len(dt)}  r={r:.4f}  斜率={b:.3f} 截距={a:+.2f}')
print(f'raw MAE={mae_raw:.2f}  corrected MAE={mae_cor:.2f}')
