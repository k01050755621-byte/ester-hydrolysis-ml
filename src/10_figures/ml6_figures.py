# -*- coding: utf-8 -*-
"""ML 补强 · 第 6 步：出版级图表（英文标注，8 图）

输出到 ml_final/figs/
"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

HOME = os.path.expanduser('~/polywork'); os.chdir(HOME)
OUT = 'fix_20260928/ml_final'; FIG = f'{OUT}/figs'
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'figure.dpi': 160,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'DejaVu Sans'})

FV = pd.read_csv(f'{OUT}/feat_vlib.csv'); FL = pd.read_csv(f'{OUT}/feat_lit.csv')
LAB = pd.read_csv(os.environ.get('LABFILE', f'{OUT}/labels_452.csv')); MAN = json.load(open(f'{OUT}/feature_manifest.json'))
BASE, MECH = MAN['base_cols'], MAN['mech_cols']
FP = [c for c in FV.columns if c.startswith('fp') and FV[c].sum() > 0]
MK = [c for c in FV.columns if c.startswith('mk') and FV[c].sum() > 0]
BEST = BASE + MECH + FP + MK
ALL = pd.concat([FV, FL], ignore_index=True).fillna(0)
tr = ALL.merge(LAB[['编号', 'dE水解_kJmol', '来源']], on='编号', how='inner')
y = tr['dE水解_kJmol'].values

# ---------- Fig 1: label distribution + DFT calibration ----------
_labf = (os.environ.get('LABFILE')
         or (f'{OUT}/labels_final.csv' if os.path.exists(f'{OUT}/labels_final.csv') else f'{OUT}/labels_452.csv'))
LABF = pd.read_csv(_labf)
yy = LABF['dE水解_kJmol'].values
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax[0].hist(yy, bins=40, color='#4C72B0', edgecolor='white', linewidth=.5)
ax[0].set_xlabel(r'Hydrolysis energy $\Delta E$ (kJ mol$^{-1}$)')
ax[0].set_ylabel('Count'); ax[0].set_title(f'(a) Label distribution (n={len(yy)})')
ax[0].axvline(0, color='k', lw=.8, ls='--')
c = pd.read_csv(f'{OUT}/../C2606_redo/calib_ester_results_fixed_v2.csv') if os.path.exists(
    f'{OUT}/../C2606_redo/calib_ester_results_fixed_v2.csv') else pd.read_csv('calib_ester_results.csv')
xt, dt = c['水解_xTB_kJmol'].values, c['水解_DFT_kJmol'].values
ax[1].scatter(xt, dt, s=42, color='#C44E52', zorder=3)
lim = [min(xt.min(), dt.min()) - 6, max(xt.max(), dt.max()) + 6]
ax[1].plot(lim, lim, 'k--', lw=.8)
ax[1].set_xlim(lim); ax[1].set_ylim(lim)
ax[1].set_xlabel(r'GFN2-xTB $\Delta E$'); ax[1].set_ylabel(r'DFT $\Delta E$')
ax[1].set_title(f'(b) xTB vs DFT calibration (n={len(dt)}, MAE={np.mean(np.abs(dt-xt)):.1f})')
fig.tight_layout(); fig.savefig(f'{FIG}/fig1_labels_calibration.png', bbox_inches='tight'); plt.close(fig)

# ---------- OOF (RF, scaffold-grouped, 3 seeds pooled) ----------
def scaf_of(s):
    m = Chem.MolFromSmiles(str(s)); sc = MurckoScaffold.GetScaffoldForMol(m) if m is not None else None
    return Chem.MolToSmiles(sc) if sc and sc.GetNumAtoms() else 'acyclic:' + Chem.MolToSmiles(m)
vmap = pd.read_csv('vlib_final.csv').set_index('编号')['SMILES'].to_dict()
d1map = pd.read_csv('dataset_v1.csv').set_index('编号')['SMILES'].to_dict()
SC = np.array([scaf_of(vmap.get(i, d1map.get(i, 'C'))) for i in tr['编号']])
UQ = np.unique(SC)
def folds(seed):
    rng = np.random.RandomState(seed); order = rng.permutation(len(UQ)); fo, sz = {}, np.zeros(5)
    for u in order:
        f = int(np.argmin(sz)); fo[UQ[u]] = f; sz[f] += int(np.sum(SC == UQ[u]))
    return np.array([fo[x] for x in SC])
oof = np.zeros((3, len(y)))
for k, sd in enumerate([0, 1, 42]):
    f = folds(sd); X = tr[BEST].values
    for j in range(5):
        te, tri = np.where(f == j)[0], np.where(f != j)[0]
        oof[k, te] = RandomForestRegressor(n_estimators=400, random_state=sd, n_jobs=4).fit(X[tri], y[tri]).predict(X[te])
pred_oof = oof.mean(axis=0)

# ---------- Fig 8: parity ----------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax[0].scatter(y, pred_oof, s=12, alpha=.55, color='#55A868')
lim = [y.min() - 5, y.max() + 5]; ax[0].plot(lim, lim, 'k--', lw=.8)
ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel(r'xTB $\Delta E$ (kJ mol$^{-1}$)'); ax[0].set_ylabel(r'Predicted $\Delta E$')
ax[0].set_title(f'(a) OOF parity  (MAE={mean_absolute_error(y,pred_oof):.2f}, R$^2$={r2_score(y,pred_oof):+.2f})')
ax[1].scatter(pred_oof, y - pred_oof, s=12, alpha=.55, color='#8172B2')
ax[1].axhline(0, color='k', lw=.8, ls='--')
ax[1].set_xlabel('Predicted'); ax[1].set_ylabel('Residual'); ax[1].set_title('(b) Residuals')
fig.tight_layout(); fig.savefig(f'{FIG}/fig8_parity.png', bbox_inches='tight'); plt.close(fig)

# ---------- Fig 2: learning curve（标签阶梯）+ 逐轮前瞻验证 ----------
lc = pd.read_csv(f'{OUT}/learning_curve_v4.csv')
ladder = pd.read_csv(f'{OUT}/label_ladder.csv', index_col=0)
ladder = ladder[ladder.index.isin(['302', '452', '771', 'final'])]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
ax[0].plot(lc['n_labels'], lc['MAE'], 'o-', color='#9ecae1', ms=3.5, lw=1,
           label='subsampled from 452-pool')
ax[0].plot(ladder['n'], ladder['MAE'], 's-', color='#4C72B0', ms=5, label='successive AL rounds')
ax[0].axhline(11.89, color='gray', ls='--', lw=.9, label='mean baseline')
ax[0].set_xlabel('Number of labels'); ax[0].set_ylabel('MAE (kJ mol$^{-1}$)')
ax[0].set_title('(a) Learning curve', fontsize=9.5); ax[0].legend(frameon=False, fontsize=6.5)
ax[1].plot(lc['n_labels'], lc['R2'], 'o-', color='#9ecae1', ms=3.5, lw=1, label='subsampled')
ax[1].plot(ladder['n'], ladder['R2'], 's-', color='#C44E52', ms=5, label='AL rounds')
ax[1].axhline(0, color='gray', ls='--', lw=.9)
ax[1].set_xlabel('Number of labels'); ax[1].set_ylabel('R$^2$ (scaffold-grouped CV)')
ax[1].set_title('(b) Explained variance', fontsize=9.5); ax[1].legend(frameon=False, fontsize=6.5)
fig.tight_layout(); fig.savefig(f'{FIG}/fig2_learning_curve.png', bbox_inches='tight'); plt.close(fig)

# ---------- Fig 3: ablation ----------
_abf = f'{OUT}/final_ablation.csv' if os.path.exists(f'{OUT}/final_ablation.csv') else f'{OUT}/ablation_table.csv'
ab = pd.read_csv(_abf)
en_fs = {'base16': 'base(16)', 'base+mech': 'base+mech', 'base+FP': 'base+FP', 'base+fp(v3口径)': 'base+FP (v3)',
         'base+mech+FP': 'base+mech+FP', 'base+mech+fp+maccs': 'base+mech+FP+MACCS', 'full(+MACCS)': 'full(+MACCS)', 'mech_only': 'mech only'}
fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.0))
lbl = [en_fs.get(s, s) for s in ab['特征集']]
x = np.arange(len(ab))
ax[0].bar(x, ab['R2'], color='#4C72B0'); ax[0].axhline(0, color='k', lw=.8)
ax[0].set_xticks(x); ax[0].set_xticklabels(lbl, fontsize=6.5, rotation=18, ha='right')
ax[0].set_ylabel('R$^2$'); ax[0].set_title('(a) Accuracy by feature block', fontsize=9.5)
for i, v in enumerate(ab['R2']):
    ax[0].text(i, v + .01, f'{v:.3f}', ha='center', fontsize=6.5)
ncap = ab['n_distinct'] if 'n_distinct' in ab.columns else pd.Series([12380]*len(ab))
ax[1].bar(x, ncap, color='#C44E52')
ax[1].set_xticks(x); ax[1].set_xticklabels(lbl, fontsize=6.5, rotation=18, ha='right')
ax[1].set_ylabel('Distinct predictions (of 12,380)')
ax[1].set_title('(b) Resolution between molecules', fontsize=9.5)
ax[1].axhline(12380, color='gray', ls='--', lw=.8)
fig.tight_layout(); fig.savefig(f'{FIG}/fig3_ablation.png', bbox_inches='tight'); plt.close(fig)

# ---------- Fig 4: isomer blindness & fix ----------
_igs = f'{OUT}/isomer_collapse_final.csv' if os.path.exists(f'{OUT}/isomer_collapse_final.csv') else f'{OUT}/isomer_collapse_summary.csv'
_ig = f'{OUT}/isomer_groups_final.csv' if os.path.exists(f'{OUT}/isomer_groups_final.csv') else f'{OUT}/isomer_groups_v4.csv'
igs = pd.read_csv(_igs); ig = pd.read_csv(_ig)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax[0].hist(ig['base不同预测数'], bins=np.arange(.5, 5.5, 1), color='#C44E52', alpha=.85,
           label='16 descriptors only', edgecolor='white')
ax[0].hist(ig['v6不同预测数'] if 'v6不同预测数' in ig else ig['v4不同预测数'],
           bins=np.arange(.5, 14.5, 1), color='#4C72B0', alpha=.7,
           label='+ mech + Morgan + MACCS', edgecolor='white')
ax[0].set_xlabel('Distinct predictions within a descriptor-identical group')
ax[0].set_ylabel('Number of groups'); ax[0].set_title(f'(a) {len(ig)} isomer families', fontsize=9.5)
ax[0].legend(frameon=False, fontsize=7)
ax[1].bar(['16 descriptors', 'full model'], igs['塌缩比'].values, color=['#C44E52', '#4C72B0'], width=.5)
ax[1].set_ylabel('Collapse ratio (molecules / distinct pred.)')
ax[1].set_title('(b) Predicted-value collapse', fontsize=9.5)
for i, v in enumerate(igs['塌缩比'].values):
    ax[1].text(i, v + .07, f'{v:.2f}', ha='center', fontsize=8)
fig.tight_layout(); fig.savefig(f'{FIG}/fig4_isomer_blindness.png', bbox_inches='tight'); plt.close(fig)

# ---------- Fig 5: SHAP ----------
EN_FEAT = {'分子量': 'MolWt', '重原子数': 'Heavy atoms', '可旋转键数': 'Rotatable bonds', '氢键供体': 'HBD',
           '氢键受体': 'HBA', '极性表面积': 'TPSA', '芳环数': 'Aromatic rings', '总环数': 'Total rings',
           '杂原子数': 'Heteroatoms', 'S原子数': 'S atoms', 'N原子数': 'N atoms', 'O原子数': 'O atoms',
           'C原子数': 'C atoms', '分子里自由度': 'Spiro+bridgehead', '环状二硫': 'Cyclic disulfide'}
sdf = pd.read_csv(f'{OUT}/shap_importance_v4.csv').head(18)
sdf['disp'] = [EN_FEAT.get(c, 'fp' + c[2:] if c.startswith('fp') else
                           'MACCS' + c[2:] if c.startswith('mk') else c) for c in sdf['feature']]
dr = json.load(open(f'{OUT}/design_rules_summary.json'))
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.4))
ax[0].barh(sdf['disp'][::-1], sdf['mean_abs_shap'][::-1], color='#4C72B0')
ax[0].set_xlabel('mean |SHAP| (kJ mol$^{-1}$)'); ax[0].set_title('(a) Top-18 features')
ax[0].tick_params(axis='y', labelsize=7)
share = [dr['mech'], dr['morgan'], dr['base'], dr['maccs']]
ax[1].pie(share, labels=['mechanistic', 'Morgan FP', 'base', 'MACCS'],
          autopct='%1.0f%%', colors=['#C44E52', '#4C72B0', '#55A868', '#CCB974'],
          textprops={'fontsize': 8})
ax[1].set_title('(b) Attribution by block')
fig.tight_layout(); fig.savefig(f'{FIG}/fig5_shap.png', bbox_inches='tight'); plt.close(fig)

# ---------- Fig 6: design rules ----------
hb = pd.read_csv(f'{OUT}/rule_halogen_bins.csv')
ct = pd.read_csv(f'{OUT}/rule_crosstab.csv')
uv = pd.read_csv(f'{OUT}/rule_univariate.csv')
fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.9))
ax[0].plot(hb['卤素数'], hb['标签均值'], 'o-', color='#4C72B0', label='xTB label')
ax[0].plot(hb['卤素数'], hb['预测均值'], 's--', color='#C44E52', label='predicted')
ax[0].set_xlabel('Number of halogen substituents'); ax[0].set_ylabel(r'$\Delta E$ (kJ mol$^{-1}$)')
ax[0].set_title('(a) Halogen loading'); ax[0].legend(frameon=False, fontsize=7)
en_ct = {'酰基脂-烷氧基脂': 'acyl:Al / alkoxy:Al', '酰基脂-烷氧基芳': 'acyl:Al / alkoxy:Ar',
         '酰基芳-烷氧基脂': 'acyl:Ar / alkoxy:Al', '酰基芳-烷氧基芳': 'acyl:Ar / alkoxy:Ar'}
ax[1].barh([en_ct.get(s, s) for s in ct['酯类型']], ct['标签均值'], color=['#55A868', '#4C72B0', '#CCB974', '#C44E52'])
ax[1].axvline(0, color='k', lw=.8)
ax[1].set_xlabel(r'mean $\Delta E$ (kJ mol$^{-1}$)'); ax[1].set_title('(b) Ester type')
ax[1].tick_params(axis='y', labelsize=6)
sel = uv[uv['因素'].isin(['无任何芳环取代基', '含 OMe 取代', '脂肪链 ≥ 6（长链酯）', '脂肪链 ≤ 3（短链酯）',
                          '无卤素取代', '卤素数 ≥ 3'])]
en = ['no Ar substituent', 'contains OMe', 'long chain (>=6)', 'short chain (<=3)', 'no halogen', 'halogen >=3']
ax[2].barh(en, sel['标签均值_kJmol'].values, color='#8172B2')
ax[2].axvline(0, color='k', lw=.8); ax[2].tick_params(axis='y', labelsize=6)
ax[2].set_xlabel(r'mean $\Delta E$ (kJ mol$^{-1}$)'); ax[2].set_title('(c) Key structural factors')
fig.tight_layout(); fig.savefig(f'{FIG}/fig6_design_rules.png', bbox_inches='tight'); plt.close(fig)

# ---------- Fig 7: conformal ----------
_sf = f'{OUT}/v6_summary.json' if os.path.exists(f'{OUT}/v6_summary.json') else f'{OUT}/final_model_summary.json'
fm = json.load(open(_sf))
if 'mondrian' in fm:
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    classes = list(fm['mondrian'].keys())
    q = [fm['mondrian'][c]['q90'] for c in classes]
    ax[0].bar(classes, q, color='#4C72B0', width=.55)
    ax[0].set_ylabel('q90 (kJ mol$^{-1}$)'); ax[0].set_title('(a) Class-conditional interval width', fontsize=9.5)
    for i, v in enumerate(q):
        ax[0].text(i, v + .15, f'{v:.1f}', ha='center', fontsize=8)
    cov = [fm['mondrian_coverage'][c] for c in classes]
    ax[1].bar(classes, [0 if c is None else c for c in cov], color='#55A868', width=.55)
    ax[1].axhline(.9, color='k', ls='--', lw=1, label='nominal 90%')
    ax[1].set_ylim(0, 1.05); ax[1].set_ylabel('Empirical coverage')
    ax[1].set_title('(b) Coverage on held-out test', fontsize=9.5); ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig7_conformal.png', bbox_inches='tight'); plt.close(fig)
else:
    cs = fm.get('conformal_splits', [])
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax[0].bar([f'split {i+1}' for i in range(len(cs))], [c['q90'] for c in cs], color='#4C72B0', width=.5)
    ax[0].set_ylabel('q90 (kJ mol$^{-1}$)'); ax[0].set_title('(a) Conformal interval width (q90)', fontsize=9.5)
    ax[1].bar([f'split {i+1}' for i in range(len(cs))], [c['coverage'] for c in cs], color='#55A868', width=.5)
    ax[1].axhline(.9, color='k', ls='--', lw=1, label='nominal 90%')
    ax[1].set_ylim(0, 1.05); ax[1].set_ylabel('Empirical coverage')
    ax[1].set_title('(b) Coverage on held-out test', fontsize=9.5); ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig7_conformal.png', bbox_inches='tight'); plt.close(fig)

# ---------- Fig 9: 逐轮前瞻验证 ----------
if 'prospective' in fm:
    pr = fm['prospective']
    ks = [k for k in ['AL3', 'AL4', 'AL5'] if k in pr]
    fig, ax = plt.subplots(1, 3, figsize=(8.2, 2.7))
    ax[0].bar(ks, [pr[k]['Spearman'] for k in ks], color='#4C72B0', width=.5)
    ax[0].set_ylabel('Spearman $\\rho$'); ax[0].set_ylim(0, 1)
    ax[0].set_title('(a) Prospective ranking', fontsize=9)
    for i, k in enumerate(ks):
        ax[0].text(i, pr[k]['Spearman'] + .02, f"{pr[k]['Spearman']:.3f}", ha='center', fontsize=7.5)
    ax[1].bar(ks, [pr[k]['MAE'] for k in ks], color='#C44E52', width=.5)
    ax[1].set_ylabel('MAE (kJ mol$^{-1}$)'); ax[1].set_title('(b) Prospective MAE', fontsize=9)
    for i, k in enumerate(ks):
        ax[1].text(i, pr[k]['MAE'] + .1, f"{pr[k]['MAE']:.2f}", ha='center', fontsize=7.5)
    ax[2].bar(ks, [pr[k]['R2'] for k in ks], color='#55A868', width=.5)
    ax[2].set_ylabel('R$^2$'); ax[2].axhline(0, color='k', lw=.8)
    ax[2].set_title('(c) Prospective R$^2$', fontsize=9)
    for i, k in enumerate(ks):
        ax[2].text(i, pr[k]['R2'] + .02, f"{pr[k]['R2']:.3f}", ha='center', fontsize=7.5)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig9_prospective.png', bbox_inches='tight'); plt.close(fig)

print('图表完成 →', FIG)
for f in sorted(os.listdir(FIG)):
    print('  ', f)
