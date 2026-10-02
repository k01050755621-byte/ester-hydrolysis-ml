# -*- coding: utf-8 -*-
"""ML 补强 · 第 3 步：设计规则提取（把"框架"变成"发现"）

三路证据：
  A 可解释线性模型：机理描述符 → 水解能，标准化系数即"规则强度"
  B 单变量趋势：每个结构因素分箱后的标签/预测均值
  C SHAP 归因：树模型的全局重要性 + 方向

输出：
  rule_linear_coefs.csv    线性模型系数（含符号与显著性）
  rule_univariate.csv      单变量趋势
  rule_crosstab.csv        化学类别交叉表
  shap_importance_v4.csv   SHAP 重要性
"""
import os, json
import numpy as np, pandas as pd
import shap
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from scipy.stats import spearmanr, pearsonr, t as tdist
from sklearn.metrics import mean_absolute_error, r2_score
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

HOME = os.path.expanduser('~/polywork'); os.chdir(HOME)
OUT = 'fix_20260928/ml_final'

FV = pd.read_csv(f'{OUT}/feat_vlib.csv'); FL = pd.read_csv(f'{OUT}/feat_lit.csv')
_lf = (os.environ.get('LABFILE')
       or (f'{OUT}/labels_final.csv' if os.path.exists(f'{OUT}/labels_final.csv') else f'{OUT}/labels_452.csv'))
LAB = pd.read_csv(_lf); MAN = json.load(open(f'{OUT}/feature_manifest.json'))
BASE, MECH = MAN['base_cols'], MAN['mech_cols']
FP = [c for c in FV.columns if c.startswith('fp') and FV[c].sum() > 0]
MK = [c for c in FV.columns if c.startswith('mk') and FV[c].sum() > 0]
BEST = BASE + MECH + FP + MK

ALL = pd.concat([FV, FL], ignore_index=True).fillna(0)
tr = ALL.merge(LAB[['编号', 'dE水解_kJmol', '来源']], on='编号', how='inner')
y = tr['dE水解_kJmol'].values
X = tr[BEST].values

# ---- A. 可解释线性模型（仅机理块，标准化）----
print('=== A. 可解释线性模型（机理描述符）===')
Xm = tr[MECH].values
pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
pred_cv = cross_val_predict(pipe, Xm, y, cv=5)
print(f'  机理线性模型 5 折 OOF: MAE={mean_absolute_error(y, pred_cv):.2f}  R2={r2_score(y, pred_cv):+.3f}  '
      f'rho={spearmanr(pred_cv, y)[0]:.3f}')
pipe.fit(Xm, y)
sc = pipe.named_steps['standardscaler']; lr = pipe.named_steps['ridgecv']
coefs = lr.coef_ / 1.0  # 已是标准化输入 → 系数即"每 1 SD 变化的 kJ/mol"
# 显著性（对 OOF 用 bootstrap）
boot = []
rng = np.random.RandomState(0)
for _ in range(400):
    idx = rng.randint(0, len(y), len(y))
    p = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))).fit(Xm[idx], y[idx])
    boot.append(p.named_steps['ridgecv'].coef_)
boot = np.array(boot)
rows = []
for i, c in enumerate(MECH):
    lo, hi = np.percentile(boot[:, i], [2.5, 97.5])
    rows.append({'描述符': c, '系数_kJmol_per_SD': round(float(coefs[i]), 3),
                 'CI2.5': round(float(lo), 3), 'CI97.5': round(float(hi), 3),
                 '显著': bool(lo * hi > 0)})
rd = pd.DataFrame(rows).sort_values('系数_kJmol_per_SD', key=np.abs, ascending=False)
rd.to_csv(f'{OUT}/rule_linear_coefs.csv', index=False)
print(rd.head(14).to_string(index=False))

# ---- B. 单变量趋势（标签 + 全库预测）----
print('\n=== B. 单变量趋势 ===')
model = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                     colsample_bytree=0.7, random_state=42, n_jobs=4).fit(X, y)
FVp = FV.copy(); FVp['pred_v4'] = model.predict(FV[BEST].values)
FVp['label'] = FVp['编号'].map(dict(zip(LAB['编号'], LAB['dE水解_kJmol'])))

uv = []
def add_uv(name, mask_lab, mask_vlib=None):
    if mask_lab.sum() == 0: return
    uv.append({'因素': name, 'n标签': int(mask_lab.sum()),
               '标签均值_kJmol': round(float(y[mask_lab.values] if hasattr(mask_lab,'values') else y[mask_lab].mean() if False else 0),2)})

# 手工构造更稳
def stat(name, m_tr, m_vl):
    d = {'因素': name, 'n标签': int(m_tr.sum())}
    if m_tr.sum() > 0:
        d['标签均值_kJmol'] = round(float(y[m_tr.values].mean()), 2)
        d['标签中位'] = round(float(np.median(y[m_tr.values])), 2)
    else:
        d['标签均值_kJmol'] = None; d['标签中位'] = None
    d['预测均值_kJmol'] = round(float(FVp.loc[m_vl.values, 'pred_v4'].mean()), 2) if m_vl.sum() else None
    d['库内占比%'] = round(100 * float(m_vl.sum()) / len(FVp), 1)
    return d

tri = tr.copy()
mv = lambda c, op: (FVp[c] > 0) if op == '>' else (FVp[c] == 0)
mt = lambda c, op: (tri[c] > 0) if op == '>' else (tri[c] == 0)

uv = []
uv.append(stat('酰基侧为芳环（苯甲酸型酯）', mt('n_acylAr','>'), mv('n_acylAr','>')))
uv.append(stat('烷氧基侧为芳环（酚酯）', mt('n_alkAr','>'), mv('n_alkAr','>')))
uv.append(stat('纯脂肪族酯（两侧皆脂肪）', (mt('n_acylAr','=')) & (mt('n_alkAr','=')), (mv('n_acylAr','=')) & (mv('n_alkAr','='))))
for k, c in [('无卤素','n_halogen'), ('含 1 个卤素','x')]:
    pass
uv.append(stat('无卤素取代', tri['n_halogen'] == 0, FVp['n_halogen'] == 0))
uv.append(stat('含卤素取代（F/Cl/Br/I）', tri['n_halogen'] > 0, FVp['n_halogen'] > 0))
uv.append(stat('卤素数 ≥ 3', tri['n_halogen'] >= 3, FVp['n_halogen'] >= 3))
uv.append(stat('含 OMe 取代', tri['n_arOMe'] > 0, FVp['n_arOMe'] > 0))
uv.append(stat('含 Me 取代', tri['n_arMe'] > 0, FVp['n_arMe'] > 0))
uv.append(stat('无任何芳环取代基', tri['ar_subst_total'] == 0, FVp['ar_subst_total'] == 0))
uv.append(stat('羰基与芳环共轭', tri['n_conjCO'] > 0, FVp['n_conjCO'] > 0))
uv.append(stat('含 S–S 键', tri['ss_flag'] > 0, FVp['ss_flag'] > 0))
uv.append(stat('脂肪链 ≥ 6（长链酯）', tri['longest_chain'] >= 6, FVp['longest_chain'] >= 6))
uv.append(stat('脂肪链 ≤ 3（短链酯）', tri['longest_chain'] <= 3, FVp['longest_chain'] <= 3))
uvdf = pd.DataFrame(uv)
uvdf.to_csv(f'{OUT}/rule_univariate.csv', index=False)
print(uvdf.to_string(index=False))

# ---- B2. 卤素数分箱 ----
bins = [(-1,0,'0'),(0,1,'1'),(1,2,'2'),(2,3,'3'),(3,99,'≥4')]
rows = []
for lo,hi,lab_ in bins:
    m_tr = (tri['n_halogen'] > lo) & (tri['n_halogen'] <= hi)
    m_vl = (FVp['n_halogen'] > lo) & (FVp['n_halogen'] <= hi)
    if m_tr.sum() or m_vl.sum():
        rows.append({'卤素数': lab_, 'n标签': int(m_tr.sum()),
                     '标签均值': round(float(y[m_tr.values].mean()),2) if m_tr.sum() else None,
                     '预测均值': round(float(FVp.loc[m_vl.values,'pred_v4'].mean()),2) if m_vl.sum() else None})
hal_df = pd.DataFrame(rows); hal_df.to_csv(f'{OUT}/rule_halogen_bins.csv', index=False)
print('\n卤素数分箱:'); print(hal_df.to_string(index=False))

# ---- B3. 交叉表：酰基/烷氧基芳香性 ----
rows = []
for a_ar in [False, True]:
    for o_ar in [False, True]:
        nm = f"酰基{'芳' if a_ar else '脂'}-烷氧基{'芳' if o_ar else '脂'}"
        m_tr = (tri['n_acylAr'] > 0) == a_ar
        m_tr &= (tri['n_alkAr'] > 0) == o_ar
        m_vl = (FVp['n_acylAr'] > 0) == a_ar
        m_vl &= (FVp['n_alkAr'] > 0) == o_ar
        rows.append({'酯类型': nm, 'n标签': int(m_tr.sum()),
                     '标签均值': round(float(y[m_tr.values].mean()),2) if m_tr.sum() else None,
                     '预测均值': round(float(FVp.loc[m_vl.values,'pred_v4'].mean()),2) if m_vl.sum() else None,
                     '库内数': int(m_vl.sum())})
ct = pd.DataFrame(rows); ct.to_csv(f'{OUT}/rule_crosstab.csv', index=False)
print('\n酯类型交叉表:'); print(ct.to_string(index=False))

# ---- C. SHAP ----
print('\n=== C. SHAP ===')
expl = shap.TreeExplainer(model)
sv = expl.shap_values(tr[BEST].values[:400])
imp = np.abs(sv).mean(axis=0)
sdf = pd.DataFrame({'feature': BEST, 'mean_abs_shap': imp}).sort_values('mean_abs_shap', ascending=False)
sdf.to_csv(f'{OUT}/shap_importance_v4.csv', index=False)
print(sdf.head(18).to_string(index=False))
# 机理块占比
mech_share = float(sdf[sdf['feature'].isin(MECH)]['mean_abs_shap'].sum() / sdf['mean_abs_shap'].sum())
base_share = float(sdf[sdf['feature'].isin(BASE)]['mean_abs_shap'].sum() / sdf['mean_abs_shap'].sum())
fp_share = float(sdf[sdf['feature'].str.startswith('fp')]['mean_abs_shap'].sum() / sdf['mean_abs_shap'].sum())
mk_share = float(sdf[sdf['feature'].str.startswith('mk')]['mean_abs_shap'].sum() / sdf['mean_abs_shap'].sum())
print(f'\n块贡献占比: 基础 {base_share:.1%} | 机理 {mech_share:.1%} | Morgan {fp_share:.1%} | MACCS {mk_share:.1%}')
json.dump({'base': base_share, 'mech': mech_share, 'morgan': fp_share, 'maccs': mk_share,
           'top20': sdf.head(20).to_dict('records'),
           'linear_model': {'MAE': round(float(mean_absolute_error(y, pred_cv)),2),
                            'R2': round(float(r2_score(y, pred_cv)),3),
                            'rho': round(float(spearmanr(pred_cv, y)[0]),3)}},
          open(f'{OUT}/design_rules_summary.json', 'w'), ensure_ascii=False, indent=2, default=str)
print('\n完成。')
