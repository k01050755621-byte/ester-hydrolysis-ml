# -*- coding: utf-8 -*-
"""核查 100k 扩库全部产物的一致性与化学有效性（只读不改）"""
import json, sys
import pandas as pd

P = "/Users/Zhuanz/polywork"
ML = f"{P}/fix_20260928/ml_final"

errors = []
def chk(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name} {detail}")
    if not cond:
        errors.append(name)

# ---------- 1. 枚举库 v4 ----------
v4 = pd.read_csv(f"{P}/枚举库_ester_v4.csv")
chk("v4 行数=87620", len(v4) == 87620, f"实际 {len(v4)}")
ids = v4["编号"].astype(str)
nums = ids.str[1:].astype(int)
chk("v4 ID 范围 C12884-C100503", nums.min() == 12884 and nums.max() == 100503,
    f"min=C{nums.min()} max=C{nums.max()}")
chk("v4 ID 无重复", ids.is_unique, f"重复 {len(ids)-ids.nunique()}")
# ID 连续性（抽样87620个自7290万，ID 应该连续分配）
chk("v4 ID 连续", (nums.diff().dropna() == 1).all(), "")
chk("v4 SMILES 无重复", v4["SMILES"].is_unique,
    f"重复 {len(v4)-v4['SMILES'].nunique()}")
chk("v4 动态键类型全为酯键", (v4["动态键类型"] == "酯键").all())

# SMILES 化学有效性 + 酯键存在（抽样 3000 + 全部 top 候选）
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import random
random.seed(0)
sample_idx = random.sample(range(len(v4)), 3000)
bad_parse, no_ester = [], []
for i in sample_idx:
    smi = v4["SMILES"].iloc[i]
    m = Chem.MolFromSmiles(smi)
    if m is None:
        bad_parse.append(smi)
    else:
        smi2 = Chem.MolToSmiles(m)
        # 酯键: O=C-O 芳香或脂肪均可
        pat = Chem.MolFromSmarts("[CX3](=O)[OX2H0]")
        if not m.HasSubstructMatch(pat):
            no_ester.append(smi)
chk("v4 抽样3000 SMILES 全部可解析", len(bad_parse) == 0, f"坏 {len(bad_parse)} 例: {bad_parse[:3]}")
chk("v4 抽样3000 全部含酯键", len(no_ester) == 0, f"缺酯 {len(no_ester)} 例: {no_ester[:3]}")

# ---------- 2. 100k 预测文件 ----------
pred = pd.read_csv(f"{ML}/vlib_100k_predictions_v7.csv")
print("\n预测文件列:", list(pred.columns))
chk("pred 行数=100000", len(pred) == 100000, f"实际 {len(pred)}")
chk("pred 编号无重复", pred["编号"].astype(str).is_unique)
# pred = 87,620 新分子(v4) + 12,380 老分子(v7 主库)，两者应正好拼成 100k
v4set = set(v4["编号"].astype(str))
pset = set(pred["编号"].astype(str))
overlap = len(v4set & pset)
chk("pred 覆盖 v4 全部 87620 + 主库 12380", v4set <= pset,
    f"交集 {overlap}，v4独有 {len(v4set-pset)}")
# 分子量、卤素数列存在性
# 库 列: 老/新划分
if "库" in pred.columns:
    vc = pred["库"].value_counts()
    print("    库列取值:", dict(vc))
new_by_col = pred[pred["编号"].astype(str).str[1:].astype(int) >= 12884]
chk("新分子(编号>=12884)=87620", len(new_by_col) == 87620, f"实际 {len(new_by_col)}")
ci_neg_new = int((new_by_col["CI上_v7"] < 0).sum())
chk("CI全负中新分子=1227", ci_neg_new == 1227, f"实际 {ci_neg_new}")
if "n_hal" in pred.columns:
    hf = pred[pred["n_hal"] == 0]
    chk("无卤分子=8084", len(hf) == 8084, f"实际 {len(hf)}")
    if "CI上_v7" in pred.columns:
        hf_neg = hf[hf["CI上_v7"] < 0]
        print(f"    无卤且CI全负 = {len(hf_neg)}（备忘录记 26）")
        chk("无卤+CI全负=26", len(hf_neg) == 26, f"实际 {len(hf_neg)}")
    # 老库无卤 2801
    old = pred[pred["编号"].astype(str).str[1:].astype(int) < 12884]
    hf_old = int((old["n_hal"] == 0).sum())
    chk("老库无卤=2801", hf_old == 2801, f"实际 {hf_old}")
else:
    print("pred 无 n_hal 列，另行核对")
# CI 全负
ci_neg = pred[pred["CI上_v7"] < 0]
chk("CI全负=1370", len(ci_neg) == 1370, f"实际 {len(ci_neg)}")
# 不同预测值（与 ml4/7/8/11/12/17 一致: round(4)）
nd = pred["预测_v7"].round(4).nunique()
chk("不同预测值=81963(82.0%) [round4口径]", nd == 81963, f"实际 {nd} ({nd/1000:.1f}%)")
# 新旧预测分布统计与汇总一致
import json as _json
S = _json.load(open(f"{ML}/expand100k_summary.json"))
pn = new_by_col["预测_v7"]
chk("pred_new.mean≈9.42", abs(pn.mean() - S["pred_new"]["mean"]) < 0.01, f"实际 {pn.mean():.2f}")
chk("pred_new.sd≈12.03", abs(pn.std() - S["pred_new"]["sd"]) < 0.01, f"实际 {pn.std():.2f}")
po = old["预测_v7"]
chk("pred_old.mean≈11.86", abs(po.mean() - S["pred_old"]["mean"]) < 0.01, f"实际 {po.mean():.2f}")
chk("pred_old.sd≈10.18", abs(po.std() - S["pred_old"]["sd"]) < 0.01, f"实际 {po.std():.2f}")
# 全库最负
i_min = pred["预测_v7"].idxmin()
row = pred.loc[i_min]
chk("全库最负是 C2001 (-25.3)", str(row["编号"]) == "C2001",
    f"实际 {row['编号']} ({row['预测_v7']:.1f})")
# 新分子最深 -19.0
new_mask = pred["编号"].astype(str).str[1:].astype(int) >= 12884
new_min = pred.loc[new_mask, "预测_v7"].min()
chk("新分子最深≈-19.0", abs(new_min - (-19.0)) < 0.1, f"实际 {new_min:.2f}")

# ---------- 3. 与 v7 主库预测的一致性（12,380 老分子预测应与 vlib_predictions_v7.csv 一致） ----------
v7 = pd.read_csv(f"{ML}/vlib_predictions_v7.csv")
print("\nv7 主库列:", list(v7.columns))
m = pred.merge(v7, on="编号", suffixes=("_100k", "_v7"))
chk("pred 与 v7 主库可对上 12380 行", len(m) == 12380, f"实际 {len(m)}")
if "预测_v7" in v7.columns and len(m) == 12380:
    d = (m["预测_v7_100k"] - m["预测_v7_v7"]).abs().max()
    chk("老分子预测与主库一致(max diff < 1e-8)", d < 1e-8, f"max diff {d:.2e}")
if "CI上_v7" in v7.columns and len(m) == 12380:
    d2 = (m["CI上_v7_100k"] - m["CI上_v7_v7"]).abs().max()
    chk("老分子CI上与主库一致", d2 < 1e-8, f"max diff {d2:.2e}")

# ---------- 4. 排序组 5 支预测不变 ----------
rk_path = f"{ML}/../exp1_ranking_set_v7.csv"
try:
    rk = pd.read_excel(rk_path)
except Exception:
    rk = pd.read_csv(rk_path, encoding="gbk")
idcol = rk.columns[0] if rk.columns[0] not in ("编号",) else "编号"
if idcol != "编号":
    rk = rk.rename(columns={idcol: "编号"})
rk_ids = rk["编号"].astype(str).tolist()
print("\n排序组:", rk_ids)
sub = pred[pred["编号"].astype(str).isin(rk_ids)]
chk("排序组 5 支全部在 100k 预测中", len(sub) == len(rk_ids), f"命中 {len(sub)}")
for _, r in sub.iterrows():
    old = v7[v7["编号"].astype(str) == str(r["编号"])]
    if len(old):
        dd = abs(float(r["预测_v7"]) - float(old["预测_v7"].iloc[0]))
        if dd > 1e-8:
            errors.append(f"排序组 {r['编号']} 预测漂移 {dd:.2e}")
            print(f"[FAIL] 排序组 {r['编号']} 预测漂移 {dd:.2e}")
chk("排序组预测无漂移", not any("排序组" in e and "漂移" in e for e in errors))

print("\n" + "=" * 50)
if errors:
    print(f"共 {len(errors)} 项失败:")
    for e in errors:
        print(" -", e)
    sys.exit(1)
else:
    print("全部检查通过 ✓")
