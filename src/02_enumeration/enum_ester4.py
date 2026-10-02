# -*- coding: utf-8 -*-
"""枚举库 v4：把酯虚拟库从 12,380 扩到 100,000。

化学空间（相对 v3 的三处扩展，其余完全继承）：
  1) 芳环取代基 6 -> 9 种：F/Cl/Br/Me/OMe/CF3 + I/NO2/OH
     （机理特征块本就含 n_arI/n_arNO2/n_arOH 计数；NO2 按 [N+](=O)[O-] 正确成键）
  2) 单取代 -> 单取代 + 双取代（不同芳环位置，含同基团双取代如二氯）
  3) 烷基帽 alkC1-8 -> alkC1-10

全组合空间 = cores_aug(585) × C(caps_aug+1, 2) ≈ 1.5e8，不做全枚举，
按 v3 的同款方法做确定性随机抽样（RandomState(42)），直到收满 TARGET 个
新分子。过滤器与 v3 完全一致：恰 2 个酯键、MW 150-500、重原子<=40、
canonical SMILES 去重、排除原料库与 v1/v2/v3。

输出：枚举库_ester_v4.csv，编号 C12884 起。
"""
import pandas as pd, itertools, sys, time
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

base = "/Users/Zhuanz/polywork/"
ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2H0]")
TARGET = 87620          # 12,380 + 87,620 = 100,000
MAX_ATTEMPTS = 2_000_000
ID_START = 12884

def canon(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToSmiles(m) if m else None

# ---------- 排除集：原料库 + v1 + v2 + v3 ----------
existing, ex_ids = set(), set()
df = pd.read_excel(base + "原料库_v1.xlsx")
for s in df["SMILES"]:
    c = canon(str(s)); ex_ids.add(str(s))
    if c: existing.add(c)
for f in ["枚举库_ester_v1.csv", "枚举库_ester_v2.csv", "枚举库_ester_v3.csv"]:
    d = pd.read_csv(base + f)
    for s in d["SMILES"]:
        c = canon(s)
        if c: existing.add(c)
    ex_ids |= set(d["编号"])
print("existing (原料库+v1+v2+v3): %d 个 SMILES / %d 个编号" % (len(existing), len(ex_ids)), flush=True)

# ---------- 1. 双切建桶（同 v2/v3）----------
cores_A, caps_B = {}, {}
for _, r in df.iterrows():
    m = Chem.MolFromSmiles(str(r["SMILES"]))
    if not m or len(m.GetSubstructMatches(ESTER)) != 2: continue
    bidx = []
    for b in m.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE: continue
        if {a1.GetSymbol(), a2.GetSymbol()} != {"C", "O"}: continue
        c_at = a1 if a1.GetSymbol() == "C" else a2
        is_carb = any(nb.GetSymbol() == "O" and
                      (lambda bb: bb is not None and bb.GetBondType() == Chem.BondType.DOUBLE)(
                          m.GetBondBetweenAtoms(nb.GetIdx(), c_at.GetIdx()))
                      for nb in c_at.GetNeighbors())
        if is_carb: bidx.append(b.GetIdx())
    if len(bidx) != 2: continue
    frag = Chem.FragmentOnBonds(m, bidx, dummyLabels=[(0, 0), (0, 0)])
    for p in Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=True):
        smi = Chem.MolToSmiles(p)
        nd = sum(1 for a in p.GetAtoms() if a.GetSymbol() == "*")
        if nd == 2: cores_A.setdefault(smi, r["编号"])
        elif nd == 1: caps_B.setdefault(smi, r["编号"])
print("cores_A: %d  caps_B: %d" % (len(cores_A), len(caps_B)), flush=True)

# ---------- 2. 装饰：9 取代基 × 单/双取代，NO2 双键正确 ----------
SUBST = {"F": ["F"], "Cl": ["Cl"], "Br": ["Br"], "I": ["I"],
         "Me": ["C"], "OMe": ["O", "C"], "CF3": ["C", "F", "F", "F"],
         "NO2": ["N", "O", "O"], "OH": ["O"]}

def attach(rw, pos, chain):
    """在 rw 的 pos 上接取代基链，返回最后的原子 idx。NO2 -> [N+](=O)[O-]。"""
    prev = pos
    news = []
    for el in chain:
        ni = rw.AddAtom(Chem.Atom(el)); rw.AddBond(prev, ni, Chem.BondType.SINGLE)
        prev = ni; news.append(ni)
    if chain == ["N", "O", "O"]:
        rw.GetAtomWithIdx(news[0]).SetFormalCharge(1)
        rw.GetAtomWithIdx(news[2]).SetFormalCharge(-1)
        b = rw.GetBondBetweenAtoms(news[0], news[1])
        b.SetBondType(Chem.BondType.DOUBLE)
    return prev

def arom_sites(m):
    return [a.GetIdx() for a in m.GetAtoms()
            if a.GetIsAromatic() and a.GetSymbol() == "C" and a.GetTotalNumHs() >= 1]

def decorate12(smi, src):
    m = Chem.MolFromSmiles(smi)
    out = {}
    if m is None: return out
    sites = arom_sites(m)
    for p in sites:
        for tag, ch in SUBST.items():
            try:
                rw = Chem.RWMol(m); attach(rw, p, ch); Chem.SanitizeMol(rw)
                out.setdefault(Chem.MolToSmiles(rw), "%s_%s%d" % (src, tag, p))
            except Exception: pass
    for p, q in itertools.combinations(sites, 2):
        for t1, c1 in SUBST.items():
            for t2, c2 in SUBST.items():
                if t1 > t2: continue
                try:
                    rw = Chem.RWMol(m); attach(rw, p, c1); attach(rw, q, c2)
                    Chem.SanitizeMol(rw)
                    out.setdefault(Chem.MolToSmiles(rw), "%s_%s%d_%s%d" % (src, t1, p, t2, q))
                except Exception: pass
    return out

cores_aug = dict(cores_A)
for smi, src in list(cores_A.items()):
    cores_aug.update(decorate12(smi, src))
caps_aug = dict(caps_B)
for n in range(1, 11):
    caps_aug.setdefault("*O" + "C" * n, "alkC%d" % n)
for smi, tag in [("*OC(C)C", "iPr"), ("*OCC(C)C", "iBu"), ("*OC(C)(C)C", "tBu"),
                 ("*OCc1ccccc1", "Bn"), ("*OC1CCCCC1", "Cy"), ("*Oc1ccccc1", "Ph")]:
    caps_aug.setdefault(smi, tag)
for smi, src in list(caps_aug.items()):
    caps_aug.update(decorate12(smi, src))
CL = list(caps_aug.items())
print("cores_aug: %d  caps_aug: %d  | 全空间 = %d × C(%d+1,2) = %s" %
      (len(cores_aug), len(CL), len(cores_aug), len(CL),
       format(len(cores_aug) * len(CL) * (len(CL) + 1) // 2, ",")), flush=True)

# ---------- 3. 重组 ----------
def join2(smi_a, smi_b):
    m = Chem.CombineMols(Chem.MolFromSmiles(smi_a), Chem.MolFromSmiles(smi_b))
    rw = Chem.RWMol(m)
    dums = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "*"]
    if len(dums) < 2: return None
    d0, d1 = dums[0], dums[-1]
    n0 = rw.GetAtomWithIdx(d0).GetNeighbors()[0].GetIdx()
    n1 = rw.GetAtomWithIdx(d1).GetNeighbors()[0].GetIdx()
    rw.AddBond(n0, n1, Chem.BondType.SINGLE)
    for d in sorted([d0, d1], reverse=True):
        rw.RemoveAtom(d)
    try:
        Chem.SanitizeMol(rw); return Chem.MolToSmiles(rw)
    except Exception:
        return None

rng = np.random.RandomState(42)
core_list = list(cores_aug.items())
nc, nk = len(core_list), len(CL)
seen, rows = set(), []
n_try = n_bad = n_dup = n_exist = n_mw = 0
t0 = time.time()
while len(rows) < TARGET and n_try < MAX_ATTEMPTS:
    ci = rng.randint(nc); k1 = rng.randint(nk); k2 = rng.randint(nk)
    if k1 > k2: k1, k2 = k2, k1
    n_try += 1
    core_smi, core_id = core_list[ci]
    c1, i1 = CL[k1]; c2, i2 = CL[k2]
    half = join2(core_smi, c1)
    prod = join2(half, c2) if half else None
    if prod is None: n_bad += 1; continue
    if prod in seen: n_dup += 1; continue
    if prod in existing: n_exist += 1; continue
    m = Chem.MolFromSmiles(prod)
    if len(m.GetSubstructMatches(ESTER)) != 2: n_bad += 1; continue
    mw = Descriptors.MolWt(m)
    if not (150 <= mw <= 500) or m.GetNumHeavyAtoms() > 40: n_mw += 1; continue
    seen.add(prod)
    rows.append({"SMILES": prod, "名称": "D(%sx%sx%s)" % (core_id, i1, i2)})
    if len(rows) % 10000 == 0:
        el = time.time() - t0
        print("  %d/%d 收集 | 尝试 %d | 速率 %.0f/s | 已用 %.1f min" %
              (len(rows), TARGET, n_try, n_try / el if el else 0, el / 60), flush=True)

print("尝试 %d | 无效 %d | 重复 %d | 已存在 %d | MW/原子过滤 %d | 收集 %d | 用时 %.1f min"
      % (n_try, n_bad, n_dup, n_exist, n_mw, len(rows), (time.time() - t0) / 60), flush=True)
assert len(rows) == TARGET, "未收满目标：%d/%d（请提高 MAX_ATTEMPTS 或检查过滤器）" % (len(rows), TARGET)

# ---------- 4. 落盘 ----------
out = pd.DataFrame(rows)
out.insert(0, "编号", ["C%d" % (ID_START + i) for i in range(len(out))])
assert not (set(out["编号"]) & ex_ids), "编号冲突！"
out["角色"] = "模型化合物"
out["动态键类型"] = "酯键"
out["文献来源DOI"] = ""
out["备注"] = "枚举虚拟库v4_9取代基单双取代_抽样10万"
out = out[["编号", "角色", "名称", "SMILES", "动态键类型", "文献来源DOI", "备注"]]
assert out["SMILES"].nunique() == len(out), "SMILES 有重复！"
out.to_csv(base + "枚举库_ester_v4.csv", index=False)
print("saved: 枚举库_ester_v4.csv, %d rows, IDs C%d..C%d"
      % (len(out), ID_START, ID_START + len(out) - 1), flush=True)
