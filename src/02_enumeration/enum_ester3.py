import pandas as pd, itertools, random
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

base = "/path/to/polywork/"
ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2H0]")

def canon(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToSmiles(m) if m else None

# ---------- 排除集：原料库 + v1 + v2 ----------
existing = set()
df = pd.read_excel(base + "原料库_v1.xlsx")
for s in df["SMILES"]:
    c = canon(s)
    if c: existing.add(c)
for f in ["枚举库_ester_v1.csv", "枚举库_ester_v2.csv"]:
    for s in pd.read_csv(base + f)["SMILES"]:
        c = canon(s)
        if c: existing.add(c)
print("existing total: %d" % len(existing))

# ---------- 1. 双切建桶（同 v2） ----------
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
print("library cores_A: %d   caps_B: %d" % (len(cores_A), len(caps_B)))

# ---------- 2. 取代基装饰 ----------
SUBST = {"F": ["F"], "Cl": ["Cl"], "Br": ["Br"],
         "Me": ["C"], "OMe": ["O", "C"], "CF3": ["C", "F", "F", "F"]}

def decorate(smi, src):
    m = Chem.MolFromSmiles(smi)
    out = {}
    for a in m.GetAtoms():
        if a.GetIsAromatic() and a.GetSymbol() == "C" and a.GetTotalNumHs() >= 1:
            pos = a.GetIdx()
            for tag, chain in SUBST.items():
                rw = Chem.RWMol(m)
                prev = pos
                for el in chain:
                    ni = rw.AddAtom(Chem.Atom(el))
                    rw.AddBond(prev, ni, Chem.BondType.SINGLE)
                    prev = ni
                try:
                    Chem.SanitizeMol(rw)
                    out.setdefault(Chem.MolToSmiles(rw), "%s_%s%d" % (src, tag, pos))
                except Exception:
                    pass
    return out

cores_aug = dict(cores_A)
for smi, src in list(cores_A.items()):
    cores_aug.update(decorate(smi, src))

caps_aug = dict(caps_B)
for n in range(1, 9):
    caps_aug.setdefault("*O" + "C" * n, "alkC%d" % n)
for smi, tag in [("*OC(C)C", "iPr"), ("*OCC(C)C", "iBu"), ("*OC(C)(C)C", "tBu"),
                 ("*OCc1ccccc1", "Bn"), ("*OC1CCCCC1", "Cy"), ("*Oc1ccccc1", "Ph")]:
    caps_aug.setdefault(smi, tag)
for smi, src in list(caps_aug.items()):
    caps_aug.update(decorate(smi, src))
print("augmented cores: %d   caps: %d" % (len(cores_aug), len(caps_aug)))

# ---------- 3. 三体重组 ----------
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
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw)
    except Exception:
        return None

combos = []
cl = list(caps_aug.items())
for core_smi, core_id in cores_aug.items():
    for (c1, i1), (c2, i2) in itertools.combinations_with_replacement(cl, 2):
        combos.append((core_smi, core_id, c1, i1, c2, i2))
print("raw combos: %d" % len(combos))

random.seed(42)
random.shuffle(combos)
seen, rows = set(), []
n_bad = n_dup = n_exist = 0
LIMIT = 12000
for core_smi, core_id, c1, i1, c2, i2 in combos:
    if len(rows) >= LIMIT: break
    half = join2(core_smi, c1)
    prod = join2(half, c2) if half else None
    if prod is None: n_bad += 1; continue
    if prod in seen: n_dup += 1; continue
    seen.add(prod)
    if prod in existing: n_exist += 1; continue
    m = Chem.MolFromSmiles(prod)
    if len(m.GetSubstructMatches(ESTER)) != 2: n_bad += 1; continue
    mw = Descriptors.MolWt(m)
    if not (150 <= mw <= 500) or m.GetNumHeavyAtoms() > 40: n_bad += 1; continue
    rows.append({"SMILES": prod, "名称": "D(%sx%sx%s)" % (core_id, i1, i2)})

print("invalid: %d  dup: %d  existing: %d" % (n_bad, n_dup, n_exist))
print("FINAL new molecules: %d" % len(rows))

# ---------- 4. 编号 C884 起 ----------
out = pd.DataFrame(rows)
out.insert(0, "编号", ["C%d" % (884 + i) for i in range(len(out))])
out["角色"] = "模型化合物"
out["动态键类型"] = "酯键"
out["文献来源DOI"] = ""
out["备注"] = "枚举虚拟库v3_取代基装饰"
out = out[["编号", "角色", "名称", "SMILES", "动态键类型", "文献来源DOI", "备注"]]
out.to_csv(base + "枚举库_ester_v3.csv", index=False)
print(out.head(5).to_string())
print("saved: 枚举库_ester_v3.csv, %d rows, IDs C884..C%d" % (len(out), 883 + len(out)))
