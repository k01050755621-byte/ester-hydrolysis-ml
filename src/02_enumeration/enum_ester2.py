import pandas as pd, itertools, random
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

base = "/Users/Zhuanz/polywork/"
ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2H0]")
CARB  = Chem.MolFromSmarts("[*]C(=O)")

def canon(smi):
    if not isinstance(smi, str): return None
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None

def frag_type(smi):
    m = Chem.MolFromSmiles(smi)
    for a in m.GetAtoms():
        if a.GetSymbol() == "*":
            nb = a.GetNeighbors()[0]
            if nb.GetSymbol() == "C":
                for n2 in nb.GetNeighbors():
                    if n2.GetSymbol() == "O":
                        b = m.GetBondBetweenAtoms(nb.GetIdx(), n2.GetIdx())
                        if b and b.GetBondType() == Chem.BondType.DOUBLE:
                            return "A"
            if nb.GetSymbol() == "O": return "B"
    return "?"

# ---------- 1. 排除集 ----------
existing = set()
df = pd.read_excel(base + "原料库_v1.xlsx")
for s in df["SMILES"]:
    c = canon(s)
    if c: existing.add(c)
try:
    v1 = pd.read_csv(base + "枚举库_ester_v1.csv")
    for s in v1["SMILES"]:
        c = canon(s)
        if c: existing.add(c)
    print("v1 excluded: %d" % len(v1))
except Exception as e:
    print("no v1: %s" % e)
print("existing total: %d" % len(existing))

# ---------- 2. 双切 ----------
cores = {"A": {}, "B": {}}   # core smiles -> 来源
caps  = {"A": {}, "B": {}}
n_di = 0
for _, r in df.iterrows():
    m = Chem.MolFromSmiles(str(r["SMILES"]))
    if not m or len(m.GetSubstructMatches(ESTER)) != 2:
        continue
    n_di += 1
    bidx = []
    for b in m.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE: continue
        pair = {a1.GetSymbol(), a2.GetSymbol()}
        if pair != {"C", "O"}: continue
        c_at = a1 if a1.GetSymbol() == "C" else a2
        is_carb = False
        for nb in c_at.GetNeighbors():
            if nb.GetSymbol() == "O":
                bb = m.GetBondBetweenAtoms(nb.GetIdx(), c_at.GetIdx())
                if bb and bb.GetBondType() == Chem.BondType.DOUBLE: is_carb = True
        if is_carb: bidx.append(b.GetIdx())
    if len(bidx) != 2: continue
    frag = Chem.FragmentOnBonds(m, bidx, dummyLabels=[(0,0),(0,0)])
    parts = [Chem.MolToSmiles(p) for p in Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=True)]
    for p in parts:
        nd = p.count("*")
        t = frag_type(p)
        if t == "?": continue
        if nd == 2: cores[t].setdefault(p, r["编号"])
        elif nd == 1: caps[t].setdefault(p, r["编号"])
print("di-ester sources: %d" % n_di)
print("A-cores: %d  B-cores: %d  A-caps: %d  B-caps: %d" %
      (len(cores["A"]), len(cores["B"]), len(caps["A"]), len(caps["B"])))

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

def build(core, c1, c2):
    half = join2(core, c1)
    if half is None: return None
    return join2(half, c2)

combos = []
for t_core, t_cap in [("A", "B"), ("B", "A")]:
    for core_smi, core_id in cores[t_core].items():
        cl = list(caps[t_cap].items())
        for (c1, i1), (c2, i2) in itertools.combinations_with_replacement(cl, 2):
            combos.append((core_smi, core_id, c1, i1, c2, i2))
print("raw 3-body combos: %d" % len(combos))

seen, rows = set(), []
n_bad = n_dup = n_exist = 0
random.seed(42)
random.shuffle(combos)
LIMIT = 15000
for core_smi, core_id, c1, i1, c2, i2 in combos:
    if len(rows) >= LIMIT: break
    prod = build(core_smi, c1, c2)
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

# ---------- 4. 编号 C513 起 ----------
out = pd.DataFrame(rows)
out.insert(0, "编号", ["C%d" % (513 + i) for i in range(len(out))])
out["角色"] = "模型化合物"
out["动态键类型"] = "酯键"
out["文献来源DOI"] = ""
out["备注"] = "枚举虚拟库v2_双酯重组"
out = out[["编号", "角色", "名称", "SMILES", "动态键类型", "文献来源DOI", "备注"]]
out.to_csv(base + "枚举库_ester_v2.csv", index=False)
print(out.head(5).to_string())
print("saved: 枚举库_ester_v2.csv, %d rows, IDs C513..C%d" % (len(out), 512 + len(out)))
