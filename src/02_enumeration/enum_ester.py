import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

base = "/Users/Zhuanz/polywork/"
ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2H0]")

def canon(smi):
    if not isinstance(smi, str):
        return None
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None

# ---------- 1. 现有库全部 SMILES ----------
existing = set()
df = pd.read_excel(base + "原料库_v1.xlsx")
for s in df["SMILES"]:
    c = canon(s)
    if c: existing.add(c)
for f in ["bde_ss.csv", "hydrolysis.csv", "ester_hyd.csv"]:
    try:
        d = pd.read_csv(base + f)
        cols = [c for c in d.columns if "smiles" in c.lower()]
        if not cols:
            print("skip %s: no SMILES-like column in %s" % (f, d.columns.tolist()))
            continue
        for s in d[cols[0]]:
            c = canon(s)
            if c: existing.add(c)
        print("loaded %s via column '%s'" % (f, cols[0]))
    except Exception as e:
        print("skip %s: %s" % (f, e))
print("existing library molecules: %d" % len(existing))

# ---------- 2. 剪酯键 ----------
acyl, alkoxy = {}, {}
def cut_ester(mol, src):
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE:
            continue
        c_at, o_at = None, None
        if a1.GetSymbol() == "C" and a2.GetSymbol() == "O":
            c_at, o_at = a1, a2
        elif a1.GetSymbol() == "O" and a2.GetSymbol() == "C":
            c_at, o_at = a2, a1
        if c_at is None:
            continue
        ok = False
        for nb in c_at.GetNeighbors():
            if nb.GetSymbol() == "O":
                bb = mol.GetBondBetweenAtoms(nb.GetIdx(), c_at.GetIdx())
                if bb is not None and bb.GetBondType() == Chem.BondType.DOUBLE:
                    ok = True
        if not ok:
            continue
        frag = Chem.FragmentOnBonds(mol, [b.GetIdx()], dummyLabels=[(0, 0)])
        parts = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=True)
        CARB = Chem.MolFromSmarts("[*]C(=O)")
        for p in parts:
            smi = Chem.MolToSmiles(p)
            if p.GetSubstructMatches(CARB):
                acyl.setdefault(smi, src)
            else:
                alkoxy.setdefault(smi, src)
        return

n_src = 0
for _, r in df.iterrows():
    m = Chem.MolFromSmiles(str(r["SMILES"]))
    if m and len(m.GetSubstructMatches(ESTER)) == 1:
        n_src += 1
        cut_ester(m, r["编号"])
print("source molecules with exactly 1 ester: %d" % n_src)
print("acyl fragments: %d   alkoxy fragments: %d" % (len(acyl), len(alkoxy)))

# ---------- 3. 重组 ----------
def join(smi_a, smi_b):
    m = Chem.CombineMols(Chem.MolFromSmiles(smi_a), Chem.MolFromSmiles(smi_b))
    rw = Chem.RWMol(m)
    dummies = sorted([a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "*"], reverse=True)
    if len(dummies) != 2:
        return None
    nbs = [rw.GetAtomWithIdx(d).GetNeighbors()[0].GetIdx() for d in dummies]
    rw.AddBond(nbs[0], nbs[1], Chem.BondType.SINGLE)
    for d in dummies:
        rw.RemoveAtom(d)
    try:
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw)
    except Exception:
        return None

seen, rows = set(), []
n_bad = n_dup = n_exist = 0
for fa, ida in acyl.items():
    for fb, idb in alkoxy.items():
        prod = join(fa, fb)
        if prod is None:
            n_bad += 1; continue
        if prod in seen:
            n_dup += 1; continue
        seen.add(prod)
        if prod in existing:
            n_exist += 1; continue
        m = Chem.MolFromSmiles(prod)
        if len(m.GetSubstructMatches(ESTER)) != 1:
            n_bad += 1; continue
        mw = Descriptors.MolWt(m)
        if not (120 <= mw <= 500) or m.GetNumHeavyAtoms() > 40:
            n_bad += 1; continue
        rows.append({"SMILES": prod, "名称": "E(%sx%s)" % (ida, idb)})

print("combos invalid: %d   internal dup: %d   already in library: %d" % (n_bad, n_dup, n_exist))
print("FINAL new molecules: %d" % len(rows))

# ---------- 4. 编号 C504 起 ----------
out = pd.DataFrame(rows)
out.insert(0, "编号", ["C%d" % (504 + i) for i in range(len(out))])
out["角色"] = "模型化合物"
out["动态键类型"] = "酯键"
out["文献来源DOI"] = ""
out["备注"] = "枚举虚拟库v1_碎片重组"
out = out[["编号", "角色", "名称", "SMILES", "动态键类型", "文献来源DOI", "备注"]]
out.to_csv(base + "枚举库_ester_v1.csv", index=False)
print("saved: 枚举库_ester_v1.csv, %d rows, IDs C504..C%d" % (len(out), 503 + len(out)))
