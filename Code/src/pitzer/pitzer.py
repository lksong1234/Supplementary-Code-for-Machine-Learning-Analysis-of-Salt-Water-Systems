# -*- coding: utf-8 -*-
"""
Corrected version: Added 'units mol/kgw' to the SOLUTION block to fix the bug where 
concentrations were scaled down by 1000x in SI diagnostics.
Workflow: SI Diagnosis -> Filter |SI| <= 0.05 valid points -> Single-phase saturation inverse calculation -> R2/RMSE/MAE
"""
import subprocess, pandas as pd, numpy as np, os, re

WORK_DIR = r"E:\phreeqc_work"
PHREEQC_EXE = os.path.join(WORK_DIR, "phreeqc.exe")
DATABASE_FILE = os.path.join(WORK_DIR, "pitzer.dat")
INP_TMP = os.path.join(WORK_DIR, "tmp_phq_in.txt")
SEL_TMP = os.path.join(WORK_DIR, "tmp_sel.txt")
LOG_TMP = os.path.join(WORK_DIR, "tmp_log.txt")

SI_TOL = 0.05

M = {"NaCl":58.44,"KCl":74.55,"MgCl2":95.21,
     "Na2SO4":142.04,"MgSO4":120.37,"K2SO4":174.26}

SYSTEM_CONFIG = {
    "NaCl-KCl-H₂O":      {"comp1":"NaCl","comp2":"KCl",
                          "ion1":[("Na",1),("Cl",1)],"ion2":[("K",1),("Cl",1)],"rep2":"K"},
    "Na₂SO₄-MgSO₄-H₂O": {"comp1":"Na2SO4","comp2":"MgSO4",
                          "ion1":[("Na",2),("S(6)",1)],"ion2":[("Mg",1),("S(6)",1)],"rep2":"Mg"},
    "KCl-MgCl₂-H₂O":     {"comp1":"KCl","comp2":"MgCl2",
                          "ion1":[("K",1),("Cl",1)],"ion2":[("Mg",1),("Cl",2)],"rep2":"Mg"},
    "NaCl-MgCl₂-H₂O":    {"comp1":"NaCl","comp2":"MgCl2",
                          "ion1":[("Na",1),("Cl",1)],"ion2":[("Mg",1),("Cl",2)],"rep2":"Mg"},
    "NaCl-Na₂SO₄-H₂O":   {"comp1":"NaCl","comp2":"Na2SO4",
                          "ion1":[("Na",1),("Cl",1)],"ion2":[("Na",2),("S(6)",1)],"rep2":"S(6)"},
    "K₂SO₄-MgSO₄-H₂O":   {"comp1":"K2SO4","comp2":"MgSO4",
                          "ion1":[("K",2),("S(6)",1)],"ion2":[("Mg",1),("S(6)",1)],"rep2":"Mg"},
}

PHASE_MAP = {
    "NaCl":"Halite","NaCl·2H₂O":"Hydrohalite",
    "KCl":"Sylvite","MgCl₂·4H₂O":"Bischofite","MgCl₂·8H₂O":"Bischofite",
    "MgCl₂·12H₂O":"Bischofite","Car":"Carnallite","Bis":"Bischofite",
    "Na₂SO₄":"Thenardite","Na₂SO₄·10H₂O":"Mirabilite","Mir":"Mirabilite","Van":"Thenardite",
    "K₂SO₄":"Arcanite","MgSO₄":"Kieserite","MgSO₄·12H₂O":"Epsomite",
    "MgSO₄·7H₂O":"Epsomite","Eps":"Epsomite","MgSO₄·6H₂O":"Hexahydrite",
    "Hex":"Hexahydrite","MgSO₄·5H₂O":"Pentahydrite","MgSO₄·4H₂O":"Starkeyite",
    "MgSO₄·H₂O":"Kieserite","Kie":"Kieserite",
    "KCl·MgCl₂·6H₂O":"Carnallite","K₂SO₄·MgSO₄·6H₂O":"Picromerite","Pic":"Picromerite",
    "K₂SO₄·MgSO₄·4H₂O":"Leonite","Leo":"Leonite","K₂SO₄·2MgSO₄":"Langbeinite","Lan":"Langbeinite",
    "Na₂SO₄·MgSO₄·4H₂O":"Bloedite","Low":"Loeweite","Na₂SO₄·MgSO₄":"Loeweite",
}

COMP2_PHASES = {
    "NaCl-KCl-H₂O":      {"KCl","Sylvite"},
    "Na₂SO₄-MgSO₄-H₂O": {"MgSO₄·12H₂O","MgSO₄·7H₂O","Eps","Hex","Kie","MgSO₄·H₂O","MgSO₄"},
    "KCl-MgCl₂-H₂O":     {"Car","Carnallite","MgCl₂·4H₂O","MgCl₂·8H₂O","Bis","Bischofite"},
    "NaCl-MgCl₂-H₂O":    {"MgCl₂·4H₂O","MgCl₂·8H₂O","Bis","Bischofite"},
    "NaCl-Na₂SO₄-H₂O":   {"Na₂SO₄","Thenardite","Van","Mir","Mirabilite"},
    "K₂SO₄-MgSO₄-H₂O":   {"Kie","Kieserite","Lan","Langbeinite","Leo","Leonite","MgSO₄","MgSO₄·H₂O"},
}

HIGH_T_UNRELIABLE = {"Carnallite","Bischofite"}

def split_phase(s):
    return [p.strip() for p in re.split(r"[+＋]", str(s)) if p.strip()]

def is_single_phase(s):
    return len(split_phase(s)) == 1

def pick_target_phase(phase_str, system):
    allowed = COMP2_PHASES.get(system, set())
    for p in split_phase(phase_str):
        if p in allowed and p in PHASE_MAP:
            return PHASE_MAP[p]
    return None

def wt_to_molality_pair(wt1, wt2, c1, c2):
    water_g = 100.0 - float(wt1) - float(wt2)
    if water_g <= 0: return None, None
    kgw = water_g/1000.0
    return (float(wt1)/M[c1])/kgw, (float(wt2)/M[c2])/kgw

def molality_to_wt_single(m, comp):
    g = m*M[comp]
    return 100.0*g/(g+1000.0)

def build_solution(ions_dict):
    return "\n".join(f"    {k} {v:.8f}" for k,v in ions_dict.items())

def add_ions(d, ion_list, mol):
    for ion,n in ion_list:
        d[ion] = d.get(ion,0.0) + n*mol

# ============ SI Diagnosis (Add units mol/kgw) ============
def diagnose_si(T, cfg, wt1, wt2, phase):
    m1, m2 = wt_to_molality_pair(wt1, wt2, cfg["comp1"], cfg["comp2"])
    if m1 is None: return None, "water<=0"
    ions = {}
    add_ions(ions, cfg["ion1"], m1)
    add_ions(ions, cfg["ion2"], m2)
    phq = f"""SOLUTION 1
    TEMP {T}
    WATER 1.0 kg
    units mol/kgw
{build_solution(ions)}
SELECTED_OUTPUT
    -file tmp_sel.txt
    -totals Na K Mg Cl S(6)
    -si {phase}
END
"""
    with open(INP_TMP,"w",encoding="utf-8") as f: f.write(phq)
    try:
        p = subprocess.run([PHREEQC_EXE,INP_TMP,LOG_TMP,DATABASE_FILE],
                           cwd=WORK_DIR,capture_output=True,timeout=60)
    except subprocess.TimeoutExpired:
        return None,"timeout"
    if p.returncode != 0: return None,f"returncode={p.returncode}"
    if not os.path.exists(SEL_TMP): return None,"no sel"
    df = pd.read_csv(SEL_TMP, sep=r"\s+")
    if df.empty: return None,"empty"
    col = f"si_{phase}"
    if col not in df.columns: return None,f"missing {col}"
    return float(df.iloc[-1][col]), None

# ============ Single-phase saturation inverse calculation (Add units mol/kgw) ============
def solve_comp2(T, cfg, wt1, wt2_exp, phase):
    m1, _ = wt_to_molality_pair(wt1, wt2_exp, cfg["comp1"], cfg["comp2"])
    if m1 is None: return None,"water<=0"
    ions = {}
    add_ions(ions, cfg["ion1"], m1)
    phq = f"""SOLUTION 1
    TEMP {T}
    WATER 1.0 kg
    units mol/kgw
{build_solution(ions)}
EQUILIBRIUM_PHASES 1
    {phase} 0.0 10.0
SELECTED_OUTPUT
    -file tmp_sel.txt
    -totals Na K Mg Cl S(6)
END
"""
    with open(INP_TMP,"w",encoding="utf-8") as f: f.write(phq)
    try:
        p = subprocess.run([PHREEQC_EXE,INP_TMP,LOG_TMP,DATABASE_FILE],
                           cwd=WORK_DIR,capture_output=True,timeout=60)
    except subprocess.TimeoutExpired:
        return None,"timeout"
    if p.returncode != 0: return None,f"returncode={p.returncode}"
    if not os.path.exists(SEL_TMP): return None,"no sel"
    df = pd.read_csv(SEL_TMP, sep=r"\s+")
    if df.empty: return None,"empty"
    row = df.iloc[-1]
    if cfg["rep2"] not in row or pd.isna(row[cfg["rep2"]]):
        return None,f"missing {cfg['rep2']}"
    m2 = float(row[cfg["rep2"]])
    return molality_to_wt_single(m2, cfg["comp2"]), None

# ============ Main Workflow ============
df_all = pd.read_excel(os.path.join(WORK_DIR,"data.xlsx"), sheet_name=None)
diag_rows = []
for sheet, g in df_all.items():
    if "System" not in g.columns: continue
    system = str(g["System"].iloc[0]).strip()
    if system not in SYSTEM_CONFIG: continue
    cfg = SYSTEM_CONFIG[system]
    c1 = [c for c in g.columns if str(c).startswith("Component1_wt")][0]
    c2 = [c for c in g.columns if str(c).startswith("Component2_wt")][0]

    for _, row in g.iterrows():
        T = row["Temperature_C"]; wt1 = row[c1]; wt2 = row[c2]
        phase_exp = row["Equilibrium_Solid_Phase"]
        if pd.isna(wt1) or pd.isna(wt2): continue
        if not is_single_phase(phase_exp): continue
        phase = pick_target_phase(phase_exp, system)
        if phase is None: continue

        si, err = diagnose_si(T, cfg, wt1, wt2, phase)
        diag_rows.append({
            "System":system,"T(C)":T,"wt1(Exp)":wt1,"wt2(Exp)":wt2,
            "Exp_Phase":phase_exp,"Target_Phase":phase,
            "SI":round(si,4) if si is not None else None,
            "Note":err if err else ("150C exceeds Pitzer recommended range" if T>=150 and phase in HIGH_T_UNRELIABLE else "")
        })

diag = pd.DataFrame(diag_rows)
print("==== Step 1: SI Diagnosis (Corrected) ====")
print(diag.to_string(index=False))
diag.to_csv(os.path.join(WORK_DIR,"step1_SI_diagnosis_fixed.csv"),index=False,encoding="utf-8-sig")

# SI Distribution Statistics
print("\n==== SI Distribution ====")
v = diag.dropna(subset=["SI"])
print(f"Total points: {len(v)}")
print(f"|SI|<=0.05: {(v['SI'].abs()<=0.05).sum()}")
print(f"|SI|<=0.10: {(v['SI'].abs()<=0.10).sum()}")
print(f"|SI|<=0.20: {(v['SI'].abs()<=0.20).sum()}")
print(f"SI range: {v['SI'].min():.4f} ~ {v['SI'].max():.4f}")

# Valid points -> Single-phase saturation inverse calculation
valid = v[v["SI"].abs() <= SI_TOL].copy()
print(f"\n==== Step 2: Valid points with |SI| <= {SI_TOL} (n={len(valid)}) ====")
if valid.empty:
    print("[Warning] No valid points found, skipping Step 3/4")
else:
    print(valid.to_string(index=False))

    cmp_rows = []
    for _, row in valid.iterrows():
        system = row["System"]; T = row["T(C)"]
        cfg = SYSTEM_CONFIG[system]
        wt2_pred, err = solve_comp2(T, cfg, row["wt1(Exp)"], row["wt2(Exp)"], row["Target_Phase"])
        cmp_rows.append({
            "System":system,"T(C)":T,
            "wt1(Exp)":row["wt1(Exp)"],"wt2(Exp)":row["wt2(Exp)"],
            "Target_Phase":row["Target_Phase"],"SI":row["SI"],
            "wt2(Pred)":round(wt2_pred,4) if wt2_pred is not None else None,
            "Note":err if err else row["Note"]
        })

    cmp = pd.DataFrame(cmp_rows)
    if cmp.empty:
        print("[Warning] Inverse calculation result is empty")
    else:
        cmp["e"] = cmp["wt2(Pred)"] - cmp["wt2(Exp)"]
        print("\n==== Step 3: Comparison of Inverse Calculations for Valid Points ====")
        print(cmp.to_string(index=False))
        cmp.to_csv(os.path.join(WORK_DIR,"step3_valid_compare_fixed.csv"),index=False,encoding="utf-8-sig")

        def metrics(x):
            x = x.dropna(subset=["wt2(Pred)","wt2(Exp)"])
            if len(x) < 2:
                return pd.Series({"n":len(x),"R2":np.nan,"RMSE":np.nan,"MAE":np.nan})
            y = x["wt2(Exp)"].values; yh = x["wt2(Pred)"].values
            ss_res = np.sum((yh-y)**2); ss_tot = np.sum((y-y.mean())**2)
            r2 = 1-ss_res/ss_tot if ss_tot>0 else np.nan
            return pd.Series({"n":len(x),"R2":round(r2,4),
                              "RMSE":round(np.sqrt(np.mean((yh-y)**2)),4),
                              "MAE":round(np.mean(np.abs(yh-y)),4)})

        print("\n==== Step 4: Metrics ====")
        print("Overall:")
        print(metrics(cmp).to_string())
        print("\nBy System:")
        print(cmp.groupby("System").apply(metrics).to_string())
        print("\nBy Target Phase:")
        print(cmp.groupby("Target_Phase").apply(metrics).to_string())

        # Save Metrics
        summary = []
        m0 = metrics(cmp); m0["Group"]="Overall"; m0["Key"]="All"; summary.append(m0)
        for k,vv in cmp.groupby("System"):
            m = metrics(vv); m["Group"]="System"; m["Key"]=k; summary.append(m)
        for k,vv in cmp.groupby("Target_Phase"):
            m = metrics(vv); m["Group"]="Target_Phase"; m["Key"]=k; summary.append(m)
        summary = pd.DataFrame(summary)[["Group","Key","n","R2","RMSE","MAE"]]
        summary.to_csv(os.path.join(WORK_DIR,"step4_metrics_fixed.csv"),index=False,encoding="utf-8-sig")
        print(f"\n✅ Saved: step1_SI_diagnosis_fixed.csv / step3_valid_compare_fixed.csv / step4_metrics_fixed.csv")

for f in [INP_TMP,SEL_TMP,LOG_TMP]:
    if os.path.exists(f): os.remove(f)