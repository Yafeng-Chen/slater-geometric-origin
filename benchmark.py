# -*- coding: utf-8 -*-
"""
Self-contained Matbench-style benchmark for Slater features vs standard features.
Formation energy prediction on 92 compounds from Materials Project.
No network required - all data hardcoded.
"""
import csv, os, sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "periodic_table_features.csv")
OUT_DIR = SCRIPT_DIR

# Load our features
our_features = {}
with open(CSV_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        our_features[row["Symbol"]] = {
            "Z_eff": float(row["Z_eff"]), "n_outer": int(row["n_outer"]), "Z": int(row["Z"]), "IE": float(row["IE_predicted_eV"]),
        }

# Standard features
standard_features = {
    "H":{"chi":2.20,"r":0.53,"IE":13.60},"Li":{"chi":0.98,"r":1.67,"IE":5.39},
    "Be":{"chi":1.57,"r":1.12,"IE":9.32},"B":{"chi":2.04,"r":0.87,"IE":8.30},
    "C":{"chi":2.55,"r":0.67,"IE":11.26},"N":{"chi":3.04,"r":0.56,"IE":14.53},
    "O":{"chi":3.44,"r":0.48,"IE":13.62},"F":{"chi":3.98,"r":0.42,"IE":17.42},
    "Na":{"chi":0.93,"r":1.90,"IE":5.14},"Mg":{"chi":1.31,"r":1.45,"IE":7.65},
    "Al":{"chi":1.61,"r":1.18,"IE":5.99},"Si":{"chi":1.90,"r":1.11,"IE":8.15},
    "P":{"chi":2.19,"r":0.98,"IE":10.49},"S":{"chi":2.58,"r":0.88,"IE":10.36},
    "Cl":{"chi":3.16,"r":0.79,"IE":12.97},"K":{"chi":0.82,"r":2.43,"IE":4.34},
    "Ca":{"chi":1.00,"r":1.94,"IE":6.11},"Sc":{"chi":1.36,"r":1.84,"IE":6.56},
    "Ti":{"chi":1.54,"r":1.76,"IE":6.83},"V":{"chi":1.63,"r":1.71,"IE":6.75},
    "Cr":{"chi":1.66,"r":1.66,"IE":6.77},"Mn":{"chi":1.55,"r":1.61,"IE":7.43},
    "Fe":{"chi":1.83,"r":1.56,"IE":7.90},"Co":{"chi":1.88,"r":1.52,"IE":7.88},
    "Ni":{"chi":1.91,"r":1.49,"IE":7.64},"Cu":{"chi":1.90,"r":1.45,"IE":7.73},
    "Zn":{"chi":1.65,"r":1.42,"IE":9.39},"Ga":{"chi":1.81,"r":1.36,"IE":6.00},
    "Ge":{"chi":2.01,"r":1.25,"IE":7.90},"As":{"chi":2.18,"r":1.14,"IE":9.79},
    "Se":{"chi":2.55,"r":1.03,"IE":9.75},"Br":{"chi":2.96,"r":0.94,"IE":11.81},
    "Rb":{"chi":0.82,"r":2.65,"IE":4.18},"Sr":{"chi":0.95,"r":2.19,"IE":5.69},
    "Y":{"chi":1.22,"r":2.12,"IE":6.22},"Zr":{"chi":1.33,"r":2.06,"IE":6.63},
    "Nb":{"chi":1.60,"r":1.98,"IE":6.76},"Mo":{"chi":2.16,"r":1.90,"IE":7.09},
    "Ru":{"chi":2.20,"r":1.78,"IE":7.36},"Rh":{"chi":2.28,"r":1.73,"IE":7.46},
    "Pd":{"chi":2.20,"r":1.69,"IE":8.34},"Ag":{"chi":1.93,"r":1.65,"IE":7.58},
    "Cd":{"chi":1.69,"r":1.61,"IE":8.99},"In":{"chi":1.78,"r":1.56,"IE":5.79},
    "Sn":{"chi":1.96,"r":1.45,"IE":7.34},"Sb":{"chi":2.05,"r":1.33,"IE":9.01},
    "Te":{"chi":2.10,"r":1.23,"IE":8.96},"I":{"chi":2.66,"r":1.15,"IE":10.45},
    "Cs":{"chi":0.79,"r":2.98,"IE":3.89},"Ba":{"chi":0.89,"r":2.53,"IE":5.21},
    "La":{"chi":1.10,"r":2.07,"IE":5.58},"Hf":{"chi":1.30,"r":2.08,"IE":6.83},
    "Ta":{"chi":1.50,"r":2.00,"IE":7.55},"W":{"chi":2.36,"r":1.93,"IE":7.86},
    "Re":{"chi":1.90,"r":1.88,"IE":7.83},"Os":{"chi":2.20,"r":1.85,"IE":8.44},
    "Ir":{"chi":2.20,"r":1.80,"IE":8.97},"Pt":{"chi":2.28,"r":1.77,"IE":8.96},
    "Au":{"chi":2.54,"r":1.74,"IE":9.23},"Hg":{"chi":2.00,"r":1.71,"IE":10.44},
    "Tl":{"chi":1.62,"r":1.56,"IE":6.11},"Pb":{"chi":2.33,"r":1.54,"IE":7.42},
    "Bi":{"chi":2.02,"r":1.43,"IE":7.29},
}

# Formation energies from Materials Project (eV/atom)
compounds = [
    ("Al2O3",-3.47,{"Al":2,"O":3}),("SiO2",-3.05,{"Si":1,"O":2}),
    ("TiO2",-3.26,{"Ti":1,"O":2}),("Fe2O3",-2.47,{"Fe":2,"O":3}),
    ("Fe3O4",-2.56,{"Fe":3,"O":4}),("MgO",-3.03,{"Mg":1,"O":1}),
    ("CaO",-3.27,{"Ca":1,"O":1}),("ZnO",-1.78,{"Zn":1,"O":1}),
    ("CuO",-0.81,{"Cu":1,"O":1}),("NiO",-1.22,{"Ni":1,"O":1}),
    ("CoO",-1.27,{"Co":1,"O":1}),("MnO",-1.96,{"Mn":1,"O":1}),
    ("Cr2O3",-2.32,{"Cr":2,"O":3}),("V2O5",-2.19,{"V":2,"O":5}),
    ("WO3",-2.64,{"W":1,"O":3}),("MoO3",-1.92,{"Mo":1,"O":3}),
    ("ZrO2",-3.47,{"Zr":1,"O":2}),("HfO2",-3.66,{"Hf":1,"O":2}),
    ("Nb2O5",-2.49,{"Nb":2,"O":5}),("Ta2O5",-3.07,{"Ta":2,"O":5}),
    ("SnO2",-1.93,{"Sn":1,"O":2}),("PbO",-0.95,{"Pb":1,"O":1}),
    ("Y2O3",-3.58,{"Y":2,"O":3}),("La2O3",-3.44,{"La":2,"O":3}),
    ("AlN",-1.64,{"Al":1,"N":1}),("TiN",-1.73,{"Ti":1,"N":1}),
    ("GaN",-0.55,{"Ga":1,"N":1}),("BN",-1.30,{"B":1,"N":1}),
    ("ZrN",-1.82,{"Zr":1,"N":1}),("HfN",-2.06,{"Hf":1,"N":1}),
    ("SiC",-0.33,{"Si":1,"C":1}),("TiC",-0.92,{"Ti":1,"C":1}),
    ("WC",-0.17,{"W":1,"C":1}),("ZrC",-1.00,{"Zr":1,"C":1}),
    ("HfC",-1.08,{"Hf":1,"C":1}),
    ("LiF",-3.17,{"Li":1,"F":1}),("NaF",-2.95,{"Na":1,"F":1}),
    ("KF",-2.79,{"K":1,"F":1}),("MgF2",-3.54,{"Mg":1,"F":2}),
    ("CaF2",-3.78,{"Ca":1,"F":2}),("AlF3",-3.61,{"Al":1,"F":3}),
    ("NaCl",-1.98,{"Na":1,"Cl":1}),("KCl",-2.20,{"K":1,"Cl":1}),
    ("LiCl",-2.05,{"Li":1,"Cl":1}),("MgCl2",-2.07,{"Mg":1,"Cl":2}),
    ("CaCl2",-2.51,{"Ca":1,"Cl":2}),
    ("ZnS",-0.97,{"Zn":1,"S":1}),("CdS",-0.72,{"Cd":1,"S":1}),
    ("PbS",-0.49,{"Pb":1,"S":1}),("MoS2",-0.73,{"Mo":1,"S":2}),
    ("WS2",-0.85,{"W":1,"S":2}),
    ("NiAl",-0.59,{"Ni":1,"Al":1}),("FeAl",-0.37,{"Fe":1,"Al":1}),
    ("TiAl",-0.40,{"Ti":1,"Al":1}),("NiTi",-0.33,{"Ni":1,"Ti":1}),
    ("CuAu",-0.05,{"Cu":1,"Au":1}),("Ni3Al",-0.45,{"Ni":3,"Al":1}),
    ("LiH",-0.45,{"Li":1,"H":1}),("NaH",-0.28,{"Na":1,"H":1}),
    ("MgH2",-0.34,{"Mg":1,"H":2}),
    ("GaP",-0.50,{"Ga":1,"P":1}),("InP",-0.40,{"In":1,"P":1}),
    ("GaAs",-0.37,{"Ga":1,"As":1}),("InAs",-0.28,{"In":1,"As":1}),
    ("CdSe",-0.47,{"Cd":1,"Se":1}),("CdTe",-0.37,{"Cd":1,"Te":1}),
    ("PbSe",-0.35,{"Pb":1,"Se":1}),("PbTe",-0.32,{"Pb":1,"Te":1}),
    ("MoSi2",-0.49,{"Mo":1,"Si":2}),("WSi2",-0.39,{"W":1,"Si":2}),
    ("TiB2",-0.90,{"Ti":1,"B":2}),("ZrB2",-0.99,{"Zr":1,"B":2}),
    ("HfB2",-1.05,{"Hf":1,"B":2}),
    ("BaTiO3",-3.31,{"Ba":1,"Ti":1,"O":3}),
    ("SrTiO3",-3.43,{"Sr":1,"Ti":1,"O":3}),
    ("LiCoO2",-1.82,{"Li":1,"Co":1,"O":2}),
    ("LiFePO4",-2.02,{"Li":1,"Fe":1,"P":1,"O":4}),
    ("YBa2Cu3O7",-2.21,{"Y":1,"Ba":2,"Cu":3,"O":7}),
    ("LaMnO3",-2.81,{"La":1,"Mn":1,"O":3}),
    ("CaTiO3",-3.41,{"Ca":1,"Ti":1,"O":3}),
    ("KTaO3",-3.03,{"K":1,"Ta":1,"O":3}),
    ("NaNbO3",-2.72,{"Na":1,"Nb":1,"O":3}),
    ("Fe",0.0,{"Fe":1}),("Al",0.0,{"Al":1}),("Cu",0.0,{"Cu":1}),
    ("Ni",0.0,{"Ni":1}),("Si",0.0,{"Si":1}),("C",0.0,{"C":1}),
    ("W",0.0,{"W":1}),("Au",0.0,{"Au":1}),("Pt",0.0,{"Pt":1}),
    ("Ag",0.0,{"Ag":1}),
]

def featurize_our(comp):
    _, _, elements = comp
    total_atoms = sum(elements.values())
    feats = []
    for key in ["Z_eff","n_outer","IE"]:
        vals, weights = [], []
        for sym, count in elements.items():
            if sym in our_features:
                vals.append(our_features[sym][key]); weights.append(count)
        vals, weights = np.array(vals), np.array(weights)/sum(weights)
        m = np.average(vals, weights=weights)
        s = np.sqrt(np.average((vals-m)**2, weights=weights))
        feats.extend([m, s])
    zvs = [our_features[s]["Z_eff"] for s in elements if s in our_features]
    feats.extend([min(zvs), max(zvs), total_atoms, len(elements)])
    return np.array(feats)

def featurize_standard(comp):
    _, _, elements = comp
    total_atoms = sum(elements.values())
    feats = []
    for key in ["chi","r","IE"]:
        vals, weights = [], []
        for sym, count in elements.items():
            if sym in standard_features:
                vals.append(standard_features[sym][key]); weights.append(count)
        vals, weights = np.array(vals), np.array(weights)/sum(weights)
        m = np.average(vals, weights=weights)
        s = np.sqrt(np.average((vals-m)**2, weights=weights))
        feats.extend([m, s])
    zvs = [standard_features[s]["IE"]*0.5 for s in elements if s in standard_features]
    feats.extend([min(zvs) if zvs else 0, max(zvs) if zvs else 0, total_atoms, len(elements)])
    return np.array(feats)

print("="*60)
print("Slater Features Benchmark")
print("="*60)
X_our = np.array([featurize_our(c) for c in compounds])
X_std = np.array([featurize_standard(c) for c in compounds])
y = np.array([c[1] for c in compounds])
print(f"{len(compounds)} compounds, {len(set(e for _,_,d in compounds for e in d))} elements")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for name, X in [("Our (derived Slater)", X_our), ("Standard (empirical)", X_std)]:
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    maes, r2s = [], []
    for tr, te in kf.split(X):
        rf.fit(X[tr], y[tr])
        yp = rf.predict(X[te])
        maes.append(mean_absolute_error(y[te], yp))
        r2s.append(r2_score(y[te], yp))
    print(f"\n{name}:")
    print(f"  MAE = {np.mean(maes):.3f} +/- {np.std(maes):.3f} eV/atom")
    print(f"  R^2 = {np.mean(r2s):.3f} +/- {np.std(r2s):.3f}")

# Plot
rf_our = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42).fit(X_our, y)
rf_std = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42).fit(X_std, y)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
lims = [-4.5, 0.5]
for ax, X, rf, title, color in [
    (axes[0], X_our, rf_our, "Our Features (derived Slater)", "steelblue"),
    (axes[1], X_std, rf_std, "Standard Features (empirical)", "darkorange")]:
    yp = rf.predict(X)
    ax.scatter(y, yp, c=color, alpha=0.6, edgecolors="k", linewidth=0.3, s=40)
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("DFT Formation Energy (eV/atom)")
    ax.set_ylabel("Predicted (eV/atom)")
    ax.set_title(f"{title}\nMAE={mean_absolute_error(y,yp):.3f}, R\262={r2_score(y,yp):.3f}")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
path = os.path.join(OUT_DIR, "17_matbench_benchmark.png")
plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
print(f"\nSaved: {path}")
