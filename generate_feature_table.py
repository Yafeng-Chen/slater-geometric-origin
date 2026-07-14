# -*- coding: utf-8 -*-
"""
generate_feature_table.py
==========================
Complete, executable generator for periodic_table_features.csv (Z=1..118).

Uses Slater rules with geometrically derived shielding constants
(0.359 / 0.862 / 0.979) plus the empirical same-subshell value (0.350).

IMPORTANT: The raw hydrogenic IE formula I = 13.598*(Z_eff/n)^2 is a crude
approximation. For many elements (noble gases, some d-block, p-block with high n),
errors exceed 100%. The Z_eff values themselves are the primary theoretical
quantity. See the README for recommended usage.

Known failure modes (documented in output and README):
  - Noble gases: filled-shell stability not captured -> IE overestimated 200-400%
  - Pd (Z=46): 4d is outermost by Slater grouping but physical ionization is
    from 5s after excitation -> Z_eff=11.98, IE_pred=122 eV vs exp=8.3 eV (1363%)
  - Other d-block with configuration exceptions (Nb,Mo,Ru,Rh,Ag): similar issues
    may apply; review outer_orbital column before use

Aufbau exceptions handled:
  Cr(24), Cu(29), Nb(41), Mo(42), Ru(44), Rh(45), Pd(46), Ag(47),
  La(57), Ce(58), Gd(64), Pt(78), Au(79), Ac(89), Th(90), Pa(91),
  U(92), Np(93), Cm(96), Lr(103)
"""

import numpy as np
import csv, os, sys

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUT_DIR, "periodic_table_features.csv")

# --- Derived Slater constants ---
S_SAME_N_DIFF_L = 0.359  # same n, different l (geometric: 0.359)
S_N1 = 0.862             # n-1 shell (geometric: 0.862)
S_DEEP = 0.979           # deep shells (geometric: 0.979)
S_SAME_SUB = 0.350       # same subshell (empirical: includes exchange)

# For d/f outer electrons: Slater rule uses all inner groups = 1.00
S_INNER_FOR_DF = 1.00

# --- Element data ---
ELEMENT_NAMES = {
    1:"Hydrogen",2:"Helium",3:"Lithium",4:"Beryllium",5:"Boron",6:"Carbon",
    7:"Nitrogen",8:"Oxygen",9:"Fluorine",10:"Neon",11:"Sodium",12:"Magnesium",
    13:"Aluminum",14:"Silicon",15:"Phosphorus",16:"Sulfur",17:"Chlorine",18:"Argon",
    19:"Potassium",20:"Calcium",21:"Scandium",22:"Titanium",23:"Vanadium",
    24:"Chromium",25:"Manganese",26:"Iron",27:"Cobalt",28:"Nickel",29:"Copper",
    30:"Zinc",31:"Gallium",32:"Germanium",33:"Arsenic",34:"Selenium",35:"Bromine",
    36:"Krypton",37:"Rubidium",38:"Strontium",39:"Yttrium",40:"Zirconium",
    41:"Niobium",42:"Molybdenum",43:"Technetium",44:"Ruthenium",45:"Rhodium",
    46:"Palladium",47:"Silver",48:"Cadmium",49:"Indium",50:"Tin",51:"Antimony",
    52:"Tellurium",53:"Iodine",54:"Xenon",55:"Cesium",56:"Barium",57:"Lanthanum",
    58:"Cerium",59:"Praseodymium",60:"Neodymium",61:"Promethium",62:"Samarium",
    63:"Europium",64:"Gadolinium",65:"Terbium",66:"Dysprosium",67:"Holmium",
    68:"Erbium",69:"Thulium",70:"Ytterbium",71:"Lutetium",72:"Hafnium",
    73:"Tantalum",74:"Tungsten",75:"Rhenium",76:"Osmium",77:"Iridium",
    78:"Platinum",79:"Gold",80:"Mercury",81:"Thallium",82:"Lead",83:"Bismuth",
    84:"Polonium",85:"Astatine",86:"Radon",87:"Francium",88:"Radium",
    89:"Actinium",90:"Thorium",91:"Protactinium",92:"Uranium",93:"Neptunium",
    94:"Plutonium",95:"Americium",96:"Curium",97:"Berkelium",98:"Californium",
    99:"Einsteinium",100:"Fermium",101:"Mendelevium",102:"Nobelium",
    103:"Lawrencium",104:"Rutherfordium",105:"Dubnium",106:"Seaborgium",
    107:"Bohrium",108:"Hassium",109:"Meitnerium",110:"Darmstadtium",
    111:"Roentgenium",112:"Copernicium",113:"Nihonium",114:"Flerovium",
    115:"Moscovium",116:"Livermorium",117:"Tennessine",118:"Oganesson"
}

ELEMENT_SYMBOLS = {
    1:"H",2:"He",3:"Li",4:"Be",5:"B",6:"C",7:"N",8:"O",9:"F",10:"Ne",
    11:"Na",12:"Mg",13:"Al",14:"Si",15:"P",16:"S",17:"Cl",18:"Ar",
    19:"K",20:"Ca",21:"Sc",22:"Ti",23:"V",24:"Cr",25:"Mn",26:"Fe",
    27:"Co",28:"Ni",29:"Cu",30:"Zn",31:"Ga",32:"Ge",33:"As",34:"Se",
    35:"Br",36:"Kr",37:"Rb",38:"Sr",39:"Y",40:"Zr",41:"Nb",42:"Mo",
    43:"Tc",44:"Ru",45:"Rh",46:"Pd",47:"Ag",48:"Cd",49:"In",50:"Sn",
    51:"Sb",52:"Te",53:"I",54:"Xe",55:"Cs",56:"Ba",57:"La",58:"Ce",
    59:"Pr",60:"Nd",61:"Pm",62:"Sm",63:"Eu",64:"Gd",65:"Tb",66:"Dy",
    67:"Ho",68:"Er",69:"Tm",70:"Yb",71:"Lu",72:"Hf",73:"Ta",74:"W",
    75:"Re",76:"Os",77:"Ir",78:"Pt",79:"Au",80:"Hg",81:"Tl",82:"Pb",
    83:"Bi",84:"Po",85:"At",86:"Rn",87:"Fr",88:"Ra",89:"Ac",90:"Th",
    91:"Pa",92:"U",93:"Np",94:"Pu",95:"Am",96:"Cm",97:"Bk",98:"Cf",
    99:"Es",100:"Fm",101:"Md",102:"No",103:"Lr",104:"Rf",105:"Db",
    106:"Sg",107:"Bh",108:"Hs",109:"Mt",110:"Ds",111:"Rg",112:"Cn",
    113:"Nh",114:"Fl",115:"Mc",116:"Lv",117:"Ts",118:"Og"
}

# Experimental IE (NIST, eV) for Z=1..103
EXP_IE = {
    1:13.598,2:24.587,3:5.392,4:9.323,5:8.298,6:11.260,7:14.534,8:13.618,
    9:17.423,10:21.565,11:5.139,12:7.646,13:5.986,14:8.152,15:10.487,16:10.360,
    17:12.968,18:15.760,19:4.341,20:6.113,21:6.561,22:6.828,23:6.746,24:6.767,
    25:7.434,26:7.902,27:7.881,28:7.640,29:7.726,30:9.394,31:5.999,32:7.899,
    33:9.789,34:9.752,35:11.814,36:13.999,37:4.177,38:5.695,39:6.217,40:6.634,
    41:6.759,42:7.092,43:7.280,44:7.361,45:7.458,46:8.337,47:7.576,48:8.994,
    49:5.786,50:7.344,51:9.010,52:8.959,53:10.451,54:12.130,
    55:3.894,56:5.212,57:5.577,58:5.539,59:5.470,60:5.525,61:5.580,62:5.644,
    63:5.670,64:6.150,65:5.864,66:5.939,67:6.022,68:6.108,69:6.184,70:6.254,
    71:5.426,72:6.825,73:7.550,74:7.864,75:7.834,76:8.438,77:8.967,78:8.959,
    79:9.226,80:10.438,81:6.108,82:7.417,83:7.286,84:8.417,85:9.320,
    86:10.749,87:4.073,88:5.278,89:5.170,90:6.307,91:5.890,92:6.194,
    93:6.266,94:6.026,95:5.974,96:5.991,97:6.198,98:6.282,99:6.420,
    100:6.500,101:6.580,102:6.650,103:4.900,
}

# ================================================================
# Electron Configurations
# ================================================================
HE_CORE  = [(1,0,2)]
NE_CORE  = HE_CORE + [(2,0,2),(2,1,6)]
AR_CORE  = NE_CORE + [(3,0,2),(3,1,6)]
KR_CORE  = AR_CORE + [(4,0,2),(3,2,10),(4,1,6)]
XE_CORE  = KR_CORE + [(5,0,2),(4,2,10),(5,1,6)]
RN_CORE  = XE_CORE + [(6,0,2),(4,3,14),(5,2,10),(6,1,6)]

def get_config(Z):
    """Return electron configuration as list of (n, l, count)."""
    if Z == 1:  return [(1,0,1)]
    if Z == 2:  return [(1,0,2)]

    # Period 2
    if 3 <= Z <= 10:
        base = HE_CORE.copy(); remaining = Z - 2
        if remaining <= 2: base.append((2,0,remaining))
        else: base.extend([(2,0,2),(2,1,remaining-2)])
        return base

    # Period 3
    if 11 <= Z <= 18:
        base = NE_CORE.copy(); remaining = Z - 10
        if remaining <= 2: base.append((3,0,remaining))
        else: base.extend([(3,0,2),(3,1,remaining-2)])
        return base

    # Period 4
    if 19 <= Z <= 36:
        base = AR_CORE.copy(); remaining = Z - 18
        s4 = min(remaining, 2); base.append((4,0,s4)); remaining -= s4
        if remaining <= 0: return base
        if Z == 24: return AR_CORE + [(4,0,1),(3,2,5)]  # Cr
        if Z == 29: return AR_CORE + [(4,0,1),(3,2,10)]  # Cu
        d3 = min(remaining, 10); base.append((3,2,d3)); remaining -= d3
        if remaining <= 0: return base
        base.append((4,1,remaining)); return base

    # Period 5
    if 37 <= Z <= 54:
        base = KR_CORE.copy(); remaining = Z - 36
        s5 = min(remaining, 2); base.append((5,0,s5)); remaining -= s5
        if remaining <= 0: return base
        if Z == 41: return KR_CORE + [(5,0,1),(4,2,4)]   # Nb
        if Z == 42: return KR_CORE + [(5,0,1),(4,2,5)]   # Mo
        if Z == 44: return KR_CORE + [(5,0,1),(4,2,7)]   # Ru
        if Z == 45: return KR_CORE + [(5,0,1),(4,2,8)]   # Rh
        if Z == 46: return KR_CORE + [(4,2,10)]           # Pd (no 5s)
        if Z == 47: return KR_CORE + [(5,0,1),(4,2,10)]  # Ag
        d4 = min(remaining, 10); base.append((4,2,d4)); remaining -= d4
        if remaining <= 0: return base
        base.append((5,1,remaining)); return base

    # Period 6
    if 55 <= Z <= 86:
        base = XE_CORE.copy(); remaining = Z - 54
        s6 = min(remaining, 2); base.append((6,0,s6)); remaining -= s6
        if remaining <= 0: return base
        if Z == 57: return XE_CORE + [(6,0,2),(5,2,1)]       # La
        if Z == 58: return XE_CORE + [(6,0,2),(4,3,1),(5,2,1)] # Ce
        if Z == 64: return XE_CORE + [(6,0,2),(4,3,7),(5,2,1)] # Gd
        if 58 <= Z <= 71:
            f4c = {58:1,59:3,60:4,61:5,62:6,63:7,64:7,65:9,66:10,67:11,68:12,69:13,70:14}
            f4n = f4c.get(Z,0)
            if Z <= 63 and Z not in [57,58]:
                return XE_CORE + [(6,0,2),(4,3,f4n)]
            elif Z == 64: pass
            elif Z <= 70:
                return XE_CORE + [(6,0,2),(4,3,f4n)]
            else:
                return XE_CORE + [(6,0,2),(4,3,14),(5,2,1)]  # Lu
        if 71 <= Z <= 80:
            if Z == 71: pass
            d5 = Z - 54 - 2 - 14; d5 = max(0, min(d5, 10))
            if Z == 78: return XE_CORE + [(6,0,1),(4,3,14),(5,2,9)]   # Pt
            if Z == 79: return XE_CORE + [(6,0,1),(4,3,14),(5,2,10)]  # Au
            if d5 > 0: return XE_CORE + [(6,0,2),(4,3,14),(5,2,d5)]
            return XE_CORE + [(6,0,2),(4,3,14)]
        remaining = Z - 54 - 2 - 14 - 10
        base = XE_CORE + [(6,0,2),(4,3,14),(5,2,10)]
        base.append((6,1,remaining)); return base

    # Period 7
    if 87 <= Z <= 118:
        base = RN_CORE.copy(); remaining = Z - 86
        s7 = min(remaining, 2); base.append((7,0,s7)); remaining -= s7
        if remaining <= 0: return base
        if Z == 89: return RN_CORE + [(7,0,2),(6,2,1)]               # Ac
        if Z == 90: return RN_CORE + [(7,0,2),(6,2,2)]               # Th
        if Z == 91: return RN_CORE + [(7,0,2),(5,3,2),(6,2,1)]       # Pa
        if Z == 92: return RN_CORE + [(7,0,2),(5,3,3),(6,2,1)]       # U
        if Z == 93: return RN_CORE + [(7,0,2),(5,3,4),(6,2,1)]       # Np
        if Z == 96: return RN_CORE + [(7,0,2),(5,3,7),(6,2,1)]       # Cm
        if Z == 103: return RN_CORE + [(7,0,2),(5,3,14),(7,1,1)]     # Lr
        if 90 <= Z <= 103:
            f5c = {90:0,91:2,92:3,93:4,94:6,95:7,96:7,97:9,98:10,99:11,100:12,101:13,102:14,103:14}
            d6c = {90:2,91:1,92:1,93:1,94:0,95:0,96:1,97:0,98:0,99:0,100:0,101:0,102:0,103:0}
            f5, d6 = f5c.get(Z,0), d6c.get(Z,0)
            orbitals = [(7,0,2)]
            if f5 > 0: orbitals.append((5,3,f5))
            if d6 > 0: orbitals.append((6,2,d6))
            if Z == 103: orbitals.append((7,1,1))
            return RN_CORE + orbitals
        if 104 <= Z <= 112:
            d6e = Z - 86 - 2 - 14; d6e = max(0, min(d6e, 10))
            return RN_CORE + [(7,0,2),(5,3,14),(6,2,d6e)] if d6e > 0 else RN_CORE + [(7,0,2),(5,3,14)]
        if 113 <= Z <= 118:
            p7e = Z - 86 - 2 - 14 - 10; p7e = max(0, min(p7e, 6))
            return RN_CORE + [(7,0,2),(5,3,14),(6,2,10),(7,1,p7e)]
    return []

# ================================================================
# Compute Z_eff
# ================================================================
def orbital_label(n, l):
    labels = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
    return labels.get(l, '?').join([str(n), ''])
def config_string(Z):
    orbitals = get_config(Z)
    if not orbitals: return "unknown"
    parts = [f"{orbital_label(n,l)}{c}" for n,l,c in orbitals]
    parts.sort(key=lambda s: (int(s[0])+{'s':0,'p':1,'d':2,'f':3}[s[1]], int(s[0])))
    return " ".join(parts)

def compute_zeff_slater(config):
    if not config: return 0.0, 0, 0
    Z = sum(c for _,_,c in config)
    max_n = max(n for n,_,_ in config)
    candidates = [(n,l,c) for n,l,c in config if n == max_n]
    candidates.sort(key=lambda x: x[0]+x[1])
    outer_n, outer_l, outer_count = candidates[-1]
    is_sp_outer = (outer_l <= 1)
    sigma = 0.0
    for n, l, count in config:
        is_outer = (n == outer_n and l == outer_l)
        if is_outer:
            other_count = count - 1
            if other_count > 0: sigma += other_count * S_SAME_SUB
        elif n == outer_n:
            sigma += count * S_SAME_N_DIFF_L
        elif n == outer_n - 1:
            sigma += count * (S_N1 if is_sp_outer else S_INNER_FOR_DF)
        else:
            sigma += count * (S_DEEP if is_sp_outer else S_INNER_FOR_DF)
    return Z - sigma, outer_n, outer_l

def get_group_period(Z, outer_n, outer_l):
    if outer_n == 1: return (1 if Z==1 else 18, 1)
    for p, limit in enumerate([2,10,18,36,54,86,118], 1):
        if Z <= limit: period = p; break
    config = get_config(Z)
    valence = sum(c for n,l,c in config if n==outer_n)
    for n,l,c in config:
        if n==outer_n-1 and l>=2: valence += c
    if outer_l <= 1: group = valence if valence<=2 else valence+10
    elif outer_l == 2: group = min(valence, 12)
    else: group = 3
    return group, period

# ================================================================
# Generate
# ================================================================
print("="*60)
print("Generating periodic_table_features.csv (Z=1..118)")
print(f"Slater: group={S_SAME_N_DIFF_L}, n-1={S_N1}, deep={S_DEEP}, same-sub={S_SAME_SUB}")
print("="*60)

rows = []
for Z in range(1, 119):
    config = get_config(Z)
    zeff, n_outer, l_outer = compute_zeff_slater(config)
    group, period = get_group_period(Z, n_outer, l_outer)
    I0 = 13.598
    ie_hydrogenic = I0 * (zeff/n_outer)**2 if n_outer>0 else 0
    ie_exp = EXP_IE.get(Z, None)
    ie_error_pct = round(abs(ie_hydrogenic-ie_exp)/ie_exp*100,1) if ie_exp else None
    outer_orb = orbital_label(n_outer, l_outer)

    rows.append({
        "Z":Z,"Symbol":ELEMENT_SYMBOLS[Z],"Name":ELEMENT_NAMES[Z],
        "Group":group,"Period":period,"Config":config_string(Z),
        "outer_orbital":outer_orb,"n_outer":n_outer,"Z_eff":round(zeff,4),
        "IE_predicted_eV":round(ie_hydrogenic,2),
        "IE_experimental_eV":round(ie_exp,2) if ie_exp else "",
        "IE_error_pct":str(ie_error_pct) if ie_error_pct is not None else "",
    })

fieldnames = ["Z","Symbol","Name","Group","Period","Config","outer_orbital",
              "n_outer","Z_eff","IE_predicted_eV","IE_experimental_eV",
              "IE_error_pct"]

with open(CSV_PATH,"w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)

print(f"Saved: {CSV_PATH} ({len(rows)} rows)")

# --- Error statistics (honest: both median AND mean) ---
errors = [float(r["IE_error_pct"]) for r in rows if r["IE_error_pct"]]
print(f"\nIE Error Statistics ({len(errors)} elements with exp. data):")
print(f"  Median error: {np.median(errors):.1f}%")
print(f"  Mean error:   {np.mean(errors):.1f}%")
print(f"  Min error:    {np.min(errors):.1f}%")
print(f"  Max error:    {np.max(errors):.1f}%")

bins = [(0,10),(10,25),(25,50),(50,100),(100,200),(200,500),(500,10000)]
for lo, hi in bins:
    cnt = sum(1 for e in errors if lo <= e < hi)
    print(f"  {lo:4d}-{hi:4d}%: {cnt} elements")

# --- Flag extreme outliers ---
print("\nExtreme outliers (>500% error):")
for r in rows:
    if r["IE_error_pct"] and float(r["IE_error_pct"]) > 500:
        print(f"  Z={r['Z']:3d} {r['Symbol']:>3s} ({r['outer_orbital']}): "
              f"{r['IE_error_pct']}% | pred={r['IE_predicted_eV']} exp={r['IE_experimental_eV']} eV")

# --- Check Pd specifically ---
pd_row = next(r for r in rows if r["Symbol"]=="Pd")
print(f"\nPd (Z=46) detail check:")
print(f"  Config: {pd_row['Config']}")
print(f"  outer_orbital: {pd_row['outer_orbital']}")
print(f"  Z_eff: {pd_row['Z_eff']}")
print(f"  IE_pred: {pd_row['IE_predicted_eV']} eV")
print(f"  IE_exp:  {pd_row['IE_experimental_eV']} eV")
print(f"  Error: {pd_row['IE_error_pct']}%")
print(f"  NOTE: Pd has [Kr]4d10 (no 5s). Slater classification treats 4d as")
print(f"  outermost, causing overestimated Z_eff. Physical ionization is from 5s")
print(f"  after excitation. This is a known limitation.")

print("\nDone.")

