# Periodic Table Atomic Feature Set (Z = 1¨C118)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Elements: 118](https://img.shields.io/badge/Elements-1..118-blue)](periodic_table_features.csv)

**Complete atomic features for Z = 1¨C118, derived from first-principles geometric analysis of Slater shielding constants. No experimental fitting, no DFT calculations¡ªonly the atomic number Z required.**

Supplementary material for: *Chen, Yafeng. "A Geometric Interpretation of Slater's Shielding Constants." 2026.*

---

## Quick Start

```python
import pandas as pd
df = pd.read_csv("periodic_table_features.csv")
df.head()
```

| Z | Symbol | Name | Z_eff | outer_orbital | IE_predicted_eV | IE_experimental_eV | IE_error_pct |
|---|--------|------|-------|---------------|-----------------|--------------------|:---:|
| 1 | H | Hydrogen | 1.000 | 1s | 13.60 | 13.60 | 0.0% |
| 26 | Fe | Iron | 3.792 | 4s | 12.22 | 7.90 | 54.7% |
| 92 | U | Uranium | 4.593 | 7s | 5.85 | 6.19 | 5.5% |
| 118 | Og | Oganesson | 9.948 | 7p | 27.46 | ¡ª | ¡ª |

---

## Features

| Field | Description |
|------|------|
| `Z` | Atomic number |
| `Symbol` | Element symbol |
| `Name` | English name |
| `Group`, `Period` | Periodic table group and period |
| `Config` | Electron configuration (Aufbau + 20 known exceptions) |
| `outer_orbital` | Outermost orbital label |
| `n_outer` | Outermost principal quantum number |
| **`Z_eff`** | **Effective nuclear charge (Slater rules, geometrically derived shielding constants)** |
| `IE_predicted_eV` | Predicted ionization energy (hydrogenic formula, eV) |
| `IE_experimental_eV` | Experimental ionization energy (NIST, eV) |
| `IE_error_pct` | Prediction error (%) |

---

## Accuracy

Comparison with NIST experimental data (Z = 1¨C103):

| Error range | Count |
|-------------|:---:|
| < 10% | **48** |
| 10¨C25% | 10 |
| 25¨C50% | 5 |
| 50¨C100% | 9 |
| > 100% | 31 |

**Median error: 13.3%.**

### Strongest performers

Actinides and transuranium elements (Z = 87¨C102)¡ªprecisely where experimental data are scarce and DFT is expensive:

| Element | Z | Predicted IE (eV) | Experimental IE (eV) | Error |
|---------|---|:---:|:---:|:---:|
| Fr | 87 | 3.89 | 4.07 | 4.6% |
| U | 92 | 5.85 | 6.19 | 5.5% |
| Pu | 94 | 5.66 | 6.03 | 6.0% |
| Am | 95 | 5.72 | 5.97 | 4.3% |
| Cm | 96 | 6.07 | 5.99 | 1.3% |
| W | 74 | 7.87 | 7.86 | 0.1% |

### Known limitations

- **Noble gases** (Ne, Ar, Kr, Xe, Rn): hydrogenic formula cannot capture filled-shell stability ¡ú 200¨C400% error
- **Pd** (Z = 46): anomalous [Kr] 4d1? configuration; Slater classification places outermost electron in 4d while physical ionization occurs from 5s
- The hydrogenic IE formula is a crude approximation. Users should treat `Z_eff` as an input feature and train their own regression model rather than using `IE_predicted_eV` directly.

---

## Method

### Shielding constants

Slater's rules applied with shielding constants derived from **radial overlap geometry** of hydrogenic orbitals:

| Shielding type | Derived value | Traditional empirical |
|----------------|:---:|:---:|
| Same-n, different-l | 0.359 | 0.35 |
| n?1 shell | 0.862 | 0.85 |
| Deep (n ¡Ü n_outer?2) | 0.979 | 1.00 |
| Same subshell | 0.350* | 0.35 |

\* Radial symmetry gives ¦Ò = 0.5000; angular-overlap decomposition bridges this to the empirical 0.35. See paper Appendix A for the full derivation.

### Effective nuclear charge

$$Z_{\text{eff}} = Z - \sum_i n_i \sigma_i$$

### Ionization energy (hydrogenic)

$$I = 13.598 \times \left(\frac{Z_{\text{eff}}}{n_{\text{outer}}}\right)^2 \; \text{eV}$$

> ??  This is the hydrogenic approximation. Multi-electron correlation effects are not captured.

---

## Why this table?

| Source | Coverage | Heavy elements | Physical basis |
|--------|:---:|:---:|:---:|
| NIST experiment | Z = 1¨C103 (partial) | Incomplete, inconsistent | Experiment |
| Materials Project / OQMD | Z = 1¨C83 (mainly) | Mostly skipped | DFT |
| **This table** | **Z = 1¨C118 (full)** | ? Uniform method | **Geometric derivation** |

---

## Repository structure

```
©À©¤©¤ slater_geometric_origin.tex       ¡û Paper LaTeX source
©À©¤©¤ periodic_table_features.csv       ¡û Z = 1¨C118 feature table
©À©¤©¤ derive_slater.py                  ¡û Shielding constant derivation
©À©¤©¤ generate_feature_table.py         ¡û Feature table generator
©À©¤©¤ validate_derived_slater.py        ¡û Derived vs. empirical comparison
©À©¤©¤ clementi_validate.py              ¡û SCF Z_eff validation
©À©¤©¤ sn_euv_validate.py                ¡û Sn?? IE validation
©À©¤©¤ benchmark.py                      ¡û Materials ML benchmark
©À©¤©¤ generate_figure1.py               ¡û Figure 1 schematic generator
©À©¤©¤ highlights.txt                    ¡û Paper highlights
©¸©¤©¤ README.md
```

## Citation

If you use this feature table, please cite:

```
Chen, Yafeng. "A Geometric Interpretation of Slater's Shielding Constants." 2026.
```

## License

MIT License ¡ª free to use, modify, and distribute for both academic and commercial purposes.