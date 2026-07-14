"""
Generate Figure 1: Geometric interpretation of sigma = P(r_inner < r_outer)
6-panel figure (3 cols x 2 rows) with proper data-driven plots.

Replaces the ChatGPT-generated 00_geometric_schematic.png which had:
  - Panel (c): question mark in title, mismatched color description
  - Panel (d): text overlap
  - Panel (f): text/table instead of comparison plot
"""

import os
import numpy as np
from scipy.special import genlaguerre, factorial
from scipy.integrate import cumulative_trapezoid, trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
rcParams["axes.unicode_minus"] = False
rcParams["mathtext.fontset"] = "dejavusans"

# ─── Radial wavefunction utilities ───────────────────────────────────────────
def radial_wavefunction(n, l, r, Z=1.0):
    a0 = 1.0
    rho = 2 * Z * r / (n * a0)
    k = n - l - 1
    if k < 0:
        return np.zeros_like(r)
    alpha = 2 * l + 1
    L = genlaguerre(k, alpha)(rho)
    norm = np.sqrt((2*Z/(n*a0))**3 * factorial(k) / (2*n*factorial(n+l)))
    R = norm * rho**l * L * np.exp(-rho/2)
    return R

def radial_probability(n, l, r, Z=1.0):
    R = radial_wavefunction(n, l, r, Z)
    return r**2 * R**2

def compute_sigma(n_outer, l_outer, n_inner, l_inner, Z=1.0, r_max=50, n_points=10000):
    r = np.linspace(0, r_max, n_points)
    P_outer = radial_probability(n_outer, l_outer, r, Z)
    P_inner = radial_probability(n_inner, l_inner, r, Z)
    P_outer /= trapezoid(P_outer, r)
    P_inner /= trapezoid(P_inner, r)
    CDF_inner = cumulative_trapezoid(P_inner, r, initial=0)
    sigma = trapezoid(P_outer * CDF_inner, r)
    return sigma

# ─── Compute all results ─────────────────────────────────────────────────────
subshells = [
    (1,0,"1s"), (2,0,"2s"), (2,1,"2p"),
    (3,0,"3s"), (3,1,"3p"), (3,2,"3d"),
    (4,0,"4s"), (4,1,"4p"), (4,2,"4d"), (4,3,"4f"),
]

results = []
for n_outer, l_outer, name_outer in subshells:
    for n_inner, l_inner, name_inner in subshells:
        if (n_outer, l_outer) <= (n_inner, l_inner):
            continue
        sigma = compute_sigma(n_outer, l_outer, n_inner, l_inner)
        dn = n_outer - n_inner
        results.append((name_outer, name_inner, n_outer, l_outer,
                        n_inner, l_inner, dn, sigma, False))
# Self-pairs
for n, l, name in subshells:
    sigma_self = compute_sigma(n, l, n, l)
    results.append((name, name, n, l, n, l, 0, sigma_self, True))

# ─── Create figure with 3x2 layout ───────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("", fontsize=14)  # No overall title, subfigure labels only

# Panel labels
panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
# Panel titles
panel_titles = [
    'Radial Probability Densities',
    'Cumulative Distribution Functions',
    'Joint Distribution $P(r_1, r_2)$',
    'Physical Picture of Shielding',
    'Overlap of Radial Densities',
    'Geometric vs. Empirical $\\sigma$',
]

# ─── Shared radial grid ──────────────────────────────────────────────────────
r = np.linspace(0, 30, 1000)
colors = plt.cm.Set2(np.linspace(0, 1, 8))

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (a): Radial probability densities for 2s and 2p
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[0, 0]
orbital_pairs_a = [(2,0,"2s"), (2,1,"2p")]
pair_colors = ['#2e86ab', '#d64933']
for i, (n, l, name) in enumerate(orbital_pairs_a):
    P = radial_probability(n, l, r)
    P /= trapezoid(P, r)
    ax.plot(r, P, color=pair_colors[i], label=name, linewidth=2)
ax.set_xlabel("$r$ ($a_0$)", fontsize=11)
ax.set_ylabel("$P_{nl}(r)$", fontsize=11)
ax.set_xlim(0, 20)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
# Add panel label and title
ax.text(-0.1, 1.05, panel_labels[0], transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom')
ax.set_title(panel_titles[0], fontsize=11, pad=8)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (b): Cumulative distribution functions for 2s and 2p
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[0, 1]
for i, (n, l, name) in enumerate(orbital_pairs_a):
    P = radial_probability(n, l, r)
    P /= trapezoid(P, r)
    CDF = cumulative_trapezoid(P, r, initial=0)
    ax.plot(r, CDF, color=pair_colors[i], label=name, linewidth=2)
ax.set_xlabel("$r$ ($a_0$)", fontsize=11)
ax.set_ylabel("CDF($r$)", fontsize=11)
ax.set_xlim(0, 20)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
ax.text(-0.1, 1.05, panel_labels[1], transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom')
ax.set_title(panel_titles[1], fontsize=11, pad=8)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (c): Joint distribution heatmap of r1 (inner) and r2 (outer)
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[0, 2]
# Compute 2D joint distribution for (2s, 2p) -> sigma(2p <- 2s)
r_c = np.linspace(0, 20, 200)
P2s = radial_probability(2, 0, r_c)
P2s /= trapezoid(P2s, r_c)
P2p = radial_probability(2, 1, r_c)
P2p /= trapezoid(P2p, r_c)

# Joint distribution: P(r1, r2) = P_outer(r1) * P_inner(r2)
# r1 = outer (2p), r2 = inner (2s)
joint = np.outer(P2p, P2s)  # shape (len(r), len(r))

im = ax.imshow(joint, extent=[0, 20, 0, 20], origin='lower',
               aspect='auto', cmap='viridis', interpolation='bilinear')

# Draw diagonal line r1 = r2 (the boundary between shielding regimes)
ax.plot([0, 20], [0, 20], 'r--', linewidth=1.5, alpha=0.7, label='$r_1 = r_2$')

# Shade the region r1 < r2 (below diagonal) with a subtle fill
ax.fill_between([0, 20], [0, 20], 0, alpha=0.08, color='green',
                label='$r_1 < r_2$ (shielding)')

ax.set_xlabel("$r_1$ (outer, $2p$) [$a_0$]", fontsize=10)
ax.set_ylabel("$r_2$ (inner, $2s$) [$a_0$]", fontsize=10)

# Add text showing the computed sigma value
sigma_val = compute_sigma(2, 1, 2, 0)  # 2p <- 2s
ax.text(0.95, 0.95, f'$\\sigma(2p \\leftarrow 2s) = {sigma_val:.3f}$',
        transform=ax.transAxes, fontsize=10, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.15)
ax.text(-0.1, 1.05, panel_labels[2], transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom')
ax.set_title(panel_titles[2], fontsize=11, pad=8)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('$P(r_1, r_2)$', fontsize=9)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (d): Physical picture - schematic of shielding
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[1, 0]
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_aspect('equal')
ax.axis('off')

# Draw concentric circles for effective charge regions
nucleus_r = 0.25
inner_r = 1.5
outer_r = 2.8

# Nucleus
nucleus = plt.Circle((0, 0), nucleus_r, color='#d62728', zorder=5)
ax.add_patch(nucleus)
ax.text(0, 0, 'Nucleus\n$+Ze$', ha='center', va='center', color='white',
        fontsize=7, fontweight='bold', zorder=6)

# Inner electron orbit (schematic dashed circle)
inner_circle = plt.Circle((0, 0), inner_r, fill=False, linestyle='--',
                          edgecolor='#2e86ab', linewidth=1.5, zorder=2)
ax.add_patch(inner_circle)

# Outer electron orbit (schematic dashed circle)
outer_circle = plt.Circle((0, 0), outer_r, fill=False, linestyle='--',
                          edgecolor='#d64933', linewidth=1.5, zorder=2)
ax.add_patch(outer_circle)

# Shading between inner and outer (screening region)
shading = plt.Circle((0, 0), outer_r, fill=True, color='#2e86ab',
                     alpha=0.05, zorder=1)
ax.add_patch(shading)
inner_shading = plt.Circle((0, 0), inner_r, fill=True, color='white',
                           zorder=1)
ax.add_patch(inner_shading)

# Electron symbols
# Inner electron
theta_i = np.pi/4
xi = inner_r * np.cos(theta_i)
yi = inner_r * np.sin(theta_i)
ax.plot(xi, yi, 'o', color='#2e86ab', markersize=10, zorder=10)
ax.text(xi+0.2, yi-0.2, '$e^-$ (inner)', fontsize=8, color='#2e86ab', fontweight='bold')

# Outer electron
theta_o = 5*np.pi/6
xo = outer_r * np.cos(theta_o)
yo = outer_r * np.sin(theta_o)
ax.plot(xo, yo, 'o', color='#d64933', markersize=10, zorder=10)
ax.text(xo-0.2, yo+0.2, '$e^-$ (outer)', fontsize=8, color='#d64933', fontweight='bold', ha='right')

# Arrow showing screening
ax.annotate('', xy=(0.8*inner_r*np.cos(np.pi/3), 0.8*inner_r*np.sin(np.pi/3)),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='green', lw=2, alpha=0.7))
ax.text(0.3, 0.8, 'Screening', fontsize=8, color='green', fontweight='bold',
        rotation=-60, alpha=0.7)

# Radius labels
ax.annotate('$r_{\\mathrm{inner}}$', xy=(inner_r, 0), xytext=(inner_r+0.3, -0.3),
            fontsize=8, color='#2e86ab',
            arrowprops=dict(arrowstyle='->', color='#2e86ab', lw=1))
ax.annotate('$r_{\\mathrm{outer}}$', xy=(outer_r, 0), xytext=(outer_r+0.3, 0.3),
            fontsize=8, color='#d64933',
            arrowprops=dict(arrowstyle='->', color='#d64933', lw=1))

# Charge label
ax.text(0.8, -1.0, '$Z_{\\mathrm{eff}} = Z - \\sigma$', fontsize=9,
        fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

ax.text(-0.1, 1.05, panel_labels[3], transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom')
ax.set_title(panel_titles[3], fontsize=11, pad=8)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (e): Overlap of radial densities - purple region
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[1, 1]
P2s_e = radial_probability(2, 0, r)
P2s_e /= trapezoid(P2s_e, r)
P2p_e = radial_probability(2, 1, r)
P2p_e /= trapezoid(P2p_e, r)

ax.plot(r, P2s_e, color='#2e86ab', label='2s', linewidth=2)
ax.plot(r, P2p_e, color='#d64933', label='2p', linewidth=2)

# Fill the overlap (intersection) in purple
overlap = np.minimum(P2s_e, P2p_e)
ax.fill_between(r, overlap, alpha=0.4, color='#7b2d8e', label='Overlap')

ax.set_xlabel("$r$ ($a_0$)", fontsize=11)
ax.set_ylabel("$P_{nl}(r)$", fontsize=11)
ax.set_xlim(0, 20)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
ax.text(-0.1, 1.05, panel_labels[4], transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom')
ax.set_title(panel_titles[4], fontsize=11, pad=8)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (f): Comparison bar chart - geometric sigma vs Slater empirical
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[1, 2]

# Compute averages
rule_groups = [
    ("Deep\n$(n\\!\\leq\\!n_{\\mathrm{out}}\\!-\\!2)$", lambda r: r[6] >= 2, 1.00),
    ("$n-1$\nshell", lambda r: r[6] == 1, 0.85),
    ("Same $n$\ndiff $l$", lambda r: r[6] == 0 and not r[8], 0.35),
    ("Same\nsubshell", lambda r: r[8], 0.50),
]

categories = []
geometric_vals = []
slater_vals = []
for rule_name, cond, target in rule_groups:
    vals = [r[7] for r in results if cond(r)]
    if vals:
        categories.append(rule_name)
        geometric_vals.append(np.mean(vals))
        slater_vals.append(target)

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, slater_vals, width, label="Slater (empirical)",
               color='#d64933', alpha=0.85, edgecolor='darkred', linewidth=0.5)
bars2 = ax.bar(x + width/2, geometric_vals, width, label="Geometric (derived)",
               color='#2e86ab', alpha=0.85, edgecolor='darkblue', linewidth=0.5)

# Add value labels on bars
for bar, val in zip(bars1, slater_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar, val in zip(bars2, geometric_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylabel("$\\sigma$ (shielding constant)", fontsize=11)
ax.set_ylim(0, 1.2)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

# Add annotation for the same-subshell gap
ax.annotate('$\\Delta\\sigma = 0.15$\n(angular-overlap reduction)',
            xy=(3, 0.50), xytext=(3.5, 0.75),
            fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

ax.text(-0.1, 1.05, panel_labels[5], transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom')
ax.set_title(panel_titles[5], fontsize=11, pad=8)

# ─── Final layout ────────────────────────────────────────────────────────────
plt.tight_layout(pad=2.0, w_pad=1.5, h_pad=1.5)

outpath = os.path.join(OUT_DIR, "00_geometric_schematic.png")
plt.savefig(outpath, dpi=200, bbox_inches='tight')
plt.close()

print(f"Figure saved: {outpath}")
print(f"Size: {os.path.getsize(outpath)/1024:.0f} KB")
print("Done!")
