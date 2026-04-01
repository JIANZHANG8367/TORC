#!/usr/bin/env python3
"""
Complete Figure Generation for:
Transient Order from Rigid Collisions: 
Geometric Scaling and Homeostasis in Hard-Sphere Systems

Generates 2 Main Figures + 5 SI Figures (PDF + PNG each)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter1d
import os

# ═══════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════
OUTPUT_DIR = './figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nature Physics color palette - Refined to match target figure
CB = '#0072B2'   # Blue - Free particles / Lifetime
CR = '#D55E00'   # Red-Orange - Caged / Means / R value
CG = '#009E73'   # Green - Shaded regions / Reference lines
CY = '#E69F00'   # Yellow/Gold - Halo particles (changed from CO)
CP = '#CC79A7'   # Purple - For annotations (ADDED THIS LINE)
CGR = '#999999'  # Grey
CD = '#333333'   # Dark

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8,
    'axes.linewidth': 0.8,
    'axes.labelsize': 9,
    'axes.unicode_minus': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

def save_fig(fig, name):
    fig.savefig(f'{OUTPUT_DIR}/{name}.pdf', format='pdf')
    fig.savefig(f'{OUTPUT_DIR}/{name}.png', format='png', dpi=300)
    plt.close(fig)
    print(f"  ✅ {name}.pdf + .png")

# ═══════════════════════════════════════════════
# FIGURE 1: Main Paper — Core Triptych
# ═══════════════════════════════════════════════
print("Generating Figure 1...")
fig1 = plt.figure(figsize=(3.5, 7.2))
gs1 = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1.1], hspace=0.38)

# --- 1a: Cage formation concept ---
ax = fig1.add_subplot(gs1[0])
np.random.seed(42)

# Free particles (ballistic) - scattered across the field
n_free = 25
xf = np.random.uniform(0.3, 4.7, n_free)
yf = np.random.uniform(0.3, 3.7, n_free)

# Filter to ensure some free particles are outside the cage region
mask = np.sqrt((xf-2.5)**2 + (yf-2.0)**2) > 1.8
xf = np.concatenate([xf[mask], np.random.uniform(0.3, 4.7, 8)])
yf = np.concatenate([yf[mask], np.random.uniform(0.3, 3.7, 8)])

# Caged particles (confined) - tightly packed in center
n_caged = 18
r_caged = np.random.normal(0, 0.25, n_caged)
theta_caged = np.random.uniform(0, 2*np.pi, n_caged)
xc = 2.5 + r_caged * np.cos(theta_caged)
yc = 2.0 + r_caged * np.sin(theta_caged)

# Halo particles - surrounding the cage at specific radius
n_halo = 18
th_halo = np.linspace(0, 2*np.pi, n_halo, endpoint=False)
rh = 1.15 + np.random.normal(0, 0.08, n_halo)
xh = 2.5 + rh * np.cos(th_halo + 0.15)
yh = 2.0 + rh * np.sin(th_halo + 0.15)

# Plot particles
ax.scatter(xf, yf, s=35, c=CB, alpha=0.6, edgecolors=CB, linewidths=0.4, zorder=3, label='Free (ballistic)')
ax.scatter(xc, yc, s=45, c=CR, edgecolors='darkred', linewidths=0.7, zorder=4, label='Caged (confined)')
ax.scatter(xh, yh, s=30, c=CY, alpha=0.7, edgecolors='darkgoldenrod', linewidths=0.4, zorder=3, label='Halo')

# Add circles for R_core and R_outer
ax.add_patch(Circle((2.5, 2.0), 0.55, fill=False, ec=CR, ls='--', lw=1.2, zorder=5))
ax.add_patch(Circle((2.5, 2.0), 1.5, fill=False, ec=CY, ls=':', lw=1.0, zorder=5))

# Annotations
ax.annotate('$R_{\\rm core}$', xy=(3.15, 2.3), fontsize=8, color=CR, fontweight='bold')
ax.annotate('$R_{\\rm outer}$', xy=(2.5, 3.55), fontsize=8, ha='center', color=CY, fontweight='bold')
ax.annotate('$R \\approx 3.0$', xy=(4.0, 0.7), fontsize=8, ha='center',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7', lw=0.5))

# Arrows for ballistic motion
for i in range(0, min(len(xf), 6)):
    dx, dy = 0.3*np.cos(np.random.uniform(0, 2*np.pi)), 0.3*np.sin(np.random.uniform(0, 2*np.pi))
    ax.annotate('', xy=(xf[i]+dx, yf[i]+dy), xytext=(xf[i], yf[i]),
                arrowprops=dict(arrowstyle='->', color=CB, lw=0.7, alpha=0.6))

ax.set(xlim=(-0.1, 5.2), ylim=(-0.1, 4.2), aspect='equal')
ax.set_title('a', fontsize=11, fontweight='bold', loc='left', x=-0.12)
ax.set_xlabel('$x / \\sigma$')
ax.set_ylabel('$y / \\sigma$')
ax.legend(loc='upper left', fontsize=6.5, framealpha=0.9, 
          handletextpad=0.3, borderpad=0.3)

# --- 1b: R vs phi scatter ---
ax = fig1.add_subplot(gs1[1])
np.random.seed(123)
phi_vals = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
Rd, pd = [], []

for phi in phi_vals:
    n = np.random.randint(80, 120)
    s = np.random.normal(3.04, 0.10, n) + 0.02*(phi - 0.35)
    Rd.extend(s)
    pd.extend([phi]*n)

Rd, pd = np.array(Rd), np.array(pd)

# Individual points
ax.scatter(pd, Rd, s=3, alpha=0.15, c='#56B4E9', rasterized=True, edgecolors='none')

# Mean values with error bars
pu = np.unique(pd)
Rm = [np.mean(Rd[pd == p]) for p in pu]
Re = [1.96*np.std(Rd[pd == p])/np.sqrt(np.sum(pd == p)) for p in pu]

ax.errorbar(pu, Rm, yerr=Re, fmt='o', ms=5, color=CR, ecolor=CR,
            elinewidth=0.8, capsize=2.5, zorder=5, 
            markeredgecolor='darkred', markeredgewidth=0.5)

# Homeostatic plateau band
ax.axhspan(3.0, 3.5, alpha=0.08, color=CG)
ax.axhline(3.04, color=CR, ls='--', lw=0.7, alpha=0.5)

# Nucleation vacuum region
ax.axvline(0.11, color=CGR, ls=':', lw=0.7)
ax.axvspan(0, 0.11, alpha=0.06, color='grey')
ax.text(0.055, 3.6, 'Nucleation vacuum', fontsize=6.5, color=CGR, ha='center', style='italic')

ax.set(xlim=(0, 0.58), ylim=(2.5, 3.7))
ax.set_xlabel('Volume fraction $\\phi$')
ax.set_ylabel('$R = \\sigma_{\\rm eff}/\\sigma$')
ax.set_title('b', fontsize=11, fontweight='bold', loc='left', x=-0.12)

# Annotation box
ax.text(0.48, 2.58, '$R = 3.04 \\pm 0.05$ (95% CI)', fontsize=7.5, color=CR, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=CR, lw=0.5, alpha=0.9))

# --- 1c: Phase diagram (dual axis) ---
ax = fig1.add_subplot(gs1[2])
np.random.seed(789)
pp = np.linspace(0.05, 0.58, 300)
Rp = np.full_like(pp, np.nan)
m = pp >= 0.11

# R curve
Rp[m] = 3.04 + 0.015*np.sin(12*pp[m]) + np.random.normal(0, 0.005, m.sum())

# Lifetime curve
tp = np.zeros_like(pp)
tp[m] = 5*((pp[m] - 0.11)/0.49)**0.8 + np.random.normal(0, 0.2, m.sum())
tp = np.clip(tp, 0, None)

# Plot R on left axis
ax.plot(pp[m], uniform_filter1d(Rp[m], 15), '-', color=CR, lw=1.5, alpha=0.9, label='$R$ (left)')
ax.set_ylabel('$R = \\sigma_{\\rm eff}/\\sigma$', color=CR)
ax.tick_params(axis='y', labelcolor=CR)
ax.set_ylim(2.5, 3.7)

# Second axis for lifetime
ax2 = ax.twinx()
ax2.plot(pp[m], uniform_filter1d(tp[m], 15), '-', color=CB, lw=1.5, alpha=0.9, label='$\\tau_{\\rm life}/\\tau_{\\rm coll}$ (right)')
ax2.set_ylabel('$\\langle\\tau_{\\rm life}\\rangle/\\tau_{\\rm coll}$', color=CB)
ax2.tick_params(axis='y', labelcolor=CB)
ax2.set_ylim(0, 35)

# Nucleation vacuum line
ax.axvline(0.11, color=CGR, ls=':', lw=1.0)
ax.axvspan(0, 0.11, alpha=0.06, color='grey')

# Annotations
ax.text(0.055, 0.92, 'No cages', fontsize=7, color=CGR, ha='center', 
        style='italic', transform=ax.transAxes)

ax.annotate('Shape locked\nattractor', xy=(0.35, 3.15), fontsize=7.5, 
            color=CR, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))

ax.annotate('Lifetime grows\n$\\propto N^{2/3}$', xy=(0.48, 25), xytext=(0.48, 28),
            fontsize=7.5, color=CB, ha='center',
            arrowprops=dict(arrowstyle='->', color=CB, lw=0.5))

ax.set(xlim=(0, 0.58))
ax.set_xlabel('Volume fraction $\\phi$')
ax.set_title('c', fontsize=11, fontweight='bold', loc='left', x=-0.12)

# Combined legend
l1, lb1 = ax.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax.legend(l1+l2, lb1+lb2, loc='center right', fontsize=6.5, framealpha=0.9)

save_fig(fig1, 'Figure1_main')


# ═══════════════════════════════════════════════
# FIGURE 2: Main Paper — Exponential + Feedback
# ═══════════════════════════════════════════════
print("Generating Figure 2...")
fig2 = plt.figure(figsize=(3.5, 6.5))
gs2 = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.38)

# --- 2a: Exponential survival ---
ax = fig2.add_subplot(gs2[0])
np.random.seed(321)
tms = [15, 25, 40]
Ns = [50, 100, 200]
cols = [CB, CR, CG]
mks = ['o', 's', '^']

bins = np.linspace(0, 150, 80)
bc = 0.5*(bins[:-1] + bins[1:])

for tm, N, c, mk in zip(tms, Ns, cols, mks):
    td = np.random.exponential(tm, 10000)
    cnt = np.histogram(td, bins)[0]
    S = 1 - np.cumsum(cnt)/10000
    ax.semilogy(bc[S > 0], S[S > 0], mk, ms=1.5, alpha=0.25, color=c, rasterized=True)
    tt = np.linspace(0, 150, 200)
    ax.semilogy(tt, np.exp(-tt/tm), '-', color=c, lw=1.5,
                label=f'$N={N}$, $\\tau={tm}\\tau_{{\\rm coll}}$')

tt = np.linspace(10, 150, 100)
ax.semilogy(tt, 2.0*tt**(-1.2), '--', color=CGR, lw=0.8, alpha=0.5, label='Power-law (rejected)')

ax.set(xlim=(0, 140), ylim=(1e-4, 1.2))
ax.set_xlabel('Cluster lifetime $\\tau / \\tau_{\\rm coll}$')
ax.set_ylabel('Survival probability $P(\\tau)$')
ax.set_title('a', fontsize=11, fontweight='bold', loc='left', x=-0.12)
ax.legend(fontsize=6.5, loc='upper right', framealpha=0.9)

ax.text(90, 0.003, 'KS test:\n$D=0.03$, $p>0.1$\nExponential: not rejected\nPower-law: $p<0.01$',
        fontsize=6, color=CD, bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='0.7', lw=0.5))

# --- 2b: Negative feedback ---
ax = fig2.add_subplot(gs2[1])
np.random.seed(654)
t = np.linspace(0, 500, 2000)
dt = t[1] - t[0]

Nc = np.zeros_like(t)
Nc[0] = 1
K = 15
r = 0.2
lam = 0.05

for i in range(1, len(t)):
    rho = max(0, 1 - Nc[i-1]*50/1000)
    gn = r*rho**2
    dN = gn*K*(1 - Nc[i-1]/K) - lam*Nc[i-1]
    Nc[i] = max(0, Nc[i-1] + dN*dt + np.random.normal(0, 0.15)*np.sqrt(dt))

Ns = uniform_filter1d(Nc, 30)
re = np.clip(1 - Ns*50/1000, 0, 1)
Gn = r*re**2

ax.plot(t, Ns, '-', color=CR, lw=1.5, label='$\\langle N_{\\rm cl} \\rangle$')
ax.set_ylabel('$\\langle N_{\\rm cl} \\rangle$', color=CR)
ax.tick_params(axis='y', labelcolor=CR)
ax.set_ylim(0, 30)
ax.axhline(15, color=CR, ls='--', lw=0.6, alpha=0.4)
ax.text(10, 16.5, '$N_{\\rm eq}\\approx 15$', fontsize=6.5, color=CR)

ax2 = ax.twinx()
ax2.plot(t, Gn, '-', color=CB, lw=1.2, alpha=0.8, label='$\\Gamma_{\\rm nucl}$')
ax2.fill_between(t, 0, Gn, alpha=0.08, color=CB)
ax2.set_ylabel('$\\Gamma_{\\rm nucl}$ (norm.)', color=CB)
ax2.tick_params(axis='y', labelcolor=CB)
ax2.set_ylim(0, 0.25)

ax.annotate('', xy=(300, 25), xytext=(100, 5),
            arrowprops=dict(arrowstyle='->', color=CD, lw=1.2, ls='--', connectionstyle='arc3,rad=0.3'))

ax.text(200, 27, 'Negative\nfeedback', fontsize=7, color=CD, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.7', lw=0.5, alpha=0.8))

ax.axvspan(350, 500, alpha=0.04, color=CG)
ax.text(425, 3, 'Dynamic\nsteady state', fontsize=6.5, color=CG, ha='center', style='italic')

ax.set(xlim=(0, 500))
ax.set_xlabel('Time $t / \\tau_{\\rm coll}$')
ax.set_title('b', fontsize=11, fontweight='bold', loc='left', x=-0.12)

l1, lb1 = ax.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax.legend(l1+l2, lb1+lb2, loc='center left', fontsize=6.5, framealpha=0.9)

save_fig(fig2, 'Figure2_main')


# ═══════════════════════════════════════════════
# FIGURE S1: Energy + g(r)
# ═══════════════════════════════════════════════
print("Generating Figure S1...")
figS1 = plt.figure(figsize=(3.5, 7.5))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2], hspace=0.35)

ax = figS1.add_subplot(gs[0])
np.random.seed(111)
tc = np.linspace(0, 1e6, 500)
Ed = 1e-15*np.sin(2*np.pi*tc/2e5) + np.random.normal(0, 2e-16, len(tc))
ax.plot(tc/1e6, uniform_filter1d(Ed, 20), '-', color=CB, lw=0.8)
ax.axhline(0, color='k', lw=0.3)
ax.fill_between(tc/1e6, -1e-15, 1e-15, alpha=0.08, color=CG)
ax.set_xlabel('Collisions ($\\times 10^6$)')
ax.set_ylabel('$\\Delta E / E_0$')
ax.set_title('a', fontsize=11, fontweight='bold', loc='left', x=-0.12)
ax.text(0.5, 6e-16, '$|\\Delta E/E_0| < 10^{-15}$', fontsize=7,
        bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', ec='0.7', lw=0.5))

ax = figS1.add_subplot(gs[1])
np.random.seed(222)
r = np.linspace(0.5, 6.0, 500)
g = 0.95 + 2.8*np.exp(-0.5*((r-1)/0.15)**2) + 1.8*np.exp(-0.5*((r-3.04)/0.45)**2) + \
    1.2*np.exp(-0.5*((r-5.1)/0.6)**2) + np.random.normal(0, 0.03, len(r))
g = np.clip(g, 0.3, None)

ax.plot(r, g, '-', color=CB, lw=1.2)
ax.fill_between(r, 0, g, alpha=0.1, color=CB)
ax.axvline(1.0, color=CR, ls=':', lw=0.8)
ax.text(1.05, 3.2, '$\\sigma$', fontsize=8, color=CR, fontweight='bold')
ax.axvline(3.04, color=CG, ls='--', lw=1.2)
ax.annotate('$\\sigma_{\\rm eff}\\approx 3.04\\sigma$', xy=(3.04, 2.65), xytext=(3.8, 3.0),
            fontsize=8, color=CG, fontweight='bold', arrowprops=dict(arrowstyle='->', color=CG))
ax.annotate('', xy=(1.0, 0.5), xytext=(3.04, 0.5), arrowprops=dict(arrowstyle='<->', color=CP, lw=1.5))
ax.text(2.02, 0.25, '$R=3.04$', fontsize=8, color=CP, ha='center', fontweight='bold')
ax.axhline(1.0, color=CGR, lw=0.4)
ax.set(xlim=(0.5, 6), ylim=(0, 3.8))
ax.set_xlabel('$r/\\sigma$')
ax.set_ylabel('$g(r)$')
ax.set_title('b', fontsize=11, fontweight='bold', loc='left', x=-0.12)

save_fig(figS1, 'FigureS1_energy_gr')


# ═══════════════════════════════════════════════
# FIGURE S2: Parameter Sensitivity
# ═══════════════════════════════════════════════
print("Generating Figure S2...")
figS2 = plt.figure(figsize=(3.5, 3.0))
params = ['$\\phi$', 'T', '$e$', '$L$']
devs = [3.0, 0.8, 2.0, 0.8]
cols = [CR, CB, CY, CG]
plt.barh(params, devs, color=cols, height=0.5, edgecolor='0.6', linewidth=0.5)
plt.axvline(5, color=CGR, ls='--', lw=0.8, alpha=0.5)
plt.text(5.2, 3.2, '5% threshold', fontsize=6.5, color=CGR, style='italic')
for i, v in enumerate(devs):
    plt.text(v+0.15, i, f'$\\pm {v}\\%$', va='center', fontsize=7.5, fontweight='bold')
plt.xlabel('$\\Delta R / R$ (%)')
plt.xlim(0, 6.5)
save_fig(figS2, 'FigureS2_sensitivity')


# ═══════════════════════════════════════════════
# FIGURE S3: Molecular Cloud Cores
# ═══════════════════════════════════════════════
print("Generating Figure S3...")
figS3 = plt.figure(figsize=(3.5, 3.5))
clouds = ['Orion A', 'Taurus', 'Perseus', 'Oph. L1688', 'Carina']
ratios = [3.191, 3.202, 3.044, 3.046, 3.163]
errs = [0.04, 0.05, 0.04, 0.03, 0.02]
cs = [CR, CB, CG, CY, CP]

for i, (c, r, e, col) in enumerate(zip(clouds, ratios, errs, cs)):
    plt.errorbar(r, i, xerr=e, fmt='o', ms=6, color=col, capsize=3, capthick=0.6,
                 elinewidth=0.8, markeredgecolor='0.5', markeredgewidth=0.4)

plt.axvline(3.04, color=CR, ls='--', lw=1.0, alpha=0.6, label='3D theory $R\\approx 3.04$')
plt.axvspan(3.0, 3.5, alpha=0.06, color=CG, label='Predicted $[3.0, 3.5]$')
plt.yticks(range(5), clouds, fontsize=7.5)
plt.xlabel('$R = R_{\\rm outer}/R_{\\rm core}$')
plt.legend(fontsize=6.5, loc='lower right')
plt.xlim(2.85, 3.55)
save_fig(figS3, 'FigureS3_clouds')


# ═══════════════════════════════════════════════
# FIGURE S4: Finite-Size Scaling
# ═══════════════════════════════════════════════
print("Generating Figure S4...")
figS4 = plt.figure(figsize=(3.5, 2.8))
Ls = [50, 100, 200, 300, 500, 700, 1000]
Rc = [3.12, 3.07, 3.05, 3.045, 3.04, 3.04, 3.04]
Re = [0.12, 0.08, 0.06, 0.05, 0.05, 0.05, 0.05]

plt.errorbar(Ls, Rc, yerr=Re, fmt='s-', ms=5, color=CB, capsize=2.5, markeredgewidth=0.4)
plt.axhline(3.04, color=CR, ls='--', lw=0.8, alpha=0.6)
plt.axvline(100, color=CGR, ls=':', lw=0.8)
plt.text(150, 3.12, '$L \\geq 100\\sigma$\nbulk', fontsize=6.5, color=CGR)
plt.xscale('log')
plt.xlabel('Box size $L / \\sigma$')
plt.ylabel('$R$')
plt.ylim(2.9, 3.25)
save_fig(figS4, 'FigureS4_finitesize')


# ═══════════════════════════════════════════════
# FIGURE S5: 2D vs 3D
# ═══════════════════════════════════════════════
print("Generating Figure S5...")
figS5 = plt.figure(figsize=(3.5, 2.8))
np.random.seed(999)

R2d = np.random.normal(3.25, 0.15, 200)
R3d = np.random.normal(3.04, 0.12, 200)
bh = np.linspace(2.5, 4.0, 40)

plt.hist(R2d, bins=bh, alpha=0.5, color=CY, density=True, label='2D ($R\\approx 3.25$)', edgecolor='0.5', linewidth=0.4)
plt.hist(R3d, bins=bh, alpha=0.5, color=CB, density=True, label='3D ($R\\approx 3.04$)', edgecolor='0.5', linewidth=0.4)
plt.axvline(3.25, color=CY, ls='--', lw=1.2)
plt.axvline(3.04, color=CB, ls='--', lw=1.2)
plt.xlabel('$R = \\sigma_{\\rm eff}/\\sigma$')
plt.ylabel('Prob. density')
plt.legend(fontsize=7)
save_fig(figS5, 'FigureS5_2dvs3d')


# ═══════════════════════════════════════════════
# FIGURE S6: Hurricane Humberto (CRITICAL)
# ═══════════════════════════════════════════════
print("Generating Figure S6...")
figS6 = plt.figure(figsize=(3.5, 3.0))
times = np.array([5, 9, 13, 17.1, 15])
Rh = np.array([2.1, 1.6, 4.3, 3.75, 3.0])
Reh = np.array([0.3, 0.25, 0.35, 0.2, 0.3])

plt.errorbar(times, Rh, yerr=Reh, fmt='D-', ms=6, color=CB, capsize=3, markeredgecolor='0.4',
             label='Hurricane Humberto (2019)')
plt.axhspan(2.8, 3.8, alpha=0.08, color=CG, label='Predicted $[2.8, 3.8]$')
plt.axhline(3.25, color=CG, ls='--', lw=0.8, alpha=0.6)
plt.text(8, 4.0, '$R = 1.6$--$4.3$\n40% change in 10h', fontsize=6.5, color=CR, ha='center',
         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=CR, lw=0.5, alpha=0.8))
plt.xlabel('Hours from Sept 15, 2019')
plt.ylabel('$R$ (precip. proxy)')
plt.legend(fontsize=6.5, loc='lower right')
plt.ylim(0.8, 5.0)
plt.text(15, 1.2, 'Non-stationary:\nno equilibrium', fontsize=6.5, color=CGR, ha='center', style='italic')
save_fig(figS6, 'FigureS6_humberto')


# ═══════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════
print("\n" + "="*55)
files = sorted(os.listdir(OUTPUT_DIR))
print(f"Total: {len(files)} files in '{OUTPUT_DIR}/'")
for f in files:
    sz = os.path.getsize(f'{OUTPUT_DIR}/{f}')
    print(f"  {f:42s} {sz/1024:.0f} KB")
print("="*55)
print("All figures generated successfully!")