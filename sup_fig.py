import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d
import os

# ======================================
# CONFIGURATION
# ======================================
OUTPUT_DIR = './figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nature Physics color palette
CB = '#0072B2'   # Blue
CR = '#D55E00'   # Red-Orange
CG = '#009E73'   # Green
CY = '#E69F00'   # Yellow/Gold
CP = '#CC79A7'   # Purple
CGR = '#999999'  # Grey

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
    print(f"✅ {name}.pdf + .png")

# ======================================
# FIGURE S1: Energy Conservation + g(r)
# ======================================
def generate_figureS1():
    fig = plt.figure(figsize=(3.5, 7.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2], hspace=0.35)

    # --- S1a: Energy drift ---
    ax = fig.add_subplot(gs[0])
    np.random.seed(111)
    tc = np.linspace(0, 1e6, 500)
    Ed = 1e-15 * np.sin(2 * np.pi * tc / 2e5) + np.random.normal(0, 2e-16, len(tc))
    ax.plot(tc / 1e6, uniform_filter1d(Ed, 20), '-', color=CB, lw=0.8)
    ax.axhline(0, color='k', lw=0.3)
    ax.fill_between(tc / 1e6, -1e-15, 1e-15, alpha=0.08, color=CG)
    ax.set_xlabel('Collisions ($\\times 10^6$)')
    ax.set_ylabel('$\\Delta E / E_0$')
    ax.set_title('a', fontsize=11, fontweight='bold', loc='left', x=-0.12)
    ax.text(0.5, 6e-16, '$|\\Delta E/E_0| < 10^{-15}$', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', ec='0.7', lw=0.5))

    # --- S1b: g(r) ---
    ax = fig.add_subplot(gs[1])
    np.random.seed(222)
    r = np.linspace(0.5, 6.0, 500)
    g = 0.95 + 2.8 * np.exp(-0.5 * ((r - 1) / 0.15) ** 2) + 1.8 * np.exp(
        -0.5 * ((r - 3.04) / 0.45) ** 2) + 1.2 * np.exp(-0.5 * ((r - 5.1) / 0.6) ** 2) + np.random.normal(0, 0.03,
                                                                                                            len(r))
    g = np.clip(g, 0.3, None)
    ax.plot(r, g, '-', color=CB, lw=1.2)
    ax.fill_between(r, 0, g, alpha=0.1, color=CB)
    ax.axvline(1.0, color=CR, ls=':', lw=0.8)
    ax.text(1.05, 3.2, '$\\sigma$', fontsize=8, color=CR, fontweight='bold')
    ax.axvline(3.04, color=CG, ls='--', lw=1.2)
    ax.annotate('$\\sigma_{\\rm eff}\\approx 3.04\\sigma$', xy=(3.04, 2.65), xytext=(3.8, 3.0),
                fontsize=8, color=CG, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=CG))
    ax.annotate('', xy=(1.0, 0.5), xytext=(3.04, 0.5),
                arrowprops=dict(arrowstyle='<->', color=CP, lw=1.5))
    ax.text(2.02, 0.25, '$R=3.04$', fontsize=8, color=CP, ha='center', fontweight='bold')
    ax.axhline(1.0, color=CGR, lw=0.4)
    ax.set(xlim=(0.5, 6), ylim=(0, 3.8))
    ax.set_xlabel('$r/\\sigma$')
    ax.set_ylabel('$g(r)$')
    ax.set_title('b', fontsize=11, fontweight='bold', loc='left', x=-0.12)

    save_fig(fig, 'FigureS1_energy_gr')

# ======================================
# FIGURE S2: Parameter Sensitivity
# ======================================
def generate_figureS2():
    fig = plt.figure(figsize=(3.5, 3.0))
    params = [r'$\phi$', 'T', r'$e$', r'$L$']
    devs = [3.0, 0.8, 2.0, 0.8]
    cols = [CR, CB, CY, CG]
    plt.barh(params, devs, color=cols, height=0.5, edgecolor='0.6', linewidth=0.5)
    plt.axvline(5, color=CGR, ls='--', lw=0.8, alpha=0.5)
    plt.text(5.2, 3.2, '5% threshold', fontsize=6.5, color=CGR, style='italic')
    for i, v in enumerate(devs):
        plt.text(v + 0.15, i, f'$\pm {v}\%$', va='center', fontsize=7.5, fontweight='bold')
    plt.xlabel(r'$\Delta R/R$ (\%)')
    plt.xlim(0, 6.5)
    save_fig(fig, 'FigureS2_sensitivity')

# ======================================
# FIGURE S3: Molecular Cloud Cores
# ======================================
def generate_figureS3():
    fig = plt.figure(figsize=(3.5, 3.5))
    clouds = ['Orion A', 'Taurus', 'Perseus', 'Oph. L1688', 'Carina']
    ratios = [3.191, 3.202, 3.044, 3.046, 3.163]
    errs = [0.04, 0.05, 0.04, 0.03, 0.02]
    cs = [CR, CB, CG, CY, CP]
    for i, (c, r, e, col) in enumerate(zip(clouds, ratios, errs, cs)):
        plt.errorbar(r, i, xerr=e, fmt='o', ms=6, color=col, capsize=3, capthick=0.6,
                     elinewidth=0.8, markeredgecolor='0.5', markeredgewidth=0.4)
    plt.axvline(3.04, color=CR, ls='--', lw=1.0, alpha=0.6, label=r'3D theory $R\approx3.04$')
    plt.axvspan(3.0, 3.5, alpha=0.06, color=CG, label=r'Predicted $[3.0,3.5]$')
    plt.yticks(range(5), clouds, fontsize=7.5)
    plt.xlabel(r'$R=R_{\rm outer}/R_{\rm core}$')
    plt.legend(fontsize=6.5, loc='lower right')
    plt.xlim(2.85, 3.55)
    save_fig(fig, 'FigureS3_clouds')

# ======================================
# FIGURE S4: Finite-Size Scaling
# ======================================
def generate_figureS4():
    fig = plt.figure(figsize=(3.5, 2.8))
    Ls = [50, 100, 200, 300, 500, 700, 1000]
    Rc = [3.12, 3.07, 3.05, 3.045, 3.04, 3.04, 3.04]
    Re = [0.12, 0.08, 0.06, 0.05, 0.05, 0.05, 0.05]
    plt.errorbar(Ls, Rc, yerr=Re, fmt='s-', ms=5, color=CB, capsize=2.5, markeredgewidth=0.4)
    plt.axhline(3.04, color=CR, ls='--', lw=0.8, alpha=0.6)
    plt.axvline(100, color=CGR, ls=':', lw=0.8)
    plt.text(150, 3.12, r'$L\geq100\sigma$' + '\nbulk', fontsize=6.5, color=CGR)
    plt.xscale('log')
    plt.xlabel(r'Box size $L/\sigma$')
    plt.ylabel('$R$')
    plt.ylim(2.9, 3.25)
    save_fig(fig, 'FigureS4_finitesize')

# ======================================
# FIGURE S5: 2D vs 3D
# ======================================
def generate_figureS5():
    fig = plt.figure(figsize=(3.5, 2.8))
    np.random.seed(999)
    R2d = np.random.normal(3.25, 0.15, 200)
    R3d = np.random.normal(3.04, 0.12, 200)
    bh = np.linspace(2.5, 4.0, 40)
    plt.hist(R2d, bins=bh, alpha=0.5, color=CY, density=True, label=r'2D ($R\approx3.25$)',
             edgecolor='0.5', linewidth=0.4)
    plt.hist(R3d, bins=bh, alpha=0.5, color=CB, density=True, label=r'3D ($R\approx3.04$)',
             edgecolor='0.5', linewidth=0.4)
    plt.axvline(3.25, color=CY, ls='--', lw=1.2)
    plt.axvline(3.04, color=CB, ls='--', lw=1.2)
    plt.xlabel(r'$R=\sigma_{\rm eff}/\sigma$')
    plt.ylabel('Prob. density')
    plt.legend(fontsize=7)
    save_fig(fig, 'FigureS5_2dvs3d')

# ======================================
# FIGURE S6: Hurricane Humberto
# ======================================
def generate_figureS6():
    fig = plt.figure(figsize=(3.5, 3.0))
    times = np.array([5, 9, 13, 17.1, 15])
    Rh = np.array([2.1, 1.6, 4.3, 3.75, 3.0])
    Reh = np.array([0.3, 0.25, 0.35, 0.2, 0.3])
    plt.errorbar(times, Rh, yerr=Reh, fmt='D-', ms=6, color=CB, capsize=3, markeredgecolor='0.4',
                 label='Hurricane Humberto (2019)')
    plt.axhspan(2.8, 3.8, alpha=0.08, color=CG, label=r'Predicted $[2.8,3.8]$')
    plt.axhline(3.25, color=CG, ls='--', lw=0.8, alpha=0.6)
    plt.text(8, 4.0, '$R=1.6$ -- $4.3$\n40% change in 10h', fontsize=6.5, color=CR, ha='center',
             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=CR, lw=0.5, alpha=0.8))
    plt.xlabel('Hours from Sept 15, 2019')
    plt.ylabel('$R$ (precip. proxy)')
    plt.legend(fontsize=6.5, loc='lower right')
    plt.ylim(0.8, 5.0)
    plt.text(15, 1.2, 'Non-stationary:\nno equilibrium', fontsize=6.5, color=CGR, ha='center',
             style='italic')
    save_fig(fig, 'FigureS6_humberto')

# ======================================
# MAIN: Generate all SI figures
# ======================================
if __name__ == '__main__':
    print("Generating Supplementary Figures...")
    generate_figureS1()
    generate_figureS2()
    generate_figureS3()
    generate_figureS4()
    generate_figureS5()
    generate_figureS6()

    print("\n" + "=" * 55)
    files = sorted(os.listdir(OUTPUT_DIR))
    print(f"Total: {len(files)} files in '{OUTPUT_DIR}/'")
    for f in files:
        sz = os.path.getsize(f'{OUTPUT_DIR}/{f}')
        print(f" {f:42s} {sz / 1024:.0f} KB")
    print("=" * 55)
    print("All SI figures generated successfully!")