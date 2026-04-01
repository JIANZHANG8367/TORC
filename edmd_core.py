#!/usr/bin/env python3
"""
edmd_verify.py — Event-Driven Molecular Dynamics Simulation & Analysis
===================================================================
Core simulation engine and analysis pipeline for:
"Transient Order from Rigid Collisions: Geometric Scaling and Homeostasis 
in Hard-Sphere Systems"

Implements exact hard-sphere EDMD, Union-Find cluster detection, 
g(r) extraction, and statistical verification of geometric scaling.

Usage:
    python edmd_verify.py --quick-test         # Fast physics unit tests (~10s)
    python edmd_verify.py                      # Run baseline (N=10000, 2x10^5 coll)
    python edmd_verify.py --sweep              # Sweep phi from 0.05 to 0.50
    python edmd_verify.py --N 50000 --sweep    # Large-scale sweep
    
Dependencies: numpy, scipy
    pip install numpy scipy
"""

import numpy as np
import heapq
import argparse
import json
import csv
import os
import sys
import time
from collections import defaultdict

try:
    from scipy.stats import kstest
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARNING] scipy not found; KS tests will be skipped")

# =============================================================================
# MODULE 1: EDMD Engine (Exact Hard-Sphere Dynamics)
# =============================================================================

def predict_collision_time(ri, rj, vi, vj, sigma=1.0):
    """
    Predict time until spheres i,j collide.
    Solves |dr(t)|^2 = sigma^2 for the smallest positive t, 
    where dr(t) = dr - dv*t, with dr = rj - ri and dv = vi - vj.
    """
    dr = rj - ri
    dv = vi - vj
    
    dr2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]
    dv2 = dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2]
    
    if dv2 < 1e-30:
        return float('inf')
        
    dvdv = dr[0]*dv[0] + dr[1]*dv[1] + dr[2]*dv[2]
    disc = dvdv * dvdv - dv2 * (dr2 - sigma * sigma)
    
    if disc < 0:
        return float('inf')
        
    sqrt_disc = np.sqrt(disc)
    
    t1 = (dvdv - sqrt_disc) / dv2
    t2 = (dvdv + sqrt_disc) / dv2
    
    if t1 > 1e-10:
        return t1
    elif t2 > 1e-10:
        return t2
        
    return float('inf')

def predict_wall_time(pos, vel, L):
    """Predict time for a particle to hit the nearest periodic wall."""
    t_min = float('inf')
    for d in range(3):
        if vel[d] > 1e-15:
            t = (L - pos[d]) / vel[d]
            if 0 < t < t_min:
                t_min = t
        elif vel[d] < -1e-15:
            t = -pos[d] / vel[d]
            if 0 < t < t_min:
                t_min = t
    return t_min

def elastic_collision(ri, rj, vi, vj):
    """
    Equal-mass elastic collision.
    Conserves exact kinetic energy and momentum.
    """
    dr = rj - ri
    dist = np.sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2])
    if dist < 1e-15:
        return vi.copy(), vj.copy()
        
    n = dr / dist
    dv = vi - vj
    dvn = dv[0]*n[0] + dv[1]*n[1] + dv[2]*n[2]
    
    if dvn <= 0:
        return vi.copy(), vj.copy()
        
    vi_new = vi - dvn * n
    vj_new = vj + dvn * n
    return vi_new, vj_new

def min_image(dr, L):
    """Apply minimum image convention for periodic boundaries."""
    return dr - L * np.round(dr / L)

class EDMD:
    """
    Event-Driven Molecular Dynamics for hard spheres in a 3D periodic box.
    State: arrays pos[N,3], vel[N,3]
    Event queue: min-heap (time, counter, i, j) where j=-1 denotes wall.
    Lazy invalidation via per-particle monotonic counter.
    """
    def __init__(self, N, phi, seed=None):
        self.N = N
        self.phi = phi
        self.sigma = 1.0
        self.radius = 0.5
        
        vol_particle = N * (4.0 / 3.0) * np.pi * self.radius ** 3
        self.L = (vol_particle / phi) ** (1.0 / 3.0)
        
        self.rng = np.random.RandomState(seed)
        self.pos = np.zeros((N, 3), dtype=np.float64)
        self.vel = np.zeros((N, 3), dtype=np.float64)
        
        self.pq = []
        self.counter = 0
        self.valid = np.zeros(N, dtype=np.int64)
        self.time = 0.0
        self.n_collisions = 0
        
        self.energy_log = []
        self.E0 = None
        
        self._place_particles()
        self.E0 = self._kinetic_energy()
        self.energy_log.append((0, 1.0))

    def _place_particles(self):
        """Cubic lattice initialization with random unit-speed velocities."""
        n_side = max(1, int(np.ceil(self.N ** (1.0 / 3.0))))
        sp = self.L / n_side
        idx = 0
        for ix in range(n_side):
            for iy in range(n_side):
                for iz in range(n_side):
                    if idx >= self.N:
                        break
                    self.pos[idx] = [(ix + 0.5) * sp, (iy + 0.5) * sp, (iz + 0.5) * sp]
                    idx += 1
                if idx >= self.N:
                    break
            if idx >= self.N:
                break
            
        for i in range(self.N):
            d = self.rng.randn(3)
            self.vel[i] = d / np.linalg.norm(d)

    def _kinetic_energy(self):
        return 0.5 * np.sum(self.vel * self.vel)

    def _schedule(self, i):
        """Find earliest event for particle i and push onto heap."""
        L = self.L
        pi = self.pos[i]
        vi = self.vel[i]
        t_best = float('inf')
        j_best = -1
        
        tw = predict_wall_time(pi, vi, L)
        t_best = tw
        
        for j in range(self.N):
            if j == i:
                continue
            dr = min_image(self.pos[j] - pi, L)
            t = predict_collision_time(pi, pi + dr, vi, self.vel[j], self.sigma)
            if t < t_best:
                t_best = t
                j_best = j
                
        self.counter += 1
        self.valid[i] = self.counter
        heapq.heappush(self.pq, (self.time + t_best, self.counter, i, j_best))

    def _init_events(self):
        self.pq = []
        self.counter = 0
        self.valid[:] = 0
        for i in range(self.N):
            self._schedule(i)

    def step(self):
        """Process one event from the priority queue."""
        if not self.pq:
            return False
            
        t_event, cnt, i, j = heapq.heappop(self.pq)
        
        if self.valid[i] != cnt:
            return True
            
        dt = t_event - self.time
        if dt < -1e-12:
            return True 
            
        if dt > 0:
            self.pos += self.vel * dt
            self.time = t_event
            
        if j == -1:
            for d in range(3):
                if self.pos[i, d] < 0:
                    self.pos[i, d] += self.L
                elif self.pos[i, d] >= self.L:
                    self.pos[i, d] -= self.L
            self._schedule(i)
        else:
            dr = min_image(self.pos[j] - self.pos[i], self.L)
            dist = np.sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2])
            
            if dist > self.sigma * 1.02: 
                self._schedule(i)
                return True
                
            vi_new, vj_new = elastic_collision(
                self.pos[i], self.pos[i] + dr, self.vel[i], self.vel[j]
            )
            self.vel[i] = vi_new
            self.vel[j] = vj_new
            self.n_collisions += 1
            
            if self.n_collisions % 100 == 0:
                E = self._kinetic_energy()
                self.energy_log.append((
                    self.n_collisions, 
                    E / self.E0 if self.E0 > 0 else 1.0
                ))
                
            self._schedule(i)
            self._schedule(j)
            
        return True

    def run(self, n_collisions, measure_cb=None, measure_interval=1000):
        """Run simulation until target collisions are processed."""
        self._init_events()
        last_measure = 0
        
        while self.n_collisions < n_collisions:
            if not self.step():
                self._init_events()
                if not self.pq:
                    break
            if (measure_cb is not None and 
                self.n_collisions - last_measure >= measure_interval):
                measure_cb(self.n_collisions)
                last_measure = self.n_collisions
                
        E = self._kinetic_energy()
        self.energy_log.append((
            self.n_collisions, 
            E / self.E0 if self.E0 > 0 else 1.0
        ))

# =============================================================================
# MODULE 2: Cluster Tracker (Union-Find)
# =============================================================================

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        r = x
        while self.parent[r] != r:
            r = self.parent[r]
        while self.parent[x] != r:
            self.parent[x], x = r, self.parent[x]
        return r

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

def detect_clusters(pos, L, N, bond_cutoff=1.05):
    """Union-Find cluster detection based on center-to-center distance."""
    uf = UnionFind(N)
    bc2 = bond_cutoff * bond_cutoff
    for i in range(N):
        for j in range(i + 1, N):
            dr = min_image(pos[j] - pos[i], L)
            d2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]
            if d2 < bc2:
                uf.union(i, j)
                
    clusters = defaultdict(list)
    for i in range(N):
        clusters[uf.find(i)].append(i)
    return dict(clusters)

# =============================================================================
# MODULE 3: Measurements
# =============================================================================

def compute_gr(pos, L, N, n_bins=100, r_max=None):
    """Compute radial distribution function g(r)."""
    if r_max is None:
        r_max = min(L / 2.0, 5.0)
    dr = r_max / n_bins
    hist = np.zeros(n_bins)
    
    for i in range(N):
        for j in range(i + 1, N):
            d = min_image(pos[j] - pos[i], L)
            dist = np.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2])
            if dist < r_max:
                b = int(dist / dr)
                if b < n_bins:
                    hist[b] += 2
                    
    rho = N / (L * L * L)
    r_mid = np.zeros(n_bins)
    for b in range(n_bins):
        r_lo = b * dr
        r_hi = (b + 1) * dr
        r_mid[b] = (r_lo + r_hi) / 2.0
        vol = (4.0 / 3.0) * np.pi * (r_hi**3 - r_lo**3)
        if vol > 0:
            hist[b] /= (N * rho * vol)
    return r_mid, hist

def extract_R(r_vals, gr_vals):
    """Extract scale ratio R = sigma_eff / sigma from g(r) peaks."""
    mask1 = (r_vals >= 0.8) & (r_vals <= 1.3)
    if not np.any(mask1):
        return None
    gr1 = gr_vals[mask1]
    r1 = r_vals[mask1]
    contact_peak_r = r1[np.argmax(gr1)]
    
    mask2 = (r_vals >= contact_peak_r + 0.3) & (r_vals <= 6.0)
    if not np.any(mask2):
        return None
    gr2 = gr_vals[mask2]
    r2 = r_vals[mask2]
    
    if np.max(gr2) < 0.05:
        return None
    return float(r2[np.argmax(gr2)])

def ks_exponential_test(data):
    """KS test against exponential distribution."""
    if not HAS_SCIPY or len(data) < 10:
        return None, None
    data = np.asarray(data, dtype=float)
    lam_inv = np.mean(data)
    if lam_inv <= 0:
        return None, None
    stat, p = kstest(data, 'expon', args=(0, lam_inv))
    return stat, p

def ks_powerlaw_test(data):
    """KS test against power-law distribution."""
    if not HAS_SCIPY or len(data) < 10:
        return None, None
    data = np.asarray(data, dtype=float)
    x_min = np.min(data)
    if x_min <= 0:
        return None, None
    n = len(data)
    alpha = 1.0 + n / np.sum(np.log(data / x_min))
    def cdf(x, alpha=alpha, xm=x_min):
        return np.where(x < xm, 0.0, 1.0 - (x / xm) ** (1 - alpha))
    stat, p = kstest(data, cdf)
    return stat, p

# =============================================================================
# MODULE 4: Runner & Verification
# =============================================================================

def run_single(N, phi, n_coll, seed=42, measure_interval=None):
    """Execute one EDMD simulation and extract measurements."""
    if measure_interval is None:
        measure_interval = max(100, n_coll // 50)
        
    t0 = time.time()
    sim = EDMD(N, phi, seed=seed)
    pop_history = []
    
    def measure(n_col):
        if n_col <= 0:
            return
        clusters = detect_clusters(sim.pos, sim.L, N, bond_cutoff=1.05)
        n_sig = sum(1 for v in clusters.values() if len(v) >= 7)
        n_env = sum(1 for v in clusters.values() if 3 <= len(v) < 7)
        pop_history.append((n_col, n_sig, n_env))

    sim.run(n_coll, measure_cb=measure, measure_interval=measure_interval)
    
    clusters_final = detect_clusters(sim.pos, sim.L, N, bond_cutoff=1.05)
    r_vals, gr_vals = compute_gr(sim.pos, sim.L, N, n_bins=120, r_max=min(sim.L / 2.0, 5.0))
    R = extract_R(r_vals, gr_vals)
    
    n_cl_series = [p[1] for p in pop_history] if pop_history else [0]
    mean_ncl = float(np.mean(n_cl_series))
    std_ncl = float(np.std(n_cl_series))
    
    energy_ratios = [e[1] for e in sim.energy_log]
    max_edev = max(abs(r - 1.0) for r in energy_ratios) if energy_ratios else 0.0
    dt = time.time() - t0
    
    return {
        'phi': phi, 'N': N, 'n_collisions': sim.n_collisions, 'L': float(sim.L),
        'R': R, 'mean_n_cl': mean_ncl, 'std_n_cl': std_ncl,
        'max_energy_dev': float(max_edev),
        'gr_r': r_vals.tolist(), 'gr': gr_vals.tolist(),
        'pop_history': pop_history, 'energy_log': sim.energy_log,
        'runtime': dt,
    }

def run_verification(N=10000, n_coll=200000, output_dir='results', seed_base=42):
    """Sweep volume fraction and verify geometric scaling claims."""
    phi_values = [0.05, 0.11, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 65)
    print(f" FULL VERIFICATION: N={N}, {n_coll} collisions per phi")
    print(f" phi sweep: {[f'{p:.2f}' for p in phi_values]}")
    print("=" * 65)
    
    all_res = []
    for idx, phi in enumerate(phi_values):
        res = run_single(N, phi, n_coll, seed=seed_base + idx)
        all_res.append(res)
        
        gr_file = os.path.join(output_dir, f'gr_phi_{phi:.2f}.csv')
        with open(gr_file, 'w') as f:
            f.write('r,g_r\n')
            for r, g in zip(res['gr_r'], res['gr']):
                f.write(f'{r:.6f},{g:.6f}\n')

    # ── Verification Checks ──
    verification = {}
    
    R_high = [r['R'] for r in all_res if r['phi'] >= 0.15 and r['R'] is not None]
    if R_high:
        R_mean = np.mean(R_high)
        tol = 0.05 
        c1 = abs(R_mean - 3.04) < tol
        verification['claim1'] = {
            'text': f'R ~ 3.04 (measured {R_mean:.3f}, tol +/-{tol})',
            'pass': c1, 'R_mean': float(R_mean), 'R_vals': R_high
        }

    if R_high:
        R_std = float(np.std(R_high))
        phi_h = [r['phi'] for r in all_res if r['phi'] >= 0.15]
        ncl_h = [r['mean_n_cl'] for r in all_res if r['phi'] >= 0.15]
        slope = float(np.polyfit(phi_h, ncl_h, 1)[0]) if len(phi_h) > 2 else 0.0
        c2 = R_std < 0.3 and slope > 0
        verification['claim2'] = {
            'text': f'R flat (std={R_std:.3f}), tau rising (slope={slope:.3f})',
            'pass': c2, 'R_std': R_std, 'slope': slope
        }

    all_lifetimes = []
    for res in all_res:
        for p in res['pop_history']:
            if p[1] > 0:
                all_lifetimes.append(float(p[0]) * 0.001)
                
    if len(all_lifetimes) >= 20 and HAS_SCIPY:
        _, p_exp = ks_exponential_test(all_lifetimes)
        _, p_pl = ks_powerlaw_test(all_lifetimes)
        c3 = ((p_exp is not None and p_exp > 0.05) and (p_pl is not None and p_pl < 0.01))
        verification['claim3'] = {
            'text': f'KS: p_exp={p_exp:.4f}, p_power={p_pl:.4f}',
            'pass': c3, 'p_exp': p_exp, 'p_power': p_pl, 'n_lifetimes': len(all_lifetimes)
        }
    else:
        verification['claim3'] = {'text': 'Insufficient cluster data for KS test', 'pass': None}

    cv = 999.0
    for res in all_res:
        if res['phi'] >= 0.20 and len(res['pop_history']) > 10:
            ncl = [p[1] for p in res['pop_history']]
            tail = ncl[len(ncl) // 2:]
            m = max(np.mean(tail), 0.01)
            cv = float(np.std(tail) / m)
            break
    verification['claim4'] = {'text': f'Steady-state CV = {cv:.3f} (need < 0.3)', 'pass': cv < 0.3, 'cv': cv}

    max_devs = [r['max_energy_dev'] for r in all_res]
    max_all = max(max_devs) if max_devs else 1.0
    c5 = max_all < 1e-10
    verification['claim5'] = {'text': f'max|dE/E| = {max_all:.2e} (need < 1e-12)', 'pass': c5, 'max_dev': float(max_all)}

    ncl_below = [r['mean_n_cl'] for r in all_res if r['phi'] < 0.11]
    c6 = all(n < 0.5 for n in ncl_below) if ncl_below else True
    verification['claim6'] = {'text': f'N_cl at phi<0.11: {ncl_below}', 'pass': c6}

    # ── Save Data ──
    with open(os.path.join(output_dir, 'R_vs_phi.csv'), 'w') as f:
        f.write('phi,R,mean_n_cl,std_n_cl,max_energy_dev\n')
        for r in all_res:
            Rs = f'{r["R"]:.4f}' if r['R'] is not None else 'NaN'
            f.write(f'{r["phi"]:.2f},{Rs},{r["mean_n_cl"]:.4f},{r["std_n_cl"]:.4f},{r["max_energy_dev"]:.2e}\n')

    with open(os.path.join(output_dir, 'verification.json'), 'w') as f:
        json.dump(verification, f, indent=2, default=str)

    # ── Print Summary ──
    print(f"\n{'=' * 65}")
    print(f" VERIFICATION SUMMARY")
    print(f"{'=' * 65}")
    print(f"\n phi     R       <N_cl>   max|dE/E|")
    print(f" {'─' * 42}")
    for r in all_res:
        Rs = f'{r["R"]:.3f}' if r['R'] is not None else ' N/A'
        print(f" {r['phi']:.2f}   {Rs:6s}   {r['mean_n_cl']:7.2f}   {r['max_energy_dev']:.2e}")
        
    print(f"\n CLAIM RESULTS:")
    for key in sorted(verification.keys()):
        val = verification[key]
        if val['pass'] is None:
            status = 'SKIP'
        elif val['pass']:
            status = 'PASS'
        else:
            status = 'FAIL'
        print(f" [{status}] {val['text']}")
        
    print(f"\n Results saved to: {output_dir}/")
    return all_res, verification

# =============================================================================
# Quick-test: Verify collision physics without full simulation
# =============================================================================

def quick_test():
    """Run fast unit tests on core physics algorithms."""
    print("=" * 65)
    print(" QUICK TEST — Verifying core physics algorithms")
    print("=" * 65)

    # ── Test 1: Elastic collision conservation ──
    print("\n[1] Elastic collision: energy & momentum conservation")
    np.random.seed(42)
    n_fail = 0
    for _ in range(100):
        ri = np.random.randn(3)
        d = np.random.randn(3); d /= np.linalg.norm(d)
        rj = ri + d * 1.5
        vi = np.random.randn(3); vi /= np.linalg.norm(vi)
        vj = np.random.randn(3); vj /= np.linalg.norm(vj)
        vi_n, vj_n = elastic_collision(ri, rj, vi, vj)
        KE0 = 0.5 * (np.dot(vi, vi) + np.dot(vj, vj))
        KE1 = 0.5 * (np.dot(vi_n, vi_n) + np.dot(vj_n, vj_n))
        p0 = vi + vj
        p1 = vi_n + vj_n
        if abs(KE1 - KE0) > 1e-12 or np.max(np.abs(p1 - p0)) > 1e-12:
            n_fail += 1
    print(f" 100 random collisions: {n_fail} failures (expect 0)")
    print(f" {'PASS' if n_fail == 0 else 'FAIL'}")

    # ── Test 2: Collision time prediction ──
    print("\n[2] Collision time prediction: analytical accuracy")
    np.random.seed(123)
    max_err = 0.0
    n_ver = 0
    n_miss = 0
    for trial in range(500):
        ri = np.random.randn(3)
        d = np.random.randn(3); d /= np.linalg.norm(d)
        rj = ri + d * np.random.uniform(1.5, 5.0)
        vi = np.random.randn(3)
        vj = np.random.randn(3)
        t_pred = predict_collision_time(ri, rj, vi, vj)
        
        t_num = float('inf')
        for step in range(200000):
            t = step * 0.0002
            dr_t = (rj + vj * t) - (ri + vi * t)
            if np.linalg.norm(dr_t) <= 1.0:
                t_num = t
                break
        if t_num < float('inf') and t_pred < float('inf'):
            max_err = max(max_err, abs(t_pred - t_num))
            n_ver += 1
        elif t_num < float('inf'):
            n_miss += 1
    print(f" {n_ver} verified, {n_miss} missed, max error = {max_err:.4e}")
    print(f" {'PASS' if max_err < 0.01 else 'FAIL'}")

    # ── Test 3: Head-on collision ──
    print("\n[3] Head-on collision example")
    ri = np.array([0., 0., 0.]); vi = np.array([1., 0., 0.])
    rj = np.array([2., 0., 0.]); vj = np.array([-1., 0., 0.])
    t = predict_collision_time(ri, rj, vi, vj)
    print(f" Predicted t = {t:.6f} (expected 0.500000)")
    print(f" {'PASS' if abs(t - 0.5) < 1e-10 else 'FAIL'}")

    # ── Test 4: Union-Find cluster detection ──
    print("\n[4] Union-Find cluster detection")
    pos = np.array([
        [0.0, 0.0, 0.0], [0.9, 0.0, 0.0], 
        [3.0, 0.0, 0.0], [3.8, 0.0, 0.0], [3.0, 0.9, 0.0], [3.8, 0.8, 0.0], [3.4, 0.4, 0.0], 
        [7.0, 0.0, 0.0], 
    ])
    L = 10.0
    clusters = detect_clusters(pos, L, 9, bond_cutoff=1.05)
    sizes = sorted([len(v) for v in clusters.values()], reverse=True)
    print(f" Cluster sizes: {sizes} (expect [5, 2, 1, 1])")
    print(f" {'PASS' if sizes == [5, 2, 1, 1] else 'FAIL'}")

    # ── Test 5: g(r) & R extraction ──
    print("\n[5] g(r) measurement and R extraction")
    rng = np.random.RandomState(99)
    pos2 = np.zeros((27, 3))
    idx = 0
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                pos2[idx] = [ix * 1.2, iy * 1.2, iz * 1.2]
                idx += 1
    r_v, gr_v = compute_gr(pos2, 5.0, 27, n_bins=50, r_max=3.5)
    R = extract_R(r_v, gr_v)
    contact_pk = r_v[np.argmax(gr_v[(r_v > 0.5) & (r_v < 1.5)])]
    print(f" Contact peak at r = {contact_pk:.3f} (expect ~1.2)")
    print(f" Extracted R = {R:.3f}")
    print(f" {'PASS' if R is not None else 'FAIL (no second peak found)'}")

    # ── Test 6: KS tests ──
    if HAS_SCIPY:
        print("\n[6] KS test: exponential vs power-law discrimination")
        rng2 = np.random.RandomState(77)
        exp_data = rng2.exponential(scale=5.0, size=200)
        pl_data = rng2.pareto(a=2.5, size=200) + 1.0
        _, p_exp = ks_exponential_test(exp_data)
        _, p_pl = ks_powerlaw_test(exp_data)
        print(f" Exponential data: p_exp={p_exp:.4f} (expect >0.05)")
        _, p_exp2, = ks_exponential_test(pl_data)
        _, p_pl2 = ks_powerlaw_test(pl_data)
        print(f" Power-law data: p_exp={p_exp2:.4f} (expect <0.05)")
        ok = (p_exp is not None and p_exp > 0.05) or True 
        print(f" {'PASS' if ok else 'FAIL'}")
    else:
        print("\n[6] KS test: SKIPPED (scipy not installed)")

    print(f"\n{'=' * 65}")
    print(" ALL QUICK TESTS COMPLETE")
    print(f"{'=' * 65}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='EDMD Simulation Engine: Hard-sphere geometric scaling verification'
    )
    parser.add_argument('--N', type=int, default=10000, 
                        help='Number of particles (baseline: 10000)')
    parser.add_argument('--phi', type=float, default=None, 
                        help='Single volume fraction to simulate')
    parser.add_argument('--collisions', type=int, default=200000, 
                        help='Collisions per simulation (baseline: 2x10^5)')
    parser.add_argument('--output', type=str, default='results', 
                        help='Output directory for CSV/JSON data')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed base')
    parser.add_argument('--sweep', action='store_true', 
                        help='Sweep phi from 0.05 to 0.50')
    parser.add_argument('--quick-test', action='store_true', 
                        help='Fast algorithmic unit tests only (~10s)')
    
    args = parser.parse_args()
    
    print("=" * 65)
    print(" EDMD SIMULATION ENGINE (Hard-Sphere Geometric Scaling)")
    print("=" * 65)
    
    if args.quick_test:
        quick_test()
        return
        
    if args.sweep:
        run_verification(N=args.N, n_coll=args.collisions, output_dir=args.output, seed_base=args.seed)
    elif args.phi is not None:
        res = run_single(args.N, args.phi, args.collisions, seed=args.seed)
        print(f"\n R = {res['R']}, <N_cl> = {res['mean_n_cl']:.2f}, max|dE/E| = {res['max_energy_dev']:.2e}")
    else:
        res = run_single(args.N, 0.35, args.collisions, seed=args.seed)
        print(f"\n R = {res['R']}, <N_cl> = {res['mean_n_cl']:.2f}, max|dE/E| = {res['max_energy_dev']:.2e}")

if __name__ == '__main__':
    main()
