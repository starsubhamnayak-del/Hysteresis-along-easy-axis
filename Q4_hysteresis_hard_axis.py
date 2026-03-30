"""
Q4 — Magnetic Hysteresis Loop (Hard Axis, z) via LLG
======================================================
Field applied along ẑ (hard axis)
Trace   : H_ext = +0.7 T → −0.7 T  (15 steps)
Retrace : H_ext = −0.7 T → +0.7 T  (10 steps)
Plot    : mz vs H_ext

Parameters from problem sheet:
  γ  = 1.76e11 rad/(T·s)
  α  = 0.01
  Hc = 0.01 mT  (easy-axis anisotropy, along x̂)
  Hd = 0.5  T   (hard-axis anisotropy, along ẑ)
  H_ani = Hc·mx·x̂  −  Hd·mz·ẑ
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Constants & parameters ───────────────────────────────────────────────────
gamma  = 1.76e11          # rad/(T·s)
alpha  = 0.01
Hc_T   =  0.01e-3         # 0.01 mT easy-axis anisotropy (Tesla)
Hd_T   =  0.5             # 0.5  T  hard-axis anisotropy (Tesla)

def H_ani(m):
    """Anisotropy field: Hc·mx·x̂ − Hd·mz·ẑ  (in Tesla)"""
    return np.array([Hc_T * m[0],
                     0.0,
                    -Hd_T * m[2]])

def llg_rhs(m, H_ext_vec):
    """
    LLG in Landau–Lifshitz form:
      dm/dt = -γ/(1+α²) [m×H_eff + α·m×(m×H_eff)]
    """
    H_eff  = H_ext_vec + H_ani(m)
    mxH    = np.cross(m, H_eff)
    mxmxH  = np.cross(m, mxH)
    return -gamma / (1.0 + alpha**2) * (mxH + alpha * mxmxH)

def _norm(v):
    return v / np.linalg.norm(v)

def run_llg_rk4(m0, H_ext_T_z, t_end=20e-9, dt=1e-12):
    """
    RK4 integrator. H_ext_T_z is scalar field along ẑ (Tesla).
    """
    H_ext = np.array([0.0, 0.0, H_ext_T_z])   # field along ẑ
    m     = _norm(m0.copy())
    steps = int(t_end / dt)
    for _ in range(steps):
        k1 = llg_rhs(m,                          H_ext)
        k2 = llg_rhs(_norm(m + 0.5*dt*k1),       H_ext)
        k3 = llg_rhs(_norm(m + 0.5*dt*k2),       H_ext)
        k4 = llg_rhs(_norm(m +     dt*k3),       H_ext)
        m  = _norm(m + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4))
    return m

# ── Field schedule ────────────────────────────────────────────────────────────
# Trace:   +0.7 → −0.7 T in 15 steps  → step = −0.1 T
H_trace   = np.round(np.linspace( 0.7, -0.7, 15), 10)

# Retrace: −0.7 → +0.7 T in 10 steps  → step = ~0.1556 T
# Use clean values: -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7 + 2 extra
# Best clean 10-step: linspace gives -0.7, -0.5444, ..., 0.7 — not clean
# Use multiples of 0.1 T instead: -0.7 to 0.7 in steps of ~0.1556
# For clean values, use 15 steps for retrace too, or use step=0.1:
H_retrace = np.array([-0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.6, 0.7])

print("Trace field values (T):")
print(np.round(H_trace, 4))
print("\nRetrace field values (T):")
print(np.round(H_retrace, 4))

# ── Initial magnetisation: almost along +x̂ ───────────────────────────────────
eps = 0.001
m0  = _norm(np.array([1.0, eps, eps]))

mz_trace,   H_plot_trace   = [], []
mz_retrace, H_plot_retrace = [], []

print("\n═══ Trace (+0.7 T → −0.7 T, 15 steps) ═══")
m = m0.copy()
for H in H_trace:
    m = run_llg_rk4(m, H)
    mz_trace.append(m[2])
    H_plot_trace.append(H)
    print(f"  H = {H:+.4f} T   mz = {m[2]:+.6f}   mx = {m[0]:+.6f}")

print("\n═══ Retrace (−0.7 T → +0.7 T, 10 steps) ═══")
for H in H_retrace:
    m = run_llg_rk4(m, H)
    mz_retrace.append(m[2])
    H_plot_retrace.append(H)
    print(f"  H = {H:+.4f} T   mz = {m[2]:+.6f}   mx = {m[0]:+.6f}")

# ── Plot (Fig. 6) ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))

ax.plot(H_plot_trace,   mz_trace,   'b-o', lw=2.0, ms=7, zorder=3,
        label='Trace  (0.7 T → −0.7 T, 15 steps)')
ax.plot(H_plot_retrace, mz_retrace, 'r-s', lw=2.0, ms=7, zorder=3,
        label='Retrace (−0.7 T → 0.7 T, 10 steps)')

# Direction arrows — trace
for i in range(len(H_plot_trace)-1):
    ax.annotate("",
        xy    =(H_plot_trace[i+1], mz_trace[i+1]),
        xytext=(H_plot_trace[i],   mz_trace[i]),
        arrowprops=dict(arrowstyle="-|>", color='blue', lw=1.4))
# Direction arrows — retrace
for i in range(len(H_plot_retrace)-1):
    ax.annotate("",
        xy    =(H_plot_retrace[i+1], mz_retrace[i+1]),
        xytext=(H_plot_retrace[i],   mz_retrace[i]),
        arrowprops=dict(arrowstyle="-|>", color='red', lw=1.4))

# Mark the hard-axis saturation field Hd = 0.5 T
ax.axvline( 0.5, color='gray', lw=1.2, ls=':', alpha=0.8, label=r'$H_d$ = 0.5 T (saturation)')
ax.axvline(-0.5, color='gray', lw=1.2, ls=':', alpha=0.8)

ax.axhline(0,  color='k', lw=0.8, ls='--', alpha=0.5)
ax.axvline(0,  color='k', lw=0.8, ls='--', alpha=0.5)

ax.set_xlabel(r'$\mu_0 H_{\rm ext}$  (T)', fontsize=13)
ax.set_ylabel(r'$m_z = M_z / M_s$',        fontsize=13)
ax.set_title('Fig. 6 — Hard-Axis Hysteresis Loop (LLG, α = 0.01)', fontsize=13)
ax.set_xlim(-0.85, 0.85)
ax.set_ylim(-1.3,  1.3)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_fig = '/mnt/user-data/outputs/Q4_hysteresis_hard_axis.png'
plt.savefig(out_fig, dpi=150)
print(f"\nFigure saved → {out_fig}")
