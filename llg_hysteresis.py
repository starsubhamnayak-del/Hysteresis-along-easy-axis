"""
LLG Equation - Magnetic Hysteresis Loop Simulation
====================================================
Simulates mx vs H_ext (hysteresis loop) along the easy axis (x).

Parameters:
  Hc = 0.01 mT  (coercive / anisotropy field)
  Hd = 0.5 T    (demagnetization field)
  alpha = 1.0   (Gilbert damping)
  gamma = 1.76e11 rad/(T·s)

Effective field model:
  B_eff_x = B_ext + Hc * mx   (net anisotropy = Hani - Hd = Hc along easy axis)
  B_eff_y = B_eff_z = 0

Sweep:
  Trace   : 0.03 T  → -0.03 T  (10 steps)
  Retrace : -0.3 T  →  0.3 T   (10 steps)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─────────────────────────────────────────────
# Physical parameters
# ─────────────────────────────────────────────
gamma = 1.76e11          # gyromagnetic ratio [rad / (T·s)]
alpha = 1.0              # Gilbert damping constant
Hc    = 0.01e-3          # coercive (anisotropy) field [T] = 0.01 mT
Hd    = 0.5              # demagnetization field [T]
# Net effective anisotropy along easy axis = Hani - Hd = Hc
# => Hani = Hd + Hc  (used only for physical bookkeeping)

T_RUN   = 5e-9           # LLG run time per field step [s] = 5 ns
N_EVAL  = 10_000         # time points per solve
SEED_EPS = 1e-3          # re-seed transverse components if they fall below this

# ─────────────────────────────────────────────
# LLG right-hand side
# ─────────────────────────────────────────────
def llg_rhs(t, m, B_ext):
    """
    Returns dm/dt for the LLG equation.

    dm/dt = -gamma/(1+alpha^2) * [m x B_eff  +  alpha * m x (m x B_eff)]

    B_eff_x = B_ext + Hc*mx   (easy-axis effective field)
    B_eff_y = B_eff_z = 0
    """
    mx, my, mz = m
    B_eff = np.array([B_ext + Hc * mx, 0.0, 0.0])

    mxB    = np.cross(m, B_eff)
    mxmxB  = np.cross(m, mxB)

    return -gamma / (1.0 + alpha**2) * (mxB + alpha * mxmxB)


# ─────────────────────────────────────────────
# Single-step LLG solver
# ─────────────────────────────────────────────
def run_step(m_in, B_ext, t_run=T_RUN):
    """
    Integrate the LLG equation for t_run seconds starting from m_in.

    Re-seeds small transverse components so a misaligned initial condition
    is always present (mimics thermal fluctuations that allow switching).

    Returns the final normalised magnetisation vector.
    """
    m = m_in.copy()

    # Re-seed if my, mz are too small to drive precession
    if abs(m[1]) < 1e-6 and abs(m[2]) < 1e-6:
        m[1] = SEED_EPS
        m[2] = SEED_EPS
        m /= np.linalg.norm(m)

    t_eval = np.linspace(0, t_run, N_EVAL)

    sol = solve_ivp(
        llg_rhs,
        (0, t_run),
        m,
        args=(B_ext,),
        method='RK45',
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )

    m_out = sol.y[:, -1]
    m_out /= np.linalg.norm(m_out)   # renormalise (removes small drift)
    return m_out


# ─────────────────────────────────────────────
# Field sweeps
# ─────────────────────────────────────────────
# Initial magnetisation: almost along +x with 0.1% tilt in y and z
eps = 0.001
m0  = np.array([1.0, eps, eps])
m0 /= np.linalg.norm(m0)

trace_fields   = np.linspace( 0.03, -0.03, 10)   # 0.03 T → -0.03 T
retrace_fields = np.linspace(-0.30,  0.30, 10)   # -0.3 T →  0.3 T

m_current = m0.copy()

trace_mx   = []
retrace_mx = []

print("=" * 55)
print("  LLG Hysteresis Simulation")
print(f"  gamma = {gamma:.3e} rad/(T·s)")
print(f"  alpha = {alpha}")
print(f"  Hc    = {Hc*1e3:.4f} mT")
print(f"  Hd    = {Hd:.2f} T")
print(f"  t_run = {T_RUN*1e9:.0f} ns per step")
print("=" * 55)

print("\n--- Trace: 0.03 T  →  -0.03 T ---")
for B_ext in trace_fields:
    m_current = run_step(m_current, B_ext)
    trace_mx.append(m_current[0])
    print(f"  H_ext = {B_ext*1e3:+8.3f} mT   mx = {m_current[0]:+.6f}")

print("\n--- Retrace: -0.3 T  →  0.3 T ---")
for B_ext in retrace_fields:
    m_current = run_step(m_current, B_ext)
    retrace_mx.append(m_current[0])
    print(f"  H_ext = {B_ext*1e3:+8.3f} mT   mx = {m_current[0]:+.6f}")


# ─────────────────────────────────────────────
# Plot hysteresis loop (Fig. 5)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(trace_fields   * 1e3, trace_mx,
        'o-', color='#2a6fbd', lw=2, ms=7,
        label='Trace (0.03 → −0.03 T)')

ax.plot(retrace_fields * 1e3, retrace_mx,
        's--', color='#c94a1e', lw=2, ms=7,
        label='Retrace (−0.3 → 0.3 T)')

# Annotations
ax.axhline(0,  color='gray', lw=0.6, ls=':')
ax.axvline(0,  color='gray', lw=0.6, ls=':')
ax.axhline( 1, color='gray', lw=0.5, ls='--', alpha=0.4)
ax.axhline(-1, color='gray', lw=0.5, ls='--', alpha=0.4)

ax.set_xlabel(r'$H_\mathrm{ext}$  (mT)', fontsize=13)
ax.set_ylabel(r'$m_x$  (normalized)', fontsize=13)
ax.set_title('Fig. 5 — Magnetic Hysteresis Loop (LLG simulation)\n'
             r'Easy axis: $\hat{x}$,  $H_c = 0.01\,\mathrm{mT}$,  '
             r'$H_d = 0.5\,\mathrm{T}$,  $\alpha = 1.0$',
             fontsize=11)
ax.set_ylim(-1.25, 1.25)
ax.legend(fontsize=11, loc='center right')
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_major_locator(ticker.AutoLocator())
ax.tick_params(labelsize=11)
ax.grid(True, alpha=0.3)

# Annotate switching events
switch_t_idx = next((i for i, v in enumerate(trace_mx) if v < 0), None)
if switch_t_idx is not None:
    ax.annotate('Switching\n(trace)',
                xy=(trace_fields[switch_t_idx]*1e3, trace_mx[switch_t_idx]),
                xytext=(-120, 0.3),
                fontsize=9, color='#2a6fbd',
                arrowprops=dict(arrowstyle='->', color='#2a6fbd'))

switch_r_idx = next((i for i, v in enumerate(retrace_mx) if v > 0), None)
if switch_r_idx is not None:
    ax.annotate('Switching\n(retrace)',
                xy=(retrace_fields[switch_r_idx]*1e3, retrace_mx[switch_r_idx]),
                xytext=(80, -0.6),
                fontsize=9, color='#c94a1e',
                arrowprops=dict(arrowstyle='->', color='#c94a1e'))

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/llg_hysteresis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to llg_hysteresis.png")
