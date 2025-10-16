#!/usr/bin/env python3
"""
First-principles DM/Baryon yields from bounce-sourced distributed-LZ transport
==============================================================================

Fast path (default for SFV/nonthermal):
  If deplete_DM_from_source=False, sigma_v=0, Gamma_wash=0  ->  use DIRECT QUADRATURE:
    Y_B = ∫_{T_lo}^{T_hi} [ S_B(T) / ( s(T) H(T) T ) ] dT,
    with S_B(T) = <P> * J_chi(T) * [A/V](y(T)) * window(y).
  This avoids ODE stiffness/hangs and is linear in incident_flux_scale.

General path (fallback):
  Radau ODE with spline-tabled A/V(T) and capped steps.

Conventions:
  - y(T) = (β/H)_p / 2 * [ (T_p/T)^2 - 1 ]   (closed form for RD with const g*).
  - [A/V](y) uses the KJMA formula with a stable integral over z.
"""
from __future__ import annotations

import argparse, json, math
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

# -------------------------
# Constants & conversions
# -------------------------
ZETA3 = 1.202056903159594
PI = math.pi
MPL_GEV = 1.220890e19     # H = 1.66 sqrt(g*) T^2 / M_Pl  (GeV)
S0_CM3 = 2891.0
S0_M3 = S0_CM3 * 1e6       # m^-3
GEV_TO_KG = 1.78266192e-27
M_PROTON_KG = 1.67262192369e-27

# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    # Microphysics / DM
    m_chi_GeV: float = 0.95
    g_chi: int = 2
    chi_stats: str = "fermion"          # "fermion" or "boson"
    regime: str = "nonthermal"          # "thermal", "nonthermal", or "auto"
    sigma_v_chi_GeV_m2: float = 0.0     # ~2.6e-9 for thermal relic

    # Transition / percolation inputs
    T_p_GeV: float = 100.0
    beta_over_H: float = 100.0
    v_w: float = 0.30
    I_p: float = 0.34                   # I(t_p)

    # g* (assumed ~const over narrow window)
    g_star: float = 106.75
    g_star_s: float = 106.75

    # Source normalization / shape
    P_chi_to_B: Optional[float] = None  # if None, try profile auto-compute
    source_shape_sigma_y: float = 15.0  # broader so the pulse is easy to resolve
    Gamma_wash_over_H: float = 0.0

    # SFV-resident χ: scale incident flux onto wall
    incident_flux_scale: float = 1.0
    # If True, subtract S_B from Y_chi; if χ is external to brane, keep False.
    deplete_DM_from_source: bool = False

    # Integration window
    T_max_over_Tp: float = 5.0
    T_min_over_Tp: float = 1e-3

    # If nonthermal
    Y_chi_init: Optional[float] = 4.90e-10
    n_chi_at_Tp_GeV3: Optional[float] = None

# -------------------------
# Thermo/cosmo helpers
# -------------------------
def H_std(T: float, g_star: float) -> float:
    return 1.66 * math.sqrt(g_star) * T * T / MPL_GEV

def s_entropy(T: float, g_star_s: float) -> float:
    return (2.0 * PI**2 / 45.0) * g_star_s * T**3

def n_chi_eq(T, m, g, stats):
    """Equilibrium number density n(T) in GeV^3 (natural units). Vectorized."""
    T_arr = np.asarray(T, dtype=float)
    out = np.empty_like(T_arr, dtype=float)

    rel = T_arr > (m / 3.0)
    if str(stats).lower().startswith("ferm"):
        c_rel = g * (3.0 * ZETA3 / (4.0 * PI**2))
    else:
        c_rel = g * (ZETA3 / (PI**2))
    out[rel] = c_rel * T_arr[rel]**3

    nr = ~rel
    if np.any(nr):
        coeff = g * (m / (2.0 * PI))**1.5
        out[nr] = coeff * (T_arr[nr]**1.5) * np.exp(-m / np.maximum(T_arr[nr], 1e-30))

    return float(out) if out.ndim == 0 else out

def vbar_chi(T, m):
    """Mean speed. Vectorized. Relativistic if T > m/3 -> v ≈ 1; else <v>=sqrt(8T/(π m))."""
    T_arr = np.asarray(T, dtype=float)
    out = np.empty_like(T_arr, dtype=float)
    rel = T_arr > (m / 3.0)
    out[rel] = 1.0
    nr = ~rel
    if np.any(nr):
        val = 8.0 * T_arr[nr] / (PI * max(m, 1e-20))
        out[nr] = np.sqrt(np.maximum(val, 0.0))
    # return scalar if scalar input
    return float(out) if out.ndim == 0 else out

def J_chi_flux(T: float, m: float, g: int, stats: str) -> float:
    return 0.25 * n_chi_eq(T, m, g, stats) * vbar_chi(T, m)

# y(T) closed form (RD, const g*), and its inverse T(y)
def y_of_T(T: float, T_p: float, beta_over_H: float) -> float:
    B = beta_over_H
    return 0.5 * B * ((T_p / max(T, 1e-30))**2 - 1.0)

def T_of_y(y: float, T_p: float, beta_over_H: float) -> float:
    B = beta_over_H
    denom = 1.0 + 2.0 * y / max(B, 1e-30)
    if denom <= 1e-12:  # outside sensible range; return a large T (integrand will be tiny)
        return T_p * 1e6
    return T_p / math.sqrt(denom)

# -------------------------
# KJMA A/V(y): stable implementation
# -------------------------
class AoverVKernel:
    def __init__(self, I_p: float, beta_over_H: float, T_p: float, v_w: float, g_star: float,
                 z_max: float = 30.0, nz: int = 1200):
        self.I_p = I_p
        self.beta_over_H = beta_over_H
        self.T_p = T_p
        self.v_w = max(v_w, 1e-12)
        self.g_star = g_star

        # Precompute constants
        self.H_p = H_std(T_p, g_star)
        self.beta = self.beta_over_H * self.H_p

        # Precompute z-grid and gamma4(z)
        self.z = np.linspace(0.0, z_max, nz)
        ez = np.exp(-self.z)
        self.g4 = 6.0 - ez * (self.z**3 + 3.0*self.z**2 + 6.0*self.z + 6.0)

    def A_over_V_y(self, y: float) -> float:
        if y > 50.0:
            return 0.0
        expy = math.exp(max(min(y, 50.0), -50.0))
        pref = (self.I_p / 2.0) * (self.beta / self.v_w) * expy
        integrand = self.z**2 * np.exp(-self.z) * np.exp(-(self.I_p / 6.0) * expy * self.g4)
        F = np.trapezoid(integrand, self.z)
        return float(pref * F)

# -------------------------
# (Optional) LZ glue
# -------------------------
def try_compute_P_from_profile(profile_csv_path: str, v_w: float) -> Optional[float]:
    try:
        import importlib
        for modname in ["lambda_local_LZ_from_profile","extended_LZ_lambda","transport_from_profile"]:
            try:
                mod = importlib.import_module(modname)
            except Exception:
                continue
            if hasattr(mod, "compute_prob_from_profile"):
                P = mod.compute_prob_from_profile(profile_csv_path, v_w)
                return float(max(min(P, 1.0), 0.0))
            if hasattr(mod, "compute_lambda_eff_from_profile"):
                lam_eff = mod.compute_lambda_eff_from_profile(profile_csv_path)
                P = 1.0 - math.exp(-2.0 * PI * max(lam_eff, 0.0))
                return float(max(min(P, 1.0), 0.0))
        return None
    except Exception:
        return None

# -------------------------
# Boltzmann system
# -------------------------
class BoltzmannSystem:
    def __init__(self, cfg: Config, P_chi_to_B: float):
        self.cfg = cfg
        self.P = float(P_chi_to_B)
        self.m = float(cfg.m_chi_GeV)
        self.aov = AoverVKernel(cfg.I_p, cfg.beta_over_H, cfg.T_p_GeV, cfg.v_w, cfg.g_star)

        # ODE fallback tables
        self._A_spline: Optional[CubicSpline] = None
        self._T_lo = None; self._T_hi = None

    # Cosmology/thermo wrappers
    def H(self, T: float) -> float: return H_std(T, self.cfg.g_star)
    def s(self, T: float) -> float: return s_entropy(T, self.cfg.g_star_s)

    # ---- fast A/V(T) for ODE fallback ----
    def build_tables(self, T_lo: float, T_hi: float, n: int = 800):
        self._T_lo, self._T_hi = float(T_lo), float(T_hi)
        Ts = np.linspace(self._T_lo, self._T_hi, n)
        Av = np.array([self.aov.A_over_V_y(y_of_T(T, self.cfg.T_p_GeV, self.cfg.beta_over_H)) for T in Ts], float)
        self._A_spline = CubicSpline(Ts, np.maximum(Av, 0.0), extrapolate=True)

    def A_over_V_T(self, T: float) -> float:
        if self._A_spline is not None:
            Tq = min(max(T, self._T_lo), self._T_hi)
            return float(self._A_spline(Tq))
        y = y_of_T(T, self.cfg.T_p_GeV, self.cfg.beta_over_H)
        return self.aov.A_over_V_y(y)

    # Flux and source
    def J_chi(self, T: float) -> float:
        return self.cfg.incident_flux_scale * J_chi_flux(T, self.m, self.cfg.g_chi, self.cfg.chi_stats)

    def S_B_T(self, T: float) -> float:
        y = y_of_T(T, self.cfg.T_p_GeV, self.cfg.beta_over_H)
        window = math.exp(-0.5 * (y / max(self.cfg.source_shape_sigma_y, 1e-6)) ** 2)
        return self.P * self.J_chi(T) * self.aov.A_over_V_y(y) * window

    # ------------- DIRECT QUADRATURE PATH -------------
    def integrate_YB_by_quadrature(self, T_lo: float, T_hi: float, n_y: int = 6000) -> float:
        """Integrate Y_B = ∫ S_B(T)/(s H T) dT using a y-grid limited to the support of the kernel."""
        # Raw bounds in y from the requested T-range
        y_lo_raw = y_of_T(T_hi, self.cfg.T_p_GeV, self.cfg.beta_over_H)  # (T_hi -> smaller y)
        y_hi_raw = y_of_T(T_lo, self.cfg.T_p_GeV, self.cfg.beta_over_H)

        # Restrict to the physical support (A/V_y ≈ 0 outside ~[-O(60), +50])
        Y_NEG_CUT = -80.0
        Y_POS_CUT = +50.0   # we hard-zero A/V for y>50 anyway
        y_lo = max(y_lo_raw, Y_NEG_CUT)
        y_hi = min(y_hi_raw, Y_POS_CUT)
        if y_hi <= y_lo:
            return 0.0

        # Build a uniform grid in y over the *useful* interval
        n_y = max(int(n_y), 2000)
        ys = np.linspace(y_lo, y_hi, n_y)

        # Map y -> T and Jacobian dT/dy
        B  = self.cfg.beta_over_H
        Tp = self.cfg.T_p_GeV
        denom = 1.0 + 2.0 * ys / max(B, 1e-30)
        denom = np.maximum(denom, 1e-12)
        Ts    = Tp / np.sqrt(denom)                    # T(y)
        dTdy  = - (Tp / max(B, 1e-30)) * denom**(-1.5) # dT/dy

        # Ingredients
        Hs = H_std(Ts, self.cfg.g_star)
        ss = s_entropy(Ts, self.cfg.g_star_s)
        Js = self.cfg.incident_flux_scale * 0.25 * n_chi_eq(Ts, self.m, self.cfg.g_chi, self.cfg.chi_stats) * vbar_chi(Ts, self.m)
        Av = np.array([self.aov.A_over_V_y(y) for y in ys], float)
        window = np.exp(-0.5 * (ys / max(self.cfg.source_shape_sigma_y, 1e-6))**2)

        SB = self.P * Js * Av * window
        integrand = SB / (ss * Hs * Ts) * np.abs(dTdy)

        return float(np.trapezoid(integrand, ys))

    # ------------- ODE fallback (only if needed) -------------
    def rhs(self, x: float, Y: ArrayLike) -> np.ndarray:
        Ychi, YB = float(Y[0]), float(Y[1])
        T = self.m / max(x, 1e-30)
        H = max(self.H(T), 1e-300)
        s = max(self.s(T), 1e-300)
        y = y_of_T(T, self.cfg.T_p_GeV, self.cfg.beta_over_H)
        window = math.exp(-0.5 * (y / max(self.cfg.source_shape_sigma_y, 1e-6)) ** 2)
        SB = self.P * self.J_chi(T) * self.A_over_V_T(T) * window

        sigmav = max(self.cfg.sigma_v_chi_GeV_m2, 0.0)
        Ychi_eq = n_chi_eq(T, self.m, self.cfg.g_chi, self.cfg.chi_stats) / s

        SB_term = (SB / s) if self.cfg.deplete_DM_from_source else 0.0
        dYchi_dx = (- sigmav * s * (Ychi**2 - Ychi_eq**2) - SB_term) / (H * x)
        gamma_w = max(self.cfg.Gamma_wash_over_H, 0.0)
        dYB_dx   = ( + SB / s - gamma_w * H * YB ) / (H * x)
        return np.array([dYchi_dx, dYB_dx], dtype=float)

# -------------------------
# Utilities
# -------------------------
def default_config() -> Dict:
    return {
        "m_chi_GeV": 0.95, "g_chi": 2, "chi_stats": "fermion", "regime": "nonthermal",
        "sigma_v_chi_GeV_m2": 0.0,
        "T_p_GeV": 100.0, "beta_over_H": 100.0, "v_w": 0.30, "I_p": 0.34,
        "g_star": 106.75, "g_star_s": 106.75,
        "P_chi_to_B": None, "source_shape_sigma_y": 15.0, "Gamma_wash_over_H": 0.0,
        "incident_flux_scale": 1.0, "deplete_DM_from_source": False,
        "T_max_over_Tp": 5.0, "T_min_over_Tp": 1.0e-3,
        "Y_chi_init": 4.90e-10, "n_chi_at_Tp_GeV3": None
    }

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    base = default_config(); base.update(raw)
    return Config(**base)

def write_template(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(default_config(), f, indent=2)
    print(f"Wrote template config to {path}")

# -------------------------
# (Optional) P from profile
# -------------------------
def maybe_P(cfg: Config, profile_csv: Optional[str]) -> float:
    P_used = cfg.P_chi_to_B
    if profile_csv:
        P_try = try_compute_P_from_profile(profile_csv, cfg.v_w)
        if P_try is not None:
            print(f"[info] Using P_chi_to_B from profile: {P_try:.6g}")
            P_used = P_try
        else:
            print("[warn] Could not compute P from profile automatically; falling back to config.")
    if P_used is None:
        raise RuntimeError("P_chi_to_B is not set and could not be computed from profile.")
    return float(P_used)

# -------------------------
# Diagnostics
# -------------------------
def print_diagnostics(bs: BoltzmannSystem, cfg: Config):
    print("\n# Diagnostics around percolation")
    Ts = np.geomspace(cfg.T_p_GeV * 0.5, cfg.T_p_GeV * 2.0, 21)
    print(" T/Tp      y(T)        A/V [GeV]         J_chi [GeV^3]      S_B [GeV^3]")
    for T in Ts:
        y = y_of_T(T, cfg.T_p_GeV, cfg.beta_over_H)
        aov = bs.aov.A_over_V_y(y); J = bs.J_chi(T)
        SB = bs.P * J * aov * math.exp(-0.5 * (y / max(cfg.source_shape_sigma_y, 1e-6)) ** 2)
        print(f"{T/cfg.T_p_GeV:7.3f}  {y:9.3f}  {aov:14.6e}  {J:16.6e}  {SB:14.6e}")

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="First-principles DM/Baryon yields from bounce-sourced transport")
    ap.add_argument("--config", required=False, help="Path to yields_config.json")
    ap.add_argument("--write-template", action="store_true", help="Write a template config and exit")
    ap.add_argument("--maybe-compute-P-from-profile", dest="profile_csv", default=None,
                    help="Try to compute P_chi_to_B from local LZ modules using this profile CSV.")
    ap.add_argument("--diagnostics", action="store_true",
                    help="Print a small table of y(T), A/V(T), J_chi(T), S_B(T) around T_p.")
    args = ap.parse_args()

    if args.write_template:
        write_template(args.config or "yields_config.json"); return
    if not args.config:
        print("ERROR: --config is required (or use --write-template)."); return

    cfg = load_config(args.config)
    P_used = maybe_P(cfg, args.profile_csv)

    bs = BoltzmannSystem(cfg, P_used)

    # Integration range
    T_p = cfg.T_p_GeV
    T_hi = cfg.T_max_over_Tp * T_p
    T_lo = cfg.T_min_over_Tp * T_p

    # Fast DIRECT QUADRATURE path?
    can_quad = (not cfg.deplete_DM_from_source) and (cfg.sigma_v_chi_GeV_m2 == 0.0) and (cfg.Gamma_wash_over_H == 0.0)
    if can_quad:
        YB_fin = bs.integrate_YB_by_quadrature(T_lo, T_hi, n_y=8000)
        # DM stays as initial (no depletion/annihilation in this mode)
        if cfg.regime.lower().startswith("therm"):
            Ychi_fin = n_chi_eq(T_hi, cfg.m_chi_GeV, cfg.g_chi, cfg.chi_stats) / s_entropy(T_hi, cfg.g_star_s)
        elif cfg.regime.lower().startswith("non"):
            if cfg.Y_chi_init is not None:
                Ychi_fin = float(cfg.Y_chi_init)
            elif cfg.n_chi_at_Tp_GeV3 is not None:
                Ychi_fin = float(cfg.n_chi_at_Tp_GeV3) / max(s_entropy(T_p, cfg.g_star_s), 1e-300)
            else:
                Ychi_fin = 1.0e-12
    else:
        # ODE fallback (rare for your setup)
        bs.build_tables(T_lo, T_hi, n=800)
        x0 = cfg.m_chi_GeV / T_hi
        x1 = cfg.m_chi_GeV / max(T_lo, 1e-30)
        if cfg.regime.lower().startswith("therm"):
            Ychi0 = n_chi_eq(T_hi, cfg.m_chi_GeV, cfg.g_chi, cfg.chi_stats) / s_entropy(T_hi, cfg.g_star_s)
        elif cfg.regime.lower().startswith("non"):
            if cfg.Y_chi_init is not None:
                Ychi0 = float(cfg.Y_chi_init)
            elif cfg.n_chi_at_Tp_GeV3 is not None:
                Ychi0 = float(cfg.n_chi_at_Tp_GeV3) / max(s_entropy(T_p, cfg.g_star_s), 1e-300)
            else:
                Ychi0 = 1.0e-12
        else:
            Ychi0 = n_chi_eq(T_hi, cfg.m_chi_GeV, cfg.g_chi, cfg.chi_stats) / s_entropy(T_hi, cfg.g_star_s)

        YB0 = 0.0
        def fun(x, y): return bs.rhs(x, y)
        x_p = cfg.m_chi_GeV / max(T_p, 1e-30)
        max_step = min(abs(x1 - x0) / 20000.0, x_p / 1000.0, 5e-4)
        sol = solve_ivp(fun, (x0, x1), np.array([Ychi0, YB0], float),
                        method="Radau", rtol=1e-8, atol=1e-12, max_step=max_step)
        if not sol.success:
            print("[warn] ODE solver reported failure:", sol.message)
        Ychi_fin = float(sol.y[0, -1]); YB_fin = float(sol.y[1, -1])

    # Today's densities
    nB0_m3  = YB_fin  * S0_M3
    nDM0_m3 = Ychi_fin * S0_M3
    rhoB0   = nB0_m3  * M_PROTON_KG
    rhoDM0  = nDM0_m3 * (cfg.m_chi_GeV * GEV_TO_KG)
    ratio   = rhoDM0 / max(rhoB0, 1e-300)

    print("\n=== Results (today) ===")
    print(f"rho_B^0   = {rhoB0:.3e} kg/m^3")
    print(f"rho_DM^0  = {rhoDM0:.3e} kg/m^3")
    print(f"DM/B ratio= {ratio:.6g}")
    with open("yields_out.json", "w", encoding="utf-8") as f:
        json.dump({"inputs": {**cfg.__dict__, "P_used": P_used},
                   "final": {"Y_B": YB_fin, "Y_chi": Ychi_fin,
                             "rho_B_kg_m3": rhoB0, "rho_DM_kg_m3": rhoDM0,
                             "DM_over_B": ratio}}, f, indent=2)
    print("Wrote yields_out.json")

    if args.diagnostics:
        print("\n# Diagnostics around percolation")
        Ts = np.geomspace(cfg.T_p_GeV * 0.5, cfg.T_p_GeV * 2.0, 21)
        print(" T/Tp      y(T)        A/V [GeV]         J_chi [GeV^3]      S_B [GeV^3]")
        for T in Ts:
            y = y_of_T(T, cfg.T_p_GeV, cfg.beta_over_H)
            aov = bs.aov.A_over_V_y(y); J = bs.J_chi(T)
            SB = bs.P * J * aov * math.exp(-0.5 * (y / max(cfg.source_shape_sigma_y, 1e-6)) ** 2)
            print(f"{T/cfg.T_p_GeV:7.3f}  {y:9.3f}  {aov:14.6e}  {J:16.6e}  {SB:14.6e}")

if __name__ == "__main__":
    main()
