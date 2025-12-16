from __future__ import annotations
from typing import Any, Dict

def process_variable_cost_usd_per_t(cfg: Dict[str, Any], route: str, total_s_pct: float) -> float:
    route = route.upper().strip()
    s = float(total_s_pct) / 100.0
    P = float(cfg["economic_parameters"]["power"]["P_usd_per_kwh"])
    pv = cfg["cost_parameters"]["process_variable_cost"]

    if route == "FLO":
        c = pv["FLO_coefficients"]
        C_FLO=float(c["C_FLO"]); D_FLO=float(c["D_FLO"]); E_FLO=float(c["E_FLO"])
        F_FLO=float(c["F_FLO"]); G_FLO=float(c["G_FLO"]); M_FLO=float(c["M_FLO"]); N_FLO=float(c["N_FLO"])
        pc1 = C_FLO*(s**4) + D_FLO*(s**3) + E_FLO*(s**2) + F_FLO*s + G_FLO
        pc2 = M_FLO*s + N_FLO
        return pc1 + pc2*P

    if route == "POX":
        c = pv["POX_coefficients"]
        C_POX=float(c["C_POX"]); D_POX=float(c["D_POX"]); E_POX=float(c["E_POX"])
        F_POX=float(c["F_POX"]); G_POX=float(c["G_POX"])
        K_POX=float(c["K_POX"]); L_POX=float(c["L_POX"]); M_POX=float(c["M_POX"]); N_POX=float(c["N_POX"])
        pc1 = C_POX*(s**4) + D_POX*(s**3) + E_POX*(s**2) + F_POX*s + G_POX
        pc2 = K_POX*(s**3) + L_POX*(s**2) + M_POX*s + N_POX
        return pc1 + pc2*P

    raise ValueError("route must be 'FLO' or 'POX'")

def total_cost_components_usd_per_t(cfg: Dict[str, Any], route: str, total_s_pct: float) -> Dict[str, float]:
    c = cfg["cost_parameters"]

    # Incremental mining cost per Excel: ((Waste+Adj) - (Ore+Adj)) * -1  -> (Waste - Ore)
    ore = float(c["mining"]["ore_mining_usd_per_t"])
    waste = float(c["mining"]["waste_mining_usd_per_t"])
    inc_mining = (waste - ore)

    rehandle = -float(c["rehandle"]["ore_rehandle_usd_per_t"])   # -2.26

    tsf_ore = float(c["tsf"]["tsf_ore_component_usd_per_t"])
    tsf_waste = float(c["tsf"]["tsf_waste_component_usd_per_t"])
    inc_tsf = -(tsf_ore - tsf_waste)  # -3.71

    pc_var = -process_variable_cost_usd_per_t(cfg, route=route, total_s_pct=total_s_pct)  # -14.76

    proc_fixed = -float(c["process_fixed_cost"]["process_fixed_usd_per_t"])  # -17.74
    limestone = -float(c["limestone"]["limestone_usd_per_t"])  # -2.83
    alloc = -float(c["other_process_allocation"]["allocation_usd_per_t"])  # +0.15
    ga = -float(c["g_and_a"]["g_and_a_usd_per_t"])  # -3.58
    csr = -float(c["csr"]["csr_usd_per_t"])  # -0.16
    closure = -float(c["closure"]["closure_usd_per_t"])  # -1.04
    sust = -float(c["sustaining_capital"]["ore_sustaining_usd_per_t"])  # -0.67

    sfix = float(c["s_pct_fixed_costs"]["coefficient"]) * float(c["s_pct_fixed_costs"]["oxidised_sulfur_terms"]["term1"] + c["s_pct_fixed_costs"]["oxidised_sulfur_terms"]["term2"] + c["s_pct_fixed_costs"]["oxidised_sulfur_terms"]["term3"])
    #sfix  = 0.0  # stays 0 given your current oxidised sulfur terms
    total = inc_mining + rehandle + inc_tsf + pc_var + proc_fixed + limestone + alloc + ga + csr + closure + sust + sfix

    return {
        "IncrementalMining": inc_mining,
        "Rehandle": rehandle,
        "IncrementalTSF": inc_tsf,
        "ProcessVariable": pc_var,
        "ProcessFixed": proc_fixed,
        "Limestone": limestone,
        "OtherProcessAllocation": alloc,
        "GandA": ga,
        "CSR": csr,
        "Closure": closure,
        "Sustaining": sust,
        "S_pct_Fixed": sfix,
        "TotalCost": total,
    }
