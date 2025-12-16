from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

from .models_bcf import au_recovery_flotation_chain_fraction
from .costs import total_cost_components_usd_per_t


@dataclass(frozen=True)
class BlockInputs:
    au_gt: float
    ag_gt: float = 0.0
    cu_pct: float = 0.0
    s_total_pct: float = 0.0
    s2_pct: float = 0.0
    dilution_pct: float = 0.0


@dataclass(frozen=True)
class BlockOutputs:
    au_gt_diluted: float
    mp_mass_pull_pct: float
    recF_pct: float
    au_concentrate_gt: float
    au_recovery_fraction: float
    revenue_usd_per_t: float
    total_cost_usd_per_t: float
    block_value_usd_per_t: float
    cost_breakdown: Dict[str, float]
    route_used: str  # auditable


def dilute(grade: float, dilution_pct: float) -> float:
    return float(grade) / (1.0 + float(dilution_pct) / 100.0)


def revenue_au_reserve_usd_per_t(cfg: Dict[str, Any], au_gt_dil: float, au_recovery_fraction: float) -> float:
    econ = cfg["economic_parameters"]
    AUP1 = float(econ["metal_prices"]["AUP1_usd_per_oz_reserve"])
    RAU1 = float(econ["refining_transport_costs"]["RAU1_usd_per_oz"])
    RYT = float(econ["royalties"]["RYT_fraction_of_net_price"])
    Gr_Oz = float(econ["unit_conversions"]["Gr_Oz_g_per_oz"])

    net_price = (AUP1 - RAU1)
    net_after_royalty = net_price - (net_price * RYT)
    return (au_gt_dil / Gr_Oz) * au_recovery_fraction * net_after_royalty


def _blend_costs(costs_a: Dict[str, float], costs_b: Dict[str, float], w_a: float) -> Dict[str, float]:
    """
    Linear blend of two cost breakdown dicts by matching keys.
    Assumes both dicts share the same keys (including 'TotalCost').
    """
    w_a = float(w_a)
    w_b = 1.0 - w_a
    keys = set(costs_a.keys()) | set(costs_b.keys())
    out: Dict[str, float] = {}
    for k in keys:
        va = float(costs_a.get(k, 0.0))
        vb = float(costs_b.get(k, 0.0))
        out[k] = w_a * va + w_b * vb
    return out


def compute_block_value(
    cfg: Dict[str, Any],
    rocktype_code: str,
    block: BlockInputs,
    route: str = "FLO",
    combined_flo_fraction: float = 0.60,  # 60% FLO / 40% POX
) -> BlockOutputs:
    """
    Computes block value for a given route:
      - FLO: uses aurec_flo_fraction + FLO cost stack
      - POX: uses aurec_pox_fraction + POX cost stack
      - COMBINED: blends FLO and POX (default 60% FLO / 40% POX)
        * Recovery: linear blend of route recoveries
        * Costs: linear blend of route cost breakdowns
    """
    route_norm = str(route).upper().strip()
    if route_norm not in {"FLO", "POX", "COMBINED"}:
        raise ValueError("route must be 'FLO', 'POX', or 'COMBINED'")

    au_dil = dilute(block.au_gt, block.dilution_pct)

    # Compute chain outputs (provides both aurec_flo_fraction and aurec_pox_fraction)
    rec = au_recovery_flotation_chain_fraction(
        cfg,
        rocktype_code=rocktype_code,
        au_gt_diluted=au_dil,
        s2_pct=block.s2_pct,
    )

    if route_norm == "FLO":
        au_rec = rec.aurec_flo_fraction
        costs = total_cost_components_usd_per_t(cfg, route="FLO", total_s_pct=block.s_total_pct)

    elif route_norm == "POX":
        au_rec = rec.aurec_pox_fraction
        costs = total_cost_components_usd_per_t(cfg, route="POX", total_s_pct=block.s_total_pct)

    else:
        w_flo = float(combined_flo_fraction)
        if not (0.0 <= w_flo <= 1.0):
            raise ValueError("combined_flo_fraction must be between 0 and 1")

        # Linear blend of recovery and costs
        au_rec = w_flo * rec.aurec_flo_fraction + (1.0 - w_flo) * rec.aurec_pox_fraction

        costs_flo = total_cost_components_usd_per_t(cfg, route="FLO", total_s_pct=block.s_total_pct)
        costs_pox = total_cost_components_usd_per_t(cfg, route="POX", total_s_pct=block.s_total_pct)
        costs = _blend_costs(costs_flo, costs_pox, w_a=w_flo)

        # Make the blended route auditable in outputs
        route_norm = f"COMBINED (FLO {w_flo:.0%} / POX {1.0 - w_flo:.0%})"

    rev = revenue_au_reserve_usd_per_t(cfg, au_dil, au_rec)
    total_cost = float(costs.get("TotalCost", 0.0))

    # Excel sign convention: TotalCost is typically negative; BV = Revenue + TotalCost
    bv = rev + total_cost

    return BlockOutputs(
        au_gt_diluted=au_dil,
        mp_mass_pull_pct=rec.mp_mass_pull_pct,
        recF_pct=rec.recF_pct,
        au_concentrate_gt=rec.au_concentrate_gt,
        au_recovery_fraction=au_rec,
        revenue_usd_per_t=rev,
        total_cost_usd_per_t=total_cost,
        block_value_usd_per_t=bv,
        cost_breakdown=costs,
        route_used=route_norm,
    )
