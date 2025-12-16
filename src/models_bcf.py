from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class RecoveryOutputs:
    mp_mass_pull_pct: float
    recF_pct: float
    au_concentrate_gt: float
    aurec_pox_fraction: float
    aurec_flo_fraction: float


def _pox_au_recovery_fraction(cfg: Dict[str, Any], au_gt: float) -> float:
    g = float(au_gt)
    pox = cfg["metallurgical_models"]["pox_cil"]
    ac = float(pox["ACco_Au_gt"])
    fact7 = float(cfg["metallurgical_models"]["global"]["fact7_recovery_reduction_factor_pctpts"])
    f1 = float(pox["gold"]["fact1_intercept_pct"])
    f2 = float(pox["gold"]["fact2_slope_pct_per_gt"])
    f3 = float(pox["gold"]["fact3_max_pct"])

    if g <= 0:
        return 0.0
    if g <= ac:
        return ((f1 + f2 * g) - fact7) / 100.0
    return (f3 - fact7) / 100.0


def au_recovery_flotation_chain_fraction(
    cfg: Dict[str, Any],
    rocktype_code: str,
    au_gt_diluted: float,
    s2_pct: float,
) -> RecoveryOutputs:
    au = float(au_gt_diluted)
    SS = float(s2_pct)

    mp_cfg = cfg["metallurgical_models"]["flotation"]["mass_pull"]
    MPisa = float(mp_cfg["MPisa_slope"])
    MPisb = float(mp_cfg["MPisb_intercept"])
    mp = MPisa * SS + MPisb

    fact7 = float(cfg["metallurgical_models"]["global"]["fact7_recovery_reduction_factor_pctpts"])
    tailsrec = float(cfg["metallurgical_models"]["flotation"]["Tailsrec_fraction"])

    consts = cfg["metallurgical_models"]["flotation"]["gold_recovery_constants_by_mettype"][rocktype_code]
    Fa = float(consts["Fa"])
    Fb = float(consts["Fb"])
    Fc = float(consts["Fc"])
    Fd = float(consts["Fd"])

    if au <= 0 or mp <= 0:
        return RecoveryOutputs(mp, 0.0, 0.0, 0.0, 0.0)

    # recF is in percent units per existing model conventions
    recF = SS * Fa + (SS / au) * Fb + (1.0 / au) * Fc + Fd - fact7

    # Guardrail: if recF is non-positive, no recovery can occur
    if recF <= 0:
        return RecoveryOutputs(mp, 0.0, 0.0, 0.0, 0.0)

    auconc = au * recF / mp

    # POX recovery fraction computed on concentrate grade per model design
    aurecPOX = _pox_au_recovery_fraction(cfg, auconc)

    # FLO overall recovery fraction (flotation recovery + tails recovery blending)
    pox = cfg["metallurgical_models"]["pox_cil"]
    ac = float(pox["ACco_Au_gt"])
    f1 = float(pox["gold"]["fact1_intercept_pct"])
    f2 = float(pox["gold"]["fact2_slope_pct_per_gt"])
    f3 = float(pox["gold"]["fact3_max_pct"])

    if auconc <= ac:
        top = (((f1 - fact7 + f2 * auconc) * recF) / 100.0) + (100.0 - recF) * tailsrec
        aurecFLO = top / 100.0
    else:
        top = (((f3 - fact7) * recF) / 100.0) + (100.0 - recF) * tailsrec
        aurecFLO = top / 100.0

    aurecFLO = max(0.0, min(1.0, aurecFLO))
    aurecPOX = max(0.0, min(1.0, aurecPOX))

    return RecoveryOutputs(mp, recF, auconc, aurecPOX, aurecFLO)
