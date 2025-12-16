# src/solver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, List


@dataclass(frozen=True)
class IterationPoint:
    iteration: int
    x: float
    fx: float


@dataclass(frozen=True)
class SolveResult:
    root: float
    f_root: float
    iterations: int
    bracket: Tuple[float, float]
    history: Tuple[IterationPoint, ...] = ()


def solve_bisection(
    f: Callable[[float], float],
    x_low: float,
    x_high: float,
    tol_x: float = 1e-6,
    tol_f: float = 1e-4,
    max_iter: int = 200,
    record_history: bool = False,
) -> SolveResult:
    """Deterministic bisection root finder (robust, reproducible)."""
    a, b = float(x_low), float(x_high)
    fa, fb = f(a), f(b)

    hist: List[IterationPoint] = []

    if fa == 0.0:
        return SolveResult(root=a, f_root=fa, iterations=0, bracket=(a, b), history=tuple(hist))
    if fb == 0.0:
        return SolveResult(root=b, f_root=fb, iterations=0, bracket=(a, b), history=tuple(hist))
    if fa * fb > 0:
        raise ValueError(f"Bisection requires sign change. f(a)={fa}, f(b)={fb}")

    m = 0.5 * (a + b)
    fm = f(m)

    for it in range(1, max_iter + 1):
        m = 0.5 * (a + b)
        fm = f(m)

        if record_history:
            hist.append(IterationPoint(it, m, fm))

        if abs(fm) <= tol_f or abs(b - a) <= tol_x:
            return SolveResult(root=m, f_root=fm, iterations=it, bracket=(a, b), history=tuple(hist))

        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    return SolveResult(root=m, f_root=fm, iterations=max_iter, bracket=(a, b), history=tuple(hist))


def solve_brent(
    f: Callable[[float], float],
    x_low: float,
    x_high: float,
    tol_x: float = 1e-6,
    tol_f: float = 1e-4,
    max_iter: int = 200,
    record_history: bool = False,
) -> SolveResult:
    """Brent's method via SciPy brentq (robust + fast).

    Notes:
    - Requires sign change between endpoints (like bisection).
    - SciPy enforces a minimum rtol; do NOT set rtol=0.
    - If record_history=True, we capture the sequence of function evaluations.
    """
    try:
        from scipy.optimize import brentq  # type: ignore
    except Exception as e:
        raise ImportError(
            "Brent's method requires SciPy. Install with: py -m pip install scipy"
        ) from e

    a, b = float(x_low), float(x_high)
    fa, fb = f(a), f(b)

    if fa == 0.0:
        return SolveResult(root=a, f_root=fa, iterations=0, bracket=(a, b), history=())
    if fb == 0.0:
        return SolveResult(root=b, f_root=fb, iterations=0, bracket=(a, b), history=())
    if fa * fb > 0:
        raise ValueError(f"Brent requires sign change. f(a)={fa}, f(b)={fb}")

    hist: List[IterationPoint] = []
    call_count = 0

    def f_wrapped(x: float) -> float:
        nonlocal call_count
        fx = f(x)
        call_count += 1
        if record_history:
            hist.append(IterationPoint(call_count, float(x), float(fx)))
        return fx

    # SciPy minimum relative tolerance is machine-epsilon–scale; use a safe rtol.
    # Keep xtol tied to tol_x. Use rtol = 1e-12 for deterministic-like behavior.
    root, r = brentq(
        f_wrapped,
        a,
        b,
        xtol=float(tol_x),
        rtol=1e-12,
        maxiter=int(max_iter),
        full_output=True,
        disp=False,
    )

    f_root = f(root)
    iters = int(getattr(r, "iterations", max_iter))

    return SolveResult(
        root=float(root),
        f_root=float(f_root),
        iterations=iters,
        bracket=(a, b),
        history=tuple(hist),
    )
