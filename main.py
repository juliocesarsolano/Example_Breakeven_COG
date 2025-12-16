from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import altair as alt

from src.config import load_config, deep_update, rocktype_code
from src.economics import BlockInputs, compute_block_value
from src.solver import solve_bisection, solve_brent


# =============================================================================
# Page configuration
# =============================================================================
st.set_page_config(
    page_title="PV Breakeven Cut-Off Grade (Au) by MetType",
    layout="wide",
)

st.title("PV Breakeven Cut-Off Grade (Au) by MetType (with Ag grade 0.0)")
st.caption(
    "Optimisation methods available: Deterministic Bisection and Brent's Method "
    "(hybrid root-finder)."
)

# =============================================================================
# Sidebar – configuration and inputs
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CFG = PROJECT_ROOT / "config" / "cutoff_params.yml"

cfg_path = st.sidebar.text_input(
    "YAML config path",
    str(DEFAULT_CFG),
)
cfg = load_config(cfg_path)

# --- MetType
mettype_key = st.sidebar.selectbox(
    "MetType",
    options=list(cfg["mettypes"].keys()),
    index=list(cfg["mettypes"].keys()).index("movc"),
)
rock = rocktype_code(cfg, mettype_key)

# --- Processing route
st.sidebar.subheader("Processing Route")
route_label = st.sidebar.radio(
    "Route",
    options=["FLO", "POX", "Combined FLO/POX"],
    index=0,  # FLO default for Excel Summary parity
)

route_token = "COMBINED" if route_label.startswith("Combined") else route_label
st.sidebar.caption(f"Route used in calculations: {route_token}")

# --- Combined split (only applicable for COMBINED)
combined_flo_fraction = 0.60
if route_token == "COMBINED":
    combined_flo_fraction = st.sidebar.slider(
        "Combined split - fraction to FLO",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
        format="%.2f",
    )
    st.sidebar.caption(
        f"Split: {combined_flo_fraction:.0%} FLO / {1.0 - combined_flo_fraction:.0%} POX"
    )

# --- Block inputs (editable)
st.sidebar.subheader("Medium Stress-case Example")

cu_pct = st.sidebar.number_input(
    "Cu (%)",
    min_value=0.0,
    max_value=100.0,
    value=0.089,
    format="%.3f",
)

s_total_pct = st.sidebar.number_input(
    "Total S (%)",
    min_value=5.244,
    max_value=11.342,
    value=8.707,
    format="%.3f",
)

s2_pct = st.sidebar.number_input(
    "S2 (%)",
    min_value=4.758,
    max_value=10.276,
    value=7.140,
    format="%.3f",
)

dilution_pct = st.sidebar.number_input(
    "Dilution (%)",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    format="%.3f",
)

# --- Reserve vs Resource context
cutoff_context = st.sidebar.radio(
    "Cut-off context",
    options=[
        "Reserve (Au price = 1500 USD/oz)",
        "Resource (Au price = 2000 USD/oz)",
    ],
    index=0,
)
default_au_price = 1500.0 if cutoff_context.startswith("Reserve") else 2000.0

# --- Session-state safe Au price handling
if "cutoff_context_prev" not in st.session_state:
    st.session_state["cutoff_context_prev"] = cutoff_context
if "au_price" not in st.session_state:
    st.session_state["au_price"] = default_au_price

if cutoff_context != st.session_state["cutoff_context_prev"]:
    st.session_state["au_price"] = default_au_price
    st.session_state["cutoff_context_prev"] = cutoff_context

# --- Scenario overrides
st.sidebar.subheader("Scenario Overrides")
au_price = st.sidebar.number_input(
    "Au price (USD/oz)",
    step=10.0,
    key="au_price",
)
power = st.sidebar.number_input(
    "Power cost P (USD/kWh)",
    value=float(cfg["economic_parameters"]["power"]["P_usd_per_kwh"]),
    format="%.5f",
)
royalty = st.sidebar.number_input(
    "Royalty RYT (fraction)",
    value=float(cfg["economic_parameters"]["royalties"]["RYT_fraction_of_net_price"]),
    format="%.4f",
)

# --- Output precision
st.sidebar.subheader("Output Precision")
st.sidebar.write("Full precision (fixed)")

# --- Optimization method
st.sidebar.subheader("Optimization Method")
opt_method = st.sidebar.radio(
    "Root finder",
    options=["Bisection (deterministic)", "Brent (hybrid)"],
    index=1,
)

# --- Apply overrides
cfg_run = deep_update(
    cfg,
    {
        "economic_parameters": {
            "metal_prices": {"AUP1_usd_per_oz_reserve": au_price},
            "power": {"P_usd_per_kwh": power},
            "royalties": {"RYT_fraction_of_net_price": royalty},
        }
    },
)

# =============================================================================
# Read-only reference panel
# =============================================================================
st.sidebar.divider()
show_ref = st.sidebar.checkbox("Show Reference Parameters (read-only)", value=True)

if show_ref:
    st.sidebar.subheader("Reference Parameters (from YAML)")

    try:
        cfg_bytes = open(cfg_path, "rb").read()
        cfg_sha256 = hashlib.sha256(cfg_bytes).hexdigest()
    except Exception:
        cfg_sha256 = "N/A"

    st.sidebar.caption(f"Config file: {cfg_path}")
    st.sidebar.caption(f"Loaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.sidebar.caption(f"SHA-256: {cfg_sha256}")

    econ = cfg_run["economic_parameters"]
    st.sidebar.markdown("**Economic Assumptions**")
    st.sidebar.write(f"AUP1: {econ['metal_prices']['AUP1_usd_per_oz_reserve']:.2f} USD/oz")
    st.sidebar.write(f"RAU1: {econ['refining_transport_costs']['RAU1_usd_per_oz']:.2f} USD/oz")
    st.sidebar.write(f"AUP2: {econ['metal_prices']['AUP2_usd_per_oz_resource']:.2f} USD/oz")
    st.sidebar.write(f"Royalty: {econ['royalties']['RYT_fraction_of_net_price'] * 100:.3f} %")
    st.sidebar.write(f"Power: {econ['power']['P_usd_per_kwh']:.5f} USD/kWh")
    st.sidebar.write("Ag grade fixed: 0.0 g/t")

    st.sidebar.markdown("**Block Inputs (current values)**")
    st.sidebar.write(f"Cu: {cu_pct:.3f} %")
    st.sidebar.write(f"Total S: {s_total_pct:.3f} %")
    st.sidebar.write(f"S2: {s2_pct:.3f} %")
    st.sidebar.write(f"Dilution: {dilution_pct:.3f} %")

if route_token == "COMBINED":
    st.subheader(
        f'MetType Selected: "{mettype_key}" | Route: COMBINED '
        f'({combined_flo_fraction:.0%} FLO / {1.0 - combined_flo_fraction:.0%} POX)'
    )
else:
    st.subheader(f'MetType Selected: "{mettype_key}" | Route: {route_token}')

# =============================================================================
# Explanation
# =============================================================================
with st.expander("Overview of Break-even Cut-off Grade Method", expanded=True):
    st.markdown(
        r"""
**Break-even cut-off grade method.**  
For each *MetType*, the gold cut-off grade \(Au_{CO}\) is defined as the grade for which:

\[
BV(Au │ MetType, Route, S, S2, Dilution, ...) = 0
\]

\[
BV = Revenue(Au) - Tot. Costs (absolute)
\]

Silver grade is fixed at **0.0 g/t** to isolate the Au-driven break-even condition.

**Processing routes supported:**
- **FLO**: flotation recovery chain and FLO cost stack
- **POX**: POX–CIL recovery and POX cost stack
- **Combined FLO/POX**: linear blend (Default: 60% FLO / 40% POX) of recoveries and costs

**Optimisation methods:**
- **Bisection**: deterministic and robust
- **Brent**: hybrid method, behaviour similar to Excel Goal Seek
"""
    )

# =============================================================================
# Tabs
# =============================================================================
tab_solver, tab_summary = st.tabs(["Cut-off Solver", "Summary Table by MetType"])


def solve_root(f, a, b, tol_f, max_iter, record_history):
    if opt_method.startswith("Bisection"):
        return solve_bisection(
            f, a, b,
            tol_f=float(tol_f),
            max_iter=int(max_iter),
            record_history=record_history,
        )
    return solve_brent(
        f, a, b,
        tol_f=float(tol_f),
        max_iter=int(max_iter),
        record_history=record_history,
    )


# =============================================================================
# Cut-off solver tab
# =============================================================================
with tab_solver:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Solve cut-off Au (g/t)")
        x_low = st.number_input("Initial Au low (g/t)", value=0.0, step=0.1)
        x_high = st.number_input("Initial Au high (g/t)", value=20.0, step=0.5)
        tol_f = st.number_input("Tolerance |BV| ($/t)", value=1e-4, format="%.1e")
        max_iter = st.number_input("Max iterations", value=200, step=10)

        if st.button("Calculate cut-off"):
            base_block = BlockInputs(
                au_gt=0.0,
                ag_gt=0.0,
                cu_pct=cu_pct,
                s_total_pct=s_total_pct,
                s2_pct=s2_pct,
                dilution_pct=dilution_pct,
            )

            def f(au_gt: float) -> float:
                blk = BlockInputs(**{**base_block.__dict__, "au_gt": au_gt})
                out = compute_block_value(
                    cfg_run,
                    rocktype_code=rock,
                    block=blk,
                    route=route_token,
                    combined_flo_fraction=combined_flo_fraction,
                )
                return out.block_value_usd_per_t

            a, b = float(x_low), float(x_high)
            fa, fb = f(a), f(b)
            expand = 0
            while fa * fb > 0 and expand < 20:
                b = max(0.1, b * 1.5)
                fb = f(b)
                expand += 1

            if fa * fb > 0:
                st.error("Could not bracket a root. Increase upper bound or review inputs.")
            else:
                res = solve_root(
                    f, a, b,
                    tol_f=tol_f,
                    max_iter=max_iter,
                    record_history=True,
                )

                st.success(f"Au cut-off = {res.root:.3f} g/t (BV = {res.f_root:.5f} $/t)")

                blk_star = BlockInputs(**{**base_block.__dict__, "au_gt": res.root})
                out = compute_block_value(
                    cfg_run,
                    rocktype_code=rock,
                    block=blk_star,
                    route=route_token,
                    combined_flo_fraction=combined_flo_fraction,
                )

                st.dataframe(
                    pd.DataFrame([{
                        "Route": out.route_used,
                        "Au_diluted_gpt": out.au_gt_diluted,
                        "Au_recovery_fraction": out.au_recovery_fraction,
                        "Revenue_usd_per_t": out.revenue_usd_per_t,
                        "TotalCost_usd_per_t": out.total_cost_usd_per_t,
                        "BlockValue_usd_per_t": out.block_value_usd_per_t,
                    }]),
                    use_container_width=True,
                )

                st.markdown("#### Cost Breakdown (USD/t)")
                st.dataframe(pd.DataFrame([out.cost_breakdown]), use_container_width=True)

                # =============================================================
                # Plots (must be inside the button-click branch)
                # =============================================================
                st.markdown("#### Convergence Plot (Block Value vs Iteration)")

                hist = getattr(res, "history", None) or getattr(res, "iterations", None)

                def _to_float(v):
                    try:
                        return float(v)
                    except Exception:
                        return None

                rows = []

                if not hist:
                    st.info(
                        "No solver iteration history available to plot. "
                        "Ensure `record_history=True` and that the solver returns `history` (or `iterations`)."
                    )
                else:
                    for i, h in enumerate(hist):
                        it = None
                        xh = None
                        fh = None

                        # 1) IterationPoint-like object with attributes (your case)
                        if hasattr(h, "x") and hasattr(h, "fx"):
                            xh = _to_float(getattr(h, "x", None))
                            fh = _to_float(getattr(h, "fx", None))
                            it = getattr(h, "iteration", None)
                            it = int(it) if it is not None else (i + 1)

                        # 2) Dict-like
                        elif isinstance(h, dict):
                            xh = _to_float(h.get("x", h.get("root", h.get("mid", None))))
                            fh = _to_float(h.get("fx", h.get("f", h.get("f_x", None))))
                            it = h.get("iteration", None)
                            it = int(it) if it is not None else (i + 1)

                        # 3) Tuple/list fallback
                        else:
                            try:
                                seq = list(h)
                                if len(seq) == 2:
                                    xh = _to_float(seq[0])
                                    fh = _to_float(seq[1])
                                    it = i + 1
                                elif len(seq) >= 3:
                                    it = int(seq[0]) if str(seq[0]).isdigit() else (i + 1)
                                    xh = _to_float(seq[1])
                                    fh = _to_float(seq[2])
                            except Exception:
                                pass

                        if it is not None and fh is not None:
                            rows.append({"iter": int(it), "Au_gpt": xh, "BV_usd_per_t": float(fh)})

                    if not rows:
                        with st.expander("Debug: solver history sample"):
                            st.write("History type:", type(hist))
                            st.write("First item type:", type(hist[0]) if len(hist) else None)
                            st.write("First item value:", hist[0] if len(hist) else None)
                        st.error(
                            "Iteration history exists, but it could not be parsed into (iteration, BV)."
                        )
                    else:
                        df_hist = pd.DataFrame(rows).sort_values("iter")

                        bv_iter_chart = alt.Chart(df_hist).mark_line(
                            color="#D4AF37",  # gold
                            point=True
                        ).encode(
                            x=alt.X("iter:Q", title="Iteration"),
                            y=alt.Y("BV_usd_per_t:Q", title="Block Value (USD/t)"),
                            tooltip=[
                                alt.Tooltip("iter:Q", title="Iteration"),
                                alt.Tooltip("Au_gpt:Q", title="Au (g/t)", format=",.6f"),
                                alt.Tooltip("BV_usd_per_t:Q", title="BV (USD/t)", format=",.6f"),
                            ],
                        )

                        zero_rule = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule().encode(y="y:Q")

                        st.altair_chart(
                            (bv_iter_chart + zero_rule).properties(height=320),
                            use_container_width=True,
                        )


    with col2:
        st.markdown(
            """
            **About this Web App**
            
            - Developed by **Julio César Solano Arroyo**  
              MRM Superintendent   
              *Development date: 13 December 2025*

            - Application designed to compute **breakeven Au cut-off grades** by MetType using a
              **deterministic economic block-value framework**.

            - Fully aligned with **official LTP cost scripts** and the **LTP "Excel" methodology**,
              ensuring numerical parity under equivalent assumptions.

            - Supports multiple **processing routes**: FLO, POX, and a **user-defined Combined FLO/POX split**.

            - Recovery models and cost stacks are **route-dependent** and sourced directly from
              validated metallurgical and cost parameter sets.

            - Implements **Brent's Method** and **Bisection** for robust root-finding,
              analogous to Excel Goal Seek but fully deterministic.

            - All calculations are performed using **full floating-point precision**;
              no intermediate rounding is applied during optimisation.

            - Economic inputs (metal prices, royalties, power) are centrally governed via **YAML configuration**
             and scenario overrides are explicitly logged.

            - Block-level variables (S%, S2%, dilution, Cu%) are treated as **explicit block descriptors**,
              ensuring transparency and reproducibility.

            - Designed for **technical governance**, traceability, and extension to future resource/reserve
            scenarios and processing configurations.
            """
        )

# =============================================================================
# Summary table tab
# =============================================================================
with tab_summary:
    st.markdown("### Summary Table by MetType")
    st.write(f"Break-even Au cut-off for each MetType (Route = {route_token}).")

    if st.button("Run Summary (all mettypes)"):
        results = []
        base_block = BlockInputs(
            au_gt=0.0,
            ag_gt=0.0,
            cu_pct=cu_pct,
            s_total_pct=s_total_pct,
            s2_pct=s2_pct,
            dilution_pct=dilution_pct,
        )

        for mt in cfg["mettypes"].keys():
            rock_mt = rocktype_code(cfg, mt)

            def f_mt(au_gt: float) -> float:
                blk = BlockInputs(**{**base_block.__dict__, "au_gt": au_gt})
                out = compute_block_value(
                    cfg_run,
                    rocktype_code=rock_mt,
                    block=blk,
                    route=route_token,
                    combined_flo_fraction=combined_flo_fraction,
                )
                return out.block_value_usd_per_t

            a, b = 0.0, 5.0
            fa, fb = f_mt(a), f_mt(b)
            expand = 0
            while fa * fb > 0 and expand < 25:
                b = max(0.1, b * 1.5)
                fb = f_mt(b)
                expand += 1

            if fa * fb > 0:
                cutoff = None
            else:
                res = solve_root(
                    f_mt, a, b,
                    tol_f=1e-4,
                    max_iter=250,
                    record_history=False,
                )
                cutoff = res.root

            results.append({
                "MetType": mt,
                "Route": route_token,
                "Au_cutoff_gpt": None if cutoff is None else round(cutoff, 3),
            })

        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        if df["Au_cutoff_gpt"].notna().any():
            st.info(
                f"Range: {df['Au_cutoff_gpt'].min():.3f} g/t – "
                f"{df['Au_cutoff_gpt'].max():.3f} g/t"
            )

        # Optional: Summary chart (Au cut-off by MetType)
        df_plot = df.dropna(subset=["Au_cutoff_gpt"]).copy()
        if not df_plot.empty:
            st.markdown("#### Plot: Au cut-off by MetType")
            chart = alt.Chart(df_plot).mark_bar().encode(
                x=alt.X("MetType:N", sort=None, title="MetType"),
                y=alt.Y("Au_cutoff_gpt:Q", title="Au cut-off (g/t)"),
                tooltip=["MetType:N", "Au_cutoff_gpt:Q"],
            )
            st.altair_chart(chart.properties(height=280), use_container_width=True)

# This script ends here.