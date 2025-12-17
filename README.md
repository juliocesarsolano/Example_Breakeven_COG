# Breakeven Cut-Off Grade (Au) — Streamlit App

This application computes the **breakeven gold (Au) cut-off grade** as the gold grade
(g/t) that makes the **block value (BV)** equal to zero for a selected **MetType**,
processing route, and block descriptors.

The app is implemented in **Streamlit** and is intended as a transparent,
deterministic, and reproducible alternative to spreadsheet-based Goal Seek
calculations.

---

## Break-even Definition

The break-even cut-off grade is defined by the condition:

BV(Au) = Revenue(Au) − Total Ore Cost = 0


Where:
- `Au` = gold grade (g/t)
- `BV` = block value (USD per tonne)

Silver grade is fixed at **0.0 g/t** in this implementation in order to isolate the
gold-driven break-even condition.

---

## Economic Framework (USD/t)

All calculations are performed on a **per-tonne basis**, consistent with standard
cut-off grade methodology.

The **Total Ore Cost** may include the following components, depending on the
configuration:

- Incremental **TSF** cost  
- **Process fixed cost**  
- **General & Administrative (G&A)** cost  
- **CSR** cost  
- **Ore sustaining capital**  
- Optional **S% fixed-cost** term (if enabled in the YAML configuration)

Cost structures and parameters are sourced from the configured **LTP-style economic
model** and are applied consistently across all processing routes.

---

## Recovery and Processing Routes

Revenue is calculated from recovered gold ounces per tonne based on the selected
processing route:

- **FLO**  
  Flotation recovery chain and FLO cost stack.

- **POX**  
  POX–CIL recovery chain and POX cost stack.

- **COMBINED (FLO / POX)**  
  Linear blend of FLO and POX recoveries and costs using a user-defined split
  (e.g. 60% FLO / 40% POX).

For FLO, the recovery chain follows the structure:

recF / MP / Au_conc / aurecFLO

Sulphur-dependent variable costs are computed from the configured sulphur terms
(S%, S2%) and applied according to the selected route.

---

## Numerical Solution

The breakeven cut-off grade is obtained by solving the scalar equation:

BV(Au) = 0


using robust one-dimensional root-finding algorithms:

- **Bisection**  
  Deterministic and robust.

- **Brent’s Method**  
  Hybrid method combining bisection, secant, and inverse quadratic interpolation,
  with behaviour similar to Excel Goal Seek but fully deterministic.

The solver iterates until:

|BV| ≤ tolerance


Optionally, the application displays a **convergence plot (BV vs iteration)** to
support QA/QC and numerical traceability.

---

## Configuration and Inputs

### Configuration
- Economic parameters, recoveries, costs, and MetType definitions are controlled
  via a **YAML configuration file**.
- Scenario overrides (Au price, power cost, royalty) are applied at runtime and
  clearly separated from baseline configuration values.

### Block Descriptors
The following block-level variables are explicitly defined by the user:
- Cu (%)
- Total S (%)
- S2 (%)
- Dilution (%)

These inputs are treated as **explicit block descriptors**, ensuring transparency
and reproducibility of results.

---

## Application Features

- Deterministic cut-off grade calculation by MetType
- Support for FLO, POX, and Combined processing routes
- Full floating-point precision (no intermediate rounding during optimisation)
- Reproducible configuration via YAML
- Solver convergence diagnostics
- Summary table of cut-off grades across all MetTypes

---

## Intended Use

This repository is provided as an **example / demonstration implementation** of a
cut-off grade calculation workflow.

It is suitable for:
- Technical validation and QA/QC
- Training and methodology demonstrations
- Replacement of spreadsheet-based Goal Seek calculations

It is **not** intended to distribute proprietary cost data or official LTP inputs.

---

## Developer

**Julio César Solano Arroyo**  
Mineral Resource Superintendent  

Development date: **13 December 2025**

---

## License

This project is provided for demonstration and educational purposes.
Please refer to the `LICENSE` file for details.