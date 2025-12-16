# PV 2025 Cut-Off Grade (Streamlit)

This application computes the **breakeven Au cut-off grade** as the gold grade (g/t) that makes the **block value (BV)** equal to zero for a selected processing route and block descriptors:

\[
BV(Au) = \text{Revenue}(Au) - \text{Total Ore Cost} \;=\; 0
\]

### Economics Included (USD/t)

The total ore cost is computed on a **per-tonne basis** and includes both variable and fixed components required for cut-off determination:

- Incremental **TSF** cost  
- **Process fixed cost**  
- **G&A** cost  
- **CSR** cost  
- **Ore sustaining capital**  
- Optional **S% fixed-cost** term (if enabled in the configuration)

### Recovery and Route Logic

Revenue is calculated from recovered gold ounces per tonne, based on the selected processing route:

- **FLO**: flotation recovery chain and FLO cost stack  
- **POX**: POX recovery chain and POX cost stack  
- **COMBINED**: linear blend of FLO and POX recoveries and costs using a user-defined split

For FLO, the recovery chain follows the structure:

- `recF / MP / Au_conc / aurecFLO`

Sulphur-dependent variable costs are computed from the configured sulphur terms and applied according to the selected route.

### Numerical Solution

The breakeven cut-off is obtained by solving for the root of \(BV(Au)\) using a robust 1-D root-finding algorithm:

- **Bisection** (deterministic and robust)
- **Brent** (hybrid and efficient)

The solver iterates until \(|BV|\) meets the selected tolerance, and the application can display a convergence plot (**BV vs Iteration**) for QA/QC.

## Processing Routes and Cut-off Calculation (FLO, POX, COMBINED)

This application computes the **breakeven Au cut-off grade** as the gold grade (g/t) that makes the **block value (BV)** equal to zero:

\[
BV(g) = \text{Revenue}(g) - \text{Total Ore Cost}(g) = 0
\]

Where \(g\) is the Au grade (g/t). All calculations are performed on a **per-tonne basis (USD/t)**.

### Route Options

#### 1) FLO (Flotation)

**Assumption:** 100% of the selected material is processed through the **FLO** route.

- **Revenue (USD/t):** computed from recovered gold ounces per tonne using the **FLO recovery chain**.
- **Costs (USD/t):** computed using the **FLO cost stack**, including applicable fixed and variable terms.

The breakeven cut-off is obtained by solving:

\[
BV_{\text{FLO}}(g) = 0
\]

---

#### 2) POX (Pressure Oxidation + downstream recovery)

**Assumption:** 100% of the selected material is processed through the **POX** route.

- **Revenue (USD/t):** computed from recovered gold ounces per tonne using the **POX recovery chain**.
- **Costs (USD/t):** computed using the **POX cost stack**, including applicable fixed and variable terms.

The breakeven cut-off is obtained by solving:

\[
BV_{\text{POX}}(g) = 0
\]

---

#### 3) COMBINED (FLO/POX split)

**Assumption:** the selected material is split between FLO and POX using a constant fraction:

- \(w\) = fraction to FLO = `combined_flo_fraction`
- \(1-w\) = fraction to POX

For a given grade \(g\), the application computes both route economics and then applies a **linear blend** of revenues and costs:

\[
\text{Revenue}_{\text{COMB}}(g) = w\cdot \text{Revenue}_{\text{FLO}}(g) + (1-w)\cdot \text{Revenue}_{\text{POX}}(g)
\]

\[
\text{Cost}_{\text{COMB}}(g) = w\cdot \text{Cost}_{\text{FLO}}(g) + (1-w)\cdot \text{Cost}_{\text{POX}}(g)
\]

\[
BV_{\text{COMB}}(g) = \text{Revenue}_{\text{COMB}}(g) - \text{Cost}_{\text{COMB}}(g)
\]

The breakeven cut-off is obtained by solving:

\[
BV_{\text{COMB}}(g) = 0
\]

**Important:** in general, the combined cut-off is **not** the weighted average of the individual cut-offs:

\[
g_{CO}^{\text{COMB}} \neq w\cdot g_{CO}^{\text{FLO}} + (1-w)\cdot g_{CO}^{\text{POX}}
\]

because the root of a weighted sum of functions is not equal to the weighted sum of the individual roots.

---

### Shared Modeling Assumptions (all routes)

- **Unit basis:** all economics are computed in **USD/t** and depend on the selected block descriptors (e.g., Cu%, Total S%, S2%, dilution%).
- **Ag grade:** fixed at **0.0 g/t** to isolate the Au-driven breakeven condition.
- **Economic parameters:** metal price, royalty, power cost, and other parameters are sourced from the configuration and can be overridden in the UI.
- **Numerical solution:** the cut-off is found by solving \(BV(g)=0\) using a 1-D root finder (**Bisection** or **Brent**) until the specified tolerance on \(|BV|\) is met.

Run:
```bash
py -m pip install -r requirements.txt
py -m streamlit run main.py
```

#### 4) Build the Executable
# build_exe.ps1. Se corre en powershell: powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
# Nota: en la carpeta dist debe quedar el .exe
 
$ErrorActionPreference = "Stop"

python -m PyInstaller --noconfirm --onefile --windowed --name PV_Cutoff_Grade `
  --add-data "main.py;." `
  --add-data "src;src" `
  --add-data "config;config" `
  --collect-all streamlit `
  .\desktop_app.py

Write-Host ""
Write-Host "Build finalizado. Revisa: .\dist\PV_Cutoff_Grade.exe"
