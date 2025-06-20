# ACI_py - CO2 Response Curve Analysis Tool

Hey! This is a preview dev release of ACI_py Ultra for analyzing CO2 response curves from Gas exchange Measurements

## What it does
- Processes raw A-Ci curve data from LI-COR 6800 gas exchange analyzers
- Supports both raw Excel (.xlsx) and CSV file formats, and you can paste temperature, A, Ci values too if using other non Licor system
- Fits C3 and C4 photosynthesis models to your data
- I am using interactive interface deployed through a Streamlit web for the moment
- I have a smart quality checking module with automated issue detection, so please try to run it before running the analysis tab

## Current Status
**Preview Release** - This is still in active development!

Should work with LI-COR 6800 data files but not tested with LI-6400 models yet, 
Batch processing is currently limited to 5 files due to Streamlit server capacity, but you can always run single file analysis includes full parameter fitting

### Fitting module methods
Based on the classical Farquhar-von Caemmerer-Berry (1980) model:

$$A = \min(W_c, W_j, W_p) - R_d$$

Where:
- $W_c$ = Rubisco-limited rate: $\frac{V_{cmax} \cdot C_i}{C_i + K_c(1 + O/K_o)}$
- $W_j$ = RuBP regeneration-limited rate: $\frac{J \cdot C_i}{4(C_i + 2\Gamma^*)}$
- $W_p$ = TPU-limited rate (triose phosphate utilization)

### C4 Photosynthesis Model
Based on the von Caemmerer (2000) model for NADP-ME type C4 plants

### Key Parameters Fitted
- **Vcmax**: Maximum carboxylation rate (μmol m⁻² s⁻¹)
- **J**: Electron transport rate (μmol m⁻² s⁻¹)
- **Rd**: Day respiration rate (μmol m⁻² s⁻¹)
- **TPU**: Triose phosphate utilization rate (μmol m⁻² s⁻¹)

Temperature Response by implementing both Bernacchi et al. (2001) and Sharkey et al. (2007) temperature response functions for parameter scaling

## To start locally
```bash
# Install all dependencies
pip install -r requirements.txt

streamlit run app.py
```

## References
- Farquhar, G.D., von Caemmerer, S. & Berry, J.A. (1980) *Planta* 149, 78-90
- von Caemmerer, S. (2000) *Biochemical Models of Leaf Photosynthesis*
- Bernacchi et al. (2001) *Plant, Cell & Environment* 24, 253-259
- Sharkey et al. (2007) *Plant, Cell & Environment* 30, 1035-1040

## About
By Mengjie Fan in 2024

Feel free to explore and provide feedback! This is a work in progress, so expect some rough edges. More features and documentation coming soon!

---
*Made with ☕*