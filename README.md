# AAO-Color-Filter-Inverse-Design-MDN
Predicting AAO Fabry–Perot structural parameters from target color (Lab) using a Mixture Density Network (MDN) implemented in PyTorch.  

## Overview
- **Goal:** Inverse design — find structural parameters that produce a desired color (L*, a*, b*).  
- **Input:** Target color (L*, a*, b*) or transmittance spectrum  
- **Output (5 parameters):**  
  - pore radius  
  - pore period  
  - AAO thickness  
  - top Ag thickness  
  - bottom Ag thickness  
- **Model:** Mixture Density Network (MDN)  
- **Number of mixtures:** K  
- **Loss Function:** Negative Log-Likelihood (NLL)  
- **Data Normalization:**  
  - Input: StandardScaler  
  - Output: MinMaxScaler

**Visualizations / Outputs include:**  
- Excel export of predicted vs. actual parameters  
- ΔE color error between target and reconstructed Lab (ΔE < 1 highlighted)

##  Model Description
The MDN predicts not a single solution but a **distribution of possible designs**.  
Each structural parameter is modeled as a Gaussian mixture with:
- **μ:** predicted mean value  
- **σ:** predicted standard deviation  
- **π:** mixture weight (sum of π = 1)
This allows handling of **non-unique inverse problems** in nanophotonic design.

## Project Structure
```text
AAO-Color-Filter-Inverse-Design-MDN/
├── mdn_inverse_design.py
├── results/
   ├── example_predictions.xlsx 

