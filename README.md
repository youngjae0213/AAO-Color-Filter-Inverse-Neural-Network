# AAO-Color-Filter-Inverse-Neural-Network
Inverse design of Fabry–Perot cavity spectra using ANN (PyTorch)
Given a target optical response (e.g. color in CIELAB space), the model predicts physically meaningful structure parameters using a Mixture Density Network (MDN) implemented in PyTorch.

## Goal
- Input: Target color (L*, a*, b*) or spectrum
- Output: 5 structural parameters of the Fabry–Perot cavity  
  - pore radius  
  - pore period  
  - AAO thickness  
  - top Ag thickness  
  - bottom Ag thickness
This solves the non-uniqueness problem in inverse design by predicting a **distribution** (mixture of Gaussians) instead of just one deterministic answer.

## Model Architecture
- Backbone: MLP to extract features from the target color/spectrum  
- MDN head: predicts (μ, σ, π) for each structural parameter  
  - **μ**: candidate design values  
  - **σ**: uncertainty  
  - **π**: mixture weights  
- K : number of mixtures

The training objective is the negative log-likelihood (NLL) of the true structure under the predicted mixture.
