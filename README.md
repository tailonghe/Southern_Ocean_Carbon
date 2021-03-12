# Southern Ocean Carbon
This Southern Ocean Carbon (SOC) model applies the convolutional neural networks and long short-term memory units to predict dissolved carbon dioxide profiles in the Southern Ocean using physical parameters that are readily available from satellite observations and climate reanalysis.

## Prerequisites
Package     | Version
---------   | -----------
Python      | 3.6.3
TensorFlow  | 2.1.0
CUDA        | 8.0.44
cuDNN       | 7.0
TensorFlow  | 2.1.0
Keras       | 2.3.1

## Multi-phase training and datasets
We propose a multi-phase training stratety to train the model. In phase 1, we use simulation diagnostics from the Biogeochemical Southern Ocean State Estimate (B-SOSE) data assimilation system. In phase 2, we train the model using observational data sets and climate reanalysis data sets. The input/output variables for both training phases are shown below:

Input variable                                          | Phase 1 source      | Phase 2 source
--------------------------------------------------------|---------------------|--------------
Sea surface height anomaly (SSHA)                       | B-SOSE              | Copernicus Marine Service
Flux of CO2 due to air-sea exchange (pCO2)              | B-SOSE              | Landschützer et al., 2016
Heat flux                                               | B-SOSE              | ERA5
Zonal component of ocean surface current velocity       | B-SOSE              | OSCAR (Ocean Surface Current Analysis Real-time)
Meridional component of ocean surface current velocity  | B-SOSE              | OSCAR (Ocean Surface Current Analysis Real-time)
Vertical component of ocean surface current velocity    | B-SOSE              | Derived from SSHA
Surface Chlorophyll-a concentration                     | B-SOSE              | The GlobColour project
Zonal component of ocean surface wind speed             | ERA5                | ERA5
Meridional component of ocean surface wind speed        | ERA5                | ERA5
Sea surface temperature                                 | ERA5                | ERA5


Output variable                                         | Phase 1 source      | Phase 2 source
--------------------------------------------------------|---------------------|--------------
Dissolved inorganic carbon (DIC)                        | B-SOSE              | Global Ocean Data Analysis Project version 2 (GLODAPv2) shipboard measurements<br />**and**<br />Southern Ocean Carbon and Climate Observations and Modeling (SOCCOM) biogeochemical Argo floats

## Getting started
### Generating data pairs
The SOC model takes vectors of 10 predictors mentioned above and outputs DIC profiles at 48 levels (from surface to 4 km depth as defined in B-SOSE).
### Train 
### Test

### DIC calculation from 1998 to 2019

## Evaluation of model performance

## References
1. Landschützer, P., Gruber, N., Bakker, D. C. E.: Decadal variations and trends of the global ocean carbon sink, Global Biogeochemical Cycles, 30, doi:10.1002/2015GB005359, 2016
