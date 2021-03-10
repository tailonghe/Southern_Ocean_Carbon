# Southern_Ocean_Carbon
This work applies the convolutional neural networks and long short-term memory units to predict dissolved carbon dioxide profiles in the Southern Ocean using physical parameters that are readily available from satellite observations and climate reanalysis.

## Requirements
Package     | Version
---------   | -----------
Python      | 3.6.3
TensorFlow  | 2.1.0
CUDA        | 8.0.44
cuDNN       |7.0

## Multi-phase training
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


### References
1. Landschützer, P., Gruber, N., Bakker, D. C. E.: Decadal variations and trends of the global ocean carbon sink, Global Biogeochemical Cycles, 30, doi:10.1002/2015GB005359, 2016
