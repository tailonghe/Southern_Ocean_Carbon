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
We propose a multi-phase training stratety to train the model. In phase 1, we use simulation diagnostics from the Biogeochemical Southern Ocean State Estimate (B-SOSE) data assimilation system. In phase 2, we train the model using observational data sets and climate reanalysis data sets. The input variables for both training phases are shown below:

Input variable   | Phase 1      | Phase 2
-----------------|--------------|--------------
Sea surface height anomaly           | 3.6.3        |
TensorFlow       | 2.1.0        |
CUDA             | 8.0.44       |
cuDNN            |7.0           |
