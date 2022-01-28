# Southern Ocean Carbon
This deep-learning (DL) model applies the convolutional neural networks (CNN) and long short-term memory (LSTM) units to predict dissolved carbon dioxide profiles in the Southern Ocean using physical parameters that are readily available from satellite observations and climate reanalysis.

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

Input variable                                            | Phase 1 source      | Phase 2 source
----------------------------------------------------------|---------------------|--------------
Sea surface height anomaly (SSHA)                         | B-SOSE  ([Link](http://sose.ucsd.edu/BSOSE_iter105_solution.html))             | Copernicus Marine Service ([Link](https://resources.marine.copernicus.eu/?option=com_csw&view=details&product_id=SEALEVEL_GLO_PHY_L4_REP_OBSERVATIONS_008_047))
Flux of CO2 due to air-sea exchange (pCO2)                | B-SOSE              | Landschützer et al., 2016 ([Link](https://www.ncei.noaa.gov/access/ocean-carbon-data-system/oceans/SPCO2_1982_present_ETH_SOM_FFN.html))
Heat flux (Tflx)                                          | B-SOSE              | ERA5
Zonal component of ocean surface current velocity (U)     | B-SOSE              | OSCAR (Ocean Surface Current Analysis Real-time) ([Link](https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_third-deg))
Meridional component of ocean surface current velocity (V)| B-SOSE              | OSCAR (Ocean Surface Current Analysis Real-time)
Vertical component of ocean surface current velocity (W)  | B-SOSE              | Derived from SSHA
Surface Chlorophyll-a concentration (CHL-a)               | B-SOSE              | The GlobColour project ([Link](https://hermes.acri.fr/))
Zonal component of ocean surface wind speed (u10m)        | ERA5 ([Link](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview))               | ERA5
Meridional component of ocean surface wind speed (v10m)   | ERA5                | ERA5
Sea surface temperature (SST)                             | ERA5                | ERA5


Output variable                                         | Phase 1 source      | Phase 2 source
--------------------------------------------------------|---------------------|--------------
Dissolved inorganic carbon (DIC)                        | B-SOSE              | Global Ocean Data Analysis Project version 2 (GLODAPv2) shipboard measurements ([Link](https://www.ncei.noaa.gov/access/ocean-carbon-data-system/oceans/GLODAPv2/))<br />**and**<br />Southern Ocean Carbon and Climate Observations and Modeling (SOCCOM) biogeochemical Argo floats ([Link1](https://soccom.princeton.edu/) or [Link2](http://www3.mbari.org/SOCCOM/))

## Getting started
### Generating data pairs
This DL model takes vectors of 10 predictors mentioned above and outputs DIC fields at 48 levels (from surface to 4 km depth as defined in B-SOSE). 
The shape of input vectors is (batch_size, latitude, longitude, predictors) and the output shape is (batch_size, latitude, longitude, 1).
The scripts used to generate data pairs are in the example_data folder.

The example.ipynb jupyter notebook contains several examples about the usage of the DL model. Operations could be also done using command lines. Below are several examples.

### Train 
To train a DL model:
```
python ./train_model.py --x sample_data/phase1_sample_data/X/X_* --y sample_data/phase1_sample_data/Y/Y_* --lvl1 1 --lvl2 2 --b 20 --o model_weights_example.h5
```
<!-- As an example, you can try to train a SOC model using the Argo floats measurements with:
```
python train_model.py --x example_data/Argo/train/*_predictors.npy --y example_data/Argo/train/*_DIC.npy --o SOC_model_Argo.h5
``` -->

### Test/predict
To use a pretrained model to generate DIC predictions:
```
python ./model_predict.py --x sample_data/phase1_sample_data/X/X_* --lvl1 1 --lvl2 2 --w ../archived_models/sensitivity/phase2/DLSOCO2_checkpt_0.09_lvl1-2_p2sens.h5 --o example_output
```
<!-- As an example, you can use a pretrained model to generate predictions with:
```
python test_model.py --x example_data/Argo/test/2019_Argo_predictors.npy --w SOC_model_Argo.h5
``` -->

<!-- ### DIC calculation from 1998 to 2019
To calculate DIC concentrations from 1998 to 2019 using the predictors regridded to 1<sup>°</sup> by 1<sup>°</sup> resolution, you can do something similar:
```
python test_model.py --x 1998_2019_predictors/*_predictors.npy --w pretrained_model --o calculated_DIC
``` -->

<!-- ## Evaluation of model performance -->

## References
1. Landschützer, P., Gruber, N., Bakker, D. C. E.: Decadal variations and trends of the global ocean carbon sink, Global Biogeochemical Cycles, 30, [doi:10.1002/2015GB005359](https://doi.org/10.1002/2015GB005359), 2016
2. Varvara E. Zemskova, Tai-Long He, Zirui Wan, Nicolas Grisouard: A deep-learning estimate of the decadal trends in the Southern Ocean carbon storage, preprint, (https://doi.org/10.31223/X52603), 2021.
