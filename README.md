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
The SOC model takes vectors of 10 predictors mentioned above and outputs DIC profiles at 48 levels (from surface to 4 km depth as defined in B-SOSE). 
Thus the shape of input vectors is (num_profiles, 1, 10) and the output shape is (num_profiles, 1, 48).
The scripts used to generate data pairs are in scripts/data_pair_generator/ folder.

### Train 
To train a SOC model:
```
python train_model.py --x list_of_predictor_files --y list_of_DIC_files --lr learning_rate --b batch_size --o output_model_name
```
As an example, you can try to train a SOC model using the Argo floats measurements with:
```
python train_model.py --x example_data/Argo/train/*_predictors.npy --y example_data/Argo/train/*_DIC.npy --o SOC_model_Argo.h5
```

### Test
To use a pretrained model to generate DIC predictions:
```
python test_model.py --x list_of_predictor_files --w name_of_pretrained_model
```
As an example, you can use a pretrained model to generate predictions with:
```
python test_model.py --x example_data/Argo/test/2019_Argo_predictors.npy --w SOC_model_Argo.h5
```

### DIC calculation from 1998 to 2019
To calculate DIC concentrations from 1998 to 2019 using the predictors regridded to 1<sup>°</sup> by 1<sup>°</sup> resolution, you can do something similar:
```
python test_model.py --x 1998_2019_predictors/*_predictors.npy --w pretrained_model --o calculated_DIC
```

## Evaluation of model performance

## References
1. Landschützer, P., Gruber, N., Bakker, D. C. E.: Decadal variations and trends of the global ocean carbon sink, Global Biogeochemical Cycles, 30, [doi:10.1002/2015GB005359](https://doi.org/10.1002/2015GB005359), 2016
