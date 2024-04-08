
[![DOI]([https://zenodo.org/badge/DOI/10.5281/zenodo.10934193.svg])](https://doi.org/10.5281/10934193)
https://zenodo.org/badge/DOI/10.5281/zenodo.10934193.svg
# ssembatya-etal_2024_earths_future

**Dual Impacts of Space Heating Electrification and Climate Change Increase Uncertainties in Peak Load Behavior and Grid Capacity Requirements in Texas**

Henry Ssembatya<sup>1\*</sup>, Jordan D. Kern<sup>1</sup>, Konstantinos Oikonomou<sup>2</sup>, Nathalie Voisin
<sup>2</sup>, Casey D. Burleyson<sup>2</sup>, and Kerem Z. Akdemir<sup>1</sup>

<sup>1 </sup> North Carolina State University, Raleigh, NC, USA   
<sup>2 </sup> Pacific Northwest National Laboratory, Richland, WA, USA  

\* corresponding author: hssemba@ncsu.edu

## Abstract
Around 60% of households in Texas currently rely on electricity for space heating. As decarbonization efforts increase, 
more households could adopt electric heat pumps, significantly increasing winter electricity demand. Simultaneously, 
anthropogenic climate change is expected to increase temperatures, the potential for summer heat waves, and associated 
electricity demand for cooling. Uncertainty regarding the timing and magnitude of these concurrent changes raises 
questions about how they will jointly affect the seasonality of peak (highest) demand, firm capacity requirements, and 
grid reliability. This study investigates the net effects of residential space heating electrification and climate 
change on long-term demand patterns and load shedding potential, using climate change projections, a predictive load 
model, and a DC OPF model of the Texas grid. Results show that full electrification via adoption of more efficient heat pumps could 
significantly improve reliability, particularly under hotter futures. Less efficient heat pumps may result in more 
severe winter peaking events and increased reliability risks. As heating electrification intensifies, system planners 
will need to balance the potential for greater resource adequacy risk caused by shifts in seasonal peaking behavior 
alongside the benefits (improved efficiency and reductions in emissions).

## Journal reference
Ssembatya, H., J. D. Kern, K. Oikonomou, N. Voisin, C. D. Burleyson, and K. Z. Akdemir (2024). The dual impacts of 
space heating electrification and climate change drive uncertainties in peak load behavior and future grid reliability. 
In revision in Earth's Future - Mar 2024

## Code reference
Ssembatya, H., J. D. Kern, K. Oikonomou, N. Voisin, C. D. Burleyson, and K. Z. Akdemir (2024). Supporting code for 
Ssembatya et al. 2024 - Earth's Future [Code]. Zenodo. https://doi.org/10.5281/zenodo.10475841.

## Data references
### Input data
|       Dataset                                   |               Repository Link                        |               DOI                |
|:-----------------------------------------------:|:----------------------------------------------------:|:--------------------------------:|
|   White et al., 2021 model output               | https://data.mendeley.com/datasets/v8mt9d3v6h/1      | https://doi.org/10.17632/v8mt9d3v6h.1            |
| Jones et al., 2022 IM3/HyperFACETS Thermodynamic Global Warming (TGW) simulations | https://tgw-data.msdlive.org | https://doi.org/10.57931/1885756 |
|   Burleyson et al., 2023 meteorology datasets   | https://data.msdlive.org/records/cnsy6-0y610 | https://doi.org/10.57931/1960530 |
|   ERCOT historical reported load                | https://www.ercot.com/gridinfo/load/load_hist        |                   -               |

### Output data
|       Dataset                                           |   Repository Link                            |                   DOI                             |
|:-------------------------------------------------------:|---------------------------------------------:|:-------------------------------------------------:|
|     ML models load output & GO ERCOT simulations    | https://zenodo.org/records/10150610          | https://doi.org/10.5281/zenodo.10150609 |

## Contributing modeling software
|  Model              | Version |         Repository Link          | DOI |
|:-------------------:|:-------:|:----------------------------------------------------------------:|:--------------------------------:|
| Ssembatya et al., 2024 Grid Operations (GO) ERCOT model version used | v1.0.0 | https://zenodo.org/records/10475965 | https://doi.org/10.5281/zenodo.10475965 | 


## Reproduce my experiment
Clone this repository to get access to the scripts used in parameretizing the Machine Learning (ML) models used to predict
residential and total load under different scenarios. Download the version of the GO ERCOT model version used in this experiment 
(https://doi.org/10.5281/zenodo.10475841). The accompanying output data contains all the output datasets from these model
runs. Run the following scripts in the workflow directory to process the raw data used in this experiment:

| Script Number | Script Name | Purpose |
| --- | --- | --- |
| 1 | `texas_ht_pred_3_mlp_github.py` | Parameterize the ML model, generate datasets (predictions) of residential load under different scenarios and the non-residential load |
| 2 | `peaking_results_peak_hourly_total.py` | Combines the residential and non-residential load to obtain the total load datasets |

Run the following scripts for the GO ERCOT model.
| Script Number | Script Name | Purpose |
| --- | --- | --- |
| 1 | `reduced_network_data_allocation_hecc.py` | Create different subfolders (using a 150 node reduced topology) each containing a scenario year of the model parameterization |
| 2 | `ERCOTDataSetup.py` | Creates the ERCOT_data.dat file under each subfolder|
| 3 | `ERCOT_simple.py` | Runs the DC OPF model as an LP |


## Reproduce my figures
Use the following scripts to reproduce figures used in this publication.

| Figure Numbers |                Script Name                              |                                  Description                                               | 
|:--------------:|:-------------------------------------------------------:|:------------------------------------------------------------------------------------------:|
|       1        |     `minmax_temp_withhistoricals_from1980_paper.py`     |      Minimum and maximum hourly annual temperature under historical and climate scenarios  |
|       3        |     `nodal_topology_with_lines_ERCOT_paper.py`     |      Reduced topology framework of the selected GO ERCOT version showing nodes and transmission lines  |
|       4        |     `pk_seas_hrly_res_results_visuals_ssp3_paper.py`     |      Season of peak hourly residential load for all future scenario simulations  |
|       5        |     `pk_seas_hrly_tot_results_visuals_ssp3_paper.py`     |      Season of peak hourly total load for all future scenario simulations  |
|       6        |     `plot_year_examples_paper.py`     |      Comparing weather and load for two selected years  |
|       7        |     `peak_totload_ssp3_paper.py`     |      Peak hourly total load for all future scenario simulations  |
|       8a        |     `lol_distribution_rcp85hotterssp3base_paper.py`     |      Nodal location of loss of load on simulation day rcp85hotterssp3_base_3rd_aug_2091  |
|       8c        |     `lol_distribution_rcp45coolerssp3stdd_paper.py`     |      Nodal location of loss of load on simulation day rcp45coolerssp3_stdd_23rd_dec_2069  |
|       8b,d        |     `ercot_temperature_maps_paper.ipynb`     |      Max and min hourly temperature distribution on selected simulation days  |
|       9        |     `lol_visuals_manuscript_SSP3_paper.py`     |      Cumulative loss of load for all scenarios  |

