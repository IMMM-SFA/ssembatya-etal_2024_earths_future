
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10475841.svg)](https://doi.org/10.5281/zenodo.10475841) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10150609.svg)](https://doi.org/10.5281/zenodo.10150609)

# ssembatya-etal_2024_tbd

**The Dual Impacts of Space Heating Electrification and Climate Change Drive Uncertainties in Peak Load Behavior and 
Future Grid Reliability**

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
model, and a DC OPF model of the Texas grid. Results show that full adoption of more efficient heat pumps could 
significantly improve reliability, particularly under hotter futures. Less efficient heat pumps may result in more 
severe winter peaking events and increased reliability risks. As heating electrification intensifies, system planners 
will need to balance the potential for greater resource adequacy risk caused by shifts in seasonal peaking behavior 
alongside the benefits (improved efficiency and reductions in emissions).

## Journal reference
Ssembatya, H., J. D. Kern, K. Oikonomou, N. Voisin, C. D. Burleyson, and K. Z. Akdemir (2023). The dual impacts of 
space heating electrification and climate change drive uncertainties in peak load behavior and future grid reliability. 
Submitted to *TBD* - December 2023.

## Code reference
Ssembatya, H., J. D. Kern, K. Oikonomou, N. Voisin, C. D. Burleyson, and K. Z. Akdemir (2023). Supporting code for 
Ssembatya et al. 2023 - TBD [Code]. Zenodo. DOI TBD.

## Data references
### Input data
|       Dataset                                   |               Repository Link                        |               DOI                |
|:-----------------------------------------------:|:----------------------------------------------------:|:--------------------------------:|
|   White et al., 2021 model output               | https://data.mendeley.com/datasets/v8mt9d3v6h/1      | 10.17632/v8mt9d3v6h.1            |
|   Burleyson et al., 2023 Meteorology datasets   | https://www.osti.gov/biblio/1960530                  | https://doi.org/10.57931/1960530 |
|   ERCOT historical reported load                | https://www.ercot.com/gridinfo/load/load_hist        |                                  |

### Output data

|       Dataset                                           |   Repository Link                            |                   DOI                             |
|:-------------------------------------------------------:|---------------------------------------------:|:-------------------------------------------------:|
|     ML models load output & GO ERCOT simulation runs    | https://zenodo.org/records/10150610          | https://zenodo.org/doi/10.5281/zenodo.10150609    |


## Contributing modeling software
|  Model              | Version |         Repository Link          | DOI |
|:-------------------:|:-------:|:----------------------------------------------------------------:|:--------------------------------:|
| GCAM-USA            |  v5.3   | https://data.msdlive.org/records/r52tb-hez28                     | https://doi.org/10.57931/1960381 |
| GO Model framework  |         | https://iopscience.iop.org/article/10.1088/2753-3751/ad1751/meta |                                  |

## Reproduce my experiment
Clone this repository to get access to the scripts used in fitting the Machine Learning (ML) models, as well as predicting
residential and total load under different scenarios. Download the version of the GO ERCOT model version used in this experiment 
(https://zenodo.org/doi/10.5281/zenodo.10475841). The accompanying output data contains all the output datasets from these model
runs.



## Reproduce my figures
Use the following notebooks to reproduce the main and supplementary figures used in this publication.

| Figure Numbers |                Script Name                 |                                  Description                                   | 
|:--------------:|:------------------------------------------:|:------------------------------------------------------------------------------:|
|       2        |        difference_calculation.ipynb        |             Shows how the mean and peak differences are calculated             |
