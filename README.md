# ssembatya-etal_2023_tbd

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
|       Dataset       |               Repository Link                |               DOI                |
|:-------------------:|:--------------------------------------------:|:--------------------------------:|
|   GCAM-USA Output   | https://data.msdlive.org/records/43sy2-n8y47 | https://doi.org/10.57931/1989373 |

### Output data
The output of the TELL model is stored in the data repository linked below. The post-processed files (resulting from the
analysis scripts itemized below) are stored in the /data directory in this meta-repository.

|       Dataset       |                                Repository Link                                |                   DOI                   |
|:-------------------:|:-----------------------------------------------------------------------------:|:---------------------------------------:|
|     TELL Output     |                 https://data.msdlive.org/records/r0rvc-kjw89                  |    https://doi.org/10.57931/2228460     |
| Post-Processed Data | https://github.com/IMMM-SFA/burleyson-etal_2023_applied_energy/tree/main/data | https://doi.org/10.5281/zenodo.10278502 |


## Contributing modeling software
|  Model   | Version |         Repository Link          | DOI |
|:--------:|:-------:|:--------------------------------:|:---:|
| GCAM-USA |  v5.3   | https://data.msdlive.org/records/r52tb-hez28 | https://doi.org/10.57931/1960381 |


## Reproduce my experiment
Clone this repository to get access to the notebooks used to execute the TELL runs for this experiment. You'll also need 
to download the input files from the accompanying data repository (https://doi.org/10.57931/2228460) and the weather 
forcing data (https://doi.org/10.57931/1960530). Once you have the input datasets downloaded you can use the following 
notebooks to rerun the TELL model and produce the post-processed data used in this analysis. For the TELL runs you 
should only have to adjust the input directories to reflect the paths to wherever you choose to store the input files. 
The accompanying data repository already contains the output from the TELL model so you can skip rerunning the TELL 
model if you want to save time.

|                Script Name                 |                                Description                                 |
|:------------------------------------------:|:--------------------------------------------------------------------------:|
|        tell_runs_exp_group_b.ipynb         | Runs the TELL model based on the GCAM-USA outputs and TGW weather forcing  |


## Reproduce my figures
Use the following notebooks to reproduce the main and supplementary figures used in this publication.

| Figure Numbers |                Script Name                 |                                  Description                                   | 
|:--------------:|:------------------------------------------:|:------------------------------------------------------------------------------:|
|       2        |        difference_calculation.ipynb        |             Shows how the mean and peak differences are calculated             |
