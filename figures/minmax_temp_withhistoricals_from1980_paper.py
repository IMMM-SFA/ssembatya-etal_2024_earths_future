
import pandas as pd
import math
import numpy as np
import os
from shutil import copy
from pathlib import Path
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import itertools
from statistics import mean
import glob
matplotlib.rcParams['font.family'] = "arial"



Year1=np.linspace(2020,2099,80).astype(int)


#comparing max temp between hottest climate scenario (rcp85hotterssp3) and least extreme scenario (rcp45coolerssp3)
#SSP3
#RCP 85 hotter
tmp_peak_r85hotts3_base= pd.read_csv('../../runs_peak/peak_temperature/rcp85hotter_ssp3_wrf_CT_peak_tmp.csv')
#RCP 45 cooler
tmp_peak_r45cools3_base= pd.read_csv('../../runs_peak/peak_temperature/rcp45cooler_ssp3_wrf_CT_peak_tmp.csv')



'''JOINT'''

#ERCOT Historical 1980-2019
ercot_historical=pd.read_csv("ercot_historicalpktemps_19802019.csv")

matplotlib.rcParams.update({'font.size': 15})  
fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(12,8))
#historical max
ax1.plot(ercot_historical["year_name"] ,ercot_historical["max_tmp/C"] , color="black", label="historical")

#85hotter ssp3
ax1.plot(Year1, tmp_peak_r85hotts3_base['max_tmp'], label="rcp85hotter_ssp3", color="darkorange")
#45cooler ssp3
ax1.plot(Year1, tmp_peak_r45cools3_base['max_tmp'], label="rcp45cooler_ssp3", color="cyan")

ax1.set_title('Maximum Hourly Annual Temperature', fontsize=16,fontweight="bold")
ax1.set_ylabel('Temperature, ($^\circ$C)',fontsize=16)


#historical min
matplotlib.rcParams.update({'font.size': 15})  
ax2.plot(ercot_historical["year_name"] ,ercot_historical["min_tmp/C"] , color="black", label="historical")
#85hotter ssp3
ax2.plot(Year1, tmp_peak_r85hotts3_base['min_tmp'], label="rcp85hotter_ssp3", color="darkorange")
#45cooler ssp3
ax2.plot(Year1, tmp_peak_r45cools3_base['min_tmp'], label="rcp45cooler_ssp3", color="cyan")


ax2.set_title('Minimum Hourly Annual Temperature', fontsize=16,fontweight="bold")
ax2.set_ylabel('Temperature, ($^\circ$C)',fontsize=16)
#plt.ylabel("Temperature,C")
#ax1.legend(fontsize=16, bbox_to_anchor=(1.01, 0.99))
ax1.legend(loc='upper left', fontsize=16)
plt.xlabel("Year", fontsize=18)
plt.tight_layout()

plt.savefig('max_min_temp_withhistoricals1980_v2.jpg',dpi=1200 ,bbox_inches ="tight")