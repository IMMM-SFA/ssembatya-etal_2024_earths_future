# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 19:29:11 2024

"""

# For Henry Plotting

import pandas as pd
import math
import numpy as np
import sklearn
import os
from shutil import copy
from pathlib import Path
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import itertools
from statistics import mode
import statsmodels.api as sm
from numpy import mean
import datetime
from datetime import datetime, timedelta
import glob
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

matplotlib.rcParams['font.family'] = "arial"
from pyhelpers.store import save_fig

#import 2016 weather data
df_wrfercot_2016=pd.read_csv("ERCO_WRF_Hourly_Mean_Meteorology_2016.csv")
#Convert the time from UTC to CT
df_wrfercot_2016=df_wrfercot_2016[6:].reset_index()
df_wrfercot_2016['Time_UTC']= pd.to_datetime(df_wrfercot_2016['Time_UTC'], errors='coerce')
df_wrfercot_2016["Time_CT"]=df_wrfercot_2016["Time_UTC"]-timedelta(hours=5)

#Import 2016 Residential dd base case
df_ddbase=pd.read_csv("ercot_2016_hourly_consumption_base.csv")

#Import 2016 Residential dd standard efficiency
df_ddstandd=pd.read_csv("ercot_2016_hourly_consumption_standard_efficiency.csv")

#Import 2016 Residential dd high efficiency
df_ddhi=pd.read_csv("ercot_2016_hourly_consumption_high_efficiency.csv")

#Import 2016 Residential dd ultra high efficiency
df_ddultra=pd.read_csv("ercot_2016_hourly_consumption_ultra_high_efficiency.csv")

#Import 2016 Non- Residential load
df_2016_nonres=pd.read_csv("non_res_load_16x.csv")

df_2016_tot=pd.DataFrame()
df_2016_tot["Base"]=df_ddbase["total_site_electricity_kwh"]/1000 + df_2016_nonres["NonResLoad_mwh"]
df_2016_tot["Stdd"]=df_ddstandd["total_site_electricity_kwh"]/1000 + df_2016_nonres["NonResLoad_mwh"]
df_2016_tot["High"]=df_ddhi["total_site_electricity_kwh"]/1000 + df_2016_nonres["NonResLoad_mwh"]
df_2016_tot["Ultra"]=df_ddultra["total_site_electricity_kwh"]/1000 + df_2016_nonres["NonResLoad_mwh"]
df_2016_tot=df_2016_tot[:8778]

#future

#rcp 4.5 hotter ssp3 2058
#weather
dfrcp45hotterss32058_wrf=pd.read_csv("ERCO_WRF_Hourly_Mean_Meteorology_2058.csv")
dfrcp45hotterss32058_wrf['Time_CT']= pd.to_datetime(dfrcp45hotterss32058_wrf['Time_CT'], errors='coerce')

#predictions of total load in rcp45hotterssp3 2058
rcp45hotterss32058_base=pd.read_csv("rcp45hotter_ssp3_base_load_tot.csv")
rcp45hotterss32058_stdd=pd.read_csv("rcp45hotter_ssp3_stdd_load_tot.csv")
rcp45hotterss32058_high=pd.read_csv("rcp45hotter_ssp3_high_load_tot.csv")
rcp45hotterss32058_ultra=pd.read_csv("rcp45hotter_ssp3_ultra_load_tot.csv")
rcp45hotterss32058=pd.DataFrame()
rcp45hotterss32058["Base"]=rcp45hotterss32058_base['2058']
rcp45hotterss32058["Stdd"]=rcp45hotterss32058_stdd['2058']
rcp45hotterss32058["High"]=rcp45hotterss32058_high['2058']
rcp45hotterss32058["Ultra"]=rcp45hotterss32058_ultra['2058']
rcp45hotterss32058=rcp45hotterss32058[:8760]

fig, axs1 = plt.subplots(2,2,figsize=(35,20), sharex='col', sharey='row',
                         gridspec_kw={
                           'width_ratios': [2, 2],
                           'height_ratios': [2, 3]
                                     })
                           


# ------Added Section)------------------>>>>>>>>------------------------------------------
# ---------------------------------------------->>>>>>>>------------------------------------------
# Generate winter blue and summer red shaded sections
winter_months = ['January', 'February', 'December']
summer_months = ['June', 'July', 'August']
# ------------------------>>>>>>>>------------------------------------------
# ---------------------------------------------->>>>>>>>------------------------------------------

paths1 =[2016 , 2058]

scenarios=['Base', 'Standard', 'High', "Ultra"]
matplotlib.rcParams.update({'font.size': 30})

# ------Added Section------------------>>>>>>>>------------------------------------------
# ---------------------------------------------->>>>>>>>------------------------------------------
# Define shaded regions for winter and summer
for month in winter_months:
    axs1[0, 0].axvspan(pd.Timestamp(month + ' 2016'), pd.Timestamp(month + ' 2016') + pd.DateOffset(months=1), color='deepskyblue', alpha=0.3, linewidth=0)
    axs1[1, 0].axvspan(pd.Timestamp(month + ' 2016'), pd.Timestamp(month + ' 2016') + pd.DateOffset(months=1), color='deepskyblue', alpha=0.3, linewidth=0)
for month in summer_months:
    axs1[0, 0].axvspan(pd.Timestamp(month + ' 2016'), pd.Timestamp(month + ' 2016') + pd.DateOffset(months=1), color='violet', alpha=0.3, linewidth=0)
    axs1[1, 0].axvspan(pd.Timestamp(month + ' 2016'), pd.Timestamp(month + ' 2016') + pd.DateOffset(months=1), color='violet', alpha=0.3, linewidth=0)

for month in winter_months:
    axs1[0, 1].axvspan(pd.Timestamp(month + ' 2058'), pd.Timestamp(month + ' 2058') + pd.DateOffset(months=1), color='deepskyblue', alpha=0.3, linewidth=0)
    axs1[1, 1].axvspan(pd.Timestamp(month + ' 2058'), pd.Timestamp(month + ' 2058') + pd.DateOffset(months=1), color='deepskyblue', alpha=0.3, linewidth=0)
for month in summer_months:
    axs1[0, 1].axvspan(pd.Timestamp(month + ' 2058'), pd.Timestamp(month + ' 2058') + pd.DateOffset(months=1), color='violet', alpha=0.3, linewidth=0)
    axs1[1, 1].axvspan(pd.Timestamp(month + ' 2058'), pd.Timestamp(month + ' 2058') + pd.DateOffset(months=1), color='violet', alpha=0.3, linewidth=0)

# ------------------------>>>>>>>>------------------------------------------
# ---------------------------------------------->>>>>>>>------------------------------------------
for n in range(len(paths1)):
    
    if n == 0:
        
        #Historical (2016) weather (top left)
        axs1[0, n].plot(df_wrfercot_2016["Time_CT"], df_wrfercot_2016["T2"]-273.15, color= "black")
        #axs00 = axs1[0, n].twinx()
        #axs00.plot(df_wrfercot_2016["Time_CT"], df_2016_tot["Base"], color= "red", alpha=0.8)
        axs1[0, n].set_ylim([-10, 45])
        
        #second row first column, historical load trend
        #Base
        axs1[1,n].plot(df_wrfercot_2016["Time_CT"], df_2016_tot["Base"], label="Base: Summer", linewidth=2, alpha=1, color='#0072B2')
        #Standard
        axs1[1, n].plot(df_wrfercot_2016["Time_CT"], df_2016_tot["Stdd"], label="Standard: Winter", linewidth=2, alpha=1, color="#D55E00")
        #High
        axs1[1, n].plot(df_wrfercot_2016["Time_CT"], df_2016_tot["High"], label="High: Winter", linewidth=2,alpha=1, color="#009E73")
        #Ultra
        axs1[1, n].plot(df_wrfercot_2016["Time_CT"], df_2016_tot["Ultra"], label="Ultra: Winter", linewidth=2,alpha=1,  color="#F0E442"  )
        
        #axs1[1, n].legend( loc="upper left", ncol=4, framealpha=0.4 ,fontsize=20,  markerscale=100)#,, labelcolor='linecolor' labelcolor='linecolor', bbox_to_anchor=(0.01, 1.1 )) #,  bbox_to_anchor=(0.1, 0.9))
        
        axs1[1, n].set_ylim([20000, 85000])
        
        axs1[0, 0].set_title("Historical Climate (2016)",fontsize=30, fontweight="bold")
        #axs1[0, 1].set_title("Future Climate (rcp45hotter_ssp3 2058)",fontsize=30, fontweight="bold")
        axs1[0, 1].set_title("Future Climate (rcp45hotter 2058)",fontsize=30, fontweight="bold")
        
        
        axs1[0, 0].set_ylabel("Temperature ($^\circ$C)",fontsize=30,  fontweight="bold")##ad=40)
        axs1[1, 0].set_ylabel('Load (Mwh)',fontsize=30,  fontweight="bold")#, labelpad=40)

       
                               
    elif n == 1:
        
        #future rcp45hotterssp3 top right
        axs1[0, n].plot(dfrcp45hotterss32058_wrf['Time_CT'], dfrcp45hotterss32058_wrf['T3'], color= "black")
        axs1[0, n].set_ylim([-10, 45])

        
        #second row second column, rcp45hotterssp3
        #base
        axs1[1, n].plot(dfrcp45hotterss32058_wrf['Time_CT'], rcp45hotterss32058["Base"], label="Base: Summer", linewidth=2,alpha=1, color='#0072B2')
        #standard
        axs1[1, n].plot(dfrcp45hotterss32058_wrf['Time_CT'], rcp45hotterss32058["Stdd"], label="Standard: Winter", linewidth=2,alpha=1, color="#D55E00")
        #High
        axs1[1, n].plot(dfrcp45hotterss32058_wrf['Time_CT'], rcp45hotterss32058["High"], label="High: Winter", linewidth=2,alpha=1, color="#009E73" )
        #Ultra
        axs1[1, n].plot(dfrcp45hotterss32058_wrf['Time_CT'], rcp45hotterss32058["Ultra"], label="Ultra: Summer", linewidth=2,alpha=1, color="#F0E442")
        
        axs1[1, n].set_ylim([20000, 85000])
        
        
plt.subplots_adjust(hspace=0.1, wspace=0.01)#,bottom=0.1)#       
matplotlib.rcParams.update({'font.size': 25}) 


#CUSTOMIZED LEGEND        
# I created 6 proxy members to generate the legend
#historical and future base
legend_seas_hfb, = plt.plot( np.NaN, np.NaN,  color= '#0072B2', linewidth=5, linestyle = 'solid', label='Base: Summer' )
#historical and future standard
legend_seas_hfs, = plt.plot( np.NaN, np.NaN,  color='#D55E00', linewidth=5, linestyle = 'solid', label='Standard: Winter' )
#historical and future high
legend_seas_hfh, = plt.plot( np.NaN, np.NaN, color='#009E73', linewidth=5, linestyle = 'solid', label='High: Winter' )
#historical ultra
legend_seas_hu, = plt.plot( np.NaN, np.NaN,  color='#F0E442', linewidth=5, linestyle = 'solid', label='Ultra: Winter' )
#future ultra
legend_seas_fu, = plt.plot( np.NaN, np.NaN, color='#F0E442', linewidth=5, linestyle = 'solid', label='Ultra: Summer' )

#legend for historical
axs1[1, 0].legend(handles=[legend_seas_hfb, legend_seas_hfs, legend_seas_hfh, legend_seas_hu],
           loc='upper left', columnspacing=2, handlelength=2, handletextpad=.8, ncol=4, frameon=True,  fontsize=20) #columnspacing=.9 handlelength=1, bbox_to_anchor=(0, 1),ncol=4, 
   
#legend for future
axs1[1, 1].legend(handles=[legend_seas_hfb, legend_seas_hfs, legend_seas_hfh, legend_seas_fu],
           loc='upper left', columnspacing=2, handlelength=2, handletextpad=.8, ncol=4, frameon=True,  fontsize=20) #columnspacing=.9 handlelength=1, bbox_to_anchor=(0, 1),ncol=4,

# ------Added Section----------------->>>>>>>>------------------------------------------
# ---------------------------------------------->>>>>>>>------------------------------------------
# CUSTOMIZED LEGEND        
# Create proxy artists for winter and summer sections
winter_proxy = plt.Rectangle((0, 0), 1, 1, fc="deepskyblue", alpha=0.3, label='Winter Period')
summer_proxy = plt.Rectangle((0, 0), 1, 1, fc="violet", alpha=0.3, label='Summer Period')

# Add proxy artists to the legend along with other legend entries
axs1[0, 0].legend(handles=[winter_proxy, summer_proxy], ncol=1, fontsize=25, bbox_to_anchor=[0.06, 1.06]) 

# ------------------------>>>>>>>>------------------------------------------
# ---------------------------------------------->>>>>>>>------------------------------------------

fig.subplots_adjust(hspace=0.1, wspace=0.1)
fig.text( 0.5, 0.06,  'Year', ha='center', va='center', fontsize = 35)


#final visuals
# ------------------------>>>>>>>>------------------------------------------
# ---------------------------------------------->>>>>>>>------------------------------------------

plt.savefig("fig6_two_year_comparison_jpg.jpg", dpi=1100, bbox_inches ="tight")

