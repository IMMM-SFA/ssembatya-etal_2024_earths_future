# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 18:08:31 2022

@author: hssemba
"""


'''Importing variables'''
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
from matplotlib.pyplot import *
import glob
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
matplotlib.rcParams['font.family'] = "arial"

'''FIG 1 SSP3 PEAK LOAD '''



scenarios=['Base', 'Standard', 'High', "Ultra"]
paths1=['rcp45cooler_ssp3_results', 'rcp45hotter_ssp3_results','rcp85cooler_ssp3_results', 'rcp85hotter_ssp3_results']
name1 = ['RCP 4.5 Cooler', 'RCP 4.5 Hotter', 'RCP 8.5 Cooler', 'RCP 8.5 Hotter']



#colorname ={"cyan" : "spring" , "red" : "summer" , "darkred" : "fall" , "blue" : "winter"}
colorname =["spring" , "summer" ,"fall" , "winter"]

Year1=np.linspace(2020,2099,80).astype(int)
 


fig, axs = plt.subplots(4,4,figsize=(30,20))#,sharex=True, sharey=True )

#fig.suptitle('Plotting Peak Total Load SSP3 and SSP5, Marked by Season', fontsize=25, y=1.0001)
plt.subplots_adjust(top=0.6)

for n in range(len(paths1)):
    #SSP3
    df_peakload_ssp3= pd.read_csv("../pk_hrly_results_peak_hourly_total/results/" + paths1[n]+'/rcp_peakload_tot.csv').transpose().reset_index()
    df_peakload_ssp3=df_peakload_ssp3.iloc[:,1:5]
    df_peakload_ssp3.columns=scenarios
    
    df_peakseas_ssp3 = pd.read_csv("../pk_hrly_results_peak_hourly_total/results/" + paths1[n]+'/rcp_peakseas_tot.csv').transpose().reset_index()
    df_peakseas_ssp3 = df_peakseas_ssp3.replace(['spring','summer','fall','winter'],["cyan", "red", "darkred", "blue"])
    df_peakseas_ssp3= df_peakseas_ssp3.iloc[:,1:5]
    df_peakseas_ssp3 .columns=scenarios
    
    #SSP5
    df_peakload_ssp5= pd.read_csv("../pk_hrly_results_peak_hourly_total/results/" + paths2[n]+'/rcp_peakload_tot.csv').transpose().reset_index()
    df_peakload_ssp5=df_peakload_ssp5.iloc[:,1:5]
    df_peakload_ssp5.columns=scenarios
    
    df_peakseas_ssp5 = pd.read_csv("../pk_hrly_results_peak_hourly_total/results/" + paths2[n]+'/rcp_peakseas_tot.csv').transpose().reset_index()
    df_peakseas_ssp5 = df_peakseas_ssp5.replace(['spring','summer','fall','winter'],["cyan", "red", "darkred", "blue"])
    df_peakseas_ssp5= df_peakseas_ssp5.iloc[:,1:5]
    df_peakseas_ssp5 .columns=scenarios
       
    if n == 0:
        axs[0, n].scatter(Year1, df_peakload_ssp3.iloc[:,0], c=df_peakseas_ssp3.iloc[:,0], marker="." ,s=150, label=colorname)
        #axs[0, n].scatter(Year1, df_peakload_ssp5.iloc[:,0], c=df_peakseas_ssp5.iloc[:,0], marker="x", s=150)
    
        axs[1, n].scatter(Year1, df_peakload_ssp3.iloc[:,1], c=df_peakseas_ssp3.iloc[:,1], marker=".", s=150, label=colorname)
        #axs[1, n].scatter(Year1, df_peakload_ssp5.iloc[:,1], c=df_peakseas_ssp5.iloc[:,1], marker="x", s=150 )

        axs[2, n].scatter(Year1, df_peakload_ssp3.iloc[:,2], c=df_peakseas_ssp3.iloc[:,2], marker=".", s=150, label=colorname)
        #axs[2, n].scatter(Year1, df_peakload_ssp5.iloc[:,2], c=df_peakseas_ssp5.iloc[:,2], marker="x" ,s=150)
        
        axs[3, n].scatter(Year1, df_peakload_ssp3.iloc[:,3], c=df_peakseas_ssp3.iloc[:,3], marker=".",  s=150, label=colorname)
        #axs[3, n].scatter(Year1, df_peakload_ssp5.iloc[:,3], c=df_peakseas_ssp5.iloc[:,3], marker="x" , s=150 )
        
        axs[n, 0].set_ylabel(scenarios[n],fontsize=30, fontweight="bold", labelpad=10)
        
        axs[n, 3].yaxis.set_label_position("right")
        
        
        #axs[0, n].legend( loc='upper left', fontsize=15)    
        
    elif n==1:
        axs[0, n].scatter(Year1, df_peakload_ssp3.iloc[:,0], c=df_peakseas_ssp3.iloc[:,0], marker=".", s=150, label=colorname)
        #axs[0, n].scatter(Year1, df_peakload_ssp5.iloc[:,0], c=df_peakseas_ssp5.iloc[:,0], marker="x",  s=150)
    
        axs[1, n].scatter(Year1, df_peakload_ssp3.iloc[:,1], c=df_peakseas_ssp3.iloc[:,1], marker=".",  s=150,label=colorname)
        #axs[1, n].scatter(Year1, df_peakload_ssp5.iloc[:,1], c=df_peakseas_ssp5.iloc[:,1], marker="x",  s=150)

        axs[2, n].scatter(Year1, df_peakload_ssp3.iloc[:,2], c=df_peakseas_ssp3.iloc[:,2], marker=".",  s=150,label=colorname)
        #axs[2, n].scatter(Year1, df_peakload_ssp5.iloc[:,2], c=df_peakseas_ssp5.iloc[:,2], marker="x",  s=150)
        
        axs[3, n].scatter(Year1, df_peakload_ssp3.iloc[:,3], c=df_peakseas_ssp3.iloc[:,3], marker=".",  s=150,label=colorname)
        #axs[3, n].scatter(Year1, df_peakload_ssp5.iloc[:,3], c=df_peakseas_ssp5.iloc[:,3], marker="x",  s=150)
        
        axs[n, 0].set_ylabel(scenarios[n],fontsize=30, fontweight="bold", labelpad=10)
        
    elif n==2:
        axs[0, n].scatter(Year1, df_peakload_ssp3.iloc[:,0], c=df_peakseas_ssp3.iloc[:,0], marker="." ,  s=150,label=colorname)
        #axs[0, n].scatter(Year1, df_peakload_ssp5.iloc[:,0], c=df_peakseas_ssp5.iloc[:,0], marker="x",  s=150)
    
        axs[1, n].scatter(Year1, df_peakload_ssp3.iloc[:,1], c=df_peakseas_ssp3.iloc[:,1], marker=".",  s=150,label=colorname)
        #axs[1, n].scatter(Year1, df_peakload_ssp5.iloc[:,1], c=df_peakseas_ssp5.iloc[:,1], marker="x",  s=150)

        axs[2, n].scatter(Year1, df_peakload_ssp3.iloc[:,2], c=df_peakseas_ssp3.iloc[:,2], marker=".",  s=150,label=colorname)
        #axs[2, n].scatter(Year1, df_peakload_ssp5.iloc[:,2], c=df_peakseas_ssp5.iloc[:,2], marker="x",  s=150)
        
        axs[3, n].scatter(Year1, df_peakload_ssp3.iloc[:,3], c=df_peakseas_ssp3.iloc[:,3], marker=".",  s=150,label=colorname)
        #axs[3, n].scatter(Year1, df_peakload_ssp5.iloc[:,3], c=df_peakseas_ssp5.iloc[:,3] ,  s=150,marker="x")
        
        axs[n, 0].set_ylabel(scenarios[n],fontsize=30, fontweight="bold", labelpad=10)
        
    elif n==3:
        axs[0, n].scatter(Year1, df_peakload_ssp3.iloc[:,0], c=df_peakseas_ssp3.iloc[:,0], marker=".",  s=150, label=colorname)
        #axs[0, n].scatter(Year1, df_peakload_ssp5.iloc[:,0], c=df_peakseas_ssp5.iloc[:,0], marker="x",  s=150)
    
        axs[1, n].scatter(Year1, df_peakload_ssp3.iloc[:,1], c=df_peakseas_ssp3.iloc[:,1], marker="." ,  s=150,label=colorname)
        #axs[1, n].scatter(Year1, df_peakload_ssp5.iloc[:,1], c=df_peakseas_ssp5.iloc[:,1], marker="x",  s=150)

        axs[2, n].scatter(Year1, df_peakload_ssp3.iloc[:,2], c=df_peakseas_ssp3.iloc[:,2], marker=".",  s=150,label=colorname)
        #axs[2, n].scatter(Year1, df_peakload_ssp5.iloc[:,2], c=df_peakseas_ssp5.iloc[:,2], marker="x",  s=150)
        
        axs[3, n].scatter(Year1, df_peakload_ssp3.iloc[:,3], c=df_peakseas_ssp3.iloc[:,3], marker=".",  s=150,label=colorname)
        #axs[3, n].scatter(Year1, df_peakload_ssp5.iloc[:,3], c=df_peakseas_ssp5.iloc[:,3], marker="x",  s=150)
        
        axs[n, 0].set_ylabel(scenarios[n],fontsize=30, fontweight="bold", labelpad=10)
        

        axs[0, 0].set_title(name1[0],fontsize=30, fontweight="bold", pad=10)
        axs[0, 1].set_title(name1[1],fontsize=30, fontweight="bold", pad=10)   
        axs[0, 2].set_title(name1[2],fontsize=30, fontweight="bold", pad=10)
        axs[0, 3].set_title(name1[3],fontsize=30, fontweight="bold", pad=10) 
        
        #axs[0, 3].legend( loc='lower right', fontsize=15)
        #axs[0, 3].legend(["spring" , "summer" ,"fall" , "winter"],loc='lower right', fontsize=15) 
plt.subplots_adjust(hspace=0.1, wspace=0.01)#,bottom=0.1)#       
matplotlib.rcParams.update({'font.size': 25}) 
#CUSTOMIZED LEGEND        
# I created 6 proxy members to generate the legend
legend_seas_1, = plt.plot( np.NaN, np.NaN, marker='s', color='cyan', linestyle = 'None', label='spring' )
legend_seas_2, = plt.plot( np.NaN, np.NaN, marker='s', color='red', linestyle = 'None', label='summer' )
legend_seas_3, = plt.plot( np.NaN, np.NaN, marker='s', color='darkred', linestyle = 'None', label='fall' )
legend_seas_4, = plt.plot( np.NaN, np.NaN, marker='s', color='blue', linestyle = 'None', label='winter' )


axs[0, 0].legend(handles=[legend_seas_1,legend_seas_2,legend_seas_3,legend_seas_4],
           loc='upper left',columnspacing=.9, handlelength=1, handletextpad=.2, ncol=4, frameon=True, bbox_to_anchor=(0, 1), fontsize=23)
        
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

    ax.set_ylim([60000, 95000]) 
    ax.yaxis.set_major_locator(MultipleLocator(10000))
    

fig.text(-0.006, 0.5, 'Peak Load (MWh) By Technology Type', ha='center', va='center', rotation='vertical',  fontsize = 35)
fig.text( 0.5, -0.005, 'Year', ha='center', va='center', fontsize = 35 )
plt.tight_layout(pad=1) 
plt.savefig('peak_load_ssp3.png',dpi=200, bbox_inches ="tight")


