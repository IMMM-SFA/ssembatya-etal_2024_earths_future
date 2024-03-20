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
matplotlib.rcParams['font.family'] = "arial"



#paths1 = ['rcp45cooler_ssp3_results', 'rcp45hotter_ssp3_results', 'rcp85cooler_ssp3_results', 'rcp85hotter_ssp3_results']
#name1 = ['RCP 4.5 Cooler SSP3', 'RCP 4.5 Hotter SSP3', 'RCP 8.5 Cooler SSP3', 'RCP 8.5 Hotter SSP3']
scenarios=['Base', 'Standard', 'High', "Ultra"]
scen =['base_', 'stdd_', 'high_', "ultra_"]

paths1=['rcp45cooler_ssp3_', 'rcp45hotter_ssp3_','rcp85cooler_ssp3_', 'rcp85hotter_ssp3_']
name1 = ['RCP 4.5 Cooler', 'RCP 4.5 Hotter', 'RCP 8.5 Cooler', 'RCP 8.5 Hotter']


paths2=['rcp45cooler_ssp5_', 'rcp45hotter_ssp5_','rcp85cooler_ssp5_', 'rcp85hotter_ssp5_']
name2 = ['RCP 4.5 Cooler SSP5', 'RCP 4.5 Hotter SSP5', 'RCP 8.5 Cooler SSP5', 'RCP 8.5 Hotter SSP5']

mycolors=[(1,0,0),(0,0,1)]

Year=np.linspace(2020,2099,80).astype(int)
 
lab =["summer" , "winter"]


'''FIG 1  SSP3'''

fig, axs = plt.subplots(4,4,figsize=(30,20),sharex=True, sharey=True)


for n in range(len(paths1)):
    df_slacksum_seas_s3_base= pd.read_csv( paths1[n]+ scen[0] + 'slacksum_seas.csv')
    df_slacksum_seas_s3_stdd= pd.read_csv( paths1[n]+ scen[1] + 'slacksum_seas.csv')
    df_slacksum_seas_s3_high= pd.read_csv( paths1[n]+ scen[2] + 'slacksum_seas.csv')
    df_slacksum_seas_s3_ultr= pd.read_csv( paths1[n]+ scen[3] + 'slacksum_seas.csv')
   # df_genmix.columns=scenarios
    
    
    if n == 0:
        
        #45coolerssp3
        #Base
        axs[0, n].stackplot(Year, df_slacksum_seas_s3_base.iloc[:,0], df_slacksum_seas_s3_base.iloc[:,1],  labels = lab, colors=mycolors)
        #Standard
        axs[1, n].stackplot(Year, df_slacksum_seas_s3_stdd.iloc[:,0], df_slacksum_seas_s3_stdd.iloc[:,1], labels = lab, colors=mycolors)
        #High
        axs[2, n].stackplot(Year, df_slacksum_seas_s3_high.iloc[:,0], df_slacksum_seas_s3_high.iloc[:,1], labels = lab, colors=mycolors)
        #Ultra
        axs[3, n].stackplot(Year, df_slacksum_seas_s3_ultr.iloc[:,0], df_slacksum_seas_s3_ultr.iloc[:,1], labels = lab, colors=mycolors)
        
        axs[n, 0].set_ylabel(scenarios[n],fontsize=25, fontweight="bold", labelpad=10)

        
        axs[n, 3].yaxis.set_label_position("right")
        
    elif n == 1:
        
        #45hotterssp3
        #Base
        axs[0, n].stackplot(Year, df_slacksum_seas_s3_base.iloc[:,0], df_slacksum_seas_s3_base.iloc[:,1],  labels = lab, colors=mycolors)
        #Standard
        axs[1, n].stackplot(Year, df_slacksum_seas_s3_stdd.iloc[:,0], df_slacksum_seas_s3_stdd.iloc[:,1], labels = lab, colors=mycolors)
        #High
        axs[2, n].stackplot(Year, df_slacksum_seas_s3_high.iloc[:,0], df_slacksum_seas_s3_high.iloc[:,1], labels = lab, colors=mycolors)
        #Ultra
        axs[3, n].stackplot(Year, df_slacksum_seas_s3_ultr.iloc[:,0], df_slacksum_seas_s3_ultr.iloc[:,1], labels = lab, colors=mycolors)
        
        axs[n, 0].set_ylabel(scenarios[n],fontsize=25, fontweight="bold", labelpad=10)

        
        
    elif n == 2:
        
        #85coolerssp3
        #Base
        axs[0, n].stackplot(Year, df_slacksum_seas_s3_base.iloc[:,0], df_slacksum_seas_s3_base.iloc[:,1],  labels = lab, colors=mycolors)
        #Standard
        axs[1, n].stackplot(Year, df_slacksum_seas_s3_stdd.iloc[:,0], df_slacksum_seas_s3_stdd.iloc[:,1], labels = lab, colors=mycolors)
        #High
        axs[2, n].stackplot(Year, df_slacksum_seas_s3_high.iloc[:,0], df_slacksum_seas_s3_high.iloc[:,1], labels = lab, colors=mycolors)
        #Ultra
        axs[3, n].stackplot(Year, df_slacksum_seas_s3_ultr.iloc[:,0], df_slacksum_seas_s3_ultr.iloc[:,1], labels = lab, colors=mycolors)
 
        axs[n, 0].set_ylabel(scenarios[n],fontsize=25, fontweight="bold", labelpad=10)

    elif n == 3:
        
        #85hotterssp3
        #Base
        axs[0, n].stackplot(Year, df_slacksum_seas_s3_base.iloc[:,0], df_slacksum_seas_s3_base.iloc[:,1],  labels = lab, colors=mycolors)
        #Standard
        axs[1, n].stackplot(Year, df_slacksum_seas_s3_stdd.iloc[:,0], df_slacksum_seas_s3_stdd.iloc[:,1], labels = lab, colors=mycolors)
        #High
        axs[2, n].stackplot(Year, df_slacksum_seas_s3_high.iloc[:,0], df_slacksum_seas_s3_high.iloc[:,1], labels = lab, colors=mycolors)
        #Ultra
        axs[3, n].stackplot(Year, df_slacksum_seas_s3_ultr.iloc[:,0], df_slacksum_seas_s3_ultr.iloc[:,1], labels = lab, colors=mycolors)
        
        axs[n, 0].set_ylabel(scenarios[n],fontsize=25, fontweight="bold", labelpad=10)
    

        axs[0, 0].set_title(name1[0],fontsize=25, fontweight="bold", pad=10)
        axs[0, 1].set_title(name1[1],fontsize=25, fontweight="bold", pad=10)   
        axs[0, 2].set_title(name1[2],fontsize=25, fontweight="bold", pad=10)
        axs[0, 3].set_title(name1[3],fontsize=25, fontweight="bold", pad=10)  
        
    axs[0, 0].legend( loc='upper left', fontsize=25)#,bbox_to_anchor=(1.03, 1))           
        
for ax in axs.flat:
    #ax.label_outer()
    #ax.grid(True)
    ax.set_ylim([0, 85000]) 
    
matplotlib.rcParams.update({'font.size': 25})    



plt.tight_layout(pad=0.3)       

#outer label
fig.text(-0.006, 0.5, 'Cumulative Loss of Load (MWh) By Technology Type', ha='center', va='center', rotation='vertical', fontsize = 32)
fig.text( 0.5, -0.005, 'Year', ha='center', va='center', fontsize = 35 )
plt.savefig('Figs/lossofload_SSP3_v25.jpg',dpi=1000, bbox_inches ="tight")

