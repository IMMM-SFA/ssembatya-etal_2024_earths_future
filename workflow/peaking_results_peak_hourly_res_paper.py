# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

'''Importing variables'''
import pandas as pd
import math
import numpy as np
import os
from shutil import copy
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from statistics import mode
from numpy import mean

import glob

rcp=["rcp45cooler_ssp3", "rcp45cooler_ssp5", "rcp45hotter_ssp3", "rcp45hotter_ssp5",
       "rcp85cooler_ssp3", "rcp85cooler_ssp5", "rcp85hotter_ssp3", "rcp85hotter_ssp5"]
#rcp=["rcp45cooler_ssp3"]

model = ["_base", "_stdd", "_high", "_ultra" ]
#model = ["_base"


#tot meaning residential + non-residential

df_rcpall=pd.DataFrame()

for rr in rcp:
    
    
    rcp_peak_hourlyload=[]
    rcp_peak_hourlyloadseas=[]

    name=[]
    
    for mm in model:
        NN= str(rr) + str(mm) + ".csv"  
        rcpmodel=str(rr) + str(mm)
        
        rcp_season=pd.read_csv("../../season/"+str(rr)+"_season.csv")
        
        #reading the nonresidential load
        df_nonres=pd.read_csv("../runs/" + str(rr) + "_nonres.csv")
        #reading the residential load
        df_res = pd.read_csv("../runs/" + NN, header=0)
        
        name.append(rcpmodel)
        peak_hourlyloadseas_yr=[]
        peak_hourlyload_yr=[]
        
        all_permodel=[]
        

        #total load
        df_total_load=pd.DataFrame()

        df_all=df_res# + df_nonres
        
        #creating dataframe with 4 columns, each summing all values for 80 years
        df_rcpall[rcpmodel]=pd.Series(df_all.values.ravel('F'))
        
        #column names
        cols=list(df_all.columns)
                
        
        for i in range(len(df_all.columns)):
            
            #Total load
            df_total_load[cols[i]]=df_res[cols[i]] #+ df_nonres[cols[i]]
            
            #find the peak load
            peak_hourlyload=max(df_all.iloc[:,i])
            
            
            #Find the season that the peak load belongs to
            idx=df_all.iloc[:,i].idxmax()
            peak_hourlyloadseas=rcp_season.iloc[idx,i]
            
            
            #save the peak load and peak loas season
            peak_hourlyload_yr.append(peak_hourlyload)
            peak_hourlyloadseas_yr.append(peak_hourlyloadseas)
            
        rcp_peak_hourlyload.append(peak_hourlyload_yr)        
        rcp_peak_hourlyloadseas.append(peak_hourlyloadseas_yr)
        

        
        #export the total load
        df_total_load.to_csv("results/"+ rr + "_results/" + rcpmodel + '_load_res.csv', index=None)
        
        
        
    #saving rcp_peak_hourlyloadseas as dataframe
    df_rcp_peak_hourlyloadseas=pd.DataFrame(rcp_peak_hourlyloadseas)
    df_rcp_peak_hourlyloadseas.index=name
    df_rcp_peak_hourlyloadseas.columns=cols
    df_rcp_peak_hourlyloadseas.to_csv("results/"+ rr +"_results/" +  'rcp_peakseas_res.csv', index=None)



    #saving the rcp_peak_hourlyload as a dataframe
    df_rcp_peak_hourlyload=pd.DataFrame(rcp_peak_hourlyload)
    df_rcp_peak_hourlyload.index=name
    df_rcp_peak_hourlyload.columns=cols
    df_rcp_peak_hourlyload.to_csv("results/"+ rr +"_results/" +  'rcp_peakload_res.csv', index=None)


    #saving rcpall
    df_rcpall.to_csv('results/' + rr + "_results/" + 'rcpall_res.csv', index=None)




