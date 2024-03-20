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


'''GLOBAL'''
colors = {1 : (0.09019607843137255, 0.7450980392156863, 0.8117647058823529), 2 : (1,0,0) , 3 : (0.49,0.05,0.17) , 4 : (0,0,1)}


scenarios=['Base', 'Standard', 'High', "Ultra"]
matplotlib.rcParams['font.family'] = "arial"
colorname ={1 : "spring" , 2 : "summer" , 3 : "fall" , 4 : "winter"}


'''FIG 1 SSP3 PEAK RESIDENTIAL LOAD SEASON '''

paths1 = ['rcp45cooler_ssp3_results', 'rcp45hotter_ssp3_results', 'rcp85cooler_ssp3_results', 'rcp85hotter_ssp3_results']

name1 = ['rcp45cooler', 'rcp45hotter', 'rcp85cooler', 'rcp85hotter']

nrow=len(paths1); ncol=1; 
fig,axes = plt.subplots(nrows=nrow, ncols=ncol, sharex=True, sharey=False, figsize=(25,5)) #


cbar_ax = fig.add_axes([1.1, .6, .008, .3])


for n in range(len(paths1)):
    df_peak_seas = pd.read_csv(paths1[n]+'/rcp_peakseas_tot.csv')
    df_peak_seas = df_peak_seas.replace(['spring','summer','fall','winter'],[1, 2, 3, 4])
    print(pd.Series(df_peak_seas.values.ravel()).unique())
    
    seas_plot = pd.Series(df_peak_seas.values.ravel()).unique()
    
    seas_plotx=seas_plot.tolist()
    seas_plotx.sort()
    mycolor = []
    for s in seas_plotx:
        mycolor.append(colors[s])
    
    df_peak_seas.index=scenarios
    var=df_peak_seas
    ax = sns.heatmap(var, square=True, linewidth=1, ax=axes[n], cmap=mycolor, cbar_ax=cbar_ax, cbar_kws={'shrink': 0.5})

    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=17)
    cbar_ax.tick_params(labelsize=17)
    ax.set_ylabel(name1[n],fontsize=22, rotation='horizontal',fontweight='bold')
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_label_coords(1.06, .5)
    #ax.axes.get_xaxis().set_ticklabels([])
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)   
   


labelcolor=[]
for s in seas_plotx:
    labelcolor.append(colorname[s])
    
    cbar_ax.set_yticklabels(labelcolor)



p=cbar_ax.get_yticks()
p1= [1., 2., 3., 4.]
p2=[2. , 2.5, 3. , 3.5, 4. ]
p3=[1.8, 1.9, 2. , 2.1, 2.2]
if p1 in p:
    if len(colors) == 4:
        cbar_ax.set_yticks([1.3, 2.2, 3., 3.8])
elif p2 in p:
    cbar_ax.set_yticks(np.arange(p[1], 4,1))
elif p3 in p:
    cbar_ax.set_yticks(np.arange(p[2], 3,1))

    


fig.subplots_adjust(hspace=0.5) #, wspace=0.2


#fig.suptitle('Season for the Peak Hourly Annual Residential Load for Different Heating Scenarios :SSP3', fontsize=20, horizontalalignment='center')
fig.tight_layout()
plt.savefig('Figs_manuscript/'+'peaking_tot_ssp3_hr.jpg', dpi=1200)   
