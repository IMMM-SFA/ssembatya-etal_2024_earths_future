import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon
import seaborn as sns
import matplotlib
import matplotlib as mpl
import networkx as nx
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pyproj import CRS

from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.rcParams['font.family'] = "arial"

#rcp= [ "rcp45cooler_ssp3_" , "rcp45cooler_ssp5_" , "rcp45hotter_ssp3_", "rcp45hotter_ssp5_",
#      "rcp85cooler_ssp3_" , "rcp85cooler_ssp5_" , "rcp85hotter_ssp3_", "rcp85hotter_ssp5_"]

# rcph= ["rcp85hotter_ssp5_"]
# #year
# #year=np.arange(2020, 2100, 1,dtype=int)
# yearh=np.arange(2091, 2092, 1,dtype=int)
# #model type
# #model= ["base" , "stdd", "high", "ultra"]
# modelh=["base"]
# #simple or coal
# UC = '_simple'



#reading in selected nodes
all_selected_nodes = pd.read_csv(r'E:\phd_work\residential_heating_ml_071223_v25\rcpuced_v25\selected_nodes_150.csv',header=0)
all_selected_nodes= [*all_selected_nodes['SelectedNodes']]
#all_selected_nodes_string = ['bus_{}'.format(a) for a in all_selected_nodes_number]


#Set the lat/lon bounds for the plot:
lat_min = 25
lat_max = 38
lon_min = -108
lon_max = -92

#Projection
projection= 4269
#Defining hours and day of year
#hours = pd.date_range(start='01-01-2091 00:00:00', end='12-31-2091 23:00:00', freq='H')

#Reading all necessary files for plotting the map
df = pd.read_csv(r'E:\phd_work\residential_heating_ml_071223_v25\rcpuced_v25\ERCOT_Bus.csv',header=0)
#crs = {'init':'epsg:4326'}
crs = CRS('epsg:4269')
# crs = {"init": "epsg:2163"}
geometry = [Point(xy) for xy in zip(df['Substation Longitude'],df['Substation Latitude'])]
#filter_nodes = gpd.GeoDataFrame(df,crs=crs,geometry=geometry)
nodes_df = gpd.GeoDataFrame(df,crs=crs,geometry=geometry)
nodes_df = nodes_df.to_crs(epsg= projection)


states_gdf = gpd.read_file(r'E:\phd_work\residential_heating_ml_071223_v25\rcpuced_v25\geo_export_9ef76f60-e019-451c-be6b-5a879a5e7c07.shp')
states_gdf = states_gdf.to_crs(epsg= projection)




# '''HOT SEASON SLACK DISTRIBUTION'''

# #fig,ax = plt.subplots(1,1,figsize=(10,5))

# for rr in rcph:

#     for mm in modelh:
        
#         for yy in yearh:
      
#             sl= "../../rcpall_output_v23/slack_Exp" + rr + str(mm) + config + str(yy) +".csv"
                
#             #Importing the slack generation for that year and scenario
#             df_slack=pd.read_csv(sl)
#             df_slack["Node_num"]=df_slack.Node.str[4:].astype(int)
                
#             #line_threshold = 100

#             lines_df = pd.read_csv("../../rcpall_output_v23/line_params_Exp" + rr + str(mm) + config + str(yy) +".csv")
#             #lines_df_filt = lines_df.loc[lines_df['limit']>=line_threshold].copy()
#             #lines_df_filt.reset_index(drop=True,inplace=True)

#             all_line_nodes = []

#             for a in range(len(lines_df)):
#                 line_name = lines_df.loc[a,'line']
#                 splitted_name = line_name.split('_')
#                 line_list = [int(splitted_name[1]),int(splitted_name[2])]
#                 all_line_nodes.append(line_list)
            
#             #slacksum_hrly=df_slack.groupby(['Time'], sort=False).agg({'Value':'sum'}).reset_index()
#             #slack sum by time(daily)
#             #slacksum_daily =slacksum_hrly.groupby(slacksum_hrly.index // 24).sum() 
                
#             #Find the selected day
#             df_maxslackday_hrly=df_slack[(df_slack["Time"] >= 5137) & (df_slack["Time"] <= 5160)]
#             df_maxslackday_sumbus=df_maxslackday_hrly.groupby(['Node_num'], sort=False).agg({'Value':'sum'}).reset_index()
            
#             #subsetting geodataframe to selected nodes
#             Selected_topology_nodes = nodes_df[nodes_df['Number'].isin(all_selected_nodes)].copy()
#             Selected_topology_nodes.reset_index(drop=True,inplace=True)
            
            
            
#             allnodes= list(Selected_topology_nodes['Number'])
#             slack_sum=[]    
#             for i in allnodes:
#                 slk_sm= df_maxslackday_sumbus.loc[df_maxslackday_sumbus["Node_num"]==i,'Value']
#                 slk_sm = slk_sm.reset_index(drop=True)
#                 slack_sum.append(slk_sm[0])
            
#             Selected_topology_nodes['sum_slack'] = slack_sum
#             #subset to where the slacksum is greater than 0
#             #Selected_topology_nodes=Selected_topology_nodes[Selected_topology_nodes['sum_slack'] >0]
            
            
            
            
#             #PLOTTING
#             colorbar_min=Selected_topology_nodes['sum_slack'].min()
#             colorbar_max=Selected_topology_nodes['sum_slack'].max()
            
            
#             fig,ax = plt.subplots()
#             divider = make_axes_locatable(ax)
            
#             #cax = divider.append_axes("right", size="1%", pad=0.01)
#             states_gdf.plot(ax=ax,color='white',edgecolor='black',linewidth=0.4)
#             #Selected_topology_nodes.plot(ax=ax,color = 'li',alpha=1)
            
#             G_all_lines = nx.Graph()
#             for i in all_selected_nodes:
    
#                 my_pos_1 = nodes_df.loc[nodes_df['Number']==i].geometry.x.values[0]
#                 my_pos_2 = nodes_df.loc[nodes_df['Number']==i].geometry.y.values[0]
    
#                 G_all_lines.add_node(i,pos=(my_pos_1,my_pos_2))     
 
#             for i in range(len(all_line_nodes)):
  
#                 G_all_lines.add_edge(all_line_nodes[i][0],all_line_nodes[i][1]) 
                
                
#             pos_lines=nx.get_node_attributes(G_all_lines,'pos')
#             nx.draw_networkx_edges(G_all_lines,pos_lines, edge_color='royalblue',alpha=0.3,width=0.5)
            
#             Selected_topology_nodes.plot(column='sum_slack' ,
#                                           ax=ax, 
#                                           edgecolor='black',
#                                           linewidth=0.2, 
#                                           markersize=20, 
#                                           cmap='turbo', 
#                                           vmin=colorbar_min, 
#                                           vmax=colorbar_max, 
#                                           alpha=1, 
#                                           marker='o', 
#                                           legend=True,
#                                           legend_kwds ={'label': ('Total Daily Outages (MWh)'), 'orientation': 'vertical'})
            
            
#             # ax.set_box_aspect(1)
#             # ax.set_xlim(-13950000,-11100000)
#             # ax.set_ylim([3500000,6250000])
#             # ax.axis('off')

#             # ax.set_box_aspect(1)
#             # ax.set_xlim(-750000,750000)
#             # ax.set_ylim([-2250000,-750000])
#             # ax.axis('off')

#             ax.set_title('Location and Magnitude of Outages in Grid Model ({}{}) on 3-August 2091'.format(rr,mm), fontsize = 10)
#             ax.set_box_aspect(1)
#             ax.set_xlim(lon_min, lon_max)
#             ax.set_ylim(lat_min, lat_max)
#             plt.axis('off')
#             name = 'slacksum_distr_' + rr + mm + '_hot_withlines2.jpg'
#             plt.tight_layout()  
#             plt.savefig(name,dpi=200, bbox_inches ="tight")
    
            
'''COLD SEASON SLACK DISTRIBUTION'''

rcpc= ["rcp45cooler_ssp3_"]
#year
#year=np.arange(2020, 2100, 1,dtype=int)
yearc=np.arange(2069, 2070, 1,dtype=int)
#model type
#model= ["base" , "stdd", "high", "ultra"]
modelc=["stdd"]

#fig,ax = plt.subplots(1,1,figsize=(10,5))

for rr in rcpc:

    for mm in modelc:
        
        for yy in yearc:
      
            #sl= "../../rcpall_output_v23/slack_Exp" + rr + str(mm) + config + str(yy) +".csv"
                
            #Importing the slack generation for that year and scenario
            #df_slack=pd.read_csv(r"../../../rcpall_output_v25/slack/slack_Exprcp45cooler_ssp5_stdd_150simple0_2069.csv",header=0)
            df_slack=pd.read_csv("E:/phd_work/residential_heating_ml_071223_v25/rcpall_output_v25/slack/slack_Exprcp45cooler_ssp3_stdd_150simple0_2069.csv",header=0)
            df_slack["Node_num"]=df_slack.Node.str[4:].astype(int)
                
            
            lines_df = pd.read_csv("E:/phd_work/residential_heating_ml_071223_v25/uced_analysis/day_of_max_slack_ssp3/Exprcp45cooler_ssp3_stdd_150simple0_2069/line_params.csv")
            all_line_nodes = []

            for a in range(len(lines_df)):
                line_name = lines_df.loc[a,'line']
                splitted_name = line_name.split('_')
                line_list = [int(splitted_name[1]),int(splitted_name[2])]
                all_line_nodes.append(line_list)      
                
                
                
            #Find the selected day
            df_maxslackday_hrly=df_slack[(df_slack["Time"] >=8545) & (df_slack["Time"] <=8568)]
            df_maxslackday_sumbus=df_maxslackday_hrly.groupby(['Node_num'], sort=False).agg({'Value':'sum'}).reset_index()
            
            #subsetting geodataframe to selected nodes
            Selected_topology_nodes = nodes_df[nodes_df['Number'].isin(all_selected_nodes)].copy()
            Selected_topology_nodes.reset_index(drop=True,inplace=True)
            
            
            
            allnodes= list(Selected_topology_nodes['Number'])
            slack_sum=[]    
            for i in allnodes:
                slk_sm= df_maxslackday_sumbus.loc[df_maxslackday_sumbus["Node_num"]==i,'Value']
                slk_sm = slk_sm.reset_index(drop=True)
                slack_sum.append(slk_sm[0])
            
            Selected_topology_nodes['sum_slack'] = slack_sum
            #subset to where the slacksum >0
            Selected_topology_nodes=Selected_topology_nodes[Selected_topology_nodes['sum_slack'] >0]
            
            
            matplotlib.rcParams.update({'font.size': 25})
            
            #PLOTTING
            
            colorbar_min=Selected_topology_nodes['sum_slack'].min()
            colorbar_max=Selected_topology_nodes['sum_slack'].max()
            
            fig,ax = plt.subplots(1, 1, figsize=(25, 10))
            #divider = make_axes_locatable(ax)
            
            states_gdf.plot(ax=ax,color='white',edgecolor='black',linewidth=1)
            #Selected_topology_nodes.plot(ax=ax,color = 'li',alpha=1)
            
            G_all_lines = nx.Graph()
            for i in all_selected_nodes:
    
                my_pos_1 = nodes_df.loc[nodes_df['Number']==i].geometry.x.values[0]
                my_pos_2 = nodes_df.loc[nodes_df['Number']==i].geometry.y.values[0]
    
                G_all_lines.add_node(i,pos=(my_pos_1,my_pos_2))     
 
            for i in range(len(all_line_nodes)):
  
                G_all_lines.add_edge(all_line_nodes[i][0],all_line_nodes[i][1]) 
                
                
            pos_lines=nx.get_node_attributes(G_all_lines,'pos')
            nx.draw_networkx_edges(G_all_lines,pos_lines, edge_color='grey',alpha=0.3,width=0.5)
            
            
            Selected_topology_nodes.plot(column='sum_slack' ,
                                          ax=ax,
                                          edgecolor='black',
                                          linewidth=0.5, 
                                          markersize=200, 
                                          cmap='turbo', 
                                          vmin=colorbar_min, 
                                          vmax=colorbar_max, 
                                          alpha=1, 
                                          marker='o',
                                          legend=True,)
                                          #legend_kwds ={'label': ('Total Daily Outages (MWh)'), 'orientation': 'vertical'})
            

                        
            
            
            
           

            ax.set_title('Outages on rcp45cooler_ssp3 "Standard" 23-December 2069',fontsize = 30, fontweight="bold")
            ax.set_box_aspect(1)
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
            plt.axis('off')
            name = 'slacksum_distr_' + rr + mm + '_cold_withlines1_take2'
            plt.tight_layout()
            plt.savefig("E:/phd_work/residential_heating_ml_071223_v25/uced_analysis/day_of_max_slack_ssp3/Exprcp45cooler_ssp3_stdd_150simple0_2069/" + name + ".jpg",dpi=300, bbox_inches ="tight")
                    
                
                