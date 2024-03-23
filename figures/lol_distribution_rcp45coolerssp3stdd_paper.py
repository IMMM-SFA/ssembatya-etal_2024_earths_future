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



#reading in selected nodes
all_selected_nodes = pd.read_csv('selected_nodes_150.csv',header=0)
all_selected_nodes= [*all_selected_nodes['SelectedNodes']]



#Set the lat/lon bounds for the plot:
lat_min = 25
lat_max = 38
lon_min = -108
lon_max = -92

#Projection
projection= 4269


#Reading all necessary files for plotting the map
df = pd.read_csv(r'E:\phd_work\residential_heating_ml_071223_v25\rcpuced_v25\ERCOT_Bus.csv',header=0)

crs = CRS('epsg:4269')

geometry = [Point(xy) for xy in zip(df['Substation Longitude'],df['Substation Latitude'])]

nodes_df = gpd.GeoDataFrame(df,crs=crs,geometry=geometry)
nodes_df = nodes_df.to_crs(epsg= projection)


states_gdf = gpd.read_file('geo_export_9ef76f60-e019-451c-be6b-5a879a5e7c07.shp')
states_gdf = states_gdf.to_crs(epsg= projection)



            
'''COLD SEASON SLACK DISTRIBUTION'''

rcpc= ["rcp45cooler_ssp3_"]
#year

yearc=np.arange(2069, 2070, 1,dtype=int)
#model type

modelc=["stdd"]



for rr in rcpc:

    for mm in modelc:
        
        for yy in yearc:
      

                
            #Importing the slack generation for that year and scenario
            df_slack=pd.read_csv("slack_Exprcp45cooler_ssp3_stdd_150simple0_2069.csv",header=0)
            df_slack["Node_num"]=df_slack.Node.str[4:].astype(int)
                
            
            lines_df = pd.read_csv("line_params.csv")
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

            Selected_topology_nodes=Selected_topology_nodes[Selected_topology_nodes['sum_slack'] >0]
            
            
            matplotlib.rcParams.update({'font.size': 25})
            
            #PLOTTING
            
            colorbar_min=Selected_topology_nodes['sum_slack'].min()
            colorbar_max=Selected_topology_nodes['sum_slack'].max()
            
            fig,ax = plt.subplots(1, 1, figsize=(25, 10))

            
            states_gdf.plot(ax=ax,color='white',edgecolor='black',linewidth=1)

            
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
            plt.savefig( name + ".jpg",dpi=300, bbox_inches ="tight")
                    
                
                