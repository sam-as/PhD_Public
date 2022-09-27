#%% importing the libraries
import pandas as pd
import geopandas as gd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%% reading the shape file first
path = r"-------Shape file data\State_shp\Admin2.shp"
gdf = gd.read_file(path)

# changing the crs
gdf = gdf.to_crs("EPSG:4326")

# %% reading the disease data
path2=  r"----------------diseasedata_population.xlsx"
df= pd.read_excel(path2)

# %% renaming the columns
gdf = gdf.rename(columns={'ST_NM': 'State/UTs'})

# %% merging the geodataframe and dataframe using the common column
da = gdf.merge(df, on = "State/UTs")

#%% getting the longitudes Aand the latitudes of the centroid of each polygon
da['Longitude'] = da['geometry'].centroid.x
da['Latitude'] = da['geometry'].centroid.y

# %% initialising the matplotlib function and calling the seaborn ticks style
sns.set(style='ticks')
plt.rcParams["figure.figsize"] = (8,9)

fig, ax = plt.subplots()
divider = make_axes_locatable(ax)

ax = da.plot("State/UTs", edgecolor='black', facecolor = "none", ax=ax )

#da.apply(lambda x: ax.annotate(text=x['State/UTs'], xy=x.geometry.centroid.coords[0], ha='center'), axis=1)

#plt.show()

# adding the mortalities to the blank map
ax1 = ax.scatter(
    x = da['Longitude'],
    y = da['Latitude'],
    s= da['Deaths']/100,
    alpha=0.7, 
    color='red',
)

# adding the legends and the labels
handles, labels = ax1.legend_elements(prop="sizes", alpha=0.5)
legend2 = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(-0.30, 0.5), 
     title="Mortality*100", fancybox=False)

plt.tight_layout()
plt.show()

#%% making a chloropleth + adding the scatter matrix
fig, ax = plt.subplots()
divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="5%", pad=0.01)

ax = da.plot("Total Cases", edgecolor='black', cmap = 'CMRmap_r', 
    alpha= 0.6, 
    legend=True,
    legend_kwds = {'label': "Positive COVID-19 Cases", "alpha" : 0.6}, ax = ax, cax = cax )

#%% 
ax1 = ax.scatter(
    x = da['Longitude'],
    y = da['Latitude'],
    s= da['Deaths']/100,
    alpha=0.7, 
    color='red',
)

# adding the legends of the scatter plot matrix
handles, labels = ax1.legend_elements(prop="sizes", alpha=0.5)
legend2 = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(-0.35, 0.5), 
     title="Mortality*100", fancybox=False)

plt.tight_layout()
plt.show()
