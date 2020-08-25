#******************************************************************************
# =============================================================================
# Anand Dari
# 6366364
# MSc Data Science
# University of Surrey
# =============================================================================

# =============================================================================
# Necessary libraries
# =============================================================================
import pandas as pd
from convertbng.util import convert_lonlat
from shapely.geometry import Point, LineString
import numpy as np 
# =============================================================================

# =============================================================================
# Reading data files
# =============================================================================
df = pd.read_csv('../Data/BusRoutesWkday.txt')
df2 = pd.read_csv('../Data/BusRoutesWkday2.txt')

# =============================================================================
# Pre-processing
# =============================================================================
df3 = pd.concat([df, df2], ignore_index=True)
df3['HOUR'] = df3['HOUR']*60
df3.pop('DATE')
df3['Start_Hour'] = pd.to_datetime(df3['HOUR'], unit='m').dt.strftime("%H:%M:%S")
df3['End_Hour'] = df3['HOUR'] + 60
df3['End_Hour'] = pd.to_datetime(df3['End_Hour'], unit='m').dt.strftime("%H:%M:%S")
df3.pop('HOUR')


bus_wkday = df3[df3.DAY_TYPE == 'Weekday']
bus_wkday['routeid'] = bus_wkday['routeid'].astype(str)

bus_wkend = df3[df3.DAY_TYPE != 'Weekday']
bus_wkend['routeid'] = bus_wkend['routeid'].astype(str)

# =============================================================================
# Export to csv
# =============================================================================
bus_wkday.to_csv('../Data/bus_wkday.csv')
bus_wkend.to_csv('../Data/bus_wkend.csv')


# =============================================================================
# ATTEMPT 1 using GEOJSON
# =============================================================================
# import requests
# import fiona
# import geopandas as gpd

# fiona.drvsupport.supported_drivers['kml'] = 'rw'
# fiona.drvsupport.supported_drivers['KML'] = 'rw'

# geo = gpd.read_file(r"C:\Users\anand\Documents\MSc\Disso_London\Bus Route\export.kml")
# geo.to_csv(r'C:\Users\anand\Documents\MSc\Disso_London\Bus Route\geo.csv')

# =============================================================================
# ATTEMPT 2 using CSV easting and northings
# =============================================================================
route = pd.read_csv('../Data/BusSequences.csv')
route = route[route.Run == 1]

drop_attr = ['Run', 'Stop_Code_LBSL', 'Bus_Stop_Code',
             'Naptan_Atco', 'Heading', 'Virtual_Bus_Stop']

for i in drop_attr:
    route.pop(i)

# Convert easting and northings to longitude and latitude
easting = route['Location_Easting'].tolist()
northing = route['Location_Northing'].tolist()
coordinate = convert_lonlat(easting, northing)

route['Long'] = coordinate[0]
route['Lat'] = coordinate[1]
route['Point'] = ""

bus_routes = np.unique(route['Route'].tolist())

coord_points = []

# Loop to generates point for each sequence in a route
for i in bus_routes:
    bus = route[route['Route'] == i]
    
    for j in bus.Sequence.tolist():
        x = bus['Long'].loc[bus['Sequence'] == j]
        y = bus['Lat'].loc[bus['Sequence'] == j]
        coord_points.append(Point(x,y))
        
 
route['Point'] = coord_points  

# Exporting route points
route.to_csv('../Data/routepoints.csv')

# loop to join coordinate points in a route in to a linestring 
string = []
for i in bus_routes:
    bus = route[route['Route'] == i]
    bus_point = bus['Point'].tolist()
    string.append(LineString(bus_point).wkt)

columns = ['routeid', 'Linestring']
final_bus = pd.DataFrame(columns = columns)

final_bus['routeid'] = bus_routes
final_bus['routeid'] = final_bus['routeid'].astype(str)
final_bus['Linestring'] = string

counted_bus_wkday = pd.merge(bus_wkday,final_bus,on='routeid',how='left')
counted_bus_wkday['AvgTaps'] = counted_bus_wkday.groupby(['routeid','Start_Hour'])['taps'].transform('mean').apply(np.ceil)
counted_bus_wkday = counted_bus_wkday.drop_duplicates(subset=['routeid','Start_Hour','End_Hour'], keep='last')
counted_bus_wkday.pop('taps')

counted_bus_wkend = pd.merge(bus_wkend,final_bus,on='routeid',how='left')
counted_bus_wkend['AvgTaps'] = counted_bus_wkend.groupby(['routeid','Start_Hour'])['taps'].transform('mean').apply(np.ceil)
counted_bus_wkend = counted_bus_wkend.drop_duplicates(subset=['routeid','Start_Hour','End_Hour'], keep='last')
counted_bus_wkend.pop('taps')


counted_bus_wkday.to_csv('../Data/bus_with_count_wkday.csv')
counted_bus_wkend.to_csv('../Data/bus_with_count_wkend.csv')
