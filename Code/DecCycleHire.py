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
from geopy.geocoders import Nominatim
import datetime as dt
geolocator = Nominatim(user_agent="cycelcoder")
#geopy.geocoders.options.default_timeout = 46543413


# =============================================================================
# Reading in file - path will need changing based on user 
# File from https://cycling.data.tfl.gov.uk/
# =============================================================================

df=pd.read_csv('../Data/CycleDecWeek.csv')


# =============================================================================
# Pre-processing
# =============================================================================
print("\nPre-processing. This may take some time...\n")


df = df.drop(df.columns[0],axis=1)

unwanted_attr = ['Bike Id', 
                 'EndStation Id', 'StartStation Id']

for i in unwanted_attr: 
    df.pop(i)


# Make changes to time format to epoch
df['StartDate'] = pd.to_datetime(df['Start Date'])
df['StartDate'] = df['StartDate'].apply(lambda x: dt.datetime.strftime(x, '%Y-%d-%m %H:%M:%S'))
df['StartDate'] = (pd.to_datetime(df['StartDate']).astype('int64')//1e9).astype(int)

df['EndDate'] = pd.to_datetime(df['End Date'])
df['EndDate'] = df['EndDate'].apply(lambda x: dt.datetime.strftime(x, '%Y-%d-%m %H:%M:%S'))
df['EndDate'] = (pd.to_datetime(df['EndDate']).astype('int64')//1e9).astype(int)

# Add information to station data for geolocating
df[['EndStationName','EndLocation','Delete']] = df['EndStation Name'].str.split(", ",expand=True,)
df['EndLocation']=df['EndLocation'].astype(str)+ ', London, UK'
df.pop('Delete')

df[['StartStationName','StartLocation','Delete']] = df['StartStation Name'].str.split(", ",expand=True,)
df['StartLocation']=df['StartLocation'].astype(str)+ ', London, UK'
df.pop('Delete')


locations =[df['EndLocation'], df['StartLocation']]


df2 = pd.DataFrame()
df2['Location']=pd.concat(locations).unique()

geo = pd.DataFrame(columns = ['location','long', 'lat'])

# Geo locate stations 
for i in df2.Location:
    point = geolocator.geocode(i,exactly_one=True, timeout=120)
    latitude = point.latitude
    longitude = point.longitude
    geo = geo.append({'location': i, 'long':longitude, 'lat':latitude}, ignore_index=True)
    
df = df.merge(geo, left_on='StartLocation',right_on='location')
df = df.rename(columns={'long': 'StartLong', 'lat':'StartLat'})
df = df.merge(geo, left_on='EndLocation',right_on='location')
df = df.rename(columns={'long': 'EndLong', 'lat':'EndLat'})



df["Linestring"] = ""
n = len(df['EndStationName'])
string = []

# Apply linestrings
from shapely.geometry import Point, LineString
for i in range(n):
    x_s = df.StartLong[i]
    y_s = df.StartLat[i]
    x_e = df.EndLong[i]
    y_e = df.EndLat[i]
    start = Point(x_s,y_s)
    end = Point(x_e,y_e)
    string.append(LineString([start,end]).wkt)
   
df["Linestring"] = string

df['Cycle_Id'] = df.index + 1
df = df.set_index('Cycle_Id')

# Export analysis csv
df.to_csv('../Data/DecCycleHireAnalysis.csv')

fromtime = '05/12/2019 00:00'
df = df[(df['Start Date'] < fromtime)]

extra_attr = ['location_x','location_y','Start Date', 'End Date',
              'EndLocation', 'StartLocation', 'StartLong', 
              'StartLat', 'EndLong', 'EndLat','EndStationName',
              'StartStationName']

for i in extra_attr: 
    df.pop(i)

df = df[~df['StartStation Name'].str.contains('West End')]
df = df[~df['EndStation Name'].str.contains('West End')]
df = df.sample(n=2000)

# =============================================================================
# Export
# =============================================================================
print(df)
print("Pre-processing complete!")
df.to_csv('../Data/DecCycleHire.csv')