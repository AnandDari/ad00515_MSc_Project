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

# =============================================================================
# Reading files
# =============================================================================
df = pd.read_csv('../Data/UberWeekend.csv')
df = df.sample(n=200000)
geo = pd.read_csv('../Data/uberboundary.csv')
# Length: 6814351
# Please be patient when running this code as lenght is quite long
print("\nPre-processing. This may take some time...\n")      

# =============================================================================
# Attribute Investigation
# =============================================================================
attributes = df.columns[0:]
attributes.tolist()

#Finding count where relecvant attributes are
#count=0
#for j in attributes:
#    count +=1
#    print(count, j)
    
# Removing uneccessary attributes
unwanted_attributes = df.columns[4:]
unwanted_attributes.tolist()
for i in unwanted_attributes:
    df.pop(i)
# ============================================================================= 
    
    
# =============================================================================
# Changing attributes for usage
# =============================================================================  

# Setting time format    
# help from https://stackoverflow.com/questions/53717929/convert-numbers-to-hours-in-python

#Convert the hour of day to actual time 
df['start_time'] = pd.to_datetime(df.hod, format='%H').dt.time

# Convert the travel time (seconds) to H:M:S format
df['mean_travel_time'] = pd.to_datetime(df["mean_travel_time"], unit='s').dt.strftime("%H:%M:%S")
# Necessary formatting to create end time attribute

df['start_time']= pd.to_timedelta(df['start_time'].astype(str))
df['mean_travel_time']= pd.to_timedelta(df['mean_travel_time'])

# Creation of the end time field based on start and travel time  
df['end_time'] = df['start_time'] + df['mean_travel_time']


# =============================================================================
# Post Import Identified Fix
# =============================================================================

# Must convert to string and remove timestamp as we do not have day count
df['start_time'] = df['start_time'].apply(lambda v: str(v))
df['mean_travel_time'] = df['mean_travel_time'].apply(lambda v: str(v))
df['end_time'] = df['end_time'].apply(lambda v: str(v))

df['start_time']= df['start_time'].str[7:]
df['mean_travel_time']= df['mean_travel_time'].str[7:]
df['end_time']= df['end_time'].str[7:]

# =============================================================================
# Remove unnecessary field
df.pop('hod')
# =============================================================================
df = df.merge(geo, left_on='sourceid',right_on='MOVEMENT_ID')
df = df.rename(columns={'longitude': 'StartLong', 'Latitude':'StartLat'})
df = df.rename(columns={'centroid_column': 'startpoint'})
df = df.merge(geo, left_on='dstid',right_on='MOVEMENT_ID')
df = df.rename(columns={'longitude': 'EndLong', 'Latitude':'EndLat'})
df = df.rename(columns={'centroid_column': 'endpoint'})

remove = ['msoa_name_x', 'MOVEMENT_ID_x',
       'msoa_name_y', 'MOVEMENT_ID_y']

for i in remove: 
    df.pop(i)


# Latitude and longitude is not good enough so we use Linestring to portray 
# routes that the uber can take 
# 09/04/2020 - will try to animate the route 
n = 200000
df["Linestring"] = ""

string = []
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


# =============================================================================
# Saving file
# =============================================================================
df.to_csv('../Data/UberWeekendAnalysis.csv',
          date_format='%H:%M:%S')

print("Pre-processing complete!")

# =============================================================================
# Dataframe Check
# =============================================================================
#print(df)

