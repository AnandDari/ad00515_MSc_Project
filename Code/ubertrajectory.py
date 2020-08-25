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
import geopandas as gpd
# =============================================================================

# =============================================================================
# Reading the file
# =============================================================================
fp = "../Data/london_lsoa.json"
print("Extracting Uber boundaries...")
# =============================================================================

# =============================================================================
# Extracting geometric boundaries
# =============================================================================
polys = gpd.read_file(fp, driver='json')
polys['centroid_column'] = polys.centroid
polys = polys.set_geometry('centroid_column')

df = pd.DataFrame(polys)
df = df[['msoa_name','MOVEMENT_ID', 'DISPLAY_NAME','geometry','centroid_column']]
df['centroid_column']= df['centroid_column'].astype(str)

df['centroid_column_split']= df['centroid_column'].str[7:-1]
df[['longitude','Latitude']] = df.centroid_column_split.str.split(" ",expand=True,)
df.pop('centroid_column_split')

print("Boundaries extracted!")
print(df)

df.to_csv ('../Data/uberboundary.csv', index = None)
# =============================================================================
