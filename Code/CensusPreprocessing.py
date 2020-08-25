#******************************************************************************
# =============================================================================
# Anand Dari
# 6366364
# MSc Data Science
# University of Surrey
# =============================================================================


# =============================================================================
# Necessary Libraries
# =============================================================================
import pandas as pd

# =============================================================================
# Reading in file - path will need changing based on user 
# File from https://data.london.gov.uk/dataset/981e9136-a06a-44ec-a067-10f3d786cd3f
# =============================================================================
df = pd.read_excel('../Data/ward-pop-ONS-GLA-Census.xls', 
                   skiprows=1,
                   sheet_name='2011 Census')
# =============================================================================


# =============================================================================
# Pre-processing
# =============================================================================
print("\nPre-processing...\n")
# Replacing header with column names
header = df.iloc[0]
print(header)
df.rename(columns = header)

#List of other attributes
attributes = df.columns[4:]
attributes.tolist()

# Correcting Ward Code to be conistent with Shape file
df.loc[(df.Borough == 'City of London'),'Ward Code']='E09000001'

# Renaming Column
df = df.rename(columns={'Persons: All Ages': 'Population'})

# Setting the Population field to numeric 
df["Population"].astype(int)

# Removing uneccessary attributes
for j in attributes:
    df.pop(j)
# =============================================================================

    
# =============================================================================
# Saving file as CSV - path needs to be changed as desired
# =============================================================================
df.to_csv('../Data/preprocessed_census.csv')
print("\nPre-processing complete, please find file in destination folder.\n")
# =============================================================================


# =============================================================================
# Attribute and dataframe check 
# uncomment to check
# =============================================================================
# print(attributes)
# print(df)
# df.dtypes
