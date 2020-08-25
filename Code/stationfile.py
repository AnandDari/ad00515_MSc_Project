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
# Reading in TrainData again
# =============================================================================
# Reading in file - path will need changing based on user 
# File from https://api-portal.tfl.gov.uk/docs
df1 = pd.read_csv('../Data/TrainWeekday.csv')

df1 = df1.loc[:, ~df1.columns.str.contains('^Unnamed')]

# Manual Correction (Brute Force Method)
df1.loc[(df1.Station == 'Bank & Monument'),'Station']='Bank'
df1.loc[(df1.Station == 'Chalfont & Latimer'),'Station']='Chalfont and Latimer'
df1.loc[(df1.Station == "Earl's Court"),'Station']='Earls Court'
df1.loc[(df1.Station == 'Edgware Road (Bak)'),'Station']='Edgware Road (Bakerloo)'
df1.loc[(df1.Station == 'Edgware Road (Cir)'),'Station']='Edgware Road (Circle/District/Hammersmith and City)'
df1.loc[(df1.Station == 'Elephant & Castle'),'Station']='Elephant and Castle'
df1.loc[(df1.Station == 'Hammersmith (Dis)'),'Station']='Hammersmith (District)'
df1.loc[(df1.Station == 'Hammersmith (H&C)'),'Station']='Hammersmith (Met.)'
df1.loc[(df1.Station == 'Harrow & Wealdstone'),'Station']='Harrow and Wealdstone'
df1.loc[(df1.Station == 'Heathrow Terminals 123'),'Station']='Heathrow Terminals 1 2 3'
df1.loc[(df1.Station == 'Highbury & Islington'),'Station']='Highbury and Islington'
df1.loc[(df1.Station == "King's Cross St. Pancras"),'Station']='Kings Cross St. Pancras'
df1.loc[(df1.Station == "Queen's Park"),'Station']='Queens Park'
df1.loc[(df1.Station == "Regent's Park"),'Station']='Regents Park'
df1.loc[(df1.Station == "Shepherd's Bush (Cen)"),'Station']='Shepherds Bush'
df1.loc[(df1.Station == "Shepherd's Bush (H&C)"),'Station']='Shepherds Bush Market'
df1.loc[(df1.Station == "St. John's Wood"),'Station']='St. Johns Wood'
df1.loc[(df1.Station == "St. Paul's"),'Station']='St. Pauls'
df1.loc[(df1.Station == 'Totteridge & Whetstone'),'Station']='Totteridge and Whetstone'
# =============================================================================


# =============================================================================
# Reading in station data
# =============================================================================
# File from https://www.doogal.co.uk/london_stations.php
df2 = pd.read_csv('../Data/Londonstations.csv')
# =============================================================================
print("\nPre-processing...\n")


# =============================================================================
#  Merging and error check
# =============================================================================

# Merge
df3 = df1.merge(df2,on="Station",how = 'left')

# Error check
error = df3[df3["Postcode"].isnull()]

if error.empty == True:
      
    # Transposing time series information from column to rows
    df3 = pd.melt(df3,['Station','Date','Note','OS X','OS Y','Latitude','Longitude','Zone','Postcode'])
    df3 = df3.sort_values('Station')
    
    # Renaming Time and value Column
    df3 = df3.rename(columns={'variable': 'timestart'})
    df3 = df3.rename(columns={'value': 'count'})
    
    # Splitting time string
    timeend = df3['timestart'].str[-4:]
    df3.insert(loc=10, column='timeend', value=timeend)
    df3['timestart'] = df3['timestart'].str[:4]
    
    
    # Setting time format
    df3 = df3.assign(timestart = pd.to_datetime(df3.timestart, format='%H%M').dt.time) 
    df3 = df3.assign(timeend = pd.to_datetime(df3.timeend, errors='coerce', format='%H%M').dt.time)
    df3['timeend'] = df3['timeend'].fillna("00:00:00")
    
    # Drop unnecessary columns
    df3 = df3.drop("Note", axis=1)
    df3 = df3.drop("OS X", axis=1)
    df3 = df3.drop("OS Y", axis=1)
    df3 = df3.drop("Postcode", axis=1)
    
    # Export file to csv
    df3.to_csv('../Data/stationtime.csv')
    
    print("Merge complete. File ready for use in directory.")
    
else:
    # Error list
    print(df3[df3["Postcode"].isnull()])


# =============================================================================
# Dataframe Check
# =============================================================================
#print(df1)
#print(df2)
#print(df3)
#df.dtypes