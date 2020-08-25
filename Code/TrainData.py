#******************************************************************************
# =============================================================================
# Anand Dari
# 6366364
# MSc Data Science
# University of Surrey
# =============================================================================


# =============================================================================
# Importing pandas
# =============================================================================
import pandas as pd
# =============================================================================

# =============================================================================
# Reading in file - path will need changing based on user 
# File from https://api-portal.tfl.gov.uk/docs
# =============================================================================
df = pd.read_csv('../Data/En17week.csv', 
                   skiprows = 6)
print("\nPre-processing...\n")
# =============================================================================


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
unwanted_attributes = df.columns[100:]
unwanted_attributes.tolist()
for i in unwanted_attributes:
    df.pop(i)
# =============================================================================


# =============================================================================
#  Removing any remaining noise
# =============================================================================
    
# Remove unnecessary nlc column
df = df.drop("nlc", axis=1)

# Remove Total row
df.drop(df.tail(1).index,inplace=True)

df.columns = df.columns.str.lstrip()
# =============================================================================

# =============================================================================
# Saving file as CSV - path needs to be changed as desired
# =============================================================================
df.to_csv('../Data/TrainWeekday.csv')
print("\nPre-processing complete, please find file in destination folder.\n")

# =============================================================================
# Dataframe Check
# =============================================================================
print(df)
#df.dtypes