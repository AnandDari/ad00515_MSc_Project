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
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np
# =============================================================================

# =============================================================================
# Reading files
# =============================================================================
df = pd.read_csv('../Data/bus_with_count_wkday.csv')
df2 = pd.read_csv('../Data/bus_with_count_wkend.csv')

df = df.drop(df.columns[0], axis=1)
df2 = df2.drop(df2.columns[0], axis=1)
# =============================================================================

# *****************************************************************************
# Hypothesis Test One:
# *****************************************************************************
# Bus routes are at full capacity (at its peak consumption) in the mornings
# and evening. In comparison to the afternoon and late at night.
#
# H0:  μ1 >= μ2    i.e. bus routes in the morning more usage than afternoon
# H1:  μ1 < μ2     i.e. bus routes in the morning have less than afternoon
# -----------------------------------------------------------------------------

# Keep seperate Weekday and Weekend
print("\nHypothesis Test One - Weekday")
print("Bus routes are at full capacity (at its peak consumption) in the mornings and evening.\n")


ts = pd.DataFrame(df[['Start_Hour', 'AvgTaps']])
ts = ts.groupby(['Start_Hour']).sum()
ts.plot(color='red', linewidth=2.0)
plt.title('Time Series of Bus Usage', fontsize=18)
plt.ylabel('Bus Card Taps', fontsize=8)
plt.xlabel('Time (24 hours)', fontsize=8)
plt.show()

# =============================================================================
# Augmented Dickey-Fuller test
# From viewing the time series we want to know if this is stationary or not.
# Using this information we can statistically confirm time dependency based
# on the null hypothesis.
# =============================================================================
# H0: suggests the time series has a unit root, meaning it is non-stationary.
#     It has some time dependent structure.
#
# H1: time series does not have a unit root, meaning it is stationary.
#     It does not have time-dependent structure.

print("Augmented Dickey-Fuller test to check for time dependency.")
result = adfuller(ts['AvgTaps'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
alpha = 0.05
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
if result[1] > alpha:
	print('Data has a unit root and is non-stationary.\n')
else:
	print('Data does not have a unit root and is stationary.\n')    
    

morningwkday = df.loc[(df.Start_Hour >= "05:00:00") & (df.Start_Hour < "12:00:00")]
morningtaps = morningwkday['AvgTaps'].mean()
morningtapsmedian = morningwkday['AvgTaps'].median()

afternoonwkday = df.loc[(df.Start_Hour >= "12:00:00") & (df.Start_Hour < "17:00:00")]
afternoontaps = afternoonwkday['AvgTaps'].mean()
afternoontapsmedian = afternoonwkday['AvgTaps'].median()

eveningwkday= df.loc[(df.Start_Hour >= "17:00:00") & (df.Start_Hour < "21:00:00")]
eveningtaps = eveningwkday['AvgTaps'].mean()
eveningtapsmedian = eveningwkday['AvgTaps'].median()

nightusage1 = df.loc[(df.Start_Hour >= "21:00:00")]
nightusage2 = df.loc[(df.Start_Hour < "05:00:00")]
nightwkday = pd.concat([nightusage1,nightusage2])
nighttaps = nightwkday['AvgTaps'].mean()
nighttapsmedian = nightwkday['AvgTaps'].median()


# H0: μ morning >= μ afternoon    
# H1: μ morning < μ afternoon 
# We want "reject H0"
print("Morning vs Afternoon")

print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(morningwkday['AvgTaps'],afternoonwkday['AvgTaps'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')      
    

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(morningwkday['AvgTaps'],afternoonwkday['AvgTaps'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')


# H0: μ morning >= μ night    
# H1: μ morning < μ night 
# We want "reject H0"
print("\nMorning vs Night")
 

print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(morningwkday['AvgTaps'],nightwkday['AvgTaps'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')      
    

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(morningwkday['AvgTaps'],nightwkday['AvgTaps'],alternative='greater')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')    


# H0: μ evening >= μ afternoon    
# H1: μ evening < μ afternoon 
# We want "reject H0"
print("\nEvening vs Afternoon")


print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(eveningwkday['AvgTaps'],afternoonwkday['AvgTaps'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')      
    

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(eveningwkday['AvgTaps'],afternoonwkday['AvgTaps'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')    
    
# H0: μ evening >= μ night    
# H1: μ evening < μ night 
# We want "reject H0"
print("\nEvening vs Night")


print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(eveningwkday['AvgTaps'],nightwkday['AvgTaps'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')      
    

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(eveningwkday['AvgTaps'],nightwkday['AvgTaps'],alternative='greater')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')      
    
# H0: μ evening >= μ morning    
# H1: μ evening < μ morning 
# We want "reject H0"
print("\nEvening vs Morning")

print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(eveningwkday['AvgTaps'],morningwkday['AvgTaps'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')      
    

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(eveningwkday['AvgTaps'],morningwkday['AvgTaps'],alternative='greater')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')      


# Keep seperate - Weekend
print("Hypothesis Test One - Weekend\n")

ts = pd.DataFrame(df2[['Start_Hour', 'AvgTaps']])
ts = ts.groupby(['Start_Hour']).sum()
ts.plot(color='red', linewidth=2.0)
plt.title('Time Series of Bus Usage', fontsize=18)
plt.ylabel('Bus Card Taps', fontsize=8)
plt.xlabel('Time (24 hours)', fontsize=8)
plt.show()

# =============================================================================
# Augmented Dickey-Fuller test
# From viewing the time series we want to know if this is stationary or not.
# Using this information we can statistically confirm time dependency based
# on the null hypothesis.
#
# H0: suggests the time series has a unit root, meaning it is non-stationary.
#     It has some time dependent structure.
#
# H1: time series does not have a unit root, meaning it is stationary.
#     It does not have time-dependent structure.
# =============================================================================

print("Augmented Dickey-Fuller test to check for time dependency.")
result = adfuller(ts['AvgTaps'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
alpha = 0.05
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
if result[1] > alpha:
	print('Data has a unit root and is non-stationary.\n')
else:
	print('Data does not have a unit root and is stationary.\n')  



morningwkend = df2.loc[(df2.Start_Hour >= "05:00:00") & (df2.Start_Hour < "12:00:00")]
morningtaps = morningwkend['AvgTaps'].mean()
morningtapsmedian = morningwkend['AvgTaps'].median()

afternoonwkend = df2.loc[(df2.Start_Hour >= "12:00:00") & (df2.Start_Hour < "17:00:00")]
afternoontaps = afternoonwkend['AvgTaps'].mean()
afternoontapsmedian = afternoonwkend['AvgTaps'].median()

eveningwkend= df2.loc[(df2.Start_Hour >= "17:00:00") & (df2.Start_Hour < "21:00:00")]
eveningtaps = eveningwkend['AvgTaps'].mean()
eveningtapsmedian = eveningwkend['AvgTaps'].median()

nightusage1 = df2.loc[(df2.Start_Hour >= "21:00:00")]
nightusage2 = df2.loc[(df2.Start_Hour < "05:00:00")]
nightwkend = pd.concat([nightusage1,nightusage2])
nighttaps = nightwkend['AvgTaps'].mean()
nighttapsmedian = nightwkend['AvgTaps'].median()


# H0: μ morning >= μ afternoon    
# H1: μ morning < μ afternoon 
# We want "reject H0"
print("\nMorning vs Afternoon")

print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(morningwkend['AvgTaps'],afternoonwkend['AvgTaps'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')      
    

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(morningwkend['AvgTaps'],afternoonwkend['AvgTaps'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')


# H0: μ morning >= μ night    
# H1: μ morning < μ night 
# We want "reject H0"
print("\nMorning vs Night")

print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(morningwkend['AvgTaps'],nightwkend['AvgTaps'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')      
    

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(morningwkend['AvgTaps'],nightwkend['AvgTaps'],alternative='greater')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')    


# H0: μ evening >= μ afternoon    
# H1: μ evening < μ afternoon 
# We want "reject H0"
print("\nEvening vs Afternoon")


print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(eveningwkend['AvgTaps'],afternoonwkend['AvgTaps'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')      
    

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(eveningwkend['AvgTaps'],afternoonwkend['AvgTaps'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')    
    
# H0: μ evening >= μ night    
# H1: μ evening < μ night 
# We want "reject H0"
print("\nEvening vs Night")


print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(eveningwkend['AvgTaps'],nightwkend['AvgTaps'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')      
    

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(eveningwkend['AvgTaps'],nightwkend['AvgTaps'],alternative='greater')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')      
    
# H0: μ evening >= μ morning    
# H1: μ evening < μ morning 
# We want "reject H0"
print("\nEvening vs Morning")


print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(eveningwkend['AvgTaps'],morningwkend['AvgTaps'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')      
    

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(eveningwkend['AvgTaps'],morningwkend['AvgTaps'],alternative='greater')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')  

# *****************************************************************************
# Hypothesis Test Two:
# *****************************************************************************
# Most active Bus routes are ones that cross central London
# Given the centre of City of London i.e. the main district for work and economy 
# has latitude anbd longitude of 51.51279 -0.09184
#
# H0:  μ1 >= μ2    i.e. bus routes in the morning more usage than afternoon
# H1:  μ1 < μ2     i.e. bus routes in the morning have less than afternoon
# -----------------------------------------------------------------------------

# Weekday Analysis
import shapely.wkt
import geopandas as gpd
df = df.dropna(subset=['Linestring'])
routes = df['Linestring'].tolist()


list_lines =  [shapely.wkt.loads(routes[i]) for i in range(len(routes))]


CoL = gpd.read_file('../Data/cityoflondon.geojson')
CoL_poly = CoL['geometry'].tolist()

intersect = []
for i in range(len(list_lines)):
    intersect.append(list_lines[i].intersects(CoL_poly[0]))
    
    
df['Intersects'] = intersect

centralbus = df[df['Intersects'] == True].sort_values(by=['Start_Hour'])
centralbusplot = centralbus.groupby(['Start_Hour', 'Intersects']).mean()
centralbustaps = centralbusplot['AvgTaps'].mean()
centralbustapsmedian = centralbusplot['AvgTaps'].median()

noncentralbus = df[df['Intersects'] == False].sort_values(by=['Start_Hour'])
noncentralbusplot = noncentralbus.groupby(['Start_Hour', 'Intersects']).mean()
noncentralbustaps = noncentralbusplot['AvgTaps'].mean()
noncentralbustapsmedian = noncentralbusplot['AvgTaps'].median()

labels = centralbus['Start_Hour'].unique()
labels = labels.tolist()
labels.sort()
ind = np.arange(len(labels)) 
width = 0.35       
plt.figure(figsize=(15,5)) 
plt.bar(ind, centralbusplot['AvgTaps'].tolist(), width, label='Intersects Central London')
plt.bar(ind + width, noncentralbusplot['AvgTaps'].tolist(), width,
    label='No Intersection with Central London')
plt.ylabel('Bus Usage')
plt.title('Weekday Bus Taps by Intersection')
plt.xticks(np.arange(min(ind), max(ind)+1, 1.0), labels, rotation=90)
plt.legend(loc='best')
plt.show()

print("Hypothesis Test Two - Central London Intersection")
print("Weekday Analysis\n")
# H0: μ central >= μ non-central    
# H1: μ central < μ non-central 
# We want "reject H0"


print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(centralbusplot['AvgTaps'],noncentralbusplot['AvgTaps'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')      
    

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(centralbusplot['AvgTaps'],noncentralbusplot['AvgTaps'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')  
    
# Weekend Analysis

df2 = df2.dropna(subset=['Linestring'])
routes = df2['Linestring'].tolist()


list_lines =  [shapely.wkt.loads(routes[i]) for i in range(len(routes))]


CoL = gpd.read_file('../Data/cityoflondon.geojson')
CoL_poly = CoL['geometry'].tolist()

intersect = []
for i in range(len(list_lines)):
    intersect.append(list_lines[i].intersects(CoL_poly[0]))
    
    
df2['Intersects'] = intersect

centralbuswknd = df2[df2['Intersects'] == True]
centralbuswkndplot = centralbuswknd.groupby(['Start_Hour', 'Intersects']).mean()
centralbuswkndtaps = centralbuswkndplot['AvgTaps'].mean()
centralbuswkndtapsmedian = centralbuswkndplot['AvgTaps'].median()

noncentralbuswknd = df[df['Intersects'] == False]
noncentralbuswkndplot = noncentralbuswknd.groupby(['Start_Hour', 'Intersects']).mean()
noncentralbuswkndtaps = noncentralbuswkndplot['AvgTaps'].mean()
noncentralbuswkndtapsmedian = noncentralbuswkndplot['AvgTaps'].median()

labels = centralbuswknd['Start_Hour'].unique()
labels = labels.tolist()
labels.sort()
ind = np.arange(len(labels)) 
width = 0.35       
plt.figure(figsize=(15,5)) 
plt.bar(ind, centralbuswkndplot['AvgTaps'].tolist(), width, label='Intersects Central London')
plt.bar(ind + width, noncentralbuswkndplot['AvgTaps'].tolist(), width,
    label='No Intersection with Central London')
plt.ylabel('Bus Usage')
plt.title('Weekend Bus Taps by Intersection')
plt.xticks(np.arange(min(ind), max(ind)+1, 1.0), labels, rotation=90)
plt.legend(loc='best')
plt.show()

# H0: μ central >= μ non-central    
# H1: μ central < μ non-central 
# We want "reject H0"
print("Weekend Analysis")


print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(centralbuswkndplot['AvgTaps'],noncentralbuswkndplot['AvgTaps'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')      
    

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(centralbuswkndplot['AvgTaps'],noncentralbuswkndplot['AvgTaps'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')      
    
    
# *****************************************************************************
# Hypothesis Test Three:
# *****************************************************************************
# Weekend vs Weekday - there is statistical significance to suggest that
# Buses are more occupied during the weekdays than the weekend (on average).
# However these will differ by hour.
#
# H0:  μ1 >= μ2    i.e. weekday taps will be greater than weekend taps
# H1:  μ1 < μ2     
# -----------------------------------------------------------------------------
print("\nHypothesis Test Three")
print("Weekend vs Weekday Taps \n")

weekday =  pd.DataFrame(df[['Start_Hour', 'AvgTaps']])
weekend = pd.DataFrame(df2[['Start_Hour', 'AvgTaps']])

weekday = round(weekday.groupby(['Start_Hour']).mean())
weekend = round(weekend.groupby(['Start_Hour']).mean())

labels = df['Start_Hour'].unique()
labels = labels.tolist()
labels.sort()
ind = np.arange(len(labels)) 
width = 0.35       
plt.figure(figsize=(15,5)) 
plt.bar(ind, weekday['AvgTaps'].tolist(), width, label='Weekday')
plt.bar(ind + width, weekend['AvgTaps'].tolist(), width,
    label='Weekend')
plt.ylabel('Average Taps')
plt.title('Weekday vs Weekend Bus Usage')
plt.xticks(np.arange(min(ind), max(ind)+1, 1.0), labels, rotation=90)
plt.legend(loc='best')
plt.show()


# H0: μ weekday >= μ weekend    
# H1: μ weekday < μ weekend 
# We want "Fail to reject H0"
print("Weekday vs Weekend Analysis")

print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(weekday['AvgTaps'],weekend['AvgTaps'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')      
    

print("\nMann-whitney U Test")
stat, p =mannwhitneyu(weekday['AvgTaps'],weekend['AvgTaps'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')    








