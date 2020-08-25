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
from scipy.stats import ttest_ind, mannwhitneyu, ttest_1samp, stats, pearsonr, spearmanr
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.stats as stats
import geopy.distance
from statsmodels.tsa.stattools import adfuller

# =============================================================================
# Reading in file - path will need changing based on user 
# =============================================================================

df = pd.read_csv('../Data/JulyCycleHireAnalysis.csv')
df2 = pd.read_csv('../Data/DecCycleHireAnalysis.csv')


# *****************************************************************************
# Hypothesis Test One:
# *****************************************************************************
# Cycle station footfall are ones closest to City of London. Take average from 
# a weeks worth of journeys
#
# H0:  μ time1 >=  μ time2
# H1:  μ time1 <  μ time2  
# ----------------------------------------------------------------------------- 
print("\nHypothesis One")
print("Highest Cycle station footfall are ones closest to City of London\n")
start = pd.DataFrame()
start = df[['StartStation Name','StartLong', 'StartLat']]
end = df[['EndStation Name', 'EndLong', 'EndLat']]

stations = df.pivot_table(index=['StartStation Name','StartLong', 'StartLat'], aggfunc='size')
stations = stations.reset_index()
stations.columns = ['Name', 'Long', 'Lat', 'Count']

stations2 = df.pivot_table(index=['EndStation Name', 'EndLong', 'EndLat'], aggfunc='size')
stations2 = stations2.reset_index()
stations2.columns = ['Name', 'Long', 'Lat', 'Count']

stationcount = pd.concat([stations, stations2]).groupby(by=['Name', 'Long', 'Lat'], as_index=False)['Count'].sum()

# City of London Coordinates
CityOfLondon = (51.51279, -0.09184)

stationcount['Distance'] = ""

Lat = stationcount["Lat"]
Long = stationcount["Long"]
dist = []

# Distance caluclation using Vincenty
for i in range(len(stationcount["Lat"])):
    dist.append(geopy.distance.geodesic(CityOfLondon, (Lat[i],Long[i])).miles)

stationcount['Distance'] = dist

meandist = stationcount['Distance'].mean()
close = stationcount[stationcount['Distance'] <= meandist]
far = stationcount[stationcount['Distance'] > meandist]


# First check the distribution of count 
density = stats.gaussian_kde(stationcount['Count'])
n, x, _ = plt.hist(stationcount['Count'], bins=np.linspace(0, 4000, 100), 
                   histtype=u'step', density=True)  
plt.plot(x, density(x))
title = "Summer Cycle Station results: mu = %.2f,  std = %.2f" % (stationcount['Count'].mean(), stationcount['Count'].std())
plt.xlabel("Station Count")
plt.ylabel("Density")
plt.title(title)
plt.show()

# H0: μ close >= μ far    
# H1: μ close < μ far
# We want "fail to reject H0"    

print("Mann-whitney U Test")
stat, p = mannwhitneyu(far['Count'], close['Count'], alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')
    
print("Two Sample, One-Tail Test")
stat, p = ttest_ind(close['Count'],far['Count'],equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))    
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')
    
stat, p = ttest_1samp(close['Count'],far['Count'].mean())
print('Statistics=%.3f, p=%.3f' % (stat, p))

# -----------------------------------------------------------------------------
# Correlation Test
# -----------------------------------------------------------------------------
stationcount = stationcount[~stationcount['Name'].str.contains('West End')]

stationdist = stationcount['Distance']
stationfootfall = stationcount['Count']

print("Correlation Test\n")
print("Pearson's Correlation Test:")
corr, p = pearsonr(stationdist , stationfootfall) 
if p > 0.05:
    print('Samples are uncorrelated (fail to reject H0)')
else:
	print('Samples are correlated (reject H0)')

print('Pearsons correlation: %.3f' % corr) 

print("\nSpearman's Correlation Test:")
corr, p = spearmanr(stationdist , stationfootfall) 
if p > 0.05:
    print('Samples are uncorrelated (fail to reject H0)')
else:
	print('Samples are correlated (reject H0)')

print('Spearman correlation: %.3f' % corr) 

sns.scatterplot(stationdist , stationfootfall)
plt.title('Summer Station Footfall vs Distance', fontsize=18)
plt.ylabel('Footfall', fontsize=16)
plt.xlabel('Distance from City of London (miles)', fontsize=16)


# *****************************************************************************
# Hypothesis Test Two:
# *****************************************************************************
# Want to test if the seasons effect the duration of cycle usage.
# Is the average duration during the summer statisitcially higher than the
# average duration in the winter
# For comparing duration, will be easier to sort stations by distance and take
# average of each station over the week.
#
# H0:  μ summer >=  μ winter
# H1:  μ summer <  μ winter  
# -----------------------------------------------------------------------------
print("Hypothesis Test Two\n")
summer = df[['StartStation Name','Duration']].sort_values(by=['StartStation Name'])
winter = df2[['StartStation Name','Duration']].sort_values(by=['StartStation Name'])

summer = summer.groupby(['StartStation Name']).mean()
winter = winter.groupby(['StartStation Name']).mean()

# Rounding duration
summer['Duration'] = round(summer['Duration']/60, 2)
winter['Duration'] = round(winter['Duration']/60, 2)

MeanDuration = summer['Duration'].mean()
DurationSD = summer['Duration'].std()
summermedian = summer['Duration'].median()

MeanDurationDec = winter['Duration'].mean()
DurationSDDec = winter['Duration'].std()
wintermedian = winter['Duration'].median()

ind = np.arange(len(summer)) 
width = 1.5       
plt.figure(figsize=(20,8))
plt.plot(winter['Duration'].tolist(), label='Winter')
plt.plot(summer['Duration'].tolist(), label='Summer')
plt.ylabel('Average Cycle Duration (Minutes)')
plt.xlabel('Every Station')
plt.title('Average Weekly Cycle Duration')
plt.legend(loc='best')
plt.show()

# H0: μ central >= μ non-central    
# H1: μ central < μ non-central 
# We want "reject H0"
print("Hypothesis Test Two")
print("Seasons effect the duration of cycle usage.\n")


print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(summer['Duration'],winter['Duration'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(winter['Duration'],summer['Duration'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')      
  
    
  
# *****************************************************************************
# Hypothesis Test Three:
# *****************************************************************************
# Investigating peak time of cycle usage based on how long bikes are out of 
# station i.e. duration
#
# H0:  μ time1 >=  μ time2
# H1:  μ time1 <  μ time2  
# -----------------------------------------------------------------------------  
        
# Keep seperate Weekday and Weekend
print("Hypothesis Test Three - Summer\n")
print("Investigating peak time of cycle usage based on how long bikes are out of station i.e. duration\n")

starttimes = []
# https://stackoverflow.com/questions/48937900/round-time-to-nearest-hour-python?rq=1
def hour_rounder(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour) 
               +timedelta(hours=t.minute//30))


for i in range(len(df['Start Date'])):
    time = hour_rounder(pd.Timestamp(df['Start Date'][i]))
    time = str(time.time())
    starttimes.append(time)
    
df['Start_Hour'] = starttimes

morning = df.loc[(df.Start_Hour >= "05:00:00") & (df.Start_Hour < "12:00:00")]
morningmean = morning['Duration'].mean()
morningmedian = morning['Duration'].median()

afternoon = df.loc[(df.Start_Hour >= "12:00:00") & (df.Start_Hour < "17:00:00")]
afternoonmean = afternoon['Duration'].mean()
afternoonmedian = afternoon['Duration'].median()

evening= df.loc[(df.Start_Hour >= "17:00:00") & (df.Start_Hour < "21:00:00")]
eveningmean = evening['Duration'].mean()
eveningmedian = evening['Duration'].median()

nightusage1 = df.loc[(df.Start_Hour >= "21:00:00")]
nightusage2 = df.loc[(df.Start_Hour < "05:00:00")]
night = pd.concat([nightusage1,nightusage2])
nightmean = night['Duration'].mean()
nightmedian = night['Duration'].median()

ts = df[['Start_Hour', 'Duration']]
ts = ts.groupby(['Start_Hour']).size().reset_index(name='Duration')
labels = df['Start_Hour'].unique()
labels = labels.tolist()
labels.sort()
ind = np.arange(len(labels)) 
ts.plot(color='red', linewidth=2.0, figsize=(10,5))
plt.ylabel('Mean Duration (Seconds)')
plt.xlabel('Time (24 hour)')
plt.title('Cycle Usage Duration Over A Day (Summer)')
plt.xticks(np.arange(min(ind), max(ind)+1, 1.0), labels, rotation=90)
plt.legend(loc='best')
plt.show()

print("Augmented Dickey-Fuller test to check for time dependency.")
result = adfuller(ts['Duration'])
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



# H0: μ morning >= μ afternoon    
# H1: μ morning < μ afternoon 
# We want "reject H0"
print("Morning vs Afternoon")

print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(morning['Duration'],afternoon['Duration'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(morning['Duration'],afternoon['Duration'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')
    
    
# H0: μ morning >= μ evening    
# H1: μ morning < μ evening 
# We want "reject H0"
print("Morning vs Evening")

print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(morning['Duration'],evening['Duration'],equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(morning['Duration'],evening['Duration'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')    

# H0: μ morning >= μ night    
# H1: μ morning < μ night 
# We want "reject H0"
print("Morning vs Night")

print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(morning['Duration'],night['Duration'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(morning['Duration'],night['Duration'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')  
    
# H0: μ afternoon >= μ evening    
# H1: μ afternoon < μ evening 
# We want "reject H0"
print("Afternoon vs Evening")

print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(afternoon['Duration'],evening['Duration'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(afternoon['Duration'],evening['Duration'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')  
    
# H0: μ afternoon >= μ night    
# H1: μ afternoon < μ night 
# We want "reject H0"
print("Afternoon vs Night")

print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(afternoon['Duration'],night['Duration'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(afternoon['Duration'],night['Duration'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')     
    
# H0: μ evening >= μ night    
# H1: μ evening < μ night 
# We want "reject H0"
print("Evening vs Night")

print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(evening['Duration'],night['Duration'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(evening['Duration'],night['Duration'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')     
    


