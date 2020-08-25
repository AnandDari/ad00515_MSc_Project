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
import numpy as np
import geopy.distance
from scipy.stats import ttest_ind, mannwhitneyu, stats, pearsonr, spearmanr, normaltest
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import warnings
from statsmodels.tsa.stattools import adfuller
# =============================================================================

# =============================================================================
# Reading data
# =============================================================================

entry = pd.read_csv('../Data/stationtime.csv')
exits = pd.read_csv('../Data/stationtimeexits.csv')

# Merging data
df = pd.concat([entry, exits]).groupby(by=['Station', 'timestart', 'timeend', 'Latitude', 'Longitude'], as_index=False)['count'].sum()


# Given the centre of City of London i.e. the main district for work and economy 
# has latitude anbd longitude of 51.51279 -0.09184
CityOfLondon = (51.51279, -0.09184)
coords_2 = (51.503071,  -0.280303)

df['Distance'] = ""

Lat = df["Latitude"]
Long = df["Longitude"]
dist = []

# Calculating distance using Vincenty formula.
for i in range(len(df["Latitude"])):
    dist.append(geopy.distance.geodesic(CityOfLondon, (Lat[i],Long[i])).miles)

df['Distance'] = dist

# Exporting the stations with distance
df.to_csv('../Data/stationwithdist.csv')


# *****************************************************************************
# Hypothesis Test One:
# *****************************************************************************
# Stations closer to central London have a higher count than ones further away.
# We will use the mean distance to test this hypothesis and the station count.
# How do we define "close", by taking average distance we determine what is 
# acceptably close based on number of stations in London.
#
# H0:  μ1 <= μ2    i.e. stations have same count regardless of how close or far
# H1:  μ1 > μ2     i.e. More station count when closer than further away
# -----------------------------------------------------------------------------

df = df.sort_values('Distance')

meandist = df['Distance'].mean()
meancount = df['count'].mean()

close = df[df['Distance'] <= meandist]
closecount = close['count'].mean()

far = df[df['Distance'] > meandist]
farcount = far['count'].mean()



print("\nHypothesis Test One")
print("Stations closer to central London have a higher count than ones further away\n")

# First check the distribution of count 
density = stats.gaussian_kde(df['count'])
n, x, _ = plt.hist(df['count'], bins=np.linspace(0, 4000, 100), 
                   histtype=u'step', density=True)  
plt.plot(x, density(x))
title = "Station results: mu = %.2f,  std = %.2f" % (df['count'].mean(), df['count'].std())
plt.title(title)
plt.show()

density = stats.gaussian_kde(close['count'])
n, x, _ = plt.hist(close['count'], bins=np.linspace(0, 4000, 200), 
                   histtype=u'step', density=True)  
plt.plot(x, density(x))
title = "Close Station Results: mu = %.2f,  std = %.2f" % (close['count'].mean(), close['count'].std())
plt.xlabel("Station Count")
plt.ylabel("Density")
plt.title(title)
plt.show()

density = stats.gaussian_kde(far['count'])
n, x, _ = plt.hist(far['count'], bins=np.linspace(0, 2000, 100), 
                   histtype=u'step', density=True)  
plt.plot(x, density(x))
title = "Far Station results: mu = %.2f,  std = %.2f" % (far['count'].mean(), far['count'].std())
plt.xlabel("Station Count")
plt.ylabel("Density")
plt.title(title)
plt.show()


size = 10000
x = np.arange(size)
h = plt.hist(df['count'], bins=range(100))

dist_names = ['gamma', 'beta', 'betaprime', 'expon', 'genpareto', 'halfcauchy', 'lomax', 'truncnorm', 'uniform']
dist_results = []
params = {}

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
   
    for dist_name in dist_names:
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(df['count'])
        pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) * size
        plt.plot(pdf_fitted, label=dist_name)
        plt.xlim(0,50)
        plt.ylim(0,500)
        
        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = stats.kstest(df['count'], dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))
        
    plt.legend(loc='upper right')
    plt.show()
    
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value
    
    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))
    


size = 10000
x = np.arange(size)
h = plt.hist(df['Distance'], bins='auto') 
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    for dist_name in dist_names:
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(df['Distance'])
        pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) * size
        plt.plot(pdf_fitted, label=dist_name)
        plt.xlim(0,30)
        plt.ylim(0,3000)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = stats.kstest(df['Distance'], dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    plt.legend(loc='upper right')
    plt.show()

    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

# H0: μ close >= μ far    
# H1: μ close < μ far
# We want "fail to reject H0"    

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(close['count'],far['count'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')
    
print("Two Sample, One-Tail Test")
stat, p = ttest_ind(close['count'],far['count'])
print('Statistics=%.3f, p=%.3f' % (stat, p))    
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')
    

# H0: μ close <= μ far    
# H1: μ close > μ far
# We want "reject H0"    
print("Mann-whitney U Test")
stat, p = mannwhitneyu(close['count'],far['count'],alternative='greater')
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')
    
print("Two Sample, One-Tail Test")
stat, p = ttest_ind(far['count'],close['count'])
print('Statistics=%.3f, p=%.3f' % (stat, p))    
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')


# H0: μ far >= μ overall    
# H1: μ far < μ overall
# We want "reject H0" 
print("Mann-whitney U Test")
stat, p = mannwhitneyu(far['count'],df['count'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')

print("Two Sample, One-Tail Test")
stat, p = ttest_ind(far['count'],df['count'])
print('Statistics=%.3f, p=%.3f' % (stat, p))    
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')
    
    
# H0: μ close >= μ overall    
# H1: μ close < μ overall
# We want "fail to reject H0" 
print("Mann-whitney U Test")
stat, p = mannwhitneyu(close['count'],df['count'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')
    
print("Two Sample, One-Tail Test")
stat, p = ttest_ind(close['count'],df['count'])
print('Statistics=%.3f, p=%.3f' % (stat, p))    
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')


# k2, p = stats.normaltest(afternoonusage['count'])

# -----------------------------------------------------------------------------
# Correlation Test
# -----------------------------------------------------------------------------
print("Correlation Test\n")
stations = df.groupby(['Station','Distance', 'Latitude', 'Longitude'], as_index=False).sum()

stationdist = stations['Distance']
stationfootfall = stations['count']

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
plt.title('Station Footfall vs Distance', fontsize=18)
plt.ylabel('Footfall', fontsize=16)
plt.xlabel('Distance from City of London (miles)', fontsize=16)


# *****************************************************************************
# Hypothesis Test Two:
# *****************************************************************************
# Station congestion mainly occurs in the morning and evening i.e. peak times.
# The peak times would occur mainly in mornings due to school and work which
# typically would mean an equal congestion rate in evenings (day end).
# The data would need to be seperated in to morning/evening and afternoon/night.
#
# H0:  μ1 >= μ2   i.e. more congestion at morning and evening than afternoon and night
# H1:  μ1 < μ2    i.e. more congestion in afternoon and night
# -----------------------------------------------------------------------------
print("\nHypothesis Test Two")
print("Station congestion mainly occurs in the morning and evening i.e. peak times.\n")

# Creating time series plot
ts = pd.DataFrame(df[['timestart', 'count' ]])
ts = ts.groupby(['timestart']).sum()
ts.plot(color='crimson', linewidth=2.0)
plt.title('Time Series of Station usage', fontsize=18)
plt.ylabel('Station Usage', fontsize=8)
plt.xlabel('Time (24 hours)', fontsize=8)
plt.show()

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

print("Augmented Dickey-Fuller test to check for time dependency.")
result = adfuller(ts['count'])
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
    
    
# Processing data in to collected form for testing.    
morningusage = df.loc[(df.timestart >= "05:00:00") & (df.timestart < "12:00:00")]
morningcount = morningusage['count'].mean()

density = stats.gaussian_kde(morningusage['count'])
n, x, _ = plt.hist(morningusage['count'], bins=np.linspace(0, 4000, 100), 
                   histtype=u'step', density=True)  
plt.plot(x, density(x))
title = "Morning Station results: mu = %.2f,  std = %.2f" % (morningusage['count'].mean(), morningusage['count'].std())
plt.title(title)
plt.show()

afternoonusage = df.loc[(df.timestart >= "12:00:00") & (df.timestart < "17:00:00")]
afternooncount = afternoonusage['count'].mean()

eveningusage = df.loc[(df.timestart >= "17:00:00") & (df.timestart < "21:00:00")]
eveningcount = eveningusage['count'].mean()

nightusage1 = df.loc[(df.timestart >= "21:00:00")]
nightusage2 = df.loc[(df.timestart < "05:00:00")]
nightusage = pd.concat([nightusage1,nightusage2])
nightcount = nightusage['count'].mean()


# H0: μ morning >= μ afternoon    
# H1: μ morning < μ afternoon 
# We want "reject H0"
print("Morning vs Afternoon")

print("Two Sample, One-Tail Test")
stat, p = ttest_ind(morningusage['count'],afternoonusage['count'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')

print("Mann-whitney U Test")
stat, p = mannwhitneyu(morningusage['count'],afternoonusage['count'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')
    
# H0: μ morning >= μ night    
# H1: μ morning < μ night 
# We want "fail to reject H0"

print("Morning vs Night")

print("Two Sample, One-Tail Test")
stat, p = ttest_ind(morningusage['count'],nightusage['count'],equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')
stat, p = mannwhitneyu(morningusage['count'],nightusage['count'],alternative='greater')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')
    
    
print("Morning vs Evening")

print("Two Sample, One-Tail Test")
stat, p = ttest_ind(morningusage['count'],eveningusage['count'],equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')
stat, p = mannwhitneyu(morningusage['count'],eveningusage['count'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')

print("Evening vs Afternoon")

print("Two Sample, One-Tail Test")
stat, p = ttest_ind(eveningusage['count'],afternoonusage['count'],equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')
stat, p = mannwhitneyu(eveningusage['count'],afternoonusage['count'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')
    

print("Evening vs Night")

print("Two Sample, One-Tail Test")
stat, p = ttest_ind(eveningusage['count'],nightusage['count'],equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')
stat, p = mannwhitneyu(eveningusage['count'],nightusage['count'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')
 

# *****************************************************************************
# Hypothesis Test Three:
# *****************************************************************************
# More people travel in to London during morning hours i.e. morning and afternoon
# Rather than out of London.
# In comparison during evenings and nights, more people travelling out of 
# central London than in to central London.
#
# H0:  μ1 >= μ2   i.e. more congestion at morning and evening than afternoon and night
# H1:  μ1 < μ2    i.e. more congestion in afternoon and night
# -----------------------------------------------------------------------------
print("\nHypothesis Test Three")
print("More people travel in to London rather than out of London\n")
print("Checking close stations\n")

# Processing close stations by time
dayentry = entry.loc[(entry.timestart >= "05:00:00") & (entry.timestart < "17:00:00")]
dayentry = dayentry.merge(df[['Station','Distance']],on='Station',how='left')
dayentry = dayentry.drop(dayentry.columns[0], axis=1)
dayentry = dayentry[dayentry['Distance'] <= meandist]
dayentrymedian = dayentry['count'].median()

dayexits = exits.loc[(exits.timestart >= "05:00:00") & (exits.timestart < "17:00:00")]
dayexits = dayexits.merge(df[['Station','Distance']],on='Station',how='left')
dayexits = dayexits.drop(dayexits.columns[0], axis=1)
dayexits = dayexits[dayexits['Distance'] <= meandist]
dayexitsmedian = dayexits['count'].median()

nightentry1 = entry.loc[(entry.timestart >= "17:00:00")].sort_values(by=['timestart'])
nightentry2 = entry.loc[(entry.timestart < "05:00:00")].sort_values(by=['timestart'])
nightentry = pd.concat([nightentry1,nightentry2])
nightentry = nightentry.merge(df[['Station','Distance']],on='Station',how='left')
nightentry = nightentry.drop(nightentry.columns[0], axis=1)
nightentry = nightentry[nightentry['Distance'] <= meandist]
nightentrymedian = nightentry['count'].median()

nightexits1 = exits.loc[(exits.timestart >= "17:00:00")].sort_values(by=['timestart'])
nightexits2 = exits.loc[(exits.timestart < "05:00:00")].sort_values(by=['timestart'])
nightexits = pd.concat([nightexits1,nightexits2])
nightexits = nightexits.merge(df[['Station','Distance']],on='Station',how='left')
nightexits = nightexits.drop(nightexits.columns[0], axis=1)
nightexits = nightexits[nightexits['Distance'] <= meandist]
nightexitsmedian = nightexits['count'].median()


# Daytime Station Plot
dayentry = dayentry.sort_values(by=['timestart'])
dayentryplot = dayentry.groupby(['timestart'], as_index=False).sum()
dayexits = dayexits.sort_values(by=['timestart'])
dayexitsplot = dayexits.groupby(['timestart'], as_index=False).sum()

labelsday = dayentry['timestart'].unique()
labelsday = labelsday.tolist()
labelsday.sort()
ind = np.arange(len(labelsday)) 
width = 0.35       
plt.figure(figsize=(20,8))
plt.bar(ind, dayentryplot['count'].tolist(), width, label='Entries')
plt.bar(ind + width, dayexitsplot['count'].tolist(), width,
    label='Exits')
plt.ylabel('Station Usage')
plt.title('Close Station Day Entries vs Exits')
plt.xticks(np.arange(min(ind), max(ind)+1, 1.0), labelsday, rotation=90)
plt.legend(loc='best')
plt.show()

# Night Station PLot

nightentryplot = nightentry.groupby(['timestart'], sort=False, as_index=False).sum()

nightexitsplot = nightexits.groupby(['timestart'], sort=False, as_index=False).sum()

labelsnight = nightentryplot['timestart'].unique()
labelsnight = labelsnight.tolist()
ind = np.arange(len(labelsnight)) 
width = 0.35 
plt.figure(figsize=(20,8))     
plt.bar(ind, nightentryplot['count'].tolist(), width, label='Entries')
plt.bar(ind + width, nightexitsplot['count'].tolist(), width,
    label='Exits')
plt.ylabel('Station Usage')
plt.title('Close Station Night Entries vs Exits')
plt.xticks(np.arange(min(ind), max(ind)+1, 1.0), labelsnight, rotation=90)
plt.legend(loc='best')
plt.show()


# H0:  μ dayentry = μ dayexits
# H1:  μ dayentry != μ dayexits
# We want "reject H0"
print("Day Entry vs Day Exits")
print("Two Sample, One-Tail Test")
stat, p = ttest_ind(dayentryplot['count'],dayexitsplot['count'], equal_var = False) 
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')
    
# H0: μ dayentry >= μ dayexits   
# H1: μ dayentry < μ dayexits     
# We want "fail to reject H0" 
print("Mann-whitney U Test")   
stat, p = mannwhitneyu(dayentryplot['count'],dayexitsplot['count'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')


# H0:  μ nightentry = μ nightexits
# H1:  μ nightentry != μ nightexits
# We want "reject H0"
print("Night Entry vs Night Exits")
print("Two Sample, One-Tail Test")
stat, p = ttest_ind(nightentryplot['count'],nightexitsplot['count'],equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')
    
# H0: μ nightentry >= μ nightexits   
# H1: μ nightentry < μ nightexits     
# We want "reject H0"     
print("Mann-whitney U Test")  
stat, p = mannwhitneyu(nightentryplot['count'],nightexitsplot['count'],alternative='less')  
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')

# -----------------------------------------------------------------------------
print("\nChecking far stations\n")
# Processing far stations by time
dayentry = entry.loc[(entry.timestart >= "05:00:00") & (entry.timestart < "17:00:00")]
dayentry = dayentry.merge(df[['Station','Distance']],on='Station',how='left')
dayentry = dayentry.drop(dayentry.columns[0], axis=1)
dayentry = dayentry[dayentry['Distance'] > meandist]
dayentrymedian = dayentry['count'].median()

dayexits = exits.loc[(exits.timestart >= "05:00:00") & (exits.timestart < "17:00:00")]
dayexits = dayexits.merge(df[['Station','Distance']],on='Station',how='left')
dayexits = dayexits.drop(dayexits.columns[0], axis=1)
dayexits = dayexits[dayexits['Distance'] > meandist]
dayexitsmedian = dayexits['count'].median()

nightentry1 = entry.loc[(entry.timestart >= "17:00:00")].sort_values(by=['timestart'])
nightentry2 = entry.loc[(entry.timestart < "05:00:00")].sort_values(by=['timestart'])
nightentry = pd.concat([nightentry1,nightentry2])
nightentry = nightentry.merge(df[['Station','Distance']],on='Station',how='left')
nightentry = nightentry.drop(nightentry.columns[0], axis=1)
nightentry = nightentry[nightentry['Distance'] > meandist]
nightentrymedian = nightentry['count'].median()

nightexits1 = exits.loc[(exits.timestart >= "17:00:00")].sort_values(by=['timestart'])
nightexits2 = exits.loc[(exits.timestart < "05:00:00")].sort_values(by=['timestart'])
nightexits = pd.concat([nightexits1,nightexits2])
nightexits = nightexits.merge(df[['Station','Distance']],on='Station',how='left')
nightexits = nightexits.drop(nightexits.columns[0], axis=1)
nightexits = nightexits[nightexits['Distance'] > meandist]
nightexitsmedian = nightexits['count'].median()


# Daytime Station Plot
dayentry = dayentry.sort_values(by=['timestart'])
dayentryplot = dayentry.groupby(['timestart'], as_index=False).sum()
dayexits = dayexits.sort_values(by=['timestart'])
dayexitsplot = dayexits.groupby(['timestart'], as_index=False).sum()

labelsday = dayentry['timestart'].unique()
labelsday = labelsday.tolist()
labelsday.sort()
ind = np.arange(len(labelsday)) 
width = 0.35    
plt.figure(figsize=(20,8))   
plt.bar(ind, dayentryplot['count'].tolist(), width, label='Entries')
plt.bar(ind + width, dayexitsplot['count'].tolist(), width,
    label='Exits')
plt.ylabel('Station Usage')
plt.title('Far Station Day Entries vs Exits')
plt.xticks(np.arange(min(ind), max(ind)+1, 1.0), labelsday, rotation=90)
plt.legend(loc='best')
plt.show()

# Night Station PLot

nightentryplot = nightentry.groupby(['timestart'], sort=False, as_index=False).sum()

nightexitsplot = nightexits.groupby(['timestart'], sort=False, as_index=False).sum()

labelsnight = nightentryplot['timestart'].unique()
labelsnight = labelsnight.tolist()
ind = np.arange(len(labelsnight)) 
width = 0.35   
plt.figure(figsize=(20,8))    
plt.bar(ind, nightentryplot['count'].tolist(), width, label='Entries')
plt.bar(ind + width, nightexitsplot['count'].tolist(), width,
    label='Exits')
plt.ylabel('Station Usage')
plt.title('Far Station Night Entries vs Exits')
plt.xticks(np.arange(min(ind), max(ind)+1, 1.0), labelsnight, rotation=90)
plt.legend(loc='best')
plt.show()


# H0:  μ dayentry = μ dayexits
# H1:  μ dayentry != μ dayexits
# We want "reject H0"
print("Day Entry vs Day Exits")
print("Two Sample, One-Tail Test")
stat, p = ttest_ind(dayentryplot['count'],dayexitsplot['count'],equal_var = False) 
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')
    
# H0: μ dayentry >= μ dayexits   
# H1: μ dayentry < μ dayexits     
# We want "fail to reject H0" 
print("Mann-whitney U Test")   
stat, p = mannwhitneyu(dayentryplot['count'],dayexitsplot['count'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')


# H0:  μ nightentry = μ nightexits
# H1:  μ nightentry != μ nightexits
# We want "reject H0"
print("Night Entry vs Night Exits")
print("Two Sample, One-Tail Test")
stat, p = ttest_ind(nightentryplot['count'],nightexitsplot['count'],equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')
    
# H0: μ nightentry >= μ nightexits   
# H1: μ nightentry < μ nightexits     
# We want "reject H0"     
print("Mann-whitney U Test")  
stat, p = mannwhitneyu(nightentryplot['count'],nightexitsplot['count'],alternative='less')  
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')
