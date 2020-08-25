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
from scipy.stats import ttest_ind, mannwhitneyu, stats, beta
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import warnings
from statsmodels.tsa.stattools import adfuller

# =============================================================================
# Reading files
# =============================================================================
df = pd.read_csv('../Data/UberWeekendAnalysis.csv')
df = df.drop(df.columns[0], axis=1)

non_analysis = ['sourceid', 'dstid', 'geometry_x', 'geometry_y']

for i in non_analysis:
    df.pop(i)
    
out_time = []
time = df['mean_travel_time'].tolist()
t2 = datetime(1900,1,1)

for i in range(len(time)):
    convert = datetime.strptime(time[i], "%H:%M:%S")
    adjust = round(((convert-t2).total_seconds()/60))
    out_time.append(adjust)
 
df['mean_travel_time'] =  out_time
MeanJourneyTime = df['mean_travel_time'].mean()

density = stats.gaussian_kde(df['mean_travel_time'])
n, x, _ = plt.hist(df['mean_travel_time'], bins=np.linspace(0, 120, 20), 
                   histtype=u'step', density=True)  
plt.plot(x, density(x))
title = "Fit results: mu = %.2f,  std = %.2f" % (MeanJourneyTime, df['mean_travel_time'].std())
plt.xlabel("Trip Duration (Minutes)")
plt.ylabel("Frequency")
plt.title(title)
plt.show()

# *****************************************************************************
# Hypothesis Test One:
# *****************************************************************************
# Statisitcal test to see if the hypothesis is true:
# The hypothesis that there is a high probability that uber journeys are only 
# used when they are below a specific travel duration. So a test to see if this 
# is true would be to perform a 1 sample test to see the probability of a journey 
# being less than the mean time of all journeys.
#
# H0:  μ travel time <=  μ of all journeys
# H1:  μ travel time >  μ of all journeys   
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# CHECKING DISTRIBUTION
# https://stackoverflow.com/questions/37487830/how-to-find-probability-distribution-and-parameters-for-real-data-python-3
# -----------------------------------------------------------------------------
size = 200000
x = np.arange(size)
h = plt.hist(df['mean_travel_time'], bins=range(120))

dist_names = ['gamma', 'beta', 'rayleigh', 'norm', 'pareto']
dist_results = []
dist_resultsD = []
params = {}


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
   
    for dist_name in dist_names:
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(df['mean_travel_time'])
        pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) * size
        plt.plot(pdf_fitted, label=dist_name)
        plt.xlabel("Trip Duration (Minutes)")
        plt.ylabel("Frequency")
        plt.xlim(0,50)
        plt.ylim(0,8000)
        
        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = stats.kstest(df['mean_travel_time'], dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        print("D value for "+dist_name+" = "+str(D)+"\n")
        dist_results.append((dist_name, p))
        dist_resultsD.append((dist_name, D))
        
    plt.legend(loc='upper right')
    plt.show()
    
    
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    
    if best_p < 0.001:
        best_dist, best_D = (min(dist_resultsD, key=lambda item: item[1]))
    # store the name of the best fit and its p value
    
    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Best KS Statistic value: "+ str(best_D))
    print("Parameters for the best fit: "+ str(params[best_dist]))
    
#Beta distribution params: (α , β, loc (lower limit), scale (upper limit - lower limit))

# Estimating alpha and beta 
# μ=α/α+β 
# σ^2=αβ/(α+β)^2(α+β+1)

a, b = 3.4867713379090257, 12.34618982515687
mean, var = beta.stats(a, b)
probability = stats.beta.cdf(mean,a,b)
print("According to Beta Distribution, we have that the probability of an uber journey being less than 24mins is " + str(round(probability*100,2)) + "%")



# *****************************************************************************
# Hypothesis Test Two:
# *****************************************************************************
# Most uber routes are ones that cross central London
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

# Importing City of London polygon
CoL = gpd.read_file('../Data/cityoflondon.geojson')
CoL_poly = CoL['geometry'].tolist()

intersect = []
within = []
for i in range(len(list_lines)):
    intersect.append(list_lines[i].intersects(CoL_poly[0]))
    within.append(list_lines[i].within(CoL_poly[0]))
    
df['Intersects'] = intersect
df['Within'] = within
fig, ax = plt.subplots()
df['Intersects'].value_counts().plot(ax=ax, kind='bar')

ts = pd.DataFrame(df[['start_time', 'Intersects']])
ts = ts.groupby(['start_time', 'Intersects']).size().reset_index(name='Uber Rides')
londonint = ts[ts["Intersects"]==True]
nolondonint = ts[ts["Intersects"]==False]

fig, ax = plt.subplots()
labels = ts['start_time'].unique()
labels = labels.tolist()
ind = np.arange(len(labels)) 
width = 0.35    
plt.figure(figsize=(15,5))      
plt.bar(ind, londonint['Uber Rides'].tolist(), width, label='Crosses City of London')
plt.bar(ind + width, nolondonint['Uber Rides'].tolist(), width,
    label='Does Not Cross City of London')
plt.ylabel('No. of Uber Rides')
plt.title('Weekend Uber Rides Crossing City of London')
plt.xticks(np.arange(min(ind), max(ind)+1, 1.0), labels, rotation=90)
plt.legend(loc='best')
plt.show()

print("\nCentral vs Not Central")
print("T-Test one sample vs mean")
stat, p = ttest_1samp(londonint['Uber Rides'],nolondonint['Uber Rides'].mean())
print('Statistics=%.3f, p=%.3f' % (stat, p))
    
print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(londonint['Uber Rides'],nolondonint['Uber Rides'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(londonint['Uber Rides'],nolondonint['Uber Rides'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')

# *****************************************************************************
# Hypothesis Test Three:
# *****************************************************************************
# When do most uber journeys occur.
# According to these visuals, peak times occur later at evening and night rather 
# than morning/afternoons for weekdays. Suggests that due to work commitments more people use 
# public transport or personal vehicles rather than uber.
#
# 
# -----------------------------------------------------------------------------

ts2 = pd.DataFrame(df[['start_time']])
ts2 = ts2.groupby(['start_time']).size().reset_index(name='Uber Rides')


fig, ax = plt.subplots()
ts2.plot(color='purple', linewidth=2.0)
plt.title('Time Series of Uber Rides', fontsize=18)
plt.ylabel('No. Of Uber Rides', fontsize=8)
plt.xlabel('Time (24 hours)', fontsize=8)
plt.show()

print("Augmented Dickey-Fuller test to check for time dependency.")
result = adfuller(ts2['Uber Rides'])
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
    
day = ts2[(ts2['start_time']>="05:00:00") & (ts2['start_time'] < "17:00:00")]
night1 = ts2[(ts2['start_time']>= "17:00:00")]
night2 = ts2[(ts2['start_time'] < "05:00:00")]
night = pd.concat([night1,night2])

print("Day vs Night")
print("T-Test one sample vs mean")
stat, p = ttest_1samp(night['Uber Rides'],day['Uber Rides'].mean())
print('Statistics=%.3f, p=%.3f' % (stat, p))

print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(night['Uber Rides'],day['Uber Rides'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(day['Uber Rides'],night['Uber Rides'],alternative='less')
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')