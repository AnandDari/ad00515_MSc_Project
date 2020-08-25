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
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
# =============================================================================

# =============================================================================
# Reading files
# =============================================================================
weekday = pd.read_csv('../Data/UberWeekdayProcessedv4.csv')
weekday = weekday.drop(weekday.columns[0], axis=1)

non_analysis = ['sourceid', 'dstid', 'geometry_x', 'geometry_y']

for i in non_analysis:
    weekday.pop(i)
    
out_time = []
time = weekday['mean_travel_time'].tolist()
t2 = datetime(1900,1,1)

for i in range(len(time)):
    convert = datetime.strptime(time[i], "%H:%M:%S")
    adjust = round(((convert-t2).total_seconds()/60))
    out_time.append(adjust)
 
weekday['mean_travel_time'] =  out_time
MeanJourneyTime = weekday['mean_travel_time'].mean()


weekend = pd.read_csv('../Data/UberWeekendAnalysis.csv')
weekend = weekend.drop(weekend.columns[0], axis=1)

non_analysis = ['sourceid', 'dstid', 'geometry_x', 'geometry_y']

for i in non_analysis:
    weekend.pop(i)
    
out_time = []
time = weekend['mean_travel_time'].tolist()
t2 = datetime(1900,1,1)

for i in range(len(time)):
    convert = datetime.strptime(time[i], "%H:%M:%S")
    adjust = round(((convert-t2).total_seconds()/60))
    out_time.append(adjust)
 
weekend['mean_travel_time'] =  out_time
MeanJourneyTime = weekend['mean_travel_time'].mean()


# =============================================================================
# Plotting Data
# =============================================================================
weekday = weekday.groupby(['start_time']).size().reset_index(name='Uber Rides')
weekend = weekend.groupby(['start_time']).size().reset_index(name='Uber Rides')

labels = weekday['start_time'].unique()
labels = labels.tolist()
ind = np.arange(len(labels)) 
width = 0.35 
plt.figure(figsize=(15,5))       
plt.bar(ind, weekday['Uber Rides'].tolist(), width, label='Weekday')
plt.bar(ind + width, weekend['Uber Rides'].tolist(), width,
    label='Weekend')
plt.ylabel('No. Of Uber Rides')
plt.xlabel('Time (24 hours)')
plt.title('Uber Rides - Weekday vs Weekend')
plt.xticks(np.arange(min(ind), max(ind)+1, 1.0), labels, rotation=90)
plt.legend(loc='best')
plt.show()

# =============================================================================
# Tests
# =============================================================================
print("Weekday vs Weekend Analysis")

print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(weekday['Uber Rides'],weekend['Uber Rides'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')    

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(weekday['Uber Rides'],weekend['Uber Rides'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')    

weekday_day = weekday[(weekday['start_time']>="05:00:00") & (weekday['start_time'] < "17:00:00")]
weekday_night1 = weekday[(weekday['start_time']>= "17:00:00")]
weekday_night2 = weekday[(weekday['start_time'] < "05:00:00")]
weekday_night = pd.concat([weekday_night1,weekday_night2])

weekend_day = weekend[(weekend['start_time']>="05:00:00") & (weekend['start_time'] < "17:00:00")]
weekend_night1 = weekend[(weekend['start_time']>= "17:00:00")]
weekend_night2 = weekend[(weekend['start_time'] < "05:00:00")]
weekend_night = pd.concat([weekend_night1,weekend_night2])

labels = weekday_day['start_time'].unique()
labels = labels.tolist()
ind = np.arange(len(labels)) 
width = 0.35 
plt.figure(figsize=(15,5))       
plt.bar(ind, weekday_day['Uber Rides'].tolist(), width, label='Weekday')
plt.bar(ind + width, weekend_day['Uber Rides'].tolist(), width,
    label='Weekend')
plt.ylabel('No. Of Uber Rides')
plt.xlabel('Time (24 hours)')
plt.title('Uber Rides Day Hours - Weekday vs Weekend')
plt.xticks(np.arange(min(ind), max(ind)+1, 1.0), labels, rotation=90)
plt.legend(loc='best')
plt.show()


labels = weekday_night['start_time'].unique()
labels = labels.tolist()
ind = np.arange(len(labels)) 
width = 0.35 
plt.figure(figsize=(15,5))       
plt.bar(ind, weekday_night['Uber Rides'].tolist(), width, label='Weekday')
plt.bar(ind + width, weekend_night['Uber Rides'].tolist(), width,
    label='Weekend')
plt.ylabel('No. Of Uber Rides')
plt.xlabel('Time (24 hours)')
plt.title('Uber Rides Night Hours - Weekday vs Weekend')
plt.xticks(np.arange(min(ind), max(ind)+1, 1.0), labels, rotation=90)
plt.legend(loc='best')
plt.show()

print("Weekday vs Weekend Analysis - Day hours")

print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(weekday_day['Uber Rides'],weekend_day['Uber Rides'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')    

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(weekday_day['Uber Rides'],weekend_day['Uber Rides'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')    

print("Weekday vs Weekend Analysis - Night hours")

print("\nTwo Sample, One-Tailed test")
stat, p = ttest_ind(weekday_night['Uber Rides'],weekend_night['Uber Rides'], equal_var = False)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')    

print("\nMann-whitney U Test")
stat, p = mannwhitneyu(weekday_night['Uber Rides'],weekend_night['Uber Rides'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
	print('Fail to reject H0\n')
else:
	print('Reject H0\n')   