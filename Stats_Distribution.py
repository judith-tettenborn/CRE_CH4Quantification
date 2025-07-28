# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:42:58 2023

@author: Judith Tettenborn (j.a.f.tettenborn@uu.nl)


There are four main assumptions underlying a linear regression model which describe the relation of a response variable Y and a predictor variable X (Von Storch and Zwiers, 2002; Flatt and Jacobs, 2019):

    Linearity: The relationship between X and the mean of Y is linear.
    Homoscedasticity: The variance of residuals is the same for any value of X.
    Independence: Observations are independent of each other.
    Normality: For any fixed value of X, the error terms (residuals) of Y are normally distributed.

Violations of these assumptions can lead to biased and misleading inferences, confidence intervals, and scientific insights.


With the analysis in this script the normality of the data are investigated, using:
    1. Shapiro-Wilk Test and Lilliefors Test
    2. Historgam Plots
    3. Quantile-Quantile Plots
    
    
To be able to plot the histograms and qq-plots you must run the cells under the
header "Shapiro-Wilk + Lilliefors Residuals Area"

"""

# Modify Python Path Programmatically -> To include the directory containing the src folder
from pathlib import Path
import sys

# HARDCODED ####################################################################################

path_base = Path('C:/Users/Judit/Documents/UNI/Utrecht/Hiwi/CRE_CH4Quantification/')
# path_base = Path('C:/Users/.../CRE_CH4Quantification/') # insert the the project path here


################################################################################################

sys.path.append(str(path_base / 'src'))


import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.diagnostic import lilliefors
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from helper_functions.constants import (
    dict_color_city,
    slope_area,
    yintercept_area
    )


#%% LOAD DATA

   
path_finaldata  = path_base / 'data' / 'final'
path_res        = path_base / 'results/'
if not (path_res / 'Figures' / 'STATS' / 'Histogram').is_dir():
       (path_res / 'Figures' / 'STATS' / 'Histogram').mkdir(parents=True)
path_histo    = path_res  / 'Figures' / 'STATS' / 'Histogram/'
if not (path_res / 'Figures' / 'STATS' / 'QQ-Plots').is_dir():
       (path_res / 'Figures' / 'STATS' / 'QQ-Plots').mkdir(parents=True)
path_qq    = path_res  / 'Figures' / 'STATS' / 'QQ-Plots/'



# Utrecht
total_peaks_U1   = pd.read_csv(path_finaldata / 'U1_TOTAL_PEAKS.csv', index_col='Datetime', parse_dates=['Datetime']) 
total_peaks_U2   = pd.read_csv(path_finaldata / 'U2_TOTAL_PEAKS.csv', index_col='Datetime', parse_dates=['Datetime'])  
 
df_U1_comb       = pd.read_csv(path_finaldata / 'U1_comb_final.csv', index_col='Datetime', parse_dates=['Datetime'])
df_U1_comb_count = df_U1_comb.groupby(['Release_rate']).size().reset_index(name='Count')

df_U2_comb       = pd.read_csv(path_finaldata / 'U2_comb_final.csv', index_col='Datetime', parse_dates=['Datetime']) 
df_U2_comb       = df_U2_comb.dropna(subset=['ln(Area)'])
df_U2_comb_count = df_U2_comb.groupby(['Release_rate']).size().reset_index(name='Count')
rr_drop          = df_U2_comb_count[df_U2_comb_count['Count'] < 10]['Release_rate'].tolist() # drop releases with less than 10 detections - to few to make statistics
df_U2_comb       = df_U2_comb[~df_U2_comb['Release_rate'].isin(rr_drop)]
df_U2_comb_count = df_U2_comb.groupby(['Release_rate']).size().reset_index(name='Count')


# Rotterdam
total_peaks_R    = pd.read_csv(path_finaldata / 'R_TOTAL_PEAKS.csv', index_col='Datetime', parse_dates=['Datetime'])  
total_peaks_R    = total_peaks_R[total_peaks_R['Release_rate'] != 0] 

df_Rloc1_comb       = pd.read_csv(path_finaldata / 'R_comb_final.csv', index_col='Datetime', parse_dates=['Datetime']) 
df_Rloc1_comb       = df_Rloc1_comb[(df_Rloc1_comb['Release_rate'] != 0) & (df_Rloc1_comb['Loc'] == 1)]
df_Rloc1_comb_count = df_Rloc1_comb.groupby(['Release_rate']).size().reset_index(name='Count')


# London
total_peaks_L1d2   = pd.read_csv(path_finaldata / 'L1_TOTAL_PEAKS_Day2.csv', index_col='Datetime', parse_dates=['Datetime'])  
total_peaks_L1d5   = pd.read_csv(path_finaldata / 'L1_TOTAL_PEAKS_Day5.csv', index_col='Datetime', parse_dates=['Datetime']) 

df_L1d2_comb       = pd.read_csv(path_finaldata / 'L1d2_comb_final.csv', index_col='Datetime', parse_dates=['Datetime'])
df_L1d2_comb_count = df_L1d2_comb.groupby(['Release_rate']).size().reset_index(name='Count')
df_L1d5_comb       = pd.read_csv(path_finaldata / 'L1d5_comb_final.csv', index_col='Datetime', parse_dates=['Datetime']) 
df_L1d5_comb_count = df_L1d5_comb.groupby(['Release_rate']).size().reset_index(name='Count')

# London II
total_peaks_L2d1 = pd.read_csv(path_finaldata / 'L2_TOTAL_PEAKS_Day1.csv', index_col='Datetime', parse_dates=['Datetime'])  
total_peaks_L2d2 = pd.read_csv(path_finaldata / 'L2_TOTAL_PEAKS_Day2.csv', index_col='Datetime', parse_dates=['Datetime'])  
total_peaks_L2d2 = total_peaks_L2d2[total_peaks_L2d2['Release_rate'] > 0.2] # only one peak at 0.2 L/min -> delete

df_L2d1_comb       = pd.read_csv(path_finaldata / 'L2d1_comb_final.csv', index_col='Datetime', parse_dates=['Datetime']) 
df_L2d1_comb_count = df_L2d1_comb.groupby(['Release_rate']).size().reset_index(name='Count')
df_L2d2_comb       = pd.read_csv(path_finaldata / 'L2d2_comb_final.csv', index_col='Datetime', parse_dates=['Datetime']) 
df_L2d2_comb       = df_L2d2_comb[df_L2d2_comb['Release_rate'] != 0.2]
df_L2d2_comb_count = df_L2d2_comb.groupby(['Release_rate']).size().reset_index(name='Count')



# Calculate residuals ------------------------------------------------------------

df_Rloc1_comb['ln(Area)_residuals'] = df_Rloc1_comb['ln(Area)'] - (slope_area * np.log(df_Rloc1_comb['Release_rate']) + yintercept_area)
df_U1_comb['ln(Area)_residuals']    = df_U1_comb['ln(Area)']    - (slope_area * np.log(df_U1_comb['Release_rate']) + yintercept_area)
df_U2_comb['ln(Area)_residuals']    = df_U2_comb['ln(Area)']    - (slope_area * np.log(df_U2_comb['Release_rate']) + yintercept_area)
df_L1d2_comb['ln(Area)_residuals']  = df_L1d2_comb['ln(Area)']  - (slope_area * np.log(df_L1d2_comb['Release_rate']) + yintercept_area)
df_L1d5_comb['ln(Area)_residuals']  = df_L1d5_comb['ln(Area)']  - (slope_area * np.log(df_L1d5_comb['Release_rate']) + yintercept_area)
df_L2d1_comb['ln(Area)_residuals']  = df_L2d1_comb['ln(Area)']  - (slope_area * np.log(df_L2d1_comb['Release_rate']) + yintercept_area)
df_L2d2_comb['ln(Area)_residuals']  = df_L2d2_comb['ln(Area)']  - (slope_area * np.log(df_L2d2_comb['Release_rate']) + yintercept_area)

# ALL
# total_peaks_all_alldist = pd.read_csv(path_finaldata / 'RU2T2L4L2_TOTAL_PEAKS.csv', index_col='Datetime', parse_dates=['Datetime'])  
# df_all_alldist =  pd.read_csv(path_finaldata / 'RU2T2L4L2_TOTAL_PEAKS_comb.csv', index_col='Datetime', parse_dates=['Datetime']) 
# total_peaks_all = total_peaks_all_alldist[total_peaks_all_alldist['Distance_to_source']<75].copy()
# df_all = df_all_alldist[df_all_alldist['Distance_to_source']<75].copy()



#%% Process Data


df_U1   = total_peaks_U1[['Loc','Release_rate','Area_mean_G23','Area_mean_G43','Max_G23','Max_G43']].copy(deep=True)
df_L1d2 = total_peaks_L1d2[['Release_height','Release_rate','Area_mean_LGR','Area_mean_G23','Max_LGR','Max_G23','Peak','Distance_to_source']].copy(deep=True)
df_L1d5 = total_peaks_L1d5[['Release_height','Release_rate','Area_mean_G23','Max_G23','Peak','Distance_to_source']].copy(deep=True)

df_L1d2 = df_L1d2[(df_L1d2['Release_height'] == 0) & (df_L1d2['Distance_to_source'] <75)]
df_L1d5 = df_L1d5[(df_L1d5['Release_height'] == 0) & (df_L1d5['Distance_to_source'] <75)]
df_L1d2.drop(['Release_height','Distance_to_source'],axis=1,inplace=True)
df_L1d5.drop(['Release_height','Distance_to_source'],axis=1,inplace=True)





#%% STATS


#%%% Shapiro-Wilk + Lilliefors Residuals Area


# Set the significance level (alpha)
alpha = 0.05

#%%%% Rotterdam LOC 1 

Release_rates = df_Rloc1_comb_count['Release_rate']
pass_or_fail_shapiro = []
p_values_shapiro = []
statistic_shapiro = []
pass_or_fail_lilliefors = []
p_values_lilliefors = []
statistic_lilliefors = []

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    x = df_Rloc1_comb.loc[df_Rloc1_comb['Release_rate'] == rr, 'ln(Area)_residuals']
    
    # 1. Perform the Shapiro-Wilk test on your data
    statistic, p_value = stats.shapiro(x)

    if p_value > alpha:
        pass_or_fail_shapiro.append('pass')
    else:
        pass_or_fail_shapiro.append('fail')
        
    p_values_shapiro.append(p_value)
    statistic_shapiro.append(statistic)
    
    # 2. Perform the Lilliefors test on your data
    statistic, p_value = lilliefors(x)

    if p_value > alpha:
        pass_or_fail_lilliefors.append('pass')
    else:
        pass_or_fail_lilliefors.append('fail')
        
    p_values_lilliefors.append(p_value)
    statistic_lilliefors.append(statistic)

df_Rloc1_tests_lognormality_residualslnArea = pd.DataFrame({'Release_rate': Release_rates, 'Shapiro_normal': pass_or_fail_shapiro, 
                                        'Shapiro_pvalue': p_values_shapiro, 'Shapiro_statistic': statistic_shapiro,
                                        'Lilliefors_normal': pass_or_fail_lilliefors, 'Lilliefors_pvalue': p_values_lilliefors,
                                        'Lilliefors_statistic': statistic_lilliefors})



    

#%%%% Utrecht I

Release_rates = df_U1_comb_count['Release_rate']
pass_or_fail_shapiro = []
p_values_shapiro = []
statistic_shapiro = []
pass_or_fail_lilliefors = []
p_values_lilliefors = []
statistic_lilliefors = []

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    x = df_U1_comb.loc[df_U1_comb['Release_rate'] == rr, 'ln(Area)_residuals']

    # 1. Perform the Shapiro-Wilk test on your data
    statistic, p_value = stats.shapiro(x)

    if p_value > alpha:
        pass_or_fail_shapiro.append('pass')
    else:
        pass_or_fail_shapiro.append('fail')
        
    p_values_shapiro.append(p_value)
    statistic_shapiro.append(statistic)
    
    # 2. Perform the Lilliefors test on your data
    statistic, p_value = lilliefors(x)

    if p_value > alpha:
        pass_or_fail_lilliefors.append('pass')
    else:
        pass_or_fail_lilliefors.append('fail')
        
    p_values_lilliefors.append(p_value)
    statistic_lilliefors.append(statistic)

df_U1_tests_lognormality_residualslnArea = pd.DataFrame({'Release_rate': Release_rates, 'Shapiro_normal': pass_or_fail_shapiro, 
                                        'Shapiro_pvalue': p_values_shapiro, 'Shapiro_statistic': statistic_shapiro,
                                        'Lilliefors_normal': pass_or_fail_lilliefors, 'Lilliefors_pvalue': p_values_lilliefors,
                                        'Lilliefors_statistic': statistic_lilliefors})


#%%%% Utrecht II

Release_rates = df_U2_comb_count['Release_rate']
pass_or_fail_shapiro = []
p_values_shapiro = []
statistic_shapiro = []
pass_or_fail_lilliefors = []
p_values_lilliefors = []
statistic_lilliefors = []

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    x = df_U2_comb.loc[df_U2_comb['Release_rate'] == rr, 'ln(Area)_residuals']

    # 1. Perform the Shapiro-Wilk test on your data
    statistic, p_value = stats.shapiro(x)

    if p_value > alpha:
        pass_or_fail_shapiro.append('pass')
    else:
        pass_or_fail_shapiro.append('fail')
        
    p_values_shapiro.append(p_value)
    statistic_shapiro.append(statistic)
    
    # 2. Perform the Lilliefors test on your data
    statistic, p_value = lilliefors(x)

    if p_value > alpha:
        pass_or_fail_lilliefors.append('pass')
    else:
        pass_or_fail_lilliefors.append('fail')
        
    p_values_lilliefors.append(p_value)
    statistic_lilliefors.append(statistic)

df_U3_tests_lognormality_residualslnArea = pd.DataFrame({'Release_rate': Release_rates, 'Shapiro_normal': pass_or_fail_shapiro, 
                                        'Shapiro_pvalue': p_values_shapiro, 'Shapiro_statistic': statistic_shapiro,
                                        'Lilliefors_normal': pass_or_fail_lilliefors, 'Lilliefors_pvalue': p_values_lilliefors,
                                        'Lilliefors_statistic': statistic_lilliefors})
  
  
#%%%% London I
 
Release_rates = [35,70,70]
Days = ['Day2','Day2','Day5']

pass_or_fail_shapiro = []
p_values_shapiro = []
statistic_shapiro = []
pass_or_fail_lilliefors = []
p_values_lilliefors = []
statistic_lilliefors = []

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    if i<2:
        x = df_L1d2_comb.loc[df_L1d2_comb['Release_rate'] == rr, 'ln(Area)_residuals']
    else:
        x = df_L1d5_comb.loc[df_L1d5_comb['Release_rate'] == rr, 'ln(Area)_residuals']

    # 1. Perform the Shapiro-Wilk test on your data
    statistic, p_value = stats.shapiro(x)

    if p_value > alpha:
        pass_or_fail_shapiro.append('pass')
    else:
        pass_or_fail_shapiro.append('fail')
        
    p_values_shapiro.append(p_value)
    statistic_shapiro.append(statistic)
    
    # 2. Perform the Lilliefors test on your data
    statistic, p_value = lilliefors(x)

    if p_value > alpha:
        pass_or_fail_lilliefors.append('pass')
    else:
        pass_or_fail_lilliefors.append('fail')
        
    p_values_lilliefors.append(p_value)
    statistic_lilliefors.append(statistic)

df_L1d2and5_tests_lognormality_residualslnArea = pd.DataFrame({'Release_rate': Release_rates, 'Day': Days,
                                         'Shapiro_normal': pass_or_fail_shapiro, 
                                        'Shapiro_pvalue': p_values_shapiro, 'Shapiro_statistic': statistic_shapiro,
                                        'Lilliefors_normal': pass_or_fail_lilliefors, 'Lilliefors_pvalue': p_values_lilliefors,
                                        'Lilliefors_statistic': statistic_lilliefors})
    
#%%%% London II 

Release_rates = pd.concat([df_L2d2_comb_count['Release_rate'], df_L2d1_comb_count['Release_rate']], ignore_index=True)
Days = ['Day2','Day2','Day2','Day2','Day2','Day2','Day2','Day1','Day1']

pass_or_fail_shapiro = []
p_values_shapiro = []
statistic_shapiro = []
pass_or_fail_lilliefors = []
p_values_lilliefors = []
statistic_lilliefors = []

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    if i<2:
        x = df_L2d2_comb.loc[df_L2d2_comb['Release_rate'] == rr, 'ln(Area)_residuals']
    else:
        x = df_L2d1_comb.loc[df_L2d1_comb['Release_rate'] == rr, 'ln(Area)_residuals']

    # 1. Perform the Shapiro-Wilk test on your data
    statistic, p_value = stats.shapiro(x)

    if p_value > alpha:
        pass_or_fail_shapiro.append('pass')
    else:
        pass_or_fail_shapiro.append('fail')
        
    p_values_shapiro.append(p_value)
    statistic_shapiro.append(statistic)
    
    # 2. Perform the Lilliefors test on your data
    statistic, p_value = lilliefors(x)

    if p_value > alpha:
        pass_or_fail_lilliefors.append('pass')
    else:
        pass_or_fail_lilliefors.append('fail')
        
    p_values_lilliefors.append(p_value)
    statistic_lilliefors.append(statistic)

df_L2d1and2_tests_lognormality_residualslnArea = pd.DataFrame({'Release_rate': Release_rates, 'Day': Days,
                                         'Shapiro_normal': pass_or_fail_shapiro, 
                                        'Shapiro_pvalue': p_values_shapiro, 'Shapiro_statistic': statistic_shapiro,
                                        'Lilliefors_normal': pass_or_fail_lilliefors, 'Lilliefors_pvalue': p_values_lilliefors,
                                        'Lilliefors_statistic': statistic_lilliefors})



#%%% Shapiro-Wilk + Lilliefors Area


# Set the significance level (alpha)
alpha = 0.05



#%%%% Rotterdam LOC 1 

Release_rates = df_Rloc1_comb_count['Release_rate']
pass_or_fail_shapiro = []
p_values_shapiro = []
statistic_shapiro = []
pass_or_fail_lilliefors = []
p_values_lilliefors = []
statistic_lilliefors = []

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    x = df_Rloc1_comb.loc[df_Rloc1_comb['Release_rate'] == rr, 'ln(Area)']
    
    # 1. Perform the Shapiro-Wilk test on your data
    statistic, p_value = stats.shapiro(x)

    if p_value > alpha:
        pass_or_fail_shapiro.append('pass')
    else:
        pass_or_fail_shapiro.append('fail')
        
    p_values_shapiro.append(p_value)
    statistic_shapiro.append(statistic)
    
    # 2. Perform the Lilliefors test on your data
    statistic, p_value = lilliefors(x)

    if p_value > alpha:
        pass_or_fail_lilliefors.append('pass')
    else:
        pass_or_fail_lilliefors.append('fail')
        
    p_values_lilliefors.append(p_value)
    statistic_lilliefors.append(statistic)

df_Rloc1_tests_lognormality = pd.DataFrame({'Release_rate': Release_rates, 'Shapiro_normal': pass_or_fail_shapiro, 
                                        'Shapiro_pvalue': p_values_shapiro, 'Shapiro_statistic': statistic_shapiro,
                                        'Lilliefors_normal': pass_or_fail_lilliefors, 'Lilliefors_pvalue': p_values_lilliefors,
                                        'Lilliefors_statistic': statistic_lilliefors})
    

#%%%% Utrecht I

Release_rates = df_U1_comb_count['Release_rate']
pass_or_fail_shapiro = []
p_values_shapiro = []
statistic_shapiro = []
pass_or_fail_lilliefors = []
p_values_lilliefors = []
statistic_lilliefors = []

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    x = df_U1_comb.loc[df_U1_comb['Release_rate'] == rr, 'ln(Area)']

    # 1. Perform the Shapiro-Wilk test on your data
    statistic, p_value = stats.shapiro(x)

    if p_value > alpha:
        pass_or_fail_shapiro.append('pass')
    else:
        pass_or_fail_shapiro.append('fail')
        
    p_values_shapiro.append(p_value)
    statistic_shapiro.append(statistic)
    
    # 2. Perform the Lilliefors test on your data
    statistic, p_value = lilliefors(x)

    if p_value > alpha:
        pass_or_fail_lilliefors.append('pass')
    else:
        pass_or_fail_lilliefors.append('fail')
        
    p_values_lilliefors.append(p_value)
    statistic_lilliefors.append(statistic)

df_U1_tests_lognormality = pd.DataFrame({'Release_rate': Release_rates, 'Shapiro_normal': pass_or_fail_shapiro, 
                                        'Shapiro_pvalue': p_values_shapiro, 'Shapiro_statistic': statistic_shapiro,
                                        'Lilliefors_normal': pass_or_fail_lilliefors, 'Lilliefors_pvalue': p_values_lilliefors,
                                        'Lilliefors_statistic': statistic_lilliefors})


#%%%% Utrecht II

Release_rates = df_U2_comb_count['Release_rate']
pass_or_fail_shapiro = []
p_values_shapiro = []
statistic_shapiro = []
pass_or_fail_lilliefors = []
p_values_lilliefors = []
statistic_lilliefors = []

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    x = df_U2_comb.loc[df_U2_comb['Release_rate'] == rr, 'ln(Area)']

    # 1. Perform the Shapiro-Wilk test on your data
    statistic, p_value = stats.shapiro(x)

    if p_value > alpha:
        pass_or_fail_shapiro.append('pass')
    else:
        pass_or_fail_shapiro.append('fail')
        
    p_values_shapiro.append(p_value)
    statistic_shapiro.append(statistic)
    
    # 2. Perform the Lilliefors test on your data
    statistic, p_value = lilliefors(x)

    if p_value > alpha:
        pass_or_fail_lilliefors.append('pass')
    else:
        pass_or_fail_lilliefors.append('fail')
        
    p_values_lilliefors.append(p_value)
    statistic_lilliefors.append(statistic)

df_U3_tests_lognormality = pd.DataFrame({'Release_rate': Release_rates, 'Shapiro_normal': pass_or_fail_shapiro, 
                                        'Shapiro_pvalue': p_values_shapiro, 'Shapiro_statistic': statistic_shapiro,
                                        'Lilliefors_normal': pass_or_fail_lilliefors, 'Lilliefors_pvalue': p_values_lilliefors,
                                        'Lilliefors_statistic': statistic_lilliefors})

    
#%%%% London I 

Release_rates = [35,70,70]
Days = ['Day2','Day2','Day5']

pass_or_fail_shapiro = []
p_values_shapiro = []
statistic_shapiro = []
pass_or_fail_lilliefors = []
p_values_lilliefors = []
statistic_lilliefors = []

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    if i<2:
        x = df_L1d2_comb.loc[df_L1d2_comb['Release_rate'] == rr, 'ln(Area)']
    else:
        x = df_L1d5_comb.loc[df_L1d5_comb['Release_rate'] == rr, 'ln(Area)']

    # 1. Perform the Shapiro-Wilk test on your data
    statistic, p_value = stats.shapiro(x)

    if p_value > alpha:
        pass_or_fail_shapiro.append('pass')
    else:
        pass_or_fail_shapiro.append('fail')
        
    p_values_shapiro.append(p_value)
    statistic_shapiro.append(statistic)
    
    # 2. Perform the Lilliefors test on your data
    statistic, p_value = lilliefors(x)

    if p_value > alpha:
        pass_or_fail_lilliefors.append('pass')
    else:
        pass_or_fail_lilliefors.append('fail')
        
    p_values_lilliefors.append(p_value)
    statistic_lilliefors.append(statistic)

df_L1d2and5_tests_lognormality = pd.DataFrame({'Release_rate': Release_rates, 'Day': Days,
                                         'Shapiro_normal': pass_or_fail_shapiro, 
                                        'Shapiro_pvalue': p_values_shapiro, 'Shapiro_statistic': statistic_shapiro,
                                        'Lilliefors_normal': pass_or_fail_lilliefors, 'Lilliefors_pvalue': p_values_lilliefors,
                                        'Lilliefors_statistic': statistic_lilliefors})


#%%%% London II 

Release_rates = pd.concat([df_L2d2_comb_count['Release_rate'], df_L2d1_comb_count['Release_rate']], ignore_index=True)
Days = ['Day2','Day2','Day2','Day2','Day2','Day2','Day2','Day1','Day1']

pass_or_fail_shapiro = []
p_values_shapiro = []
statistic_shapiro = []
pass_or_fail_lilliefors = []
p_values_lilliefors = []
statistic_lilliefors = []

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    if i<2:
        x = df_L2d2_comb.loc[df_L2d2_comb['Release_rate'] == rr, 'ln(Area)']
    else:
        x = df_L2d1_comb.loc[df_L2d1_comb['Release_rate'] == rr, 'ln(Area)']

    # 1. Perform the Shapiro-Wilk test on your data
    statistic, p_value = stats.shapiro(x)

    if p_value > alpha:
        pass_or_fail_shapiro.append('pass')
    else:
        pass_or_fail_shapiro.append('fail')
        
    p_values_shapiro.append(p_value)
    statistic_shapiro.append(statistic)
    
    # 2. Perform the Lilliefors test on your data
    statistic, p_value = lilliefors(x)

    if p_value > alpha:
        pass_or_fail_lilliefors.append('pass')
    else:
        pass_or_fail_lilliefors.append('fail')
        
    p_values_lilliefors.append(p_value)
    statistic_lilliefors.append(statistic)

df_L2d1and2_tests_lognormality = pd.DataFrame({'Release_rate': Release_rates, 'Day': Days,
                                         'Shapiro_normal': pass_or_fail_shapiro, 
                                        'Shapiro_pvalue': p_values_shapiro, 'Shapiro_statistic': statistic_shapiro,
                                        'Lilliefors_normal': pass_or_fail_lilliefors, 'Lilliefors_pvalue': p_values_lilliefors,
                                        'Lilliefors_statistic': statistic_lilliefors})
    

#%%%% London II Day1

Release_rates = df_L2d1_comb_count['Release_rate']
pass_or_fail_shapiro = []
p_values_shapiro = []
statistic_shapiro = []
pass_or_fail_lilliefors = []
p_values_lilliefors = []
statistic_lilliefors = []

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    x = df_L2d1_comb.loc[df_L2d1_comb['Release_rate'] == rr, 'ln(Area)']

    # 1. Perform the Shapiro-Wilk test on your data
    statistic, p_value = stats.shapiro(x)

    if p_value > alpha:
        pass_or_fail_shapiro.append('pass')
    else:
        pass_or_fail_shapiro.append('fail')
        
    p_values_shapiro.append(p_value)
    statistic_shapiro.append(statistic)
    
    # 2. Perform the Lilliefors test on your data
    statistic, p_value = lilliefors(x)

    if p_value > alpha:
        pass_or_fail_lilliefors.append('pass')
    else:
        pass_or_fail_lilliefors.append('fail')
        
    p_values_lilliefors.append(p_value)
    statistic_lilliefors.append(statistic)

df_L2d1_tests_lognormality = pd.DataFrame({'Release_rate': Release_rates, 'Shapiro_normal': pass_or_fail_shapiro, 
                                        'Shapiro_pvalue': p_values_shapiro, 'Shapiro_statistic': statistic_shapiro,
                                        'Lilliefors_normal': pass_or_fail_lilliefors, 'Lilliefors_pvalue': p_values_lilliefors,
                                        'Lilliefors_statistic': statistic_lilliefors})


#%%%% London II Day2


Release_rates = df_L2d2_comb_count['Release_rate']
pass_or_fail_shapiro = []
p_values_shapiro = []
statistic_shapiro = []
pass_or_fail_lilliefors = []
p_values_lilliefors = []
statistic_lilliefors = []

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    x = df_L2d2_comb.loc[df_L2d2_comb['Release_rate'] == rr, 'ln(Area)']

    # 1. Perform the Shapiro-Wilk test on your data
    statistic, p_value = stats.shapiro(x)

    if p_value > alpha:
        pass_or_fail_shapiro.append('pass')
    else:
        pass_or_fail_shapiro.append('fail')
        
    p_values_shapiro.append(p_value)
    statistic_shapiro.append(statistic)
    
    # 2. Perform the Lilliefors test on your data
    statistic, p_value = lilliefors(x)

    if p_value > alpha:
        pass_or_fail_lilliefors.append('pass')
    else:
        pass_or_fail_lilliefors.append('fail')
        
    p_values_lilliefors.append(p_value)
    statistic_lilliefors.append(statistic)

df_L2d2_tests_lognormality = pd.DataFrame({'Release_rate': Release_rates, 'Shapiro_normal': pass_or_fail_shapiro, 
                                        'Shapiro_pvalue': p_values_shapiro, 'Shapiro_statistic': statistic_shapiro,
                                        'Lilliefors_normal': pass_or_fail_lilliefors, 'Lilliefors_pvalue': p_values_lilliefors,
                                        'Lilliefors_statistic': statistic_lilliefors})

    

#%%% Shapiro-Wilk + Lilliefors Max


# Set the significance level (alpha)
alpha = 0.05


    

# Utrecht ----------------------------------------------------------------------------------------------------------------------------------------------------

Release_rates = [2.18,3,15]
pass_or_fail_shapiro = []
p_values_shapiro = []
statistic_shapiro = []
pass_or_fail_lilliefors = []
p_values_lilliefors = []
statistic_lilliefors = []

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    x = df_U1_comb.loc[df_U1_comb['Release_rate'] == rr, 'ln(Max)']

    # 1. Perform the Shapiro-Wilk test on your data
    statistic, p_value = stats.shapiro(x)

    if p_value > alpha:
        pass_or_fail_shapiro.append('pass')
    else:
        pass_or_fail_shapiro.append('fail')
        
    p_values_shapiro.append(p_value)
    statistic_shapiro.append(statistic)
    
    # 2. Perform the Lilliefors test on your data
    statistic, p_value = lilliefors(x)

    if p_value > alpha:
        pass_or_fail_lilliefors.append('pass')
    else:
        pass_or_fail_lilliefors.append('fail')
        
    p_values_lilliefors.append(p_value)
    statistic_lilliefors.append(statistic)

df_U1_tests_lognormality_max = pd.DataFrame({'Release_rate': Release_rates, 'Shapiro_normal': pass_or_fail_shapiro, 
                                        'Shapiro_pvalue': p_values_shapiro, 'Shapiro_statistic': statistic_shapiro,
                                        'Lilliefors_normal': pass_or_fail_lilliefors, 'Lilliefors_pvalue': p_values_lilliefors,
                                        'Lilliefors_statistic': statistic_lilliefors})
    
# London ----------------------------------------------------------------------------------------------------------------------------------------------------

Release_rates = [35,70,70]
Days = ['Day2','Day2','Day5']

pass_or_fail_shapiro = []
p_values_shapiro = []
statistic_shapiro = []
pass_or_fail_lilliefors = []
p_values_lilliefors = []
statistic_lilliefors = []

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    if i<2:
        x = df_L1d2_comb.loc[df_L1d2_comb['Release_rate'] == rr, 'ln(Max)']
    else:
        x = df_L1d5_comb.loc[df_L1d5_comb['Release_rate'] == rr, 'ln(Max)']

    # 1. Perform the Shapiro-Wilk test on your data
    statistic, p_value = stats.shapiro(x)

    if p_value > alpha:
        pass_or_fail_shapiro.append('pass')
    else:
        pass_or_fail_shapiro.append('fail')
        
    p_values_shapiro.append(p_value)
    statistic_shapiro.append(statistic)
    
    # 2. Perform the Lilliefors test on your data
    statistic, p_value = lilliefors(x)

    if p_value > alpha:
        pass_or_fail_lilliefors.append('pass')
    else:
        pass_or_fail_lilliefors.append('fail')
        
    p_values_lilliefors.append(p_value)
    statistic_lilliefors.append(statistic)

df_L1d2and5_tests_lognormality_max = pd.DataFrame({'Release_rate': Release_rates, 'Day': Days,
                                         'Shapiro_normal': pass_or_fail_shapiro, 
                                        'Shapiro_pvalue': p_values_shapiro, 'Shapiro_statistic': statistic_shapiro,
                                        'Lilliefors_normal': pass_or_fail_lilliefors, 'Lilliefors_pvalue': p_values_lilliefors,
                                        'Lilliefors_statistic': statistic_lilliefors})
    




    


#%% PLOTS

#%%% Relevant -----------------------------------------------------------------





#%%% Histo + gaussian fit - Residuals ln(Area)

save_fig = False

'''
Version 1
display the test result of the tests applied to ln(Area)
Version 2
display the test result of the tests applied to the residuals

-> it yields the same output, but for testing normality it is correct to apply it to the residuals.
    '''

#%%%% Rotterdam LOC 1


# Gaussian Fit V2 


# Rotterdam LOC 1 ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~

bin_size=10

fig, axes = plt.subplots(3, 2, figsize=(16, 8), sharex='col')
#fig.suptitle(f"Rotterdam - Histogramm of ln(Peak Area)\n# of bins: {bin_size}", fontsize=16)  # Set the figure title
fig.subplots_adjust(hspace=0.4)  # Adjust the vertical spacing between subplots

# Iterate over a range of 10 Release_rate values
Release_rates = df_Rloc1_comb_count['Release_rate']

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    
    x = df_Rloc1_comb.loc[df_Rloc1_comb['Release_rate'] == rr, 'ln(Area)']
    
    # Define the text and its relative coordinates
    # text  = f"1: {df_Rloc1_tests_lognormality['Shapiro_normal'][i]}\n2: {df_Rloc1_tests_lognormality['Lilliefors_normal'][i]}"
    text  = f"1: {df_Rloc1_tests_lognormality_residualslnArea['Shapiro_normal'][i]}\n2: {df_Rloc1_tests_lognormality_residualslnArea['Lilliefors_normal'][i]}"
    x_rel = 0.02  # Relative x-coordinate (0.0 to 1.0) #Shapiro-Wilk Lilliefors
    y_rel = 0.65  # Relative y-coordinate (0.0 to 1.0)

 
    
    # Create a subplot in the gridspec layout
    row = i % 3
    col = i // 3
    
    #-------
    # Define the range of values
    mean = np.mean(x)
    std = np.std(x)
    x_g = np.linspace(mean - 3 * std, mean + 3 * std, 1000)
    # Calculate the probability density function (PDF) for each x value
    pdf = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-(x_g - mean)**2 / (2 * std**2))

    #Scale
    if i<5:
        hist, bins = np.histogram(x, bins=bin_size)
    else:
        hist, bins = np.histogram(x, bins=4)
    bin_width = bins[1] - bins[0]
    scaling_factor = bin_width * np.sum(hist)
    pdf *= scaling_factor
    
    # Create a plot of the Gaussian distribution
    axes[row, col].plot(x_g, pdf,color='black',alpha=0.8)
    
    #----------
    if i<5:    
        axes[row, col].hist(x, bins=bin_size,alpha=0.6,color='orange', label=f'Size: {len(x)}') #int(np.sqrt(len(x)))
    else:
        axes[row, col].hist(x, bins=4,alpha=0.6,color='orange', label=f'Size: {len(x)}')
    axes[row, col].set_title(f'{rr} ' r'$\mathrm{\mathbf{L \, min^{-1}}}$',fontsize=14,fontweight='bold')
    axes[row, col].tick_params(axis='y', labelsize=14)
    axes[row, col].set_xlim(0,8.5)
    if i<3:
        axes[row, col].legend(handlelength=0,fontsize=14)
    else:
        axes[row, col].legend(handlelength=0,fontsize=14, loc='upper right')
    
    # Add result of Shapiro and Lilliefors test (pass/fail) as text to the plot
    # Calculate the data coordinates from relative coordinates
    x_loc = x_rel * (axes[row, col].get_xlim()[1] - axes[row, col].get_xlim()[0]) + axes[row, col].get_xlim()[0]
    y_loc = y_rel * (axes[row, col].get_ylim()[1] - axes[row, col].get_ylim()[0]) + axes[row, col].get_ylim()[0]
    axes[row, col].text(x_loc, y_loc, text,fontsize=14, transform=axes[row, col].transData)

    
    if row == 2:
        axes[row, col].set_xlabel(r'ln$\left(\mathrm{\left[CH_4\right]_{area}}\ \left[ \frac{\mathrm{ppm*m}}{1\ \mathrm{ppm*m}} \right]\right)$',fontsize=18) #,fontweight = 'bold'
        axes[row, col].tick_params(axis='x', labelsize=14)
    if col == 0:
        axes[row, col].set_ylabel('Frequency',fontsize=18)

if save_fig:
    plt.savefig(path_histo+f'Rloc1_histo_lnArea_bs{bin_size}.png',bbox_inches='tight')
    plt.savefig(path_histo+f'Rloc1_histo_lnArea_bs{bin_size}.pdf',bbox_inches='tight')
    plt.savefig(path_histo+f'Rloc1_histo_lnArea_bs{bin_size}.svg',bbox_inches='tight')




#%%%% Utrecht I

# test fit gaussian

bin_size = 10

# a) Gaussian Fit V2 ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~

fig, axes = plt.subplots(3,1, figsize=(16, 12), sharex='col')
#fig.suptitle(f"Utrecht - Histogramm of ln(Peak Area)\n# of bins: {bin_size}", fontsize=16)  # Set the figure title
fig.subplots_adjust(hspace=0.4)  # Adjust the vertical spacing between subplots

# Iterate over a range of 10 Release_rate values
Release_rates = df_U1_comb_count['Release_rate']

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    x = df_U1_comb.loc[df_U1_comb['Release_rate'] == rr, 'ln(Area)']
    
    # Define the text and its relative coordinates (results of Shapiro and Lilliefors test)
    # text  = f"Shapiro-Wilk: {df_U1_tests_lognormality['Shapiro_normal'][i]}\nLilliefors: {df_U1_tests_lognormality['Lilliefors_normal'][i]}"
    text  = f"Shapiro-Wilk: {df_U1_tests_lognormality_residualslnArea['Shapiro_normal'][i]}\nLilliefors: {df_U1_tests_lognormality_residualslnArea['Lilliefors_normal'][i]}"
    x_rel = 0.02  # Relative x-coordinate (0.0 to 1.0)
    y_rel = 0.77  # Relative y-coordinate (0.0 to 1.0)

    # Create a subplot in the gridspec layout
    row = i
    col = 0
    
    #-------
    # Define the range of values
    mean = np.mean(x)
    std = np.std(x)
    x_g = np.linspace(mean - 3 * std, mean + 3 * std, 1000)
    # Calculate the probability density function (PDF) for each x value
    pdf = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-(x_g - mean)**2 / (2 * std**2))

    #Scale
    hist, bins = np.histogram(x, bins=bin_size)
    bin_width = bins[1] - bins[0]
    scaling_factor = bin_width * np.sum(hist)
    pdf *= scaling_factor
    
    # Create a plot of the Gaussian distribution
    axes[row].plot(x_g, pdf,color='black',alpha=0.8)
    
    #----------
    
    axes[row].hist(x, bins=bin_size,alpha=0.6,color='orchid',label=f'Size: {len(x)}') #int(np.sqrt(len(x)))  
    axes[row].set_title(f'{rr} ' r'$\mathrm{\mathbf{L \, min^{-1}}}$',fontsize=16,fontweight='bold')
    axes[row].set_ylabel('Frequency',fontsize=18)
    axes[row].set_xlim(0,7.5)
    axes[row].tick_params(axis='y', labelsize=16)
    axes[row].legend(handlelength=0,fontsize=14)
    
    # Add result of Shapiro and Lilliefors test (pass/fail) as text to the plot
    # Calculate the data coordinates from relative coordinates
    x_loc = x_rel * (axes[row].get_xlim()[1] - axes[row].get_xlim()[0]) + axes[row].get_xlim()[0]
    y_loc = y_rel * (axes[row].get_ylim()[1] - axes[row].get_ylim()[0]) + axes[row].get_ylim()[0]
    axes[row].text(x_loc, y_loc, text,fontsize=14, transform=axes[row].transData)


    if row == (len(Release_rates)-1):
            axes[row].set_xlabel(r'ln$\left(\mathrm{\left[CH_4\right]_{area}}\ \left[ \frac{\mathrm{ppm*m}}{1\ \mathrm{ppm*m}} \right]\right)$',fontsize=18) #,fontweight = 'bold'
            axes[row].tick_params(axis='x', labelsize=16)
        

if save_fig:
    plt.savefig(path_histo+f'U1_histo_lnArea_bs{bin_size}.png',bbox_inches='tight')
    plt.savefig(path_histo+f'U1_histo_lnArea_bs{bin_size}.pdf',bbox_inches='tight')
    plt.savefig(path_histo+f'U1_histo_lnArea_bs{bin_size}.svg',bbox_inches='tight')
    

#%%%% Utrecht II

# test fit gaussian
# save_fig = False
bin_size = 10

# a) Gaussian Fit V2 ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~ ~~~

fig, axes = plt.subplots(6,2, figsize=(22, 12), sharex='col')
#fig.suptitle(f"Utrecht - Histogramm of ln(Peak Area)\n# of bins: {bin_size}", fontsize=16)  # Set the figure title
fig.subplots_adjust(hspace=0.4)  # Adjust the vertical spacing between subplots

# Iterate over a range of 10 Release_rate values
Release_rates = df_U2_comb_count['Release_rate']

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    x = df_U2_comb.loc[df_U2_comb['Release_rate'] == rr, 'ln(Area)']
    
    # Define the text and its relative coordinates (results of Shapiro and Lilliefors test)
    # text  = f"1: {df_U3_tests_lognormality['Shapiro_normal'][i]}\n2: {df_U3_tests_lognormality['Lilliefors_normal'][i]}"
    text  = f"1: {df_U3_tests_lognormality_residualslnArea['Shapiro_normal'][i]}\n2: {df_U3_tests_lognormality_residualslnArea['Lilliefors_normal'][i]}"
    x_rel = 0.02  # Relative x-coordinate (0.0 to 1.0)
    y_rel = 0.62  # Relative y-coordinate (0.0 to 1.0)

    # Create a subplot in the gridspec layout
    row = i % 6
    col = i // 6
    
    #-------
    # Define the range of values
    mean = np.mean(x)
    std = np.std(x)
    x_g = np.linspace(mean - 3 * std, mean + 3 * std, 1000)
    # Calculate the probability density function (PDF) for each x value
    pdf = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-(x_g - mean)**2 / (2 * std**2))

    #Scale
    hist, bins = np.histogram(x, bins=bin_size)
    bin_width = bins[1] - bins[0]
    scaling_factor = bin_width * np.sum(hist)
    pdf *= scaling_factor
    
    # Create a plot of the Gaussian distribution
    axes[row, col].plot(x_g, pdf,color='black',alpha=0.8)
    
    #----------
    
    # Create a histogram in the appropriate subplot
    axes[row, col].hist(x, bins=bin_size,alpha=0.6,color=dict_color_city['Utrecht II'],label=f'Size: {len(x)}') #int(np.sqrt(len(x)))  
    axes[row, col].set_title(f'{rr} ' r'$\mathrm{\mathbf{L \, min^{-1}}}$',fontsize=16,fontweight='bold')
    axes[row, col].set_ylabel('Frequency',fontsize=18)
    if i<6:
        axes[row, col].set_xlim(-1,5.5)
    else:
        axes[row, col].set_xlim(0,8)
    axes[row, col].tick_params(axis='y', labelsize=16)
    axes[row, col].legend(handlelength=0,fontsize=14, loc='upper right')
    
    # Add result of Shapiro and Lilliefors test (pass/fail) as text to the plot
    # Calculate the data coordinates from relative coordinates
    x_loc = x_rel * (axes[row, col].get_xlim()[1] - axes[row, col].get_xlim()[0]) + axes[row, col].get_xlim()[0]
    y_loc = y_rel * (axes[row, col].get_ylim()[1] - axes[row, col].get_ylim()[0]) + axes[row, col].get_ylim()[0]
    axes[row, col].text(x_loc, y_loc, text,fontsize=14, transform=axes[row, col].transData)

    
    if row == (5):
            axes[row, col].set_xlabel(r'ln$\left(\mathrm{\left[CH_4\right]_{area}}\ \left[ \frac{\mathrm{ppm*m}}{1\ \mathrm{ppm*m}} \right]\right)$',fontsize=18) #,fontweight = 'bold'
            axes[row, col].tick_params(axis='x', labelsize=16)
        

if save_fig:
    plt.savefig(path_histo+f'U3_histo_lnArea_bs{bin_size}.png',bbox_inches='tight')
    plt.savefig(path_histo+f'U3_histo_lnArea_bs{bin_size}.pdf',bbox_inches='tight')
    plt.savefig(path_histo+f'U3_histo_lnArea_bs{bin_size}.svg',bbox_inches='tight')


#%%%% London I

bin_size = 10

fig, axes = plt.subplots(3,1, figsize=(16, 12), sharex='col')
#fig.suptitle(f"London - Histogramm of ln(Peak Area)\n# of bins: {bin_size}", fontsize=16)  # Set the figure title
fig.subplots_adjust(hspace=0.5)  # Adjust the vertical spacing between subplots

# Iterate over a range of 10 Release_rate values
Release_rates = [35,70,70]
colors = ['mediumseagreen','mediumseagreen','lime']
Days = ['Day2','Day2','Day5']

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    if i<2:
        x = df_L1d2_comb.loc[df_L1d2_comb['Release_rate'] == rr, 'ln(Area)']
    else:
        x = df_L1d5_comb.loc[df_L1d5_comb['Release_rate'] == rr, 'ln(Area)']
        
    # Define the text and its relative coordinates
    # text  = f"Shapiro-Wilk: {df_L1d2and5_tests_lognormality['Shapiro_normal'][i]}\nLilliefors: {df_L1d2and5_tests_lognormality['Lilliefors_normal'][i]}"
    text  = f"Shapiro-Wilk: {df_L1d2and5_tests_lognormality_residualslnArea['Shapiro_normal'][i]}\nLilliefors: {df_L1d2and5_tests_lognormality_residualslnArea['Lilliefors_normal'][i]}"
    x_rel = 0.02  # Relative x-coordinate (0.0 to 1.0) #Shapiro-Wilk Lilliefors
    y_rel = 0.77  # Relative y-coordinate (0.0 to 1.0)

    # Create a subplot in the gridspec layout
    row = i
    
    #-------
    # Define the range of values
    mean = np.mean(x)
    std = np.std(x)
    x_g = np.linspace(mean - 3 * std, mean + 3 * std, 1000)
    # Calculate the probability density function (PDF) for each x value
    pdf = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-(x_g - mean)**2 / (2 * std**2))

    #Scale
    hist, bins = np.histogram(x, bins=bin_size)
    bin_width = bins[1] - bins[0]
    scaling_factor = bin_width * np.sum(hist)
    pdf *= scaling_factor
    
    # Create a plot of the Gaussian distribution
    axes[row].plot(x_g, pdf,color='black',alpha=0.8)  
    #----------
    

    # Create a histogram in the appropriate subplot
    axes[row].hist(x, bins=bin_size,alpha=0.6,color=colors[i],label=f'Size: {len(x)}') #int(np.sqrt(len(x)))  
    axes[row].set_title(f'{Days[i]}: {rr} ' r'$\mathrm{\mathbf{L \, min^{-1}}}$',fontsize=16,fontweight='bold')
    axes[row].set_ylabel('Frequency',fontsize=18)
    axes[row].set_xlim(0,8)
    axes[row].tick_params(axis='y', labelsize=16)
    axes[row].legend(handlelength=0,fontsize=14)
    
    # Add result of Shapiro and Lilliefors test (pass/fail) as text to the plot
    # Calculate the data coordinates from relative coordinates
    x_loc = x_rel * (axes[row].get_xlim()[1] - axes[row].get_xlim()[0]) + axes[row].get_xlim()[0]
    y_loc = y_rel * (axes[row].get_ylim()[1] - axes[row].get_ylim()[0]) + axes[row].get_ylim()[0]
    axes[row].text(x_loc, y_loc, text,fontsize=14, transform=axes[row].transData)

    
    if row == (len(Release_rates)-1):
            axes[row].set_xlabel(r'ln$\left(\mathrm{\left[CH_4\right]_{area}}\ \left[ \frac{\mathrm{ppm*m}}{1\ \mathrm{ppm*m}} \right]\right)$',fontsize=18) #,fontweight = 'bold'
            axes[row].tick_params(axis='x', labelsize=16)
    
if save_fig:
    plt.savefig(path_histo+f'Ld2and5_histo_lnArea_bs{bin_size}.png',bbox_inches='tight')
    plt.savefig(path_histo+f'Ld2and5_histo_lnArea_bs{bin_size}.pdf',bbox_inches='tight')
    plt.savefig(path_histo+f'Ld2and5_histo_lnArea_bs{bin_size}.svg',bbox_inches='tight')
    
    
#%%%% London II

bin_size = 10

fig, axes = plt.subplots(5,2, figsize=(22, 12), sharex='col') 
#fig.suptitle(f"London - Histogramm of ln(Peak Area)\n# of bins: {bin_size}", fontsize=16)  # Set the figure title
fig.subplots_adjust(hspace=0.5)  # Adjust the vertical spacing between subplots

# Hide the last subplot (10th subplot)
fig.delaxes(axes[4, 1])

# Iterate over a range of 10 Release_rate values
Release_rates = pd.concat([df_L2d2_comb_count['Release_rate'], df_L2d1_comb_count['Release_rate']], ignore_index=True)
colors = ['chocolate','chocolate','brown','brown','brown','brown','brown','brown','brown']
Days = ['Day2','Day2','Day1','Day1','Day1','Day1','Day1','Day1','Day1']

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    if i<2:
        x = df_L2d2_comb.loc[df_L2d2_comb['Release_rate'] == rr, 'ln(Area)']
    else:
        x = df_L2d1_comb.loc[df_L2d1_comb['Release_rate'] == rr, 'ln(Area)']
        
    # Define the text and its relative coordinates
    # text  = f"1: {df_L2d1and2_tests_lognormality['Shapiro_normal'][i]}\n2: {df_L2d1and2_tests_lognormality['Lilliefors_normal'][i]}"
    text  = f"1: {df_L2d1and2_tests_lognormality_residualslnArea['Shapiro_normal'][i]}\n2: {df_L2d1and2_tests_lognormality_residualslnArea['Lilliefors_normal'][i]}"
    if i<5:
        x_rel = 0.85  # Relative x-coordinate (0.0 to 1.0) #Shapiro-Wilk Lilliefors
        y_rel = 0.05  # Relative y-coordinate (0.0 to 1.0)
    else:
        x_rel = 0.02  # Relative x-coordinate (0.0 to 1.0) #Shapiro-Wilk Lilliefors
        y_rel = 0.65

    # Create a subplot in the gridspec layout
    row = i % 5
    col = i // 5
    
    #-------
    # Define the range of values
    mean = np.mean(x)
    std = np.std(x)
    x_g = np.linspace(mean - 3 * std, mean + 3 * std, 1000)
    # Calculate the probability density function (PDF) for each x value
    pdf = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-(x_g - mean)**2 / (2 * std**2))

    #Scale
    hist, bins = np.histogram(x, bins=bin_size)
    bin_width = bins[1] - bins[0]
    scaling_factor = bin_width * np.sum(hist)
    pdf *= scaling_factor
    
    # Create a plot of the Gaussian distribution
    axes[row, col].plot(x_g, pdf,color='black',alpha=0.8)  
    #----------
    

    # Create a histogram in the appropriate subplot
    axes[row, col].hist(x, bins=bin_size,alpha=0.6,color=colors[i],label=f'Size: {len(x)}') #int(np.sqrt(len(x)))  
    axes[row, col].set_title(f'{Days[i]}: {rr} ' r'$\mathrm{\mathbf{L \, min^{-1}}}$',fontsize=16,fontweight='bold')
    axes[row, col].set_ylabel('Frequency',fontsize=18)
    if i<5:
        axes[row, col].set_xlim(-1,5.5)
    else:
        axes[row, col].set_xlim(1.5,6)
    axes[row, col].tick_params(axis='y', labelsize=16)
    axes[row, col].legend(handlelength=0,fontsize=14)
    
    # Add result of Shapiro and Lilliefors test (pass/fail) as text to the plot
    # Calculate the data coordinates from relative coordinates
    x_loc = x_rel * (axes[row, col].get_xlim()[1] - axes[row, col].get_xlim()[0]) + axes[row, col].get_xlim()[0]
    y_loc = y_rel * (axes[row, col].get_ylim()[1] - axes[row, col].get_ylim()[0]) + axes[row, col].get_ylim()[0]
    axes[row, col].text(x_loc, y_loc, text,fontsize=14, transform=axes[row, col].transData)

    
    if row == 4:
            axes[row, col].set_xlabel(r'ln$\left(\mathrm{\left[CH_4\right]_{area}}\ \left[ \frac{\mathrm{ppm*m}}{1\ \mathrm{ppm*m}} \right]\right)$',fontsize=18) #,fontweight = 'bold'
            axes[row, col].tick_params(axis='x', labelsize=16)
axes[3, 1].set_xlabel(r'ln$\left(\mathrm{\left[CH_4\right]_{area}}\ \left[ \frac{\mathrm{ppm*m}}{1\ \mathrm{ppm*m}} \right]\right)$',fontsize=18) #,fontweight = 'bold'
axes[3, 1].tick_params(axis='x', labelsize=16, labelbottom=True)
# axes[3, 1].set_xticks(np.arange(2, 7, 1))  # Adjust range and step as needed
# axes[3, 1].set_xticklabels(np.arange(2, 7, 1), fontsize=16)  # Set labels as integers
    
if save_fig:
    plt.savefig(path_histo+f'L2d1and2_histo_lnArea_bs{bin_size}.png',bbox_inches='tight')
    plt.savefig(path_histo+f'L2d1and2_histo_lnArea_bs{bin_size}.pdf',bbox_inches='tight')
    plt.savefig(path_histo+f'L2d1and2_histo_lnArea_bs{bin_size}.svg',bbox_inches='tight')






#%%% QQ Plot - Data vs Normal Distribution

'''
Quantile-Quantile Plot
When the quantiles of two variables are plotted against each other, then the plot obtained 
is known as quantile – quantile plot or qqplot. This plot provides a summary of whether the 
distributions of two variables are similar or not with respect to the locations.

1. Sort your data in ascending order.

2. Calculate the empirical quantiles of your data points. These are the values that correspond 
to the cumulative distribution function (CDF) of your data.

3. Calculate the expected quantiles for a normal distribution with the same sample size and the 
mean and standard deviation of your data.

4. Plot the empirical quantiles on the x-axis and the expected quantiles on the y-axis. If your 
data follows a normal distribution, the points should roughly fall along a straight line.
    '''
    
save_fig = False

#%%%% Rotterdam  loc 1

fig, axes = plt.subplots(2,3, figsize=(14, 8))
#fig.suptitle("Rotterdam - Quantile-Quantile Plot: Normal Distribution vs. ln(Peak Area)", fontsize=16)  # Set the figure title
fig.subplots_adjust(hspace=0.3,wspace=0.3)  # Adjust the vertical spacing between subplots

# Iterate over a range of 10 Release_rate values
Release_rates = df_Rloc1_comb_count['Release_rate']

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    x = df_Rloc1_comb.loc[df_Rloc1_comb['Release_rate'] == rr, 'ln(Area)']
    
    # Define the text and its relative coordinates (results of Shapiro and Lilliefors test)
    text  = f"1: {df_Rloc1_tests_lognormality_residualslnArea['Shapiro_normal'][i]}\n2: {df_Rloc1_tests_lognormality_residualslnArea['Lilliefors_normal'][i]}"
    x_rel = 0.7  # Relative x-coordinate (0.0 to 1.0) #Shapiro-Wilk Lilliefors
    y_rel = 0.05  # Relative y-coordinate (0.0 to 1.0)

    
    sorted_data = np.sort(x)

    empirical_quantiles = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    expected_quantiles = stats.norm.ppf(empirical_quantiles, loc=np.mean(x), scale=np.std(x))

    row, col = divmod(i, 3)

    axes[row, col].plot(expected_quantiles[:-1], expected_quantiles[:-1], color='gray', linestyle='--') #,label='1:1 Line'
    axes[row, col].scatter(expected_quantiles, sorted_data,color='orange',s=15,alpha=0.65, label=f'Size: {len(x)}') 
    axes[row, col].set_title(f'{rr} ' r'$\mathrm{\mathbf{L \, min^{-1}}}$',fontsize=14,fontweight='bold')
    axes[row, col].tick_params(axis='x', labelsize=16)
    axes[row, col].tick_params(axis='y', labelsize=16)
    axes[row, col].legend(handlelength=0,fontsize=14)
    
    # Add result of Shapiro and Lilliefors test (pass/fail) as text to the plot
    # Calculate the data coordinates from relative coordinates
    x_loc = x_rel * (axes[row, col].get_xlim()[1] - axes[row, col].get_xlim()[0]) + axes[row, col].get_xlim()[0]
    y_loc = y_rel * (axes[row, col].get_ylim()[1] - axes[row, col].get_ylim()[0]) + axes[row, col].get_ylim()[0]
    axes[row, col].text(x_loc, y_loc, text,fontsize=14, transform=axes[row, col].transData)

    
    if row == 1:
        axes[row, col].set_xlabel('Theoretical Quantiles\n(Expected)',fontsize=16) #,fontweight = 'bold'
    if col == (0):
        axes[row, col].set_ylabel('Sample Quantiles\n(Empirical)',fontsize=16)
      
  
if save_fig:
    plt.savefig(path_qq+f'Rloc1_QQ_lnArea.png',bbox_inches='tight')
    plt.savefig(path_qq+f'Rloc1_QQ_lnArea.pdf',bbox_inches='tight')
    plt.savefig(path_qq+f'Rloc1_QQ_lnArea.svg',bbox_inches='tight')


#%%%% Utrecht I

fig, axes = plt.subplots(1,3, figsize=(18, 5))
#fig.suptitle("Utrecht - Quantile-Quantile Plot: Normal Distribution vs. ln(Peak Area)", fontsize=16)  # Set the figure title
fig.subplots_adjust(hspace=0.4)  # Adjust the vertical spacing between subplots

# Iterate over a range of 10 Release_rate values
Release_rates = df_U1_comb_count['Release_rate']

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    x = df_U1_comb.loc[df_U1_comb['Release_rate'] == rr, 'ln(Area)']
    
    # Define the text and its relative coordinates (results of Shapiro and Lilliefors test)
    text  = f"Shapiro-Wilk: {df_U1_tests_lognormality_residualslnArea['Shapiro_normal'][i]}\nLilliefors: {df_U1_tests_lognormality_residualslnArea['Lilliefors_normal'][i]}"
    x_rel = 0.55  # Relative x-coordinate (0.0 to 1.0) #Shapiro-Wilk Lilliefors
    y_rel = 0.05  # Relative y-coordinate (0.0 to 1.0)

    
    sorted_data = np.sort(x)

    empirical_quantiles = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    expected_quantiles = stats.norm.ppf(empirical_quantiles, loc=np.mean(x), scale=np.std(x))

    row, col = divmod(i, 3)

    axes[col].plot(expected_quantiles[:-1], expected_quantiles[:-1], color='gray', linestyle='--') #,label='1:1 Line'
    axes[col].scatter(expected_quantiles, sorted_data, color=dict_color_city['Utrecht'], alpha=0.5, label=f'Size: {len(x)}') 
    axes[col].set_title(f'{rr} ' r'$\mathrm{\mathbf{L \, min^{-1}}}$',fontsize=16,fontweight='bold')
    axes[col].set_xlabel('Theoretical Quantiles\n(Expected)',fontsize=16)
    #axes[col].set_ylabel('Sample Quantiles (Empirical)',fontsize=12)
    axes[col].tick_params(axis='x', labelsize=16)
    axes[col].tick_params(axis='y', labelsize=16)
    axes[col].legend(handlelength=0,fontsize=14)
    
    if col == (0):
        axes[col].set_ylabel('Sample Quantiles\n(Empirical)',fontsize=16)
    
    # Add result of Shapiro and Lilliefors test (pass/fail) as text to the plot
    # Calculate the data coordinates from relative coordinates
    x_loc = x_rel * (axes[col].get_xlim()[1] - axes[col].get_xlim()[0]) + axes[col].get_xlim()[0]
    y_loc = y_rel * (axes[col].get_ylim()[1] - axes[col].get_ylim()[0]) + axes[col].get_ylim()[0]
    axes[col].text(x_loc, y_loc, text,fontsize=14, transform=axes[col].transData)



if save_fig:
    plt.savefig(path_qq+f'U1_QQ_lnArea.png',bbox_inches='tight')
    plt.savefig(path_qq+f'U1_QQ_lnArea.pdf',bbox_inches='tight')
    plt.savefig(path_qq+f'U1_QQ_lnArea.svg',bbox_inches='tight')
    
    
#%%%% Utrecht II

fig, axes = plt.subplots(2,6, figsize=(22, 6.5))
#fig.suptitle("Rotterdam - Quantile-Quantile Plot: Normal Distribution vs. ln(Peak Area)", fontsize=16)  # Set the figure title
fig.subplots_adjust(hspace=0.3,wspace=0.3)  # Adjust the vertical spacing between subplots

# Iterate over a range of 10 Release_rate values
Release_rates = df_U2_comb_count['Release_rate']

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    x = df_U2_comb.loc[df_U2_comb['Release_rate'] == rr, 'ln(Area)']
    
    # Define the text and its relative coordinates (results of Shapiro and Lilliefors test)
    text  = f"1: {df_U3_tests_lognormality_residualslnArea['Shapiro_normal'][i]}\n2: {df_U3_tests_lognormality_residualslnArea['Lilliefors_normal'][i]}"
    x_rel = 0.65  # Relative x-coordinate (0.0 to 1.0) #Shapiro-Wilk Lilliefors
    y_rel = 0.05  # Relative y-coordinate (0.0 to 1.0)

    
    sorted_data = np.sort(x)

    empirical_quantiles = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    expected_quantiles = stats.norm.ppf(empirical_quantiles, loc=np.mean(x), scale=np.std(x))

    row, col = divmod(i, 6)

    axes[row, col].plot(expected_quantiles[:-1], expected_quantiles[:-1], color='gray', linestyle='--') #,label='1:1 Line'
    axes[row, col].scatter(expected_quantiles, sorted_data,color=dict_color_city['Utrecht II'],s=15,alpha=0.65, label=f'Size: {len(x)}') 
    axes[row, col].set_title(f'{rr} ' r'$\mathrm{\mathbf{L \, min^{-1}}}$',fontsize=14,fontweight='bold')
    axes[row, col].tick_params(axis='x', labelsize=16)
    axes[row, col].tick_params(axis='y', labelsize=16)
    axes[row, col].legend(handlelength=0,fontsize=14)
    
    # Add result of Shapiro and Lilliefors test (pass/fail) as text to the plot
    # Calculate the data coordinates from relative coordinates
    x_loc = x_rel * (axes[row, col].get_xlim()[1] - axes[row, col].get_xlim()[0]) + axes[row, col].get_xlim()[0]
    y_loc = y_rel * (axes[row, col].get_ylim()[1] - axes[row, col].get_ylim()[0]) + axes[row, col].get_ylim()[0]
    axes[row, col].text(x_loc, y_loc, text,fontsize=14, transform=axes[row, col].transData)

    
    if row == 1:
        axes[row, col].set_xlabel('Theoretical Quantiles\n(Expected)',fontsize=16) #,fontweight = 'bold'
    if col == (0):
        axes[row, col].set_ylabel('Sample Quantiles\n(Empirical)',fontsize=16)


if save_fig:
    plt.savefig(path_qq+f'U3_QQ_lnArea.png',bbox_inches='tight')
    plt.savefig(path_qq+f'U3_QQ_lnArea.pdf',bbox_inches='tight')
    plt.savefig(path_qq+f'U3_QQ_lnArea.svg',bbox_inches='tight')
    

#%%%% London I


fig, axes = plt.subplots(1,3, figsize=(18, 5))
#fig.suptitle("London - Quantile-Quantile Plot: Normal Distribution vs. ln(Peak Area)", fontsize=16)  # Set the figure title
fig.subplots_adjust(hspace=0.4)  # Adjust the vertical spacing between subplots

# Iterate over a range of 10 Release_rate values
Release_rates = [35,70,70]
colors = ['mediumseagreen','mediumseagreen','lime']
Days = ['Day1','Day1','Day3']

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    if i<2:
        x = df_L1d2_comb.loc[df_L1d2_comb['Release_rate'] == rr, 'ln(Area)']

    else:
        x = df_L1d5_comb.loc[df_L1d5_comb['Release_rate'] == rr, 'ln(Area)']
    
    # Define the text and its relative coordinates (results of Shapiro and Lilliefors test)
    text  = f"Shapiro-Wilk: {df_L1d2and5_tests_lognormality_residualslnArea['Shapiro_normal'][i]}\nLilliefors: {df_L1d2and5_tests_lognormality_residualslnArea['Lilliefors_normal'][i]}"
    x_rel = 0.55  # Relative x-coordinate (0.0 to 1.0) #Shapiro-Wilk Lilliefors
    y_rel = 0.05  # Relative y-coordinate (0.0 to 1.0)

    
    sorted_data = np.sort(x)

    empirical_quantiles = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    expected_quantiles = stats.norm.ppf(empirical_quantiles, loc=np.mean(x), scale=np.std(x))

    col = i

    axes[col].plot(expected_quantiles[:-1], expected_quantiles[:-1], color='gray', linestyle='--') #,label='1:1 Line'
    axes[col].scatter(expected_quantiles, sorted_data, color=colors[i], alpha=0.5, label=f'Size: {len(x)}') 
    axes[col].set_title(f'{Days[i]}: {rr} ' r'$\mathrm{\mathbf{L \, min^{-1}}}$',fontsize=16,fontweight='bold')
    axes[col].set_xlabel(f'Theoretical Quantiles\n(Expected)',fontsize=16)
    #axes[col].set_ylabel('Sample Quantiles (Empirical)',fontsize=12)
    axes[col].tick_params(axis='x', labelsize=16)
    axes[col].tick_params(axis='y', labelsize=16)
    axes[col].legend(handlelength=0,fontsize=14)
    
    if col == 0:
        axes[col].set_ylabel('Sample Quantiles\n(Empirical)',fontsize=16)
        
    # Add result of Shapiro and Lilliefors test (pass/fail) as text to the plot
    # Calculate the data coordinates from relative coordinates
    x_loc = x_rel * (axes[col].get_xlim()[1] - axes[col].get_xlim()[0]) + axes[col].get_xlim()[0]
    y_loc = y_rel * (axes[col].get_ylim()[1] - axes[col].get_ylim()[0]) + axes[col].get_ylim()[0]
    axes[col].text(x_loc, y_loc, text,fontsize=14, transform=axes[col].transData)



if save_fig:
    plt.savefig(path_qq+f'L1_QQ_lnArea.png',bbox_inches='tight')
    plt.savefig(path_qq+f'L1_QQ_lnArea.pdf',bbox_inches='tight')
    plt.savefig(path_qq+f'L1_QQ_lnArea.svg',bbox_inches='tight')
    
    
#%%%% London II

fig, axes = plt.subplots(2,5, figsize=(22, 8))
#fig.suptitle("Rotterdam - Quantile-Quantile Plot: Normal Distribution vs. ln(Peak Area)", fontsize=16)  # Set the figure title
fig.subplots_adjust(hspace=0.3,wspace=0.3)  # Adjust the vertical spacing between subplots

# Hide the last subplot (10th subplot)
fig.delaxes(axes[1, 4])

# Iterate over a range of 10 Release_rate values
Release_rates = pd.concat([df_L2d2_comb_count['Release_rate'], df_L2d1_comb_count['Release_rate']], ignore_index=True)
colors = ['chocolate','chocolate','chocolate','chocolate','chocolate','chocolate','chocolate','brown','brown']
Days = ['Day2','Day2','Day2','Day2','Day2','Day2','Day2','Day1','Day1']

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    if i<2:
        x = df_L2d2_comb.loc[df_L2d2_comb['Release_rate'] == rr, 'ln(Area)']
    else:
        x = df_L2d1_comb.loc[df_L2d1_comb['Release_rate'] == rr, 'ln(Area)']
    
    # Define the text and its relative coordinates (results of Shapiro and Lilliefors test)
    text  = f"1: {df_L2d1and2_tests_lognormality_residualslnArea['Shapiro_normal'][i]}\n2: {df_L2d1and2_tests_lognormality_residualslnArea['Lilliefors_normal'][i]}"
    
    x_rel = 0.67  # Relative x-coordinate (0.0 to 1.0) #Shapiro-Wilk Lilliefors
    y_rel = 0.05  # Relative y-coordinate (0.0 to 1.0)
    
    
    sorted_data = np.sort(x)

    empirical_quantiles = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    expected_quantiles = stats.norm.ppf(empirical_quantiles, loc=np.mean(x), scale=np.std(x))

    row, col = divmod(i, 5)

    axes[row, col].plot(expected_quantiles[:-1], expected_quantiles[:-1], color='gray', linestyle='--') #,label='1:1 Line'
    axes[row, col].scatter(expected_quantiles, sorted_data,color=colors[i],s=15,alpha=0.65, label=f'Size: {len(x)}') 
    axes[row, col].set_title(f'{Days[i]}: {rr} ' r'$\mathrm{\mathbf{L \, min^{-1}}}$',fontsize=14,fontweight='bold')
    axes[row, col].tick_params(axis='x', labelsize=16)
    axes[row, col].tick_params(axis='y', labelsize=16)
    axes[row, col].legend(handlelength=0,fontsize=14)
    
    # Add result of Shapiro and Lilliefors test (pass/fail) as text to the plot
    # Calculate the data coordinates from relative coordinates
    x_loc = x_rel * (axes[row, col].get_xlim()[1] - axes[row, col].get_xlim()[0]) + axes[row, col].get_xlim()[0]
    y_loc = y_rel * (axes[row, col].get_ylim()[1] - axes[row, col].get_ylim()[0]) + axes[row, col].get_ylim()[0]
    axes[row, col].text(x_loc, y_loc, text,fontsize=14, transform=axes[row, col].transData)

    
    if row == 1:
        axes[row, col].set_xlabel('Theoretical Quantiles\n(Expected)',fontsize=16) #,fontweight = 'bold'
    if col == (0):
        axes[row, col].set_ylabel('Sample Quantiles\n(Empirical)',fontsize=16)
        
    axes[0, 4].set_xlabel('Theoretical Quantiles\n(Expected)',fontsize=16)


if save_fig:
    plt.savefig(path_qq+f'L2_QQ_lnArea.png',bbox_inches='tight')
    plt.savefig(path_qq+f'L2_QQ_lnArea.pdf',bbox_inches='tight')
    plt.savefig(path_qq+f'L2_QQ_lnArea.svg',bbox_inches='tight')

#%%% QQ Plot - Residuals Area

# Quantile-Quantile Plot
# When the quantiles of two variables are plotted against each other, then the plot obtained 
# is known as quantile – quantile plot or qqplot. This plot provides a summary of whether the 
# distributions of two variables are similar or not with respect to the locations.

# 1. Sort your data in ascending order.

# 2. Calculate the empirical quantiles of your data points. These are the values that correspond 
# to the cumulative distribution function (CDF) of your data.

# 3. Calculate the expected quantiles for a normal distribution with the same sample size and the 
# mean and standard deviation of your data.

# 4. Plot the empirical quantiles on the x-axis and the expected quantiles on the y-axis. If your 
# data follows a normal distribution, the points should roughly fall along a straight line.


#%%%% Rotterdam  loc 1

fig, axes = plt.subplots(2,3, figsize=(14, 8))
#fig.suptitle("Rotterdam - Quantile-Quantile Plot: Normal Distribution vs. ln(Peak Area)", fontsize=16)  # Set the figure title
fig.subplots_adjust(hspace=0.3,wspace=0.3)  # Adjust the vertical spacing between subplots

# Iterate over a range of 10 Release_rate values
Release_rates = df_Rloc1_comb_count['Release_rate']

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    x = df_Rloc1_comb.loc[df_Rloc1_comb['Release_rate'] == rr, 'ln(Area)_residuals']
    
    # Define the text and its relative coordinates (results of Shapiro and Lilliefors test)
    text  = f"1: {df_Rloc1_tests_lognormality_residualslnArea['Shapiro_normal'][i]}\n2: {df_Rloc1_tests_lognormality_residualslnArea['Lilliefors_normal'][i]}"
    x_rel = 0.7  # Relative x-coordinate (0.0 to 1.0) #Shapiro-Wilk Lilliefors
    y_rel = 0.05  # Relative y-coordinate (0.0 to 1.0)

    
    sorted_data = np.sort(x)

    empirical_quantiles = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    expected_quantiles = stats.norm.ppf(empirical_quantiles, loc=np.mean(x), scale=np.std(x))

    row, col = divmod(i, 3)

    axes[row, col].plot(expected_quantiles[:-1], expected_quantiles[:-1], color='gray', linestyle='--') #,label='1:1 Line'
    axes[row, col].scatter(expected_quantiles, sorted_data,color='orange',s=15,alpha=0.65, label=f'Size: {len(x)}') 
    axes[row, col].set_title(f'{rr} ' r'$\mathrm{\mathbf{L \, min^{-1}}}$',fontsize=14,fontweight='bold')
    axes[row, col].tick_params(axis='x', labelsize=16)
    axes[row, col].tick_params(axis='y', labelsize=16)
    axes[row, col].legend(handlelength=0,fontsize=14)
    
    # Add result of Shapiro and Lilliefors test (pass/fail) as text to the plot
    # Calculate the data coordinates from relative coordinates
    x_loc = x_rel * (axes[row, col].get_xlim()[1] - axes[row, col].get_xlim()[0]) + axes[row, col].get_xlim()[0]
    y_loc = y_rel * (axes[row, col].get_ylim()[1] - axes[row, col].get_ylim()[0]) + axes[row, col].get_ylim()[0]
    axes[row, col].text(x_loc, y_loc, text,fontsize=14, transform=axes[row, col].transData)

    
    if row == 1:
        axes[row, col].set_xlabel('Theoretical Quantiles\n(Expected)',fontsize=16) #,fontweight = 'bold'
    if col == (0):
        axes[row, col].set_ylabel('Sample Quantiles\n(Empirical)',fontsize=16)
      
  

if save_fig:
    plt.savefig(path_qq+f'R_QQ_RESIDUALSlnArea.png',bbox_inches='tight')
    plt.savefig(path_qq+f'R_QQ_RESIDUALSlnArea.pdf',bbox_inches='tight')
    plt.savefig(path_qq+f'R_QQ_RESIDUALSlnArea.svg',bbox_inches='tight')


#%%%% Utrecht 

fig, axes = plt.subplots(1,3, figsize=(18, 6))
#fig.suptitle("Utrecht - Quantile-Quantile Plot: Normal Distribution vs. ln(Peak Area)", fontsize=16)  # Set the figure title
fig.subplots_adjust(hspace=0.4)  # Adjust the vertical spacing between subplots

# Iterate over a range of 10 Release_rate values
Release_rates = df_U1_comb_count['Release_rate']

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    x = df_U1_comb.loc[df_U1_comb['Release_rate'] == rr, 'ln(Area)_residuals']
    
    # Define the text and its relative coordinates (results of Shapiro and Lilliefors test)
    text  = f"Shapiro-Wilk: {df_U1_tests_lognormality_residualslnArea['Shapiro_normal'][i]}\nLilliefors: {df_U1_tests_lognormality_residualslnArea['Lilliefors_normal'][i]}"
    x_rel = 0.55  # Relative x-coordinate (0.0 to 1.0) #Shapiro-Wilk Lilliefors
    y_rel = 0.05  # Relative y-coordinate (0.0 to 1.0)

    
    sorted_data = np.sort(x)

    empirical_quantiles = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    expected_quantiles = stats.norm.ppf(empirical_quantiles, loc=np.mean(x), scale=np.std(x))

    col = i

    axes[col].plot(expected_quantiles[:-1], expected_quantiles[:-1], color='gray', linestyle='--') #,label='1:1 Line'
    axes[col].scatter(expected_quantiles, sorted_data, color='orchid', alpha=0.5, label=f'Size: {len(x)}') 
    axes[col].set_title(f'{rr} L/min',fontsize=16,fontweight='bold')
    axes[col].set_xlabel('Theoretical Quantiles\n(Expected)',fontsize=16)
    #axes[col].set_ylabel('Sample Quantiles (Empirical)',fontsize=12)
    axes[col].tick_params(axis='x', labelsize=16)
    axes[col].tick_params(axis='y', labelsize=16)
    axes[col].legend(handlelength=0,fontsize=14)
    
    if col == (0):
        axes[col].set_ylabel('Sample Quantiles\n(Empirical)',fontsize=16)
    
    # Add result of Shapiro and Lilliefors test (pass/fail) as text to the plot
    # Calculate the data coordinates from relative coordinates
    x_loc = x_rel * (axes[col].get_xlim()[1] - axes[col].get_xlim()[0]) + axes[col].get_xlim()[0]
    y_loc = y_rel * (axes[col].get_ylim()[1] - axes[col].get_ylim()[0]) + axes[col].get_ylim()[0]
    axes[col].text(x_loc, y_loc, text,fontsize=14, transform=axes[col].transData)



if save_fig:
    plt.savefig(path_qq+f'U1_QQ_RESIDUALSlnArea.png',bbox_inches='tight')
    plt.savefig(path_qq+f'U1_QQ_RESIDUALSlnArea.pdf',bbox_inches='tight')
    plt.savefig(path_qq+f'U1_QQ_RESIDUALSlnArea.svg',bbox_inches='tight')


#%%%% London I


fig, axes = plt.subplots(1,3, figsize=(18, 6))
#fig.suptitle("London - Quantile-Quantile Plot: Normal Distribution vs. ln(Peak Area)", fontsize=16)  # Set the figure title
fig.subplots_adjust(hspace=0.4)  # Adjust the vertical spacing between subplots

# Iterate over a range of 10 Release_rate values
Release_rates = [35,70,70]
colors = ['mediumseagreen','mediumseagreen','lime']
Days = ['Day2','Day2','Day5']

for i, rr in enumerate(Release_rates):
    # Select the data for the current Release_rate
    if i<2:
        x = df_L1d2_comb.loc[df_L1d2_comb['Release_rate'] == rr, 'ln(Area)_residuals']

    else:
        x = df_L1d5_comb.loc[df_L1d5_comb['Release_rate'] == rr, 'ln(Area)_residuals']
    
    # Define the text and its relative coordinates (results of Shapiro and Lilliefors test)
    text  = f"Shapiro-Wilk: {df_L1d2and5_tests_lognormality_residualsArea['Shapiro_normal'][i]}\nLilliefors: {df_L1d2and5_tests_lognormality_residualsArea['Lilliefors_normal'][i]}"
    x_rel = 0.55  # Relative x-coordinate (0.0 to 1.0) #Shapiro-Wilk Lilliefors
    y_rel = 0.05  # Relative y-coordinate (0.0 to 1.0)

    
    sorted_data = np.sort(x)

    empirical_quantiles = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    expected_quantiles = stats.norm.ppf(empirical_quantiles, loc=np.mean(x), scale=np.std(x))

    col = i

    axes[col].plot(expected_quantiles[:-1], expected_quantiles[:-1], color='gray', linestyle='--') #,label='1:1 Line'
    axes[col].scatter(expected_quantiles, sorted_data, color=colors[i], alpha=0.5, label=f'Size: {len(x)}') 
    axes[col].set_title(f'{Days[i]} - {rr} L/min',fontsize=16,fontweight='bold')
    axes[col].set_xlabel(f'Theoretical Quantiles\n(Expected)',fontsize=16)
    #axes[col].set_ylabel('Sample Quantiles (Empirical)',fontsize=12)
    axes[col].tick_params(axis='x', labelsize=16)
    axes[col].tick_params(axis='y', labelsize=16)
    axes[col].legend(handlelength=0,fontsize=14)
    
    if col == 0:
        axes[col].set_ylabel('Sample Quantiles\n(Empirical)',fontsize=16)
        
    # Add result of Shapiro and Lilliefors test (pass/fail) as text to the plot
    # Calculate the data coordinates from relative coordinates
    x_loc = x_rel * (axes[col].get_xlim()[1] - axes[col].get_xlim()[0]) + axes[col].get_xlim()[0]
    y_loc = y_rel * (axes[col].get_ylim()[1] - axes[col].get_ylim()[0]) + axes[col].get_ylim()[0]
    axes[col].text(x_loc, y_loc, text,fontsize=14, transform=axes[col].transData)



if save_fig:
    plt.savefig(path_qq+f'L1_QQ_RESIDUALSlnArea.png',bbox_inches='tight')
    plt.savefig(path_qq+f'L1_QQ_RESIDUALSlnArea.pdf',bbox_inches='tight')
    plt.savefig(path_qq+f'L1_QQ_RESIDUALSlnArea.svg',bbox_inches='tight')





