# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:43:21 2023

@author: Judith Tettenborn (j.a.f.tettenborn@uu.nl)

Follow analysis from Luetschwager et al. (2021):
"Our next analysis examined the effect of the number of detections on the variability in estimated leak emission rates. 
1. For this analysis, we subset the data to only include verified peaks with 20 or more observed peaks (i.e., where elevated 
CH4 had been observed 20 or more times). 

2. For each of these verified peaks, we calculated the estimated emission rate 
using the average natural log of the maximum excess CH4 of all the observed peaks within that verified peak. We refer to 
this estimated emission rate derived from all observations of the leak as the leak indication’s reference emission rate 
for this simulation. 

3. We then performed a Monte Carlo simulation to assess the variation in estimated emission rates under 
a different number of detections relative to this reference. For a given number of detections and each verified peak, we 
randomly sampled that number of detections (observed peaks) from the entire set of detections of the verified peak. For 
example, we randomly select three observed peaks from the set of 45 observed peaks that compose the verified peak. We used 
the randomly selected observed peaks to compute a new, simulated leak emission rate. We repeated this random sampling 2000 
times for each verified peak and number of detections combination. We compiled these results for each of 2–10 detections. 

4. Next, we calculated the percentage difference between the reference emission rate and the simulated emission rates for 
each verified peak, ([simulated–reference]/reference *100). " (Luetschwager et al. (2021))


For point 2:
    Since we know the actual release rate, we can do both: 
        1. comparison to true rr
        2. comparison to reference rr to compare to Luetschwager findings


"""

# Modify Python Path Programmatically -> To include the directory containing the src folder
from pathlib import Path
import sys

# HARDCODED ####################################################################################

path_base = Path('C:/Users/Judit/Documents/UNI/Utrecht/Hiwi/CRE_CH4Quantification/')
# path_base = Path('C:/Users/.../CRE_CH4Quantification/') # insert the the project path here


################################################################################################

sys.path.append(str(path_base / 'src'))
# In Python, the "Python path" refers to the list of directories where Python looks for modules
# and packages when you try to import them in your scripts or interactive sessions. This path 
# is stored in the sys.path list. When you execute an import statement in Python, it searches 
# for the module or package you're trying to import in the directories listed in sys.path. 
# If the directory containing the module or package is not included in sys.path, Python won't 
# be able to find and import it.

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm


from helper_functions.utils import *
from stats_analysis.stats_functions import *

from helper_functions.constants import (
    CAP
    )



#%% LOAD DATA

path_finaldata  = path_base / 'data' / 'final' 
if not (path_base / 'results' / 'Figures' / 'STATS' / 'analysis_montecarlo').is_dir():
       (path_base / 'results' / 'Figures' / 'STATS' / 'analysis_montecarlo').mkdir(parents=True)
path_savefig     = path_base / 'results' / 'Figures' / 'STATS' / 'analysis_montecarlo/'
if not (path_base / 'data' / 'final' / 'analysis_montecarlo').is_dir():
       (path_base / 'data' / 'final' / 'analysis_montecarlo').mkdir(parents=True)
path_savedata     = path_base / 'data' / 'final' / 'analysis_montecarlo/'
   


# ALL
total_peaks_all_alldist = pd.read_csv(path_finaldata / 'RU2T2L3L2_TOTAL_PEAKS.csv', index_col='Datetime', parse_dates=['Datetime'])  
df_all_alldist =  pd.read_csv(path_finaldata / 'RU2T2L3L2_TOTAL_PEAKS_comb.csv', index_col='Datetime', parse_dates=['Datetime']) 

total_peaks_all = total_peaks_all_alldist[total_peaks_all_alldist['Distance_to_source']<75].copy() # only distance to source of < 75 m
df_all = df_all_alldist[(df_all_alldist['Distance_to_source']<75)].copy()
df_all['Loc'] = df_all['Loc'].replace({10: 1, 20: 2, 30:3}) # loc 10 indicated a peak attributed to loc 1, but further away (similar for 2, 3), not needed for this analyis 
total_peaks_all['Loc'] = total_peaks_all['Loc'].replace({10: 1, 20: 2, 30:3})
# Drop rows where 'City' is 'Rotterdam' AND 'Loc' is NOT 1 + London I-Day3
total_peaks_all = total_peaks_all[~(((total_peaks_all['City'] == 'Rotterdam') & (total_peaks_all['Loc'] != 1)) | 
                                    (total_peaks_all['City'] == 'London I-Day3'))]

total_peaks_count = total_peaks_all.groupby(['City','Loc', 'Release_rate']).size().reset_index(name='Count')
total_peaks_count10 = total_peaks_count[total_peaks_count['Count']>=10]
total_peaks_count16 = total_peaks_count[total_peaks_count['Count']>=16]
total_peaks_count20 = total_peaks_count[total_peaks_count['Count']>=20]

df_all_count = df_all.groupby(['City','Loc', 'Release_rate']).size().reset_index(name='Count_peaks')

df_count10 = df_all_count.merge(total_peaks_count10, on=['City', 'Loc', 'Release_rate'], how='inner')

df_count10['Count'].mean()
df_count10['Count'].median()




#%% Theoretical Considerations

# WITHOUT replacement
# Number of possible combnations, given a subset of size k from set of size n
# math.comb(n, k)
math.comb(80, 2)
math.comb(80, 15)
math.comb(20, 10)
math.comb(10, 2)
math.comb(10, 10)

# WITH replacement
# Number of choices (n) and selections (k)
n = 60
k = 20
math.comb(n + k - 1, k)

# Maximum percentage deviations give a cap = 200 L/min

# test = df_R_comb.copy()
# test['rr_area'] = np.exp((test['ln(Area)'] - yintercept_area)/slope_area)

rr_test = pd.DataFrame([5,10,20,40,80,0.15,0.31,0.515,1], columns=['rr'])
rr_test['max_rrpercentage'] = round((200-rr_test['rr'])/rr_test['rr']*100)
print(rr_test)





#%% 2. Determine Reference Leak Rate

df_log = df_all.copy()

# Similar to Luetschwager et al. (2021)

# 1. Calculate ln
# 2. Take mean of ln's
# 3. Calculate leak rate using averaged ln


# Group by 'loc' and 'Release_rate', then calculate the count of occurences per release rate and the mean for each group
df_count = df_log.groupby(['City','Loc', 'Release_rate']).size().reset_index(name='count')
df_ref2_0 = df_log.groupby(['City','Loc', 'Release_rate']).mean(numeric_only=True).reset_index()  



# With Cap ================================================================================================    

df_ref2 = calc_rE_cap(df_ref2_0,CAP) # calculate the emission rates with the different regression model, applying an upper bound of 200 L/min

# Percentage difference vs. real rr (compare diff. calc. methods) 
df_ref2_percent = df_ref2.copy(deep=True)
divisor = df_ref2_percent.iloc[:, 1]
# Divide the entire DataFrame by the divisor using broadcasting
df_ref2_percent.iloc[:, 2:] = (df_ref2_percent.iloc[:, 2:].values-divisor.values[:, np.newaxis]) / divisor.values[:, np.newaxis] *100

# Calculate the mean of the absolute values in the DataFrame
# Create mean_row based on the calculated abs_mean
mean_row = np.abs(df_ref2_percent.iloc[:, 2:]).mean(axis=0,numeric_only=True)
mean_row = mean_row.to_frame().T
mean_row = mean_row.reset_index(drop=True) # Reset the index of mean_row
median_row = np.abs(df_ref2_percent.iloc[:, 2:]).median(axis=0)
median_row = median_row.to_frame().T
median_row = median_row.reset_index(drop=True)
df_ref2_percent = pd.concat([df_ref2_percent, mean_row,median_row], ignore_index=True)

df_log_percent = df_log.copy(deep=True)
df_log_percent= calc_rE_cap(df_log_percent,CAP)
# Percentage difference vs. real rr (compare diff. calc. methods) 
divisor = df_log_percent.loc[:, 'Release_rate']
df_log_percent.loc[:, ['rE_Weller','rE_ALL_max','rE_ALL_area']] = (df_log_percent.loc[:, ['rE_Weller','rE_ALL_max','rE_ALL_area']].values-divisor.values[:, np.newaxis]) / divisor.values[:, np.newaxis] *100
print(df_log_percent.std(numeric_only=True))
print(np.abs(df_log_percent.loc[:, ['rE_Weller','rE_ALL_max','rE_ALL_area']]).mean())



#%% 3. Monte Carlo Simulation


#%%% Prepare Data

# Apply the function to each row to create a new column with tuples
df_count['Loc_tuple'] = df_count.apply(combine_city_loc_and_rr, axis=1)
df_log['Loc_tuple'] = df_log.apply(combine_city_loc_and_rr, axis=1)
df_log.drop(['Loc','Release_rate','City'],axis=1,inplace=True)
df_ref2['Loc_tuple'] = df_ref2.apply(combine_city_loc_and_rr, axis=1)


# # Delete Locations with <20 measurements + outlier
# loc_delete = [('Rotterdam',1.0,120.0),('London-Day2',1,35),('London-Day3',1,35),('London-Day4',1,35),('London-Day4',1,70),
#               ('London_II-Day2',1,0.2),('Utrecht_III',1,0.15),('Utrecht_III',1,1),('Utrecht_III',1,3.95),('Utrecht_III',1,4),
#               ('Utrecht_III',1,10),('Utrecht_III',1,80),('Utrecht_III',1,100),('Utrecht_III',2,0.15),('Utrecht_III',2,0.3),
#               ('Utrecht_III',2,0.5),('Utrecht_III',2,1),('Utrecht_III',2,2.2),('Utrecht_III',2,2.5),('Utrecht_III',2,20),
#               ('Utrecht_III',2,60),('Utrecht_III',2,80),('Utrecht_III',3,4)]

# Delete Locations with <15 measurements
# loc_delete = [('Rotterdam',1.0,120.0),('London-Day2',1,35),('London-Day4',1,35),('Toronto-2c',1,0.12),('Toronto-2c',1,1),('Toronto-2c',1,5),('Toronto-2c',1,9.9),
#               ('London_II-Day2',1,0.2),('Utrecht_III',1,0.15),('Utrecht_III',1,1),('Utrecht_III',1,3.95),
#               ('Utrecht_III',1,10),('Utrecht_III',1,80),('Utrecht_III',1,100),('Utrecht_III',2,0.15),('Utrecht_III',2,0.3),
#               ('Utrecht_III',2,0.5),('Utrecht_III',2,20),('Utrecht_III',2,60),('Utrecht_III',3,4)]

# Delete Locations with <10 measurements
loc_delete = [('Rotterdam',1.0,120.0),('London-Day4',1,35),('Toronto-2c',1,0.12),('Toronto-2c',1,5),('Toronto-2c',1,9.9),
              ('London II-Day2',1,0.2),('Utrecht II',1,0.15),('Utrecht II',1,1),('Utrecht II',1,3.95),
              ('Utrecht II',1,10),('Utrecht II',1,80),('Utrecht II',2,0.3),('Utrecht II',2,60)]

# filenames = ["MonteCarlo_Results_London_II-Day1_loc1.0_rr0.99_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr5.64_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr10.63_MC2000.xlsx",
#               "MonteCarlo_Results_London_II-Day1_loc1.0_rr30.6_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr30.58_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr50.52_MC2000.xlsx",
#               "MonteCarlo_Results_London_II-Day1_loc1.0_rr70.48_MC2000.xlsx","MonteCarlo_Results_London_II-Day2_loc1.0_rr0.49_MC2000.xlsx","MonteCarlo_Results_London_II-Day2_loc1.0_rr0.99_MC2000.xlsx",
#               "MonteCarlo_Results_London-Day2_loc1.0_rr70.0_MC2000.xlsx","MonteCarlo_Results_London-Day5_loc1.0_rr70.0_MC2000.xlsx","MonteCarlo_Results_Rotterdam_loc1.0_rr5.0_MC2000.xlsx",
#               "MonteCarlo_Results_Rotterdam_loc1.0_rr10.0_MC2000.xlsx","MonteCarlo_Results_Rotterdam_loc1.0_rr20.0_MC2000.xlsx","MonteCarlo_Results_Rotterdam_loc1.0_rr40.0_MC2000.xlsx",
#               "MonteCarlo_Results_Rotterdam_loc1.0_rr80.0_MC2000.xlsx","MonteCarlo_Results_Utrecht II_loc1.0_rr15.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_III_loc1.0_rr20.0_MC2000.xlsx",
#               "MonteCarlo_Results_Utrecht_loc1.0_rr2.18_MC2000.xlsx","MonteCarlo_Results_Utrecht_loc1.0_rr3.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_loc2.0_rr15.0_MC2000.xlsx"] 
# release_rates = [0.99,5.64,10.63,30.6,30.58,50.52,70.48,0.49,0.99,70,70,5,10,20,40,80,15,20,2.18,3,15] 



# Create a boolean mask for rows where 'Loc_tuple' is not in values_to_delete
mask = ~df_count['Loc_tuple'].isin(loc_delete)
# Use the mask to filter the DataFrame
df_count = df_count[mask]
mask = ~df_log['Loc_tuple'].isin(loc_delete)
df_log = df_log[mask]
mask = ~df_ref2['Loc_tuple'].isin(loc_delete)
df_ref2 = df_ref2[mask]

df = df_log.copy(deep=True)

methods_determineRR = ['Weller eq.','ALL - max','ALL - area']


    
#%%% MC

# allow for duplicates, otherwise the code is not running bcs of runtime limitations
# even with that the code takes long to run for all releases
# for testing take fewer MC iterations (e.g. 200 instead of 2000) and only simulate
# certain N transects (only for averaging e.g. 3 and 6 transects)




num_repetitions = 20 #200 for testing, 2000 in publication

# Perform the Monte Carlo analysis
results = []

for loc in df_count['Loc_tuple']:  
    loc_results = []
    loc_data = df[df['Loc_tuple'] == loc]  # Filter data for the current location
    print(loc)
    
    sample_range = range(1,len(loc_data)+1) # from 1 to the maximal number of detections for that release
    #sample_range = range(2, 4) # for testing
    
    for N in sample_range:
        print('___________' + str(N) + '____________')
        N_results = []
        
        for i in range(num_repetitions):
            
            # Draw N entries for the current location
            entries = loc_data.sample(N,replace=False, axis=0)
            
            # Perform calculations on the drawn entries
            result = calc_rE_cap_MonteCarlo(entries.loc[:,['ln(Max)','ln(Area)']],CAP) # :-1 -> only give peak and area to function, not last column which is the loc tuple
            
            N_results.append(result)
        
        # Store the results for this N value
        concatenated_df = pd.concat(N_results,ignore_index=True)
        
        loc_results.append(concatenated_df)
            
        
        
    # Store the results for this location
    results.append(loc_results)



save_results = False
if save_results:
    
    for i,loc in enumerate(results, start=0):
        
        # Specify the Excel file name
        excel_file = path_savedata+f"MonteCarlo_Results_{df_count.iloc[i,0]}_loc{df_count.iloc[i,1]}_rr{df_count.iloc[i,2]}_MC{num_repetitions}.xlsx"
        # Create an ExcelWriter object
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:        
        # Loop through the list of DataFrames and save them on different sheets
            for N, df in enumerate(loc, start=0):
                sheet_name = f"N={N+1}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)



                



#%% 4. Stats

#%%% dataframes for analysis

'''
In the following the percentage difference to the 
1. true release rate (diff_true_rr)
2. calculated emission rate based on the mean of all transects (diff_calc_ref_rE)
is calculated for each entry from the MC analysis.

results: list of lists of DataFrames. 
Each list represent one release (a specific release rate at a specific location).
The number of entries per list is determined by the number of available peak detections 
(which is not equal to the number of transects, in case several instruemnts were mounted
 on the same vehicle). The first dataframe in the list represents emission rate (rE) caluclation
when taking into account N=1 measurement, the second with N=2 measurements and so on.
Each dataframe contains rE calculations for all three methods (columns) for num_repetitions rows (2000).

diff_true_list: list of lists of DataFrames
Each list represents one release. It gives the percentage deviation of the
calculated emission rate for all 2000 MC repetitions per N.

diff_true_rr: list of DataFrames
Each DataFrame represents one release. It gives the average percentage deviation of the
calculated emission rate. First the percentage deviation is calculated for all 2000 MC repetitions per N,
then the mean is taken per N. 

    '''

df_ref=df_ref2


diff_true_rr = []
diff_true_list = []
i = 0
for loc in results:
    if loc:  # if loc not empty
        N_array = []
        N_array_list = []
        for N in loc:
            N_copy = N.copy()  # Create a copy of N to avoid modifying the original
            for j in range(3): #N.shape[1]

                rE_true = df_ref['Release_rate'].iloc[i]
                print(rE_true)
                print(N.columns[j])
                
                N_copy[N.columns[j]] = ((N_copy[N.columns[j]] - rE_true) / rE_true) * 100  # Calculate the difference in %
            
            N_array_list.append(np.abs(N_copy))
            N_mean = np.mean(np.abs(N_copy), axis=0)
            N_array.append(N_mean.to_frame().T)

        diff_true_list.append(N_array_list)
        N_array = pd.concat(N_array, ignore_index=True)
        diff_true_rr.append(N_array)
    else:
        diff_true_list.append()
        N_array = pd.DataFrame()  # Create an empty DataFrame when loc is empty
        diff_true_rr.append(N_array)
    i += 1         
    
    
diff_calc_ref_rE = []
diff_calc_ref_list = []
i = 0
for loc in results:
    if loc:  # if loc not empty
        N_array = []
        N_array_list = []
        for N in loc:
            N_copy = N.copy()  # Create a copy of N to avoid modifying the original
            for j in range(3): #N.shape[1]
                rE_true = df_ref['Release_rate'].iloc[i]  # Use true rr (column 1)
                rE_ref = df_ref[N.columns[j]].iloc[i] # each rE estimation method should be compared with itself (Weller estimations with reference value calculated with Weller)
                
                print(rE_true)
                print(rE_ref)
                N_copy[N.columns[j]] = ((N_copy[N.columns[j]] - rE_ref) / rE_ref) * 100  # Calculate the difference in %
              
            N_array_list.append(np.abs(N_copy))
            N_mean = np.mean(np.abs(N_copy), axis=0)
            N_array.append(N_mean.to_frame().T)

        diff_calc_ref_list.append(N_array_list)
        N_array = pd.concat(N_array, ignore_index=True)
        diff_calc_ref_rE.append(N_array)
    else:
        diff_calc_ref_list.append(N_array_list)
        N_array = pd.DataFrame()  # Create an empty DataFrame when loc is empty
        diff_calc_ref_rE.append(N_array)
    i += 1 
    
    
# Merge dfs with same number of transects N of different locations together--------------    
# True rr
difftrue_list_merged = diff_montecarlo_list_merge(diff_true_list)
# Calc rE
diffcalc_list_merged = diff_montecarlo_list_merge(diff_calc_ref_list)




save_results = False
if save_results:

    # Specify the Excel file name
    excel_file = path_savedata / f"diff_true_rr_MC{num_repetitions}.xlsx"
    
    # Create an ExcelWriter object
    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        # Loop through the list of DataFrames and save them on different sheets
        for i, df in enumerate(diff_true_rr, start=0):
            sheet_name = f"{df_count.iloc[i,0]}_loc{df_count.iloc[i,1]}_rr{df_count.iloc[i,2]}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
    # Specify the Excel file name
    excel_file = path_savedata / f"diff_calc_ref_rE_MC{num_repetitions}.xlsx"
    
    # Create an ExcelWriter object
    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        # Loop through the list of DataFrames and save them on different sheets
        for i, df in enumerate(diff_calc_ref_rE, start=0):
            sheet_name = f"{df_count.iloc[i,0]}_loc{df_count.iloc[i,1]}_rr{df_count.iloc[i,2]}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            

   
    
    

    
    
    

#%% PLOTS
save_plots = False


df_dict_truerr = {}

# Loop through the values in df_ref2['Loc_tuple'] and the DataFrames in diff_true_rr
for loc, df in zip(df_ref2['Loc_tuple'], diff_true_rr):
    df_dict_truerr[loc] = df  # Use the value in 'Loc_tuple' as the key and the DataFrame as the value



#%%% Optional: Read In Data

#%%%% Directly diff_true_rr + diff_calc_ref

# in case previous run was saved
read_in_montecarlo_run = False

# Function to read Excel files and store sheets as DataFrames in a list
def read_excel_to_list(file_path):
    dataframes = []
    excel_data = pd.ExcelFile(file_path)
    for sheet_name in excel_data.sheet_names:
        df = pd.read_excel(excel_data, sheet_name=sheet_name)
        dataframes.append(df)
    return dataframes

if read_in_montecarlo_run:
    
    # Define the file names
    file_diff_true_rr = f"diff_true_rr_MC2000.xlsx"
    file_diff_calc_ref_rE = f"diff_calc_ref_rE_MC2000.xlsx"
    
    # Initialize lists to hold the dataframes
    diff_true_rr = []
    diff_calc_ref_rE = []
    
    # Read the "diff_true_rr" file
    diff_true_rr_path = os.path.join(path_savedata, file_diff_true_rr)
    if os.path.exists(diff_true_rr_path):
        diff_true_rr = read_excel_to_list(diff_true_rr_path)
    
    # Read the "diff_calc_ref_rE" file
    diff_calc_ref_rE_path = os.path.join(path_savedata, file_diff_calc_ref_rE)
    if os.path.exists(diff_calc_ref_rE_path):
        diff_calc_ref_rE = read_excel_to_list(diff_calc_ref_rE_path)
    

#%%%% MonteCarlo runs
 

# must be in same order as df_count/df_ref2 - otherwise calculation of diff_calc_rE
# & plotting is wrong       
        
  
# filenames = ["MonteCarlo_Results_loc0.15_MC2000.xlsx","MonteCarlo_Results_loc0.31_MC2000.xlsx","MonteCarlo_Results_loc0.515_MC2000.xlsx","MonteCarlo_Results_loc1.0_MC2000.xlsx","MonteCarlo_Results_loc5.0_MC2000.xlsx","MonteCarlo_Results_loc10.0_MC2000.xlsx","MonteCarlo_Results_loc20.0_MC2000.xlsx","MonteCarlo_Results_loc40.0_MC2000.xlsx","MonteCarlo_Results_loc80.0_MC2000.xlsx"] 
# release_rates = [0.15,0.31,0.515,1,5,10,20,40,80] 

# All - #N >= 20 =============================================================================

# filenames = ["MonteCarlo_Results_London-Day2_loc1.0_rr70.0_MC2000.xlsx","MonteCarlo_Results_London-Day5_loc1.0_rr70.0_MC2000.xlsx",
#               "MonteCarlo_Results_London_II-Day1_loc1.0_rr0.99_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr5.64_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr10.63_MC2000.xlsx",
#               "MonteCarlo_Results_London_II-Day1_loc1.0_rr30.6_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr30.58_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr50.52_MC2000.xlsx",
#               "MonteCarlo_Results_London_II-Day1_loc1.0_rr70.48_MC2000.xlsx","MonteCarlo_Results_London_II-Day2_loc1.0_rr0.49_MC2000.xlsx","MonteCarlo_Results_London_II-Day2_loc1.0_rr0.99_MC2000.xlsx",
#               "MonteCarlo_Results_Rotterdam_loc1.0_rr5.0_MC2000.xlsx","MonteCarlo_Results_Rotterdam_loc1.0_rr10.0_MC2000.xlsx","MonteCarlo_Results_Rotterdam_loc1.0_rr20.0_MC2000.xlsx",
#               "MonteCarlo_Results_Rotterdam_loc1.0_rr40.0_MC2000.xlsx","MonteCarlo_Results_Rotterdam_loc1.0_rr80.0_MC2000.xlsx",
#               "MonteCarlo_Results_Utrecht_loc1.0_rr2.18_MC2000.xlsx","MonteCarlo_Results_Utrecht_loc1.0_rr3.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_loc2.0_rr15.0_MC2000.xlsx",
#               "MonteCarlo_Results_Utrecht_II_loc1.0_rr15.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc1.0_rr20.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc2.0_rr4.0_MC2000.xlsx"] 
# release_rates_read_in = [70,70,0.99,5.64,10.63,30.58,30.6,50.52,70.48,0.49,0.99,5,10,20,40,80,2.18,3,15,15,20,4] 
# city_names = ['London-Day2','London-Day5','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day2','London_II-Day2',
#               'Rotterdam','Rotterdam','Rotterdam','Rotterdam','Rotterdam','Utrecht','Utrecht','Utrecht','Utrecht II','Utrecht II','Utrecht II']
# locs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0]
  

# All - #N >= 10 =============================================================================

filenames = ["MonteCarlo_Results_London-Day2_loc1.0_rr35.0_MC2000.xlsx","MonteCarlo_Results_London-Day2_loc1.0_rr70.0_MC2000.xlsx","MonteCarlo_Results_London-Day5_loc1.0_rr70.0_MC2000.xlsx",
              "MonteCarlo_Results_London_II-Day1_loc1.0_rr0.99_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr5.64_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr10.63_MC2000.xlsx",
              "MonteCarlo_Results_London_II-Day1_loc1.0_rr30.58_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr30.6_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr50.52_MC2000.xlsx",
              "MonteCarlo_Results_London_II-Day1_loc1.0_rr70.48_MC2000.xlsx","MonteCarlo_Results_London_II-Day2_loc1.0_rr0.49_MC2000.xlsx","MonteCarlo_Results_London_II-Day2_loc1.0_rr0.99_MC2000.xlsx",
              "MonteCarlo_Results_Rotterdam_loc1.0_rr5.0_MC2000.xlsx","MonteCarlo_Results_Rotterdam_loc1.0_rr10.0_MC2000.xlsx",
              "MonteCarlo_Results_Rotterdam_loc1.0_rr20.0_MC2000.xlsx","MonteCarlo_Results_Rotterdam_loc1.0_rr40.0_MC2000.xlsx","MonteCarlo_Results_Rotterdam_loc1.0_rr80.0_MC2000.xlsx",
              "MonteCarlo_Results_Toronto-2c_loc1.0_rr0.5_MC2000.xlsx","MonteCarlo_Results_Toronto-2c_loc1.0_rr1.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_loc1.0_rr2.18_MC2000.xlsx",
              "MonteCarlo_Results_Utrecht_loc1.0_rr3.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_loc2.0_rr15.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc1.0_rr4.0_MC2000.xlsx",
              "MonteCarlo_Results_Utrecht_II_loc1.0_rr15.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc1.0_rr20.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc1.0_rr100.0_MC2000.xlsx",
              "MonteCarlo_Results_Utrecht_II_loc2.0_rr0.15_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc2.0_rr0.5_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc2.0_rr1.0_MC2000.xlsx",
              "MonteCarlo_Results_Utrecht_II_loc2.0_rr2.2_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc2.0_rr2.5_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc2.0_rr4.0_MC2000.xlsx",
              "MonteCarlo_Results_Utrecht_II_loc2.0_rr20.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc2.0_rr80.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc3.0_rr4.0_MC2000.xlsx"] 
release_rates_read_in = [35,70,70,0.99,5.64,10.63,30.58,30.6,50.52,70.48,0.49,0.99,5,10,20,40,80,0.5,1,2.18,3,15,4,15,20,100,0.15,0.5,1,2.2,2.5,4,20,80,4] 
city_names = ['London I-Day2','London I-Day2','London I-Day5','London II-Day1','London II-Day1','London II-Day1','London II-Day1','London II-Day1','London II-Day1','London  II-Day1','London II-Day2','London II-Day2',
              'Rotterdam','Rotterdam','Rotterdam','Rotterdam','Rotterdam','Toronto-2c','Toronto-2c','Utrecht I','Utrecht I','Utrecht I','Utrecht_II','Utrecht II','Utrecht II','Utrecht II','Utrecht II','Utrecht II',
              'Utrecht II','Utrecht II','Utrecht II','Utrecht II','Utrecht II','Utrecht II','Utrecht II',]
locs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0]



list_data = []
    
for i in range(len(filenames)):
    filepath = path_savedata + filenames[i]

    # Load all sheets into a dictionary of DataFrames
    xls = pd.ExcelFile(filepath)
    sheet_names = xls.sheet_names

    # As list
    data = list(pd.read_excel(filepath, sheet_name=sheet_names[:20]).values())
    add_characteristics(data,release_rates_read_in[i],city_names[i],locs[i])
    list_data.append(data)
    
# apply cap -------
# # Function to cap values in a dataframe
# def cap_values(df, CAP):
#     return df.applymap(lambda x: min(x, CAP))


# # Iterate through the list of lists and update dataframes
# for sublist in list_data:
#     for i, df in enumerate(sublist):
#         sublist[i] = cap_values(df, 200)
# -----------------

# list_data_0 = list_data.copy()

# # For testing: ---------------------
# # Define a function to check if any dataframe in the list contains 'London-Day4' in the 'City' column
# def contains_cityx(df_list,city_x):
#     for df in df_list:
#         if city_x in df['City'].values:
#             return True
#     return False

# # Filter out the lists where any dataframe contains 'London-Day4' in 'City'
# list_data = [df_list for df_list in list_data if not contains_cityx(df_list, 'London-Day4')]
# # ------------------------------------

data_comb_0 = [pd.concat(dataframes, axis=0, ignore_index=True) for dataframes in zip(*list_data)]



df_ref = df_ref2.copy()
# df_ref = df_ref[df_ref['City'] != 'London-Day3']
df_ref = df_ref[(df_ref['City'] != 'London-Day3') & (df_ref['City'] != 'London-Day4')]
df_ref = df_ref.reset_index(drop=True)


diff_true_rr = []
diff_true_list = []
i = 0
for loc in list_data:
    if loc:  # if loc not empty
        N_array = []
        N_array_list = []
        for N in loc:
            N_copy = N.copy()  # Create a copy of N to avoid modifying the original
            rE_true = N_copy.loc[0,'Release_rate']
            for j in range(3): #N.shape[1]

                print(rE_true)
                print(N.columns[j])
                N_copy = N_copy[['rE_Weller', 'rE_ALL_max', 'rE_ALL_area']]
                N_copy[N.columns[j]] = ((N_copy[N.columns[j]] - rE_true) / rE_true) * 100  # Calculate the difference in %
                
            N_array_list.append(np.abs(N_copy))
            N_mean = np.mean(np.abs(N_copy), axis=0)
            N_array.append(N_mean.to_frame().T)

        diff_true_list.append(N_array_list)
        N_array = pd.concat(N_array, ignore_index=True)
        diff_true_rr.append(N_array)
    else:
        diff_true_list.append()
        N_array = pd.DataFrame()  # Create an empty DataFrame when loc is empty
        diff_true_rr.append(N_array)
    i += 1         
    

    
diff_calc_ref_rE = []
diff_calc_ref_list = []
i = 0
for loc in list_data:
    if loc:  # if loc not empty
        N_array = []
        N_array_list = []
        for N in loc:
            N_copy = N.copy()  # Create a copy of N to avoid modifying the original
            rr_index = df_ref.index[(df_ref['Release_rate'] == N_copy.loc[0, 'Release_rate']) & 
                                     (df_ref['City'] == N_copy.loc[0, 'City']) & 
                                     (df_ref['Loc'] == N_copy.loc[0, 'Loc'])]
            for j in range(3): #N.shape[1]
                #i=5 #test
                rE_true = df_ref['Release_rate'].iloc[i]  # Use true rr (column 1)
                # rE_ref = df_ref[N.columns[j]].iloc[i] # each rE estimation method should be compared with itself (Weller estimations with reference value calculated with Weller)
                rE_ref = df_ref.loc[rr_index, N_copy.columns[j]].iloc[0]
                print(rE_true)
                print(rE_ref)
                N_copy = N_copy[['rE_Weller', 'rE_ALL_max', 'rE_ALL_area']]
                N_copy[N.columns[j]] = ((N_copy[N.columns[j]] - rE_ref) / rE_ref) * 100  # Calculate the difference in %
              
            N_array_list.append(np.abs(N_copy))
            N_mean = np.mean(np.abs(N_copy), axis=0)
            N_array.append(N_mean.to_frame().T)

        diff_calc_ref_list.append(N_array_list)
        N_array = pd.concat(N_array, ignore_index=True)
        diff_calc_ref_rE.append(N_array)
    else:
        diff_calc_ref_list.append(N_array_list)
        N_array = pd.DataFrame()  # Create an empty DataFrame when loc is empty
        diff_calc_ref_rE.append(N_array)
    i += 1 


# Merge dfs with same number of transects N of different locations together--------------    
# True rr
difftrue_list_merged = diff_montecarlo_list_merge(diff_true_list)
# Calc rE
diffcalc_list_merged = diff_montecarlo_list_merge(diff_calc_ref_list)
# ----------------------------------------------------------------------------------------




dev_max = [df['rE_ALL_area'].max() for df in difftrue_list_merged]
 
    


    
    
  
    
#%%% FINAL: Area only

with_mean = True




#%%%% FINAL: calc. rE - Zoom: All rR

## ADJUST ## --------------------------------------
save_fig = False
N_show =10
rr = np.array(df_ref['Release_rate'])  # uncomment, if running the script in one go
# rr = np.array(release_rates_read_in) # uncomment, when reading in MonteCarlo data that were stored previously
# ------------------------------------------------

colors9 = ['#f5abb9','#feb47b','#ff7e5f','#765285','#351c4d','#c7fdf7','#a7e351','#2ace82','#32e7c8']
colors9 = ['#f5abb9','#feb47b','#ff7e5f','#765285','#351c4d','#c7fdf7','#a7e351','#2ace82','#32e7c8','#f5abb9','#feb47b','#ff7e5f','#765285','#351c4d','#c7fdf7','#a7e351','#2ace82','#32e7c8','#c7fdf7','#a7e351','#2ace82','#32e7c8']

# With colorbar -----------------
# Normalize the rr array to [0, 1] for mapping to colors
norm = Normalize(vmin=min(rr), vmax=max(rr))
cmap = cm.get_cmap('gist_rainbow')  # You can choose any colormap, e.g., 'plasma', 'coolwarm', etc.
# -------------------------------

labels3 = [r'$r_E$ Weller', r'$r_E$ ALL_max', r'$r_E$ ALL_area']
line_handles = []

j = 2 # area
i=0
fig,ax1 = plt.subplots(figsize=(20,12))

for loc in diff_calc_ref_rE:
    print(len(loc))
    color = cmap(norm(rr[i]))  # Map rr[i] to a color
    line, = plt.plot(loc[0:N_show].index+1,loc[0:N_show].iloc[:,j],linewidth=2,color=color,label=f'{df_ref.iloc[i,-1]}') 
    line_handles.append(line)
    i+=1

# with mean over all rr
diff_concatenated = pd.concat(diff_calc_ref_rE[:-1], axis=0)
mean_df = diff_concatenated.groupby(diff_concatenated.index).mean()
line, = plt.plot(mean_df[0:N_show].index+1,mean_df[0:N_show].iloc[:,j],linewidth=7,color='black') #label=f'{loc.columns[i]}' #,label=labels3[j]
plt.scatter(mean_df[0:N_show].index+1,mean_df[0:N_show].iloc[:,j],s=160,color='black', label=f'mean', zorder=3) # zorder = Ensure mean line is on top - give higher order than lines before
line_handles.append(line)
    
plt.xlabel('Number of transects',fontsize=34)
plt.ylabel(r'$\Delta$ % of estimated $r_E$ from mean $r_\mathrm{E,mean}  $',fontsize=34)
# plt.title(f'Area eq.',fontsize=24,fontweight='bold')
# Set custom ticks on the x-axis
ax1.set_ylim(0,700)
custom_ticks = np.arange(2, N_show+2, 2)
plt.xticks(custom_ticks)
ax1.tick_params(axis='x', labelsize=34)
ax1.tick_params(axis='y', labelsize=34)

# Colorbar instead of displaying rr explicitely ---------------------
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(rr)  # This makes the colorbar know the range of values
cbar = fig.colorbar(sm, ax=ax1)  # Link colorbar to the axis
cbar.set_label(r'Emission rate [$\mathrm{Lmin^{-1}}$]',fontsize=34)  # Label for the colorbar
cbar.ax.tick_params(labelsize=34) # increase fontsize of color bar ticks
# ----------------------------------------------------------



if save_fig:
    # plt.savefig(path_savefig+f'Calc_rE/CalcrE_AllrR_Area_t10N10.png',bbox_inches='tight')
    # plt.savefig(path_savefig+f'Calc_rE/CalcrE_AllrR_Area_t10N10.pdf',bbox_inches='tight')
    # plt.savefig(path_savefig+f'Calc_rE/CalcrE_AllrR_Area_t10N10.svg',bbox_inches='tight')
    
    plt.savefig(path_savefig+f'Calc_rE/CalcrE_AllrR_Area_t10N10_yaxis700.png',bbox_inches='tight')
    plt.savefig(path_savefig+f'Calc_rE/CalcrE_AllrR_Area_t10N10_yaxis700.pdf',bbox_inches='tight')
    plt.savefig(path_savefig+f'Calc_rE/CalcrE_AllrR_Area_t10N10_yaxis700.svg',bbox_inches='tight')




#%%%% FINAL: true rR - Zoom: All rR


# ADJUST --------------------------------------------------------------

save_fig = False
N_show = 10 #19

line_handles = []

# either uncomment all a) (read in saved datasets) or all b) (letting script run directly)

# 1. Which DATA to plot 
# a)
# df_true_plot = diff_true_rr.copy()
# b)
df_true_plot = [df for df in diff_true_rr if df['rE_ALL_area'].iloc[-1] < df['rE_ALL_area'].iloc[0]]

# 2. Get corresponding reference dataframe (which stores release rates and mean emission rates):
# a) Track indices while building diff_concatenated ----
# indices_of_concatenated = [i for i, df in enumerate(diff_true_rr) if df['rE_ALL_area'].iloc[-1] < df['rE_ALL_area'].iloc[0]] # negative curves
# indices_of_concatenated = [i for i, df in enumerate(diff_true_rr) if df['rE_ALL_area'].iloc[-1] > df['rE_ALL_area'].iloc[0]] # positive curves
# Use these indices to get the corresponding dataframes from diff_true_list
# df_ref_plot = pd.DataFrame([df_ref.iloc[i,:] for i in indices_of_concatenated])
# b) ----
df_ref_plot = df_ref.copy()

# 3. Release rates of corresponding dataset
# a)
# rr = np.array(release_rates_read_in)
# b)
rr = np.array(df_ref_plot['Release_rate'])



# With colorbar -----------------
# Normalize the rr array to [0, 1] for mapping to colors
norm = Normalize(vmin=min(rr), vmax=max(rr))
cmap = cm.get_cmap('gist_rainbow')
# -------------------------------

j = 2 # area
i=0
fig,ax1 = plt.subplots(figsize=(20,12))

# -------------
for loc in df_true_plot[:-1]:
    print(len(loc))
    color = cmap(norm(rr[i]))  # Map rr[i] to a color
    line, = plt.plot(loc[0:N_show].index+1,loc[0:N_show].iloc[:,j],linewidth=2,color=color,label=f'{df_ref_plot.iloc[i,-1]}') #label=f'{loc.columns[i]}' #,label=labels3[j],
    
    line_handles.append(line)
    i+=1
# -------------

# with mean over all rr
diff_concatenated = pd.concat(diff_true_rr[:-1], axis=0)
mean_df = diff_concatenated.groupby(diff_concatenated.index).mean()
#mean_df = diff_concatenated.groupby(diff_concatenated.index % (N_show+1)).mean()
line, = plt.plot(mean_df[0:N_show].index+1,mean_df[0:N_show].iloc[:,j],linewidth=7,color='black', zorder=3) #label=f'{loc.columns[i]}' #,label=labels3[j]
plt.scatter(mean_df[0:N_show].index+1,mean_df[0:N_show].iloc[:,j],s=160,color='black', label=f'mean', zorder=3) # zorder = Ensure mean line is on top - give higher order than lines before
line_handles.append(line)

 
plt.xlabel('Number of transects',fontsize=36)
plt.ylabel(r'$\Delta$ % of estimated $r_E$ from true $r_\mathrm{E,true}$',fontsize=36)
# plt.title(f'Area eq.',fontsize=24,fontweight='bold')
custom_ticks = np.arange(2, N_show+2, 2) # Set custom ticks on the x-axis
plt.xticks(custom_ticks)
ax1.tick_params(axis='x', labelsize=36)
ax1.tick_params(axis='y', labelsize=36)
#ax1.set_ylim(0,400)


# Colorbar instead of displaying rr explicitely ---------------------
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(rr)  # This makes the colorbar know the range of values
cbar = fig.colorbar(sm, ax=ax1)  # Link colorbar to the axis
cbar.set_label(r'Emission rate [$\mathrm{Lmin^{-1}}$]',fontsize=36)  # Label for the colorbar
cbar.ax.tick_params(labelsize=36) # increase fontsize of color bar ticks
# ----------------------------------------------------------

plt.tight_layout()
plt.show()

if save_fig:
    plt.savefig(path_savefig+f'True_RR/TrueRR_AllrR_Area_t10N10.png',bbox_inches='tight')
    plt.savefig(path_savefig+f'True_RR/TrueRR_AllrR_Area_t10N10.pdf',bbox_inches='tight')
    plt.savefig(path_savefig+f'True_RR/TrueRR_AllrR_Area_t10N10.svg',bbox_inches='tight')
    
    
    

#%%% All regression models


#%%%% calc. rE: All calc. methods per loc.

# Single Plots ------------------------------------------------------------------

colors3 = ['gray','blue','orange']
colors3 = ['gray','#5142f5','darkorange']
labels3 = [r'$r_E$ Weller', r'$r_E$ ALL_max', r'$r_E$ ALL_area']
line_handles = []

j=0
for loc in diff_calc_ref_rE: 
    fig,ax1 = plt.subplots(figsize=(16,12))
    
    
    for i in [0,1,2]:
        line, = plt.plot(loc.index+1,loc.iloc[:,i],linewidth=2,color=colors3[i],label=labels3[i]) #label=f'{loc.columns[i]}'
        plt.scatter(loc.index+1,loc.iloc[:,i],s=50,color=colors3[i], label='')
        
        line_handles.append(line)
        
    plt.xlabel('Number of transects',fontsize=24)
    plt.ylabel(r'$\Delta$ % of estimated $r_E$ from reference $r_\mathrm{E,ref}$',fontsize=24)
    plt.title(f'{df_count.iloc[j,0]}, loc {df_count.iloc[j,1]}, rr {df_count.iloc[j,2]} L/min',fontsize=24,fontweight='bold')
    # Set custom ticks on the x-axis
    custom_ticks = np.arange(10, len(loc), 24)
    plt.xticks(custom_ticks)
    ax1.tick_params(axis='x', labelsize=24)
    ax1.tick_params(axis='y', labelsize=24)
    # Modify the legend labels
    new_labels = ['Weller eq.', 'Maximum eq.', 'Area eq.']  # Replace these with your desired labels 
    line_legend = plt.legend(handles=line_handles, labels=new_labels, fontsize=20)
    ax1.add_artist(line_legend)

    # plt.savefig(path_savefig+f'Calc_rE/CalcrE_{df_count.iloc[j,0]}_rr{df_count.iloc[j,2]}_AllMethods_N{len(loc)}.png',bbox_inches='tight')
    # plt.savefig(path_savefig+f'Calc_rE/CalcrE_{df_count.iloc[j,0]}_rr{df_count.iloc[j,2]}_AllMethods_N{len(loc)}.pdf',bbox_inches='tight')
    # plt.savefig(path_savefig+f'Calc_rE/CalcrE_{df_count.iloc[j,0]}_rr{df_count.iloc[j,2]}_AllMethods_N{len(loc)}.svg',bbox_inches='tight')
    
   
    j+=1
    
    
#%%%% calc. rE - Zoom: All calc. methods per loc. 1-20

# Single Plots ------------------------------------------------------------------

colors3 = ['gray','blue','orange']
colors3 = ['gray','#5142f5','darkorange']
labels3 = [r'$r_E$ Weller', r'$r_E$ ALL_max', r'$r_E$ ALL_area']
line_handles = []

j=0
for loc in diff_calc_ref_rE: #
    fig,ax1 = plt.subplots(figsize=(16,12))
    
    
    for i in [0,1,2]:
        line, = plt.plot(loc[0:19].index+1,loc[0:19].iloc[:,i],linewidth=2,color=colors3[i],label=labels3[i]) #label=f'{loc.columns[i]}'
        plt.scatter(loc[0:19].index+1,loc[0:19].iloc[:,i],s=60,color=colors3[i], label='')
        
        line_handles.append(line)
        
    plt.xlabel('Number of transects',fontsize=24)
    plt.ylabel(r'$\Delta$ % of estimated $r_E$ from reference $r_\mathrm{E,ref}$',fontsize=24)
    plt.title(f'{df_count.iloc[j,0]}, loc {df_count.iloc[j,1]}, rr {df_count.iloc[j,2]} L/min',fontsize=24,fontweight='bold')
    # Set custom ticks on the x-axis
    custom_ticks = np.arange(2, 21, 2)
    plt.xticks(custom_ticks)
    ax1.tick_params(axis='x', labelsize=24)
    ax1.tick_params(axis='y', labelsize=24)
    # Modify the legend labels
    new_labels = ['Weller eq.', 'Maximum eq.', 'Area eq.']  # Replace these with your desired labels 
    line_legend = plt.legend(handles=line_handles, labels=new_labels, fontsize=20)
    ax1.add_artist(line_legend)

    # plt.savefig(path_savefig+f'Calc_rE/CalcrE_{df_count.iloc[j,0]}_rr{df_count.iloc[j,2]}_AllMethods_N{len(loc)}_ZOOM.png',bbox_inches='tight')
    # plt.savefig(path_savefig+f'Calc_rE/CalcrE_{df_count.iloc[j,0]}_rr{df_count.iloc[j,2]}_AllMethods_N{len(loc)}_ZOOM.pdf',bbox_inches='tight')
    # plt.savefig(path_savefig+f'Calc_rE/CalcrE_{df_count.iloc[j,0]}_rr{df_count.iloc[j,2]}_AllMethods_N{len(loc)}_ZOOM.svg',bbox_inches='tight')

    
    j+=1
    
    


 

#%%%% true RR: All calc. methods per loc.
  
# Single Plots - Only 3 ---------------------------------------------------------

#colors3 = ['gray','blue','orange']
colors3 = ['gray','#5142f5','darkorange']
labels3 = [r'$r_E$ Weller', r'$r_E$ ALL_max', r'$r_E$ ALL_area']
line_handles = []

j=0
for loc in diff_true_rr: 
    fig,ax1 = plt.subplots(figsize=(16,12))
    
    
    for i in [0,1,2]:
        line, = plt.plot(loc.index+1,loc.iloc[:,i],linewidth=2,color=colors3[i],label=labels3[i]) #label=f'{loc.columns[i]}'
        plt.scatter(loc.index+1,loc.iloc[:,i],s=50,color=colors3[i], label='')
        
        line_handles.append(line)
        
    plt.xlabel('Number of transects',fontsize=24)
    plt.ylabel(r'$\Delta$ % of estimated $r_E$ from true $r_R$',fontsize=24)
    plt.title(f'{df_count.iloc[j,0]}, loc {df_count.iloc[j,1]}, rr {df_count.iloc[j,2]} L/min',fontsize=24,fontweight='bold')
    # Set custom ticks on the x-axis
    custom_ticks = np.arange(10, len(loc), 10)
    plt.xticks(custom_ticks)
    ax1.tick_params(axis='x', labelsize=24)
    ax1.tick_params(axis='y', labelsize=24)
    # Modify the legend labels
    new_labels = ['Weller eq.', 'Maximum eq.', 'Area eq.']  # Replace these with your desired labels 
    line_legend = plt.legend(handles=line_handles, labels=new_labels, fontsize=20)
    ax1.add_artist(line_legend)

    
    # plt.savefig(path_savefig+f'True_RR/TrueRR_{df_count.iloc[j,0]}_rr{df_count.iloc[j,2]}_AllMethods_N{len(loc)}.png',bbox_inches='tight')
    # plt.savefig(path_savefig+f'True_RR/TrueRR_{df_count.iloc[j,0]}_rr{df_count.iloc[j,2]}_AllMethods_N{len(loc)}.pdf',bbox_inches='tight')
    # plt.savefig(path_savefig+f'True_RR/TrueRR_{df_count.iloc[j,0]}_rr{df_count.iloc[j,2]}_AllMethods_N{len(loc)}.svg',bbox_inches='tight')

    j+=1
     
    
#Single Plots with R regression ------------------------------------------------------------------

# colors3 = ['gray','blue','orange']
# colors5 = ['gray','turquoise','blue','violet','orangered']
# labels5 = [r'$r_E$ Weller', r'$r_E$ R_max', r'$r_E$ ALL_max', r'$r_E$ R_area', r'$r_E$ ALL_area']

# j=0
# for loc in diff_calc_ref_rE: #diff_true_rr   #diff_calc_ref_rE
#     fig,ax1 = plt.subplots(figsize=(16,12))
    
#     for i in range(loc.shape[1]):
#         plt.plot(loc.index+1,loc.iloc[:,i],linewidth=2,color=colors5[i],label=labels5[i]) #label=f'{loc.columns[i]}'
#         plt.scatter(loc.index+1,loc.iloc[:,i],s=40,color=colors5[i])
        
#     plt.xlabel('Number of transects',fontsize=20)
#     plt.ylabel(r'$\Delta$ % of estimated $r_E$ from true $r_R$',fontsize=20)
#     plt.title(f'{df_count.iloc[j,1]} L/min',fontsize=20)
#     # Set custom ticks on the x-axis
#     custom_ticks = np.arange(2, 21, 2)
#     plt.xticks(custom_ticks)
#     ax1.tick_params(axis='x', labelsize=20)
#     ax1.tick_params(axis='y', labelsize=20)
#     plt.legend(fontsize=18)
#     # plt.savefig(path_savefig+f'True_RR/R_trueRR_loc{df_count.iloc[j,1]}_AllMethods_N{max(sample_range)}_MC{num_repetitions}.png',bbox_inches='tight')
#     # plt.savefig(path_savefig+f'True_RR/R_trueRR_loc{df_count.iloc[j,1]}_AllMethods_N{max(sample_range)}_MC{num_repetitions}.pdf',bbox_inches='tight')

#     j+=1



#%%%% true RR Zoom: All calc. methods per loc. 1-20
  
# Single Plots - Only 3 ---------------------------------------------------------

colors3 = ['gray','blue','orange']
colors3 = ['gray','#5142f5','darkorange']
labels3 = [r'$r_E$ Weller', r'$r_E$ ALL_max', r'$r_E$ ALL_area']
line_handles = []

j=0
for loc in diff_true_rr: 
    fig,ax1 = plt.subplots(figsize=(16,12))
    
    
    for i in [0,1,2]:
        line, = plt.plot(loc[0:19].index+1,loc[0:19].iloc[:,i],linewidth=2,color=colors3[i],label=labels3[i]) #label=f'{loc.columns[i]}'
        plt.scatter(loc[0:19].index+1,loc[0:19].iloc[:,i],s=60,color=colors3[i], label='')
        
        line_handles.append(line)
        
        
    plt.xlabel('Number of transects',fontsize=24)
    plt.ylabel(r'$\Delta$ % of estimated $r_E$ from true $r_R$',fontsize=24)
    # use if after run:
    # plt.title(f'{df_count.iloc[j,0]}, loc {df_count.iloc[j,1]}, rr {df_count.iloc[j,2]} L/min',fontsize=24,fontweight='bold')
    # use if after reading runs in from excel
    plt.title(f'{city_names[j]}, loc {locs[j]}, rr {release_rates_read_in[j]} L/min',fontsize=24,fontweight='bold')
    # Set custom ticks on the x-axis
    custom_ticks = np.arange(2, 21, 2)
    plt.xticks(custom_ticks)
    ax1.tick_params(axis='x', labelsize=24)
    ax1.tick_params(axis='y', labelsize=24)
    # Modify the legend labels
    new_labels = ['Weller eq.', 'Maximum eq.', 'Area eq.']  # Replace these with your desired labels 
    line_legend = plt.legend(handles=line_handles, labels=new_labels, fontsize=20)
    ax1.add_artist(line_legend)

    # plt.savefig(path_savefig+f'True_RR/TrueRR_{df_count.iloc[j,0]}_rr{df_count.iloc[j,2]}_AllMethods_N{len(loc)}_ZOOM.png',bbox_inches='tight')
    # plt.savefig(path_savefig+f'True_RR/TrueRR_{df_count.iloc[j,0]}_rr{df_count.iloc[j,2]}_AllMethods_N{len(loc)}_ZOOM.pdf',bbox_inches='tight')
    # plt.savefig(path_savefig+f'True_RR/TrueRR_{df_count.iloc[j,0]}_rr{df_count.iloc[j,2]}_AllMethods_N{len(loc)}_ZOOM.svg',bbox_inches='tight')


    
    
    j+=1
    



#%% Hypothetical Distributions

'''
For testing purposes: investigate how the Monte Carlo analysis results look for
different distributions, e.g. perfect gaussian distribution, gaussian pluss offset or outlier...

In Supplement of Tettenborn et al. (2025), in section  S10 Influence of Sampling 
Effort, from Figure S24 onwards.
    '''



from scipy.stats import truncnorm



slope_max = 0.854
yintercept_max = -1.25
slope_area = 0.774
yintercept_area = 1.84

size_distr = 60

df_distr = pd.DataFrame()

rr = 3
area_linreg = np.log(rr)*slope_area+yintercept_area # area that should be measured according to the regression when release 3 L/min

# Distribution 1 - perfect ----------------------------------------------------
mean =  area_linreg # Mean of the distribution = 2.7
std_dev = 1  # Standard deviation of the distribution
lower_bound = mean-2  # Lower boundary
upper_bound = mean+2  # Upper boundary
# Calculate the a and b parameters for truncnorm
a, b = (lower_bound - mean) / std_dev, (upper_bound - mean) / std_dev
# Create the truncated normal distribution
trunc_gauss = truncnorm(a, b, loc=mean, scale=std_dev)
# Generate samples
distr = trunc_gauss.rvs(size=size_distr)  # Specify the number of samples
df_distr['1_perfect'] = distr

# Distribution 2 - too low ----------------------------------------------------
mean = area_linreg - 2.7 
std_dev = 1  
lower_bound = mean-2  
upper_bound = mean+2  
# Calculate the a and b parameters for truncnorm
a, b = (lower_bound - mean) / std_dev, (upper_bound - mean) / std_dev
# Create the truncated normal distribution
trunc_gauss = truncnorm(a, b, loc=mean, scale=std_dev)
# Generate samples
distr = trunc_gauss.rvs(size=size_distr)  # Specify the number of samples
df_distr['2_large-neg-offset'] = distr

# Distribution 3 - small negative offset ----------------------------------------------------
mean = area_linreg - 0.7  
std_dev = 1  
lower_bound = mean-2  
upper_bound = mean+2  
# Calculate the a and b parameters for truncnorm
a, b = (lower_bound - mean) / std_dev, (upper_bound - mean) / std_dev
# Create the truncated normal distribution
trunc_gauss = truncnorm(a, b, loc=mean, scale=std_dev)
# Generate samples
distr = trunc_gauss.rvs(size=size_distr)  # Specify the number of samples
df_distr['3_small-neg-offset'] = distr

# Distribution 4 - too high ----------------------------------------------------
mean = area_linreg + 2.7  
std_dev = 1  
lower_bound = mean-2  
upper_bound = mean+2  
# Calculate the a and b parameters for truncnorm
a, b = (lower_bound - mean) / std_dev, (upper_bound - mean) / std_dev
# Create the truncated normal distribution
trunc_gauss = truncnorm(a, b, loc=mean, scale=std_dev)
# Generate samples
distr = trunc_gauss.rvs(size=size_distr)  # Specify the number of samples

df_distr['4_large-pos-offset'] = distr


# Distribution 5 -  small positive offset ----------------------------------------------------
mean = area_linreg + 0.7   
std_dev = 1.2 
lower_bound = mean-2  
upper_bound = mean+2  
# Calculate the a and b parameters for truncnorm
a, b = (lower_bound - mean) / std_dev, (upper_bound - mean) / std_dev
# Create the truncated normal distribution
trunc_gauss = truncnorm(a, b, loc=mean, scale=std_dev)
# Generate samples
distr = trunc_gauss.rvs(size=size_distr)  # Specify the number of samples

df_distr['5_small-pos-offset'] = distr


# Distribution 6 - too low + Outliers ----------------------------------------------------
mean = area_linreg - 2.7 
std_dev = 1  
lower_bound = mean-2  
upper_bound = mean+2  
# Calculate the a and b parameters for truncnorm
a, b = (lower_bound - mean) / std_dev, (upper_bound - mean) / std_dev
# Create the truncated normal distribution
trunc_gauss = truncnorm(a, b, loc=mean, scale=std_dev)
# Generate samples
distr = trunc_gauss.rvs(size=size_distr-3)  # Specify the number of samples
distr = np.append(distr, [area_linreg+0.5,area_linreg+0.9, area_linreg+1.1])
df_distr['6_large-neg-offset-outlier'] = distr


# Use melt to stack the columns
df_distr_stack = df_distr.melt(var_name='Type_distr', value_name='ln(Area)')
df_distr_stack_count = df_distr_stack.groupby('Type_distr').size().reset_index(name='count')

df_ref = df_distr_stack.groupby(['Type_distr']).mean(numeric_only=True).reset_index()  
df_ref['rE_ALL_area'] = np.exp((df_ref['ln(Area)'] - yintercept_area)/slope_area)
df_ref['Release_rate'] = rr


#%%% MC Simulation

def perform_calculation(entries,cap):
    # Compute "simulated" leak rate
    
    entries = entries.mean(numeric_only=True).to_frame().T # take mean of log value and ensure it stays a df
    
    entries_area = entries.filter(like='Area')
    
    # Estimate r_E from that mean value
    rE_ALL_area = np.exp((entries_area - yintercept_area)/slope_area)
    rE_ALL_area.loc[rE_ALL_area['ln(Area)'] > cap, 'ln(Area)'] = cap
    
    rE_ALL_area.rename(columns={'ln(Area)': 'rE_ALL_area'}, inplace=True)
    df_rE = pd.concat([rE_ALL_area,entries_area],axis=1)
    
    return df_rE

cap=200




#sample_range = range(1, 20) #test
#sample_range = range(2, 4)
num_repetitions = 2000

# Perform the Monte Carlo analysis
results_hypo = []
entries_hypo = []
#test = df_count.iloc[5].to_frame().T
for loc in df_distr_stack_count['Type_distr']: 
    loc_results = []
    loc_entries = []
    loc_data = df_distr_stack[df_distr_stack['Type_distr'] == loc]  # Filter data for the current location
    print(loc)
    
    sample_range = range(1,len(loc_data)+1)
    
    for N in sample_range:
        print('___________' + str(N) + '____________')
        N_results = []

        for i in range(num_repetitions):

            # Draw N entries for the current location        
            entries = loc_data.sample(N,replace=False, axis=0) #[['column1', 'column2']]  # Adjust columns as needed
            loc_entries.append(entries)
            # Perform calculations on the drawn entries
            result = perform_calculation(entries.loc[:,['ln(Area)']],cap) # :-1 -> only give peak and area to function, not last column which is the loc tuple
            N_results.append(result)
        
        # Store the results for this N value
        concatenated_df = pd.concat(N_results,ignore_index=True)
        loc_results.append(concatenated_df)
        
    # Store the results for this location
    results_hypo.append(loc_results)
    entries_hypo.append(loc_entries)



#%%% Stats

results = results_hypo
# results_hypo = results_hypo_3.copy()
# entries_hypo = entries_hypo_3.copy()
# results_hypo_50 = results_hypo.copy()
# entries_hypo_50 = entries_hypo.copy()
# results_hypo_3 = results_hypo.copy()
# entries_hypo_3 = entries_hypo.copy()


diff_true_rr_hypo = []
diff_true_list_hypo = []
i = 0
for loc in results:
    if loc:  # if loc not empty
        N_array = []
        N_array_list = []
        for N in loc:
            N_copy = N.copy()  # Create a copy of N to avoid modifying the original
            for j in range(1): #N.shape[1]

                rE_true = df_ref['Release_rate'].iloc[i]
                print(rE_true)
                print(N.columns[j])
                
                N_copy[N.columns[j]] = ((N_copy[N.columns[j]] - rE_true) / rE_true) * 100  # Calculate the difference in %
            
            N_array_list.append(np.abs(N_copy))
            N_mean = np.mean(np.abs(N_copy), axis=0)
            N_array.append(N_mean.to_frame().T)

        diff_true_list_hypo.append(N_array_list)
        N_array = pd.concat(N_array, ignore_index=True)
        diff_true_rr_hypo.append(N_array)
    else:
        diff_true_list_hypo.append()
        N_array = pd.DataFrame()  # Create an empty DataFrame when loc is empty
        diff_true_rr_hypo.append(N_array)
    i += 1         
    
    
diff_calcref_rE_hypo = []
diff_calcref_list_hypo = []
i = 0
for loc in results:
    if loc:  # if loc not empty
        N_array = []
        N_array_list = []
        for N in loc:
            N_copy = N.copy()  # Create a copy of N to avoid modifying the original
            for j in range(1): #N.shape[1]
                #i=5 #test
                rE_true = df_ref['Release_rate'].iloc[i]  # Use true rr (column 1)
                rE_ref = df_ref['rE_ALL_area'].iloc[i] # each rE estimation method should be compared with itself (Weller estimations with reference value calculated with Weller)
                
                print(rE_true)
                print(rE_ref)
                N_copy[N.columns[j]] = ((N_copy[N.columns[j]] - rE_ref) / rE_ref) * 100  # Calculate the difference in %
              
            N_array_list.append(np.abs(N_copy))
            N_mean = np.mean(np.abs(N_copy), axis=0)
            N_array.append(N_mean.to_frame().T)

        diff_calcref_list_hypo.append(N_array_list)
        N_array = pd.concat(N_array, ignore_index=True)
        diff_calcref_rE_hypo.append(N_array)
    else:
        diff_calcref_list_hypo.append(N_array_list)
        N_array = pd.DataFrame()  # Create an empty DataFrame when loc is empty
        diff_calcref_rE_hypo.append(N_array)
    i += 1 
    
    
# Merge dfs with same number of transects N of different locations together 

# True rr
difftrue_list_merged_hypo = []
# Iterate over the index positions of the DataFrames (in this case, 0 to 12)
for i in range(min(len(lst) for lst in diff_true_list_hypo)):  # Assuming all sublists have the same length len(diff_true_list[0])
    # Collect the DataFrames at position i from each sublist
    dfs_to_merge = [sublist[i] for sublist in diff_true_list_hypo]
    
    # Merge the DataFrames at this index (axis=0 will concatenate them row-wise)
    merged_df = pd.concat(dfs_to_merge, axis=0)
    
    # Append the merged DataFrame to the result list
    difftrue_list_merged_hypo.append(merged_df)
    
# Calc rE
diffcalc_list_merged_hypo = []
# Iterate over the index positions of the DataFrames (in this case, 0 to 12)
for i in range(min(len(lst) for lst in diff_calcref_list_hypo)):  # Assuming all sublists have the same length len(diff_true_list[0])
    # Collect the DataFrames at position i from each sublist
    dfs_to_merge = [sublist[i] for sublist in diff_calcref_list_hypo]
    
    # Merge the DataFrames at this index (axis=0 will concatenate them row-wise)
    merged_df = pd.concat(dfs_to_merge, axis=0)
    
    # Append the merged DataFrame to the result list
    diffcalc_list_merged_hypo.append(merged_df)


 


#%%% Plots

save_fig = True
 
#%%%% Distributions

colors = ['yellowgreen','deepskyblue', 'lightskyblue', 'deeppink', 'pink','blue']
x_distr = [3, 2.6, 2.8, 3.4, 3.2, 2.7]
op = 0.08 # offset for plot
x_distr = [np.log(rr), np.log(rr)-(3*op), np.log(rr)-(1*op), np.log(rr)+(2*op), np.log(rr)+(1*op), np.log(rr)-(2*op)]

fig,ax1 = plt.subplots(figsize=(16,12))

x_linreg = np.linspace(np.log(rr)-1.5,np.log(rr)+1.5,11)

for i in range(df_distr.shape[1]):
    
    plt.scatter([x_distr[i]] * len(df_distr), df_distr.iloc[:,i],color=colors[i], alpha=0.7, label=f'{df_ref.iloc[i,0]}')

plt.plot(x_linreg, (x_linreg*slope_area+yintercept_area), color='red',linewidth=2, label = 'Area eq.')

plt.xlabel(r'ln(Emission Rate $\left[ \frac{\mathrm{Lmin^{-1}}}{1\ \mathrm{Lmin^{-1}}} \right]$)', fontsize=24)
plt.ylabel(r'ln($\mathrm{\left[CH_4\right]_{area}}\ \left[ \frac{\mathrm{ppm*m}}{1\ \mathrm{ppm*m}} \right]$)', fontsize=24)
# plt.title(r'Sample distributions with different offsets for $r_\mathrm{E,true}$ = ' f'{rr} ' r'$\mathrm{Lmin^{-1}}$ ', fontsize=24)
plt.legend(fontsize=20)
ax1.tick_params(axis='x', labelsize=24)
ax1.tick_params(axis='y', labelsize=24)

# Set x-ticks to display only np.log(rr)
ax1.set_xticks([np.log(rr)])
ax1.set_xticklabels([f'{np.log(rr):.2f}'])  # Format as needed (e.g., 2 decimal places)


if save_fig:
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/Distributions_lnArea_rr{rr}.png',bbox_inches='tight')
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/Distributions_lnArea_rr{rr}.pdf',bbox_inches='tight')
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/Distributions_lnArea_rr{rr}.svg',bbox_inches='tight')



#%%%% MonteCarlo Plots

colors = ['yellowgreen','deepskyblue', 'lightskyblue', 'deeppink', 'pink','blue']


# True RR ---------------------------------------------------------------------

fig,ax = plt.subplots(1,1,figsize=(16,12))

i=0
for loc in diff_true_rr_hypo:

    ax.plot(loc.index+1,loc.iloc[:,0],color=colors[i],label=f'{df_ref.iloc[i,0]}')
    ax.scatter(loc.index+1,loc.iloc[:,0],color=colors[i])
    i+=1
    
ax.set_xlabel('Number of transects',fontsize=24)
ax.set_ylabel(r'$\Delta$ %',fontsize=24)
# plt.title(r'MonteCarlo plot - comparison to true release rate $r_\mathrm{E,true}$',fontsize=24)
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)
plt.legend(fontsize=20)
ax1.tick_params(axis='x', labelsize=24)
ax1.tick_params(axis='y', labelsize=24)

if save_fig:
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/MonteCarlo_hypodistr_truerr_rr{rr}.png',bbox_inches='tight')
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/MonteCarlo_hypodistr_truerr_rr{rr}.pdf',bbox_inches='tight')
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/MonteCarlo_hypodistr_truerr_rr{rr}.svg',bbox_inches='tight')


# True RR - Zoom ---------------------------------------------------------------------

# fig,ax = plt.subplots(1,1,figsize=(16,12))

# i=0
# for loc in diff_true_rr_hypo:

#     ax.plot(loc.index+1,loc.iloc[:,0],color=colors[i],label=f'{df_ref.iloc[i,0]}')
#     ax.scatter(loc.index+1,loc.iloc[:,0],color=colors[i])
#     i+=1
    
# ax.set_ylim(-10,120)
# ax.set_xlabel('Number of transects',fontsize=16)
# ax.set_ylabel(r'$\Delta$ %',fontsize=16)
# plt.title('MonteCarlo plot - comparison to true release rate rR - Zoom',fontsize=18)
# ax.tick_params(axis='x', labelsize=16)
# ax.tick_params(axis='y', labelsize=16)
# plt.legend(fontsize=16)
# ax1.tick_params(axis='x', labelsize=16)
# ax1.tick_params(axis='y', labelsize=16)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

# First subplot with y-axis limited to 290-302
for i, loc in enumerate(diff_true_rr_hypo):
    ax1.plot(loc.index + 1, loc.iloc[:, 0], color=colors[i], label=f'{df_ref.iloc[i, 0]}')
    ax1.scatter(loc.index + 1, loc.iloc[:, 0], color=colors[i])

ax1.set_ylim(0, 490) # rr 3
# ax1.set_ylim(298, 301) # rr 50
ax1.set_ylabel(r'$\Delta$ %', fontsize=24)
# ax1.set_title(r'MonteCarlo plot - comparison to true release rate $r_\mathrm{E,true}$ - Zoom (290-302)', fontsize=24)
ax1.tick_params(axis='y', labelsize=24)

# Second subplot with y-axis limited to 91-102
for i, loc in enumerate(diff_true_rr_hypo):
    ax2.plot(loc.index + 1, loc.iloc[:, 0], color=colors[i])
    ax2.scatter(loc.index + 1, loc.iloc[:, 0], color=colors[i])

ax2.set_ylim(91, 100) # rr 3
# ax2.set_ylim(91, 100) # rr 50
ax2.set_xlabel('Number of transects', fontsize=24)
ax2.set_ylabel(r'$\Delta$ %', fontsize=24)
# ax2.set_title(r'MonteCarlo plot - comparison to true release rate $r_\mathrm{E,true}$ - Zoom (91-102)', fontsize=24)
ax2.tick_params(axis='x', labelsize=24)
ax2.tick_params(axis='y', labelsize=24)

# Legend for the first subplot only
ax1.legend(fontsize=20, loc='upper right')

plt.tight_layout()
plt.show()


if save_fig:
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/MonteCarlo_hypodistr_truerr_zoom_rr{rr}.png',bbox_inches='tight')
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/MonteCarlo_hypodistr_truerr_zoom_rr{rr}.pdf',bbox_inches='tight')
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/MonteCarlo_hypodistr_truerr_zoom_rr{rr}.svg',bbox_inches='tight')


# Calc rE ---------------------------------------------------------------------

fig,ax = plt.subplots(1,1,figsize=(16,12))

i=0
for loc in diff_calcref_rE_hypo:
    ax.plot(loc.index+1,loc.iloc[:,0],color=colors[i],label=f'{df_ref.iloc[i,0]}')
    ax.scatter(loc.index+1,loc.iloc[:,0],color=colors[i])
    i+=1
    
ax.set_xlabel('Number of transects',fontsize=24)
ax.set_ylabel(r'$\Delta$ %',fontsize=24)
# plt.title('MonteCarlo plot - comparison to mean calculated emission rate $r_\mathrm{E,calc.}$',fontsize=24)
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)
plt.legend(fontsize=20)
ax1.tick_params(axis='x', labelsize=24)
ax1.tick_params(axis='y', labelsize=24)

if save_fig:
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/MonteCarlo_hypodistr_calcrE_rr{rr}.png',bbox_inches='tight')
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/MonteCarlo_hypodistr_calcrE_rr{rr}.pdf',bbox_inches='tight')
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/MonteCarlo_hypodistr_calcrE_rr{rr}.svg',bbox_inches='tight')



#%%%% Visualize Calc. Steps


'''
 Visualize the different steps of the percentage deviation calculation to understand the outcome better.
1. Plot distribution of all ln(Area) which are randomly selected during the MC Analysis
    e.g. for N=3 -> 3*2000 ln(Area) values
2. Distribution of means over N ln(Area) values
    -> 2000 means per N, each taken over N ln(Area) values
3. Distribution of emission rates calclated on basis of the means
    -> 2000 rE
4.1 Distribution of percentage deviations of the emission rates (in both positive and negative direction)
    -> 2000 delta %
4.2 Distribution of absolute percentage deviations of the emission rates (only in positive direction)
    -> 2000 delta %
     
     
    '''


loc=5
save_fig = True
print(loc)



###############################################################################
# Figure 1&2 combined --------------------------------------------------------------------
###############################################################################

# loc=1
loc_list = results_hypo[loc]

#-----------------
# 1. Actually use tracked sampled ln(Area):
# Initialize a list to hold y_values for different DataFrame sizes
# y_values = [[] for _ in range(len(entries_hypo[loc]))]  # Create a list for each size from 1 to 19
# # Loop through each DataFrame in the list
# for df in entries_hypo[loc]:
#     num_rows = df.shape[0]  # Get the number of rows in the DataFrame
#     # if 1 <= num_rows <= len(df):  # Check if it's between 1 and 19
#         # Append the 'ln(Area)' values to the corresponding list
#     y_values[num_rows - 1].extend(df['ln(Area)'].values)  # Extend to add all values
    
# 2. Or use just the distribution -> the same
y_values = df_distr.iloc[:,loc]


#-----------------
x_values_mean = []
y_values_mean = []

# Loop through each DataFrame in the list
for i, df in enumerate(loc_list):
    
    # Get all values in the 'ln(Area)' column for the current DataFrame
    ln_area_values = df['ln(Area)'].values
    
    # Append the DataFrame index 'i' for each 'ln(Area)' value
    x_values_mean.extend([i+1] * len(ln_area_values))
    
    # Append the corresponding 'ln(Area)' values
    y_values_mean.extend(ln_area_values)
        
        
#-----------------
fig,ax = plt.subplots(figsize=(14,8))


plt.scatter([1] * len(y_values), y_values, color='silver', alpha=0.7, label=r'individual ln($\mathrm{\left[CH_4\right]_{area}}$) values')
for i in range(2, len(loc_list) + 1):
    plt.scatter([i] * len(y_values), y_values, color='silver', alpha=0.7)

plt.scatter(x_values_mean, y_values_mean, color='black', alpha=0.7, label='mean of sample with size N')


# Customize the plot
plt.xticks(range(5, len(loc_list) + 1, 5))  # Set x-axis ticks at increments of 5 
plt.xlabel('Number of transects N',fontsize=24)
plt.ylabel(r'ln($\mathrm{\left[CH_4\right]_{area}}\ \left[ \frac{\mathrm{ppm*m}}{1\ \mathrm{ppm*m}} \right]$)',fontsize=24)
label= df_ref.loc[loc,'Type_distr']
# plt.title(f'{label}: Distribution of ln(Area) sampled and mean of ln(Area)-sample in MC Analysis')
plt.legend(loc='lower right',fontsize=20)
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)

# Show the plot
plt.show()
if save_fig:
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/Distribution_MClnArea_{df_ref.iloc[loc,0]}.png',bbox_inches='tight')
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/Distribution_MClnArea_{df_ref.iloc[loc,0]}.pdf',bbox_inches='tight')
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/Distribution_MClnArea_{df_ref.iloc[loc,0]}.svg',bbox_inches='tight')




###############################################################################
# Figure 3 --------------------------------------------------------------------
###############################################################################

# loc = 0

x_values = []
y_values = []
x_values_mean = []
y_values_mean = []
loc_list = results_hypo[loc]
# Loop through each DataFrame in the list
for i, df in enumerate(loc_list):
    if 'rE_ALL_area' in df.columns:
        # Get all values in the 'ln(Area)' column for the current DataFrame
        ln_rr_values = df['rE_ALL_area'].values
        
        # Append the DataFrame index 'i' for each 'ln(Area)' value
        x_values.extend([i+1] * len(ln_rr_values))
        
        # Append the corresponding 'ln(Area)' values
        y_values.extend(ln_rr_values)
        y_values_mean.append(np.mean(ln_rr_values))
        x_values_mean  = np.arange(1,len(y_values_mean)+1)


fig,ax = plt.subplots(figsize=(14,8))
# Scatter plot where x-values are the DataFrame indices and y-values are the ln(Area) values
plt.scatter(x_values, y_values, color='gray', alpha=0.7, label=r'individual emission rate estimate $r_\mathrm{E,i}$')
plt.plot(x_values_mean, y_values_mean, color='black', alpha=0.7, label=r'mean estimated $r_\mathrm{E,mean}$')
plt.scatter(x_values_mean, y_values_mean, color='black', alpha=0.7)
plt.plot(x_values_mean, [df_ref.loc[loc,'Release_rate']] * len(x_values_mean), color='red', alpha=0.7, linewidth=2, label=r'true $r_\mathrm{E,true}$')

# Customize the plot
plt.xticks(range(5, len(loc_list) + 1, 5))  # Set x-axis ticks at increments of 5 
plt.xlabel('Number of transects',fontsize=24)
# plt.ylabel('Emission rate estimate',fontsize=24)
plt.ylabel(r'Emission rate estimate $\mathrm{\left[Lmin^{-1}\right]}$',fontsize=24) # with unit
label= df_ref.loc[loc,'Type_distr']
# plt.title(f'{label}: Distribution of emission rate estimates and its mean over all 2000 MC repetitions')
plt.legend(fontsize=20)
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)

# Show the plot
plt.show()
if save_fig:
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/Distribution_MCtrueRR_{df_ref.iloc[loc,0]}.png',bbox_inches='tight')
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/Distribution_MCtrueRR_{df_ref.iloc[loc,0]}.pdf',bbox_inches='tight')
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/Distribution_MCtrueRR_{df_ref.iloc[loc,0]}.svg',bbox_inches='tight')

    # plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/Distribution_MCtrueRR_{df_ref.iloc[loc,0]}_withunit.png',bbox_inches='tight')



###############################################################################
# Figure 4.1 --------------------------------------------------------------------
###############################################################################

# Pecentage deviations in positive and negative direction

# loc = 0

x_values = []
y_values = []
x_values_mean = []
y_values_mean = []
loc_list = results_hypo[loc]
# Loop through each DataFrame in the list
for i, df in enumerate(loc_list):
    if 'rE_ALL_area' in df.columns:
        # Get all values in the 'ln(Area)' column for the current DataFrame
        ln_rr_values = df['rE_ALL_area'].values
        perc_dev = (ln_rr_values-df_ref.loc[loc,'Release_rate'])/df_ref.loc[loc,'Release_rate'] * 100
        
        # Append the DataFrame index 'i' for each 'ln(Area)' value
        x_values.extend([i+1] * len(perc_dev))
        
        # Append the corresponding 'ln(Area)' values
        y_values.extend(perc_dev)
        y_values_mean.append(np.mean(perc_dev))
        x_values_mean  = np.arange(1,len(y_values_mean)+1)


fig,ax = plt.subplots(figsize=(14,8))
# Scatter plot where x-values are the DataFrame indices and y-values are the ln(Area) values
plt.scatter(x_values, y_values, color='gray', alpha=0.7, label=r'individual percentage deviation')
plt.plot(x_values_mean, y_values_mean, color='black', alpha=0.7, label=r'$\Delta$ % of mean estimated $r_\mathrm{E,mean}$')
plt.scatter(x_values_mean, y_values_mean, color='black', alpha=0.7)
plt.plot(x_values_mean, [0] * len(x_values_mean), color='red', alpha=0.7, linewidth=2, label=r'$\Delta$ % of true $r_\mathrm{E,true}$')

# Customize the plot
plt.xticks(range(5, len(loc_list) + 1, 5))  # Set x-axis ticks at increments of 5 
plt.xlabel('Number of transects',fontsize=24)
plt.ylabel(r'Percentage deviation $\Delta$ %',fontsize=24)
label= df_ref.loc[loc,'Type_distr']
# plt.title(f'{label}: Distribution of percentage deviations and its mean over all 2000 MC repetitions')
plt.legend(fontsize=20)
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)

# Show the plot
plt.show()
if save_fig:
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/Distribution_MCpercdev_{df_ref.iloc[loc,0]}.png',bbox_inches='tight')
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/Distribution_MCpercdev_{df_ref.iloc[loc,0]}.pdf',bbox_inches='tight')
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/Distribution_MCpercdev_{df_ref.iloc[loc,0]}.svg',bbox_inches='tight')



###############################################################################
# Figure 4.2 --------------------------------------------------------------------
###############################################################################

# Pecentage deviations only in positive direction (take absolute percentage deviation)

# loc = 0

x_values = []
y_values = []
x_values_mean = []
y_values_mean = []
loc_list = results_hypo[loc]
# Loop through each DataFrame in the list
for i, df in enumerate(loc_list):
    if 'rE_ALL_area' in df.columns:
        # Get all values in the 'ln(Area)' column for the current DataFrame
        ln_rr_values = df['rE_ALL_area'].values
        perc_dev = (ln_rr_values-df_ref.loc[loc,'Release_rate'])/df_ref.loc[loc,'Release_rate'] * 100
        
        # Append the DataFrame index 'i' for each 'ln(Area)' value
        x_values.extend([i+1] * len(perc_dev))
        
        # Append the corresponding 'ln(Area)' values
        y_values.extend(np.abs(perc_dev))
        y_values_mean.append(np.mean(np.abs(perc_dev)))
        x_values_mean  = np.arange(1,len(y_values_mean)+1)


fig,ax = plt.subplots(figsize=(14,8))
# Scatter plot where x-values are the DataFrame indices and y-values are the ln(Area) values
plt.scatter(x_values, y_values, color='gray', alpha=0.7, label=r'individual abs. percentage deviation')
plt.plot(x_values_mean, y_values_mean, color='black', alpha=0.7, label=r'|$\Delta$ %| of mean estimated $r_\mathrm{E,mean}$')
plt.scatter(x_values_mean, y_values_mean, color='black', alpha=0.7)
plt.plot(x_values_mean, [0] * len(x_values_mean), color='red', alpha=0.7, linewidth=2, label=r'|$\Delta$ %| of true $r_\mathrm{E,true}$')

# Customize the plot
plt.xticks(range(5, len(loc_list) + 1, 5))  # Set x-axis ticks at increments of 5 
plt.xlabel('Number of transects',fontsize=24)
plt.ylabel(r'Absolute Percentage deviation |$\Delta$ %|',fontsize=24)
label= df_ref.loc[loc,'Type_distr']
# plt.title(f'{label}: Distribution of absolute percentage deviations and its mean over all 2000 MC repetitions')
plt.legend(fontsize=20)
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)

# Show the plot
plt.show()
if save_fig:
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/Distribution_MCpercdevABS_{df_ref.iloc[loc,0]}.png',bbox_inches='tight')
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/Distribution_MCpercdevABS_{df_ref.iloc[loc,0]}.pdf',bbox_inches='tight')
    plt.savefig(path_savefig+f'Tests/Hypothetical_Distributions/Distribution_MCpercdevABS_{df_ref.iloc[loc,0]}.svg',bbox_inches='tight')











    

