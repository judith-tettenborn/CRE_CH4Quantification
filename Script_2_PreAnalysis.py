# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:35:33 2024

@author: Judith Tettenborn (j.a.f.tettenborn@uu.nl) & Daan Stroeken


# Pre ANALYSIS FOR PEAKS THAT PASSED QC
# =============================================================================

- calculates Spatial Peak Area and main features important for later analysis
- data are saved in several ways:
    1. Files containing all detected peaks per CRE, with area values calculated. 
    Each instrument has separate columns for Max and Area values.
       total_peaks_T1b -> "T_TOTAL_PEAKS_1b.csv"           - for each experiment
       total_peaks_all -> "RU2T2L4L2_TOTAL_PEAKS.csv"      - all CRE combined
       
    2. Files with all instruments merged: measurements are consolidated into two 
    columns: 'Max' and 'Area'.
       df_R_comb       -> "R_comb_final.csv"               - for each experiment
       df_RU2T2L4L2    -> "RU2T2L4L2_TOTAL_PEAKS_comb.csv" - all CRE combined
       
    3. Files with count of valid peaks per release rate
       df_R_rr_count   -> "R_comb_count.csv"
       df_RU2T2L4L2_rr_count -> "RU2T2L4L2_count.csv"
- plots of each quality-checked peak can be created (optional, up to several 
  hundreds per CRE)



"""

#%% Load Packages & Data

# Modify Python Path Programmatically -> To include the directory containing the src folder
from pathlib import Path
import sys

# HARDCODED ####################################################################################

# path_base = Path('C:/Users/.../CRE_CH4Quantification/') # insert the the project path here

save_to_csv = True
indiv_peak_plots = True # plot individual peaks for QC? ATTENTION: when True this will create several hundreds of figures PER CRE


################################################################################################


path_base = Path('C:/Users/Judit/Documents/UNI/Utrecht/Hiwi/CRE_CH4Quantification/')
sys.path.append(str(path_base / 'src'))

# In Python, the "Python path" refers to the list of directories where Python looks for modules
# and packages when you try to import them in your scripts or interactive sessions. This path 
# is stored in the sys.path list. When you execute an import statement in Python, it searches 
# for the module or package you're trying to import in the directories listed in sys.path. 
# If the directory containing the module or package is not included in sys.path, Python won't 
# be able to find and import it.

import pandas as pd
import numpy as np


# CRE code functions:
from preprocessing.read_in_data import *
from peak_analysis.find_analyze_peaks import *
from plotting.general_plots import *
from helper_functions.utils import *


from helper_functions.dataset_dicts import (
    U1_vars_G43,
    U1_vars_G23,
    U2_vars_aeris,
    U2_vars_G23,
    R_vars_G43,
    R_vars_G23,
    R_vars_aeris,
    R_vars_miro,
    R_vars_aerodyne,
    T_vars_1b_LGR,
    T_vars_1c_G24,
    T_vars_2c_G24,
    L1_vars_d2_LGR,
    L1_vars_d2_G23,
    L1_vars_d3_Licor,
    L1_vars_d3_G23,
    L1_vars_d4_Licor,
    L1_vars_d4_G23,
    L1_vars_d5_G23,
    L2_vars_d1_Licor,
    L2_vars_d2_Licor
    )


# DEFINE PATHS

path_fig        = path_base / 'results' / 'Figures/'
path_procdata   = path_base / 'data' / 'processed/'

if not (path_base / 'data' / 'final').is_dir(): # output: processed data
    (path_base / 'data' / 'final').mkdir(parents=True)
path_finaldata = path_base / 'data' / 'final'


# -----------------------------------------------------------------------------
# Read in datafiles containing identified and quality-checked methane plumes
# -----------------------------------------------------------------------------
# Each row corresponds to a peak identified with the scipy function find_peaks, the index gives the datetime 
# of the maximum pint of the peak. In the quality check (QC) the peak were manually checked for validity,
# based on vehicle speed (e.g. car standing), instrument failures, distance to source, availability of GPS information... 
# The columns with the respective 'CH4 instrument name' (e.g. G2301 or Aeris), 'GPS', 'Loc', 'QC' are indicative
# if the peak is valid (if it is not valid one or more of those columns contain a 0 for this specific peak). 
# In the following script the peaks which are valid are further analysed (e.g. calculation of integrated area).

corrected_peaks_U1      = pd.read_excel(path_procdata / "U1_CH4peaks_QC.xlsx",sheet_name='G4302',index_col='Datetime')
corrected_peaks_U2      = pd.read_excel(path_procdata / "U2_CH4peaks_QC.xlsx",sheet_name='G2301',index_col='Datetime')
corrected_peaks_R_mUU   = pd.read_excel(path_procdata / "R_CH4peaks_morningUU_QC.xlsx",sheet_name='G4302',index_col='Datetime')
corrected_peaks_R_mTNO  = pd.read_excel(path_procdata / "R_CH4peaks_morningTNO_QC.xlsx",sheet_name='miro',index_col='Datetime')
corrected_peaks_R_aTNO  = pd.read_excel(path_procdata / "R_CH4peaks_afternoon_QC.xlsx",sheet_name='miro',index_col='Datetime')
corrected_peaks_L1d2    = pd.read_excel(path_procdata / "L1_CH4peaks_day2_QC.xlsx",sheet_name='LGR',index_col='Datetime')
corrected_peaks_L1d3    = pd.read_excel(path_procdata / "L1_CH4peaks_day3_QC.xlsx",sheet_name='G2301',index_col='Datetime')
corrected_peaks_L1d4    = pd.read_excel(path_procdata / "L1_CH4peaks_day4_QC.xlsx",sheet_name='G2301',index_col='Datetime')
corrected_peaks_L1d5    = pd.read_excel(path_procdata / "L1_CH4peaks_day5_QC.xlsx",sheet_name='G2301',index_col='Datetime')
corrected_peaks_L2d1    = pd.read_excel(path_procdata / "L2_CH4peaks_day1_QC.xlsx",sheet_name='Licor',index_col='Datetime')
corrected_peaks_L2d2    = pd.read_excel(path_procdata / "L2_CH4peaks_day2_QC.xlsx",sheet_name='Licor',index_col='Datetime')
corrected_peaks_T1b     = pd.read_excel(path_procdata / "T_CH4peaks_1b_QC.xlsx",sheet_name='Day1-Bike',index_col='Datetime')
corrected_peaks_T1c     = pd.read_excel(path_procdata / "T_CH4peaks_1c_QC.xlsx",sheet_name='Day1-Car',index_col='Datetime')
corrected_peaks_T2c     = pd.read_excel(path_procdata / "T_CH4peaks_2c_QC.xlsx",sheet_name='Day2-Car',index_col='Datetime') 

# Drop columns which are not necessary for the analysis
# 'Sure?' and 'corrected' contain comments on the quality check (e.g. why a peak was regarded, if a peak was kept, but there is uncertainty about its validity)
corrected_peaks_U1      = corrected_peaks_U1.drop(columns={'Peakstart','Peakend','corrected','Sure?','Overlap?','Car_passed? (Both)','Car_passed? (G4)'})
corrected_peaks_U2      = corrected_peaks_U2.drop(columns={'Peakstart','Peakend','corrected','Sure?'})
corrected_peaks_R_mUU   = corrected_peaks_R_mUU.drop(columns={'Peakstart','Peakend','corrected','Peak_old', 'Sure?'})
corrected_peaks_R_aTNO  = corrected_peaks_R_aTNO.drop(columns={'Peakstart','Peakend','Peakend_peakfinder','corrected','Sure?','Distance','CH4_ele_miro_TNO'})
corrected_peaks_R_mTNO  = corrected_peaks_R_mTNO.drop(columns={'Peakstart','Peakend','corrected','Distance','CH4_ele_miro_TNO'})
corrected_peaks_L1d2    = corrected_peaks_L1d2.drop(columns={'Peakstart','Peakend','Peakend_peakfinder','corrected','Sure?','CH4_bg05','CH4_bg10'})
corrected_peaks_L1d3    = corrected_peaks_L1d3.drop(columns={'Peakstart','Peakend','corrected','Sure?'})
corrected_peaks_L1d4    = corrected_peaks_L1d4.drop(columns={'Peakstart','Peakend','corrected','Sure?'})
corrected_peaks_L1d5    = corrected_peaks_L1d5.drop(columns={'Peakstart','Peakend','Peakend_peakfinder','corrected','Sure?','CH4_bg05','CH4_bg10','Run','species','Distance'})
corrected_peaks_L2d1    = corrected_peaks_L2d1.drop(columns={'Peakstart','Peakend','corrected','Sure?'})
corrected_peaks_L2d2    = corrected_peaks_L2d2.drop(columns={'Peakstart','Peakend','corrected','Sure?'})
corrected_peaks_T1b     = corrected_peaks_T1b.drop(columns={'Peakstart','Peakend','corrected','Sure?'})
corrected_peaks_T1c     = corrected_peaks_T1c.drop(columns={'Peakstart','Peakend','corrected','Sure?'})
corrected_peaks_T2c     = corrected_peaks_T2c.drop(columns={'Peakstart','Peakend','corrected','Sure?'})




#%% Treat Data: Analyse Peaks (Calculate Area)

from peak_analysis.find_analyze_peaks import *
# 
# analyse_peak is a function in module find_analyze_peaks
# based on the validity of the peaks it  
# 1. assigns 0 or 1 in column QC (0- not valid, 1- valid)
# 2. calculates the integrated area
# 3. assigns the release rate to each peak based on timestamp and location

analyse_peak(corrected_peaks_R_mUU,R_vars_G43,R_vars_aeris, R_vars_G23)
analyse_peak(corrected_peaks_R_mTNO,R_vars_miro,R_vars_aerodyne)
analyse_peak(corrected_peaks_R_aTNO,R_vars_miro,R_vars_aerodyne,R_vars_aeris, R_vars_G43)
 
analyse_peak(corrected_peaks_U1,U1_vars_G43, U1_vars_G23)
analyse_peak_U2(corrected_peaks_U2,U2_vars_G23,U2_vars_aeris)

analyse_peak(corrected_peaks_L1d2,L1_vars_d2_LGR, L1_vars_d2_G23)
analyse_peak(corrected_peaks_L1d3,L1_vars_d3_Licor, L1_vars_d3_G23)
analyse_peak(corrected_peaks_L1d4,L1_vars_d4_Licor, L1_vars_d4_G23)
analyse_peak(corrected_peaks_L1d5,L1_vars_d5_G23)

analyse_peak(corrected_peaks_L2d1,L2_vars_d1_Licor)
analyse_peak(corrected_peaks_L2d2,L2_vars_d2_Licor)

analyse_peak(corrected_peaks_T1b,T_vars_1b_LGR)
analyse_peak(corrected_peaks_T1c,T_vars_1c_G24)
analyse_peak(corrected_peaks_T2c,T_vars_2c_G24)



# The natural logarithm of the Maximum and Area for each instrument and the Release rate is added to the dataframe
apply_log_transform(corrected_peaks_R_mUU,  [R_vars_G43, R_vars_G23, R_vars_aeris])
apply_log_transform(corrected_peaks_R_mTNO, [R_vars_miro, R_vars_aerodyne])
apply_log_transform(corrected_peaks_R_aTNO, [R_vars_miro, R_vars_aerodyne, R_vars_aeris, R_vars_G43])
apply_log_transform(corrected_peaks_U1,     [U1_vars_G43, U1_vars_G23])
apply_log_transform(corrected_peaks_U2,     [U2_vars_G23, U2_vars_aeris])
apply_log_transform(corrected_peaks_L1d2,   [L1_vars_d2_LGR, L1_vars_d2_G23])
apply_log_transform(corrected_peaks_L1d3,   [L1_vars_d3_Licor, L1_vars_d3_G23])
apply_log_transform(corrected_peaks_L1d4,   [L1_vars_d4_Licor, L1_vars_d4_G23])
apply_log_transform(corrected_peaks_L1d5,   [L1_vars_d5_G23])
apply_log_transform(corrected_peaks_L2d1,   [L2_vars_d1_Licor])
apply_log_transform(corrected_peaks_L2d2,   [L2_vars_d2_Licor])
apply_log_transform(corrected_peaks_T1b,    [T_vars_1b_LGR])
apply_log_transform(corrected_peaks_T1c,    [T_vars_1c_G24])
apply_log_transform(corrected_peaks_T2c,    [T_vars_2c_G24])


# create column 'City' filled with the associated city name (+ day of experiment if applicable)
corrected_peaks_R_mUU['City']   = 'Rotterdam'
corrected_peaks_R_mTNO['City']  = 'Rotterdam'
corrected_peaks_R_aTNO['City']  = 'Rotterdam'
corrected_peaks_U1['City']      = 'Utrecht I'
corrected_peaks_U2['City']      = 'Utrecht II'
corrected_peaks_L1d2['City']    = 'London I-Day2'
corrected_peaks_L1d3['City']    = 'London I-Day3'
corrected_peaks_L1d4['City']    = 'London I-Day4'
corrected_peaks_L1d5['City']    = 'London I-Day5'
corrected_peaks_T1b['City']     = 'Toronto-1b'
corrected_peaks_T1c['City']     = 'Toronto-1c'
corrected_peaks_T2c['City']     = 'Toronto-2c'
corrected_peaks_L2d1['City']    = 'London II-Day1'
corrected_peaks_L2d2['City']    = 'London II-Day2'

# CRE with different release locations alread have a column 'Loc', here add a
# column to those that do not have it yet for consistency
corrected_peaks_L1d2['Loc']     = 1
corrected_peaks_L1d3['Loc']     = 1
corrected_peaks_L1d4['Loc']     = 1
corrected_peaks_L1d5['Loc']     = 1
corrected_peaks_T1b['Loc']      = 1
corrected_peaks_T1c['Loc']      = 1
corrected_peaks_T2c['Loc']      = 1
corrected_peaks_L2d1['Loc']     = 1
corrected_peaks_L2d2['Loc']     = 1

#%% Save Total Peaks


# Dataframes are filtered for valid peaks (QC = 1) and saved as an csv file
def fct_save_to_csv(df, city, name, path, save_to_csv, Day=None):
    total_peaks = df[(df['QC'] == True) & (df['Release_rate'] != 0)].copy(deep=True)
    total_peaks = add_distance_to_df(total_peaks,city,Day=Day) # Add distance to source
    if save_to_csv:
        total_peaks.to_csv(path / f"{name}.csv")
    return total_peaks


# London I
corrected_peaks_L1d2 = corrected_peaks_L1d2[corrected_peaks_L1d2['Release_height'] == 0] # since we are only investigating ground releases, delete peaks from releases from higher altitudes
corrected_peaks_L1d3 = corrected_peaks_L1d3[corrected_peaks_L1d3['Release_height'] == 0]
corrected_peaks_L1d4 = corrected_peaks_L1d4[corrected_peaks_L1d4['Release_height'] == 0]
corrected_peaks_L1d5 = corrected_peaks_L1d5[corrected_peaks_L1d5['Release_height'] == 0]

total_peaks_L1d2 = fct_save_to_csv(corrected_peaks_L1d2,'London I',"L1_TOTAL_PEAKS_Day2", path_finaldata,save_to_csv)
total_peaks_L1d3 = fct_save_to_csv(corrected_peaks_L1d3,'London I',"L1_TOTAL_PEAKS_Day3", path_finaldata,save_to_csv)
total_peaks_L1d4 = fct_save_to_csv(corrected_peaks_L1d4,'London I',"L1_TOTAL_PEAKS_Day4", path_finaldata,save_to_csv)
total_peaks_L1d5 = fct_save_to_csv(corrected_peaks_L1d5,'London I',"L1_TOTAL_PEAKS_Day5", path_finaldata,save_to_csv)

# London II
total_peaks_L2d1 = fct_save_to_csv(corrected_peaks_L2d1,'London II',"L2_TOTAL_PEAKS_Day1", path_finaldata,save_to_csv)
total_peaks_L2d2 = fct_save_to_csv(corrected_peaks_L2d2,'London II',"L2_TOTAL_PEAKS_Day2", path_finaldata,save_to_csv)

# Toronto
total_peaks_T1b = fct_save_to_csv(corrected_peaks_T1b,'Toronto',"T_TOTAL_PEAKS_1b", path_finaldata,save_to_csv,Day=1)
total_peaks_T1c = fct_save_to_csv(corrected_peaks_T1c,'Toronto',"T_TOTAL_PEAKS_1c", path_finaldata,save_to_csv,Day=1)
total_peaks_T2c = fct_save_to_csv(corrected_peaks_T2c,'Toronto',"T_TOTAL_PEAKS_2c", path_finaldata,save_to_csv,Day=2)

# Utrecht I
corrected_peaks_U1['Loc'] = corrected_peaks_U1['Loc'].replace({10: 1, 20: 2})
total_peaks_U1 = fct_save_to_csv(corrected_peaks_U1,'Utrecht I',"U1_TOTAL_PEAKS", path_finaldata,save_to_csv)

# Utrecht II
total_peaks_U2 = fct_save_to_csv(corrected_peaks_U2,'Utrecht II',"U2_TOTAL_PEAKS", path_finaldata,save_to_csv)

# Rotterdam
peaks_UUm     = corrected_peaks_R_mUU[corrected_peaks_R_mUU['QC']]
peaks_TNOm    = corrected_peaks_R_mTNO[corrected_peaks_R_mTNO['QC']]
peaks_TNOa    = corrected_peaks_R_aTNO[corrected_peaks_R_aTNO['QC']]
cars          = ['UUAQ', 'TNO']

total_peaks_R        = pd.concat((peaks_UUm, peaks_TNOm, peaks_TNOa))
total_peaks_R['Loc'] = total_peaks_R['Loc'].replace({10: 1, 20: 2})
total_peaks_R = fct_save_to_csv(total_peaks_R,'Rotterdam',"R_TOTAL_PEAKS", path_finaldata,save_to_csv)

  
# RU2T2L4L2 ------

total_peaks_all     = pd.concat((total_peaks_R, total_peaks_U1, total_peaks_U2, total_peaks_T1b, total_peaks_T1c, total_peaks_T2c, total_peaks_L1d2, total_peaks_L1d3, total_peaks_L1d4, total_peaks_L1d5, total_peaks_L2d1, total_peaks_L2d2))

if save_to_csv:
    total_peaks_all.to_csv(path_finaldata / 'RU2T2L4L2_TOTAL_PEAKS.csv')

    

#%% P: Detailed Peak Plots


''' ===== Utrecht I ===== '''

# Define other necessary variables
coord_extent = [5.1633, 5.166, 52.0873, 52.0888]
release_loc1 = (5.1647191, 52.0874256) 
release_loc2 = (5.164405555555556, 52.0885444)
column_names = {'G2301': 'CH4_ele_G23', 'G4302': 'CH4_ele_G43'}


if not (path_fig / 'Utrecht_I_2022' / 'U1_Peakplots_QCpassed').is_dir():
    (path_fig / 'Utrecht_I_2022' / 'U1_Peakplots_QCpassed').mkdir(parents=True)
path_save =  path_fig / 'Utrecht_I_2022' / 'U1_Peakplots_QCpassed'


# First location on lane 1 (in excel file named location 3)
plot_indivpeaks_afterQC(total_peaks_U1, path_save, coord_extent_1, release_loc1, release_loc2, indiv_peak_plots, column_names_1, U1_vars_G23, U1_vars_G43)



''' ===== Utrecht II ===== '''

# Define other necessary variables
coord_extent_1  = [total_peaks_U2['Longitude'].min()-0.001,total_peaks_U2['Longitude'].max(), total_peaks_U2['Latitude'].min()-0.0005, total_peaks_U2['Latitude'].max()+0.0005] #r_loc: 43.782970N, (-)79.46952W 
coord_extent_1  = [5.1633, 5.166, 52.0873, 52.0888]
release_loc1    = (5.164652777777778, 52.0874472)
release_loc1_2  = (5.16506388888889, 52.0875333) 
release_loc2    = (5.164452777777778, 52.0885333) # 
column_names_1  = {'G2301': 'CH4_ele_G23','Aeris': 'CH4_ele_aeris'}


if not (path_fig / 'Utrecht_II_2024' / 'U2_Peakplots_QCpassed').is_dir():
    (path_fig / 'Utrecht_II_2024' / 'U2_Peakplots_QCpassed').mkdir(parents=True)
path_save =  path_fig / 'Utrecht_II_2024' / 'U2_Peakplots_QCpassed'


# First location on lane 1 (in excel file named location 3)
plot_indivpeaks_afterQC(total_peaks_U2[:'2024-06-11 11:22:00'], path_save, coord_extent_1, release_loc1, release_loc2, indiv_peak_plots, column_names_1, U2_vars_G23, U2_vars_aeris)

# Second location on lane 1 (in excel file named location 1)
plot_indivpeaks_afterQC(total_peaks_U2['2024-06-11 11:22:00':], path_save, coord_extent_1, release_loc1_2, release_loc2, indiv_peak_plots, column_names_1, U2_vars_G23, U2_vars_aeris)


''' ===== Rotterdam ===== '''


# Define other necessary variables
coord_extent = [4.51832, 4.52830, 51.91921, 51.92288]
release_loc1_R = (4.5237450, 51.9201216)
release_loc2_R = (4.5224917, 51.9203931) #51.9203931,4.5224917
release_loc3_R = (4.523775, 51.921028) # estimated from Daans plot (using google earth)
column_names_mUU = {'G4302': 'CH4_ele_G43','G2301': 'CH4_ele_G23', 'Aeris':'CH4_ele_aeris'}
column_names_mTNO = {'Miro': 'CH4_ele_miro', 'Aerodyne': 'CH4_ele_aero'}
column_names_aTNO = {'Miro': 'CH4_ele_miro', 'Aerodyne': 'CH4_ele_aero', 'G4302': 'CH4_ele_G43', 'Aeris':'CH4_ele_aeris'}
indiv_peak_plots = True  # or False based on your requirement


''' --- Morning UU --- '''

if not (path_fig / 'Rotterdam_2022' / 'R_Peakplots_QCpassed' / 'Morning_UU').is_dir():
    (path_fig / 'Rotterdam_2022' / 'R_Peakplots_QCpassed' / 'Morning_UU').mkdir(parents=True)
path_save =  path_fig / 'Rotterdam_2022' / 'R_Peakplots_QCpassed' / 'Morning_UU'

# Call the function with necessary arguments
plot_indivpeaks_afterQC(corrected_peaks_R_mUU, path_save, coord_extent, release_loc1_R, release_loc2_R, indiv_peak_plots, column_names_mUU, R_vars_G43, R_vars_G23,R_vars_aeris)
# Note: put main instrument first in args* (for Morning UU: G4302 -> R_vars_G43)

''' --- Morning TNO --- '''

if not (path_fig / 'Rotterdam_2022' / 'R_Peakplots_QCpassed' / 'Morning_TNO').is_dir():
    (path_fig / 'Rotterdam_2022' / 'R_Peakplots_QCpassed' / 'Morning_TNO').mkdir(parents=True)
path_save =  path_fig / 'Rotterdam_2022' / 'R_Peakplots_QCpassed' / 'Morning_TNO'

# Call the function with necessary arguments
plot_indivpeaks_afterQC(corrected_peaks_R_mTNO, path_save, coord_extent, release_loc1_R, release_loc2_R, indiv_peak_plots, column_names_mTNO, R_vars_miro, R_vars_aerodyne)


''' --- Afternoon TNO --- '''

if not (path_fig / 'Rotterdam_2022' / 'R_Peakplots_QCpassed' / 'Afternoon_TNO').is_dir():
    (path_fig / 'Rotterdam_2022' / 'R_Peakplots_QCpassed' / 'Afternoon_TNO').mkdir(parents=True)
path_save =  path_fig / 'Rotterdam_2022' / 'R_Peakplots_QCpassed' / 'Afternoon_TNO'

# Call the function with necessary arguments
plot_indivpeaks_afterQC(corrected_peaks_R_aTNO, path_save, coord_extent, release_loc1_R, release_loc3_R, indiv_peak_plots, column_names_aTNO, R_vars_miro, R_vars_aerodyne,R_vars_G43,R_vars_aeris)




''' ===== Toronto ===== '''

# Define other necessary variables
passed_peaks = corrected_peaks_T1b.loc[(corrected_peaks_T1b['QC'] == True)]
coord_extent_1 = [passed_peaks['Longitude'].min()-0.003,passed_peaks['Longitude'].max()+0.003, passed_peaks['Latitude'].min()-0.003, passed_peaks['Latitude'].max()+0.003]
passed_peaks = corrected_peaks_T2c.loc[(corrected_peaks_T2c['QC'] == True)]
coord_extent_2 = [passed_peaks['Longitude'].min()-0.002,passed_peaks['Longitude'].max()+0.002, passed_peaks['Latitude'].min()-0.002, passed_peaks['Latitude'].max()+0.002]
column_names_1 = {'LGR': 'CH4_ele_LGR'}
column_names_2 = {'G2401': 'CH4_ele_G24'}
release_loc1_T = (-79.325254, 43.655007)
release_loc2_T = (-79.46952, 43.782970)


''' --- Day 1 - Bike --- '''

if not (path_fig / 'Toronto_2021' / 'T_Peakplots_QCpassed' / 'Day1_Bike').is_dir():
    (path_fig / 'Toronto_2021' / 'T_Peakplots_QCpassed' / 'Day1_Bike').mkdir(parents=True)
path_save =  path_fig / 'Toronto_2021' / 'T_Peakplots_QCpassed' / 'Day1_Bike'

# Call the plot function to plot individual peaks
plot_indivpeaks_afterQC(corrected_peaks_T1b, path_save, coord_extent_1, release_loc1_T, release_loc2==None, indiv_peak_plots, column_names_1, T_vars_1b_LGR)

''' --- Day 1 - Car --- '''

if not (path_fig / 'Toronto_2021' / 'T_Peakplots_QCpassed' / 'Day1_Car').is_dir():
    (path_fig / 'Toronto_2021' / 'T_Peakplots_QCpassed' / 'Day1_Car').mkdir(parents=True)
path_save =  path_fig / 'Toronto_2021' / 'T_Peakplots_QCpassed' / 'Day1_Car'

# Call the plot function to plot individual peaks
plot_indivpeaks_afterQC(corrected_peaks_T1c, path_save, coord_extent_1, release_loc1_T, release_loc2==None, indiv_peak_plots, column_names_2, T_vars_1c_G24)

''' --- Day 2 - Car --- '''

if not (path_fig / 'Toronto_2021' / 'T_Peakplots_QCpassed' / 'Day2_Car').is_dir():
    (path_fig / 'Toronto_2021' / 'T_Peakplots_QCpassed' / 'Day2_Car').mkdir(parents=True)
path_save =  path_fig / 'Toronto_2021' / 'T_Peakplots_QCpassed' / 'Day2_Car'

# Call the plot function to plot individual peaks
plot_indivpeaks_afterQC(corrected_peaks_T2c, path_save, coord_extent_2, release_loc2_T, release_loc2==None, indiv_peak_plots, column_names_2, T_vars_2c_G24)



''' ===== London ===== '''

# Define other necessary variables
passed_peaks = corrected_peaks_L1d2.loc[(corrected_peaks_L1d2['QC'] == True)]
coord_extent_2 = [passed_peaks['Longitude'].min()-0.001,passed_peaks['Longitude'].max(), passed_peaks['Latitude'].min()-0.0005, passed_peaks['Latitude'].max()+0.0005] #r_loc: 43.782970N, (-)79.46952W 
passed_peaks = corrected_peaks_L1d3.loc[(corrected_peaks_L1d3['QC'] == True)]
coord_extent_3 = [passed_peaks['Longitude'].min()-0.001,passed_peaks['Longitude'].max(), passed_peaks['Latitude'].min()-0.0005, passed_peaks['Latitude'].max()+0.0005] #r_loc: 43.782970N, (-)79.46952W 
passed_peaks = corrected_peaks_L1d4.loc[(corrected_peaks_L1d4['QC'] == True)]
coord_extent_4 = [passed_peaks['Longitude'].min()-0.001,passed_peaks['Longitude'].max(), passed_peaks['Latitude'].min()-0.0005, passed_peaks['Latitude'].max()+0.0005] #r_loc: 43.782970N, (-)79.46952W 
passed_peaks = corrected_peaks_L1d5.loc[(corrected_peaks_L1d5['QC'] == True)]
coord_extent_5 = [passed_peaks['Longitude'].min()-0.003,passed_peaks['Longitude'].max()+0.003, passed_peaks['Latitude'].min()-0.003, passed_peaks['Latitude'].max()+0.003] #r_loc: 43.782970N, (-)79.46952W 
column_names_2 = {'LGR': 'CH4_ele_LGR', 'G2301': 'CH4_ele_G23'}
column_names_3 = {'Licor': 'CH4_ele_Licor', 'G2301': 'CH4_ele_G23'}
column_names_5 = {'G2301': 'CH4_ele_G23'}
release_loc_L = (-0.437888,52.233343)


''' --- Day 2 --- '''

if not (path_fig / 'London_I_2019' / 'L1_Peakplots_QCpassed' / 'Day2').is_dir():
    (path_fig / 'London_I_2019' / 'L1_Peakplots_QCpassed' / 'Day2').mkdir(parents=True)
path_save =  path_fig / 'London_I_2019' / 'L1_Peakplots_QCpassed' / 'Day2'

# Call the plot function to plot individual peaks
plot_indivpeaks_afterQC(corrected_peaks_L1d2, path_save, coord_extent_2, release_loc_L, release_loc2==None, indiv_peak_plots, column_names_2, L1_vars_d2_LGR, L1_vars_d2_G23)


''' --- Day 3 --- '''

if not (path_fig / 'London_I_2019' / 'L1_Peakplots_QCpassed' / 'Day3').is_dir():
    (path_fig / 'London_I_2019' / 'L1_Peakplots_QCpassed' / 'Day3').mkdir(parents=True)
path_save =  path_fig / 'London_I_2019' / 'L1_Peakplots_QCpassed' / 'Day3'

# Call the plot function to plot individual peaks
plot_indivpeaks_afterQC(corrected_peaks_L1d3, path_save, coord_extent_3, release_loc_L, release_loc2==None, indiv_peak_plots, column_names_3, L1_vars_d3_Licor, L1_vars_d3_G23)


''' --- Day 4 --- '''

if not (path_fig / 'London_I_2019' / 'L1_Peakplots_QCpassed' / 'Day4').is_dir():
    (path_fig / 'London_I_2019' / 'L1_Peakplots_QCpassed' / 'Day4').mkdir(parents=True)
path_save =  path_fig / 'London_I_2019' / 'L1_Peakplots_QCpassed' / 'Day4'

# Call the plot function to plot individual peaks
plot_indivpeaks_afterQC(corrected_peaks_L1d4, path_save, coord_extent_4, release_loc_L, release_loc2==None, indiv_peak_plots, column_names_3, L1_vars_d4_Licor, L1_vars_d4_G23)



''' --- Day 5 --- '''

if not (path_fig / 'London_I_2019' / 'L1_Peakplots_QCpassed' / 'Day5').is_dir():
    (path_fig / 'London_I_2019' / 'L1_Peakplots_QCpassed' / 'Day5').mkdir(parents=True)
path_save =  path_fig / 'London_I_2019' / 'L1_Peakplots_QCpassed' / 'Day5'

# Call the plot function to plot individual peaks
plot_indivpeaks_afterQC(corrected_peaks_L1d5, path_save, coord_extent_5, release_loc_L, release_loc2==None, indiv_peak_plots, column_names_5, L1_vars_d5_G23)




''' ===== London II ===== '''

# Define other necessary variables
passed_peaks = corrected_peaks_L2d1.loc[(corrected_peaks_L2d1['QC'] == True)]
coord_extent_1 = [passed_peaks['Longitude'].min()-0.001,passed_peaks['Longitude'].max()+0.0018, passed_peaks['Latitude'].min()-0.0005, passed_peaks['Latitude'].max()+0.0005] #r_loc: 43.782970N, (-)79.46952W 
passed_peaks = corrected_peaks_L2d2.loc[(corrected_peaks_L2d2['QC'] == True)]
coord_extent_2 = [passed_peaks['Longitude'].min()-0.001,passed_peaks['Longitude'].max()+0.0018, passed_peaks['Latitude'].min()-0.0005, passed_peaks['Latitude'].max()+0.0005] #r_loc: 43.782970N, (-)79.46952W 
column_names_1 = {'Licor': 'CH4_ele_Licor'}
release_L2_loc1 = (-0.44161,52.23438)


''' --- Day 1 --- '''

if not (path_fig / 'London_II_2024' / 'L2_Peakplots_QCpassed' / 'Day1').is_dir():
    (path_fig / 'London_II_2024' / 'L2_Peakplots_QCpassed' / 'Day1').mkdir(parents=True)
path_save =  path_fig / 'London_II_2024' / 'L2_Peakplots_QCpassed' / 'Day1'

# Call the plot function to plot individual peaks
plot_indivpeaks_afterQC(corrected_peaks_L2d1, path_save, coord_extent_1, release_L2_loc1, release_loc2==None, indiv_peak_plots, column_names_1, L2_vars_d1_Licor)


''' --- Day 2 --- '''

if not (path_fig / 'London_II_2024' / 'L2_Peakplots_QCpassed' / 'Day2').is_dir():
    (path_fig / 'London_II_2024' / 'L2_Peakplots_QCpassed' / 'Day2').mkdir(parents=True)
path_save =  path_fig / 'London_II_2024' / 'L2_Peakplots_QCpassed' / 'Day2'

# Call the plot function to plot individual peaks
plot_indivpeaks_afterQC(corrected_peaks_L2d2, path_save, coord_extent_2, release_L2_loc1, release_loc2==None, indiv_peak_plots, column_names_1, L2_vars_d2_Licor)



#%% Process&Save for further analysis

'''
The dataframes total_peaks_x/corrected_peaks_x contain, for each deployed instrument, 
separate columns for the corresponding maximum and area values. For subsequent 
statistical analysis, it is more practical to consolidate these values into single columns.

In the following, the dataframes for each CRE are reorganized so that only one 
column for the maximum 'Max' and one for the area 'Area' remain. The instrument 
used for each measurement is recorded in the column 'Instruments_max'.
-> df_x_comb

A dataframe is created using .groupby(['Release_rate']).size().reset_index(name='count') 
to count the number of valid peaks per release rate for each CRE
-> df_x_rr_count

Finally, a combined dataframe is created that includes the measurements from all CREs.
    '''



# Combine -------------------------------------------------------------------------------------------------------------------------------

df_U1   = total_peaks_U1[['Loc','Release_rate','Area_mean_G23','Area_mean_G43','Max_G23','Max_G43','Latitude','Longitude','Mean_speed']].copy(deep=True)
df_U2   = total_peaks_U2[['Loc','Release_rate','Area_mean_G23','Area_mean_aeris','Max_G23','Max_aeris','Latitude','Longitude','Mean_speed']].copy(deep=True)
df_L1d2 = total_peaks_L1d2[['Release_rate','Area_mean_LGR','Area_mean_G23','Max_LGR','Max_G23','Peak','Latitude','Longitude','Mean_speed']].copy(deep=True)
df_L1d3 = total_peaks_L1d3[['Release_rate','Area_mean_Licor','Area_mean_G23','Max_Licor','Max_G23','Peak','Latitude','Longitude','Mean_speed']].copy(deep=True)
df_L1d4 = total_peaks_L1d4[['Release_rate','Area_mean_Licor','Area_mean_G23','Max_Licor','Max_G23','Peak','Latitude','Longitude','Mean_speed']].copy(deep=True)
df_L1d5 = total_peaks_L1d5[['Release_rate','Area_mean_G23','Max_G23','Peak','Latitude','Longitude','Mean_speed']].copy(deep=True)
df_T1b  = total_peaks_T1b[['Release_rate','Area_mean_LGR','Max_LGR','Peak','Latitude','Longitude','Mean_speed']].copy(deep=True)
df_T1c  = total_peaks_T1c[['Release_rate','Area_mean_G24','Max_G24','Peak','Latitude','Longitude','Mean_speed']].copy(deep=True)
df_T2c  = total_peaks_T2c[['Release_rate','Area_mean_G24','Max_G24','Peak','Latitude','Longitude','Mean_speed']].copy(deep=True)
df_L2d1 = total_peaks_L2d1[['Release_rate','Area_mean_Licor','Max_Licor','Peak','Latitude','Longitude','Mean_speed']].copy(deep=True)
df_L2d2 = total_peaks_L2d2[['Release_rate','Area_mean_Licor','Max_Licor','Peak','Latitude','Longitude','Mean_speed']].copy(deep=True)


df_T1b_comb = df_T1b.rename(columns={'Area_mean_LGR':'Area','Max_LGR':'Max'})
df_T1c_comb = df_T1c.rename(columns={'Area_mean_G24':'Area','Max_G24':'Max'})
df_T2c_comb = df_T2c.rename(columns={'Area_mean_G24':'Area','Max_G24':'Max'})
df_T1b_comb['Instruments_max'] = 'Max_LGR'
df_T1c_comb['Instruments_max'] = 'Max_G24'
df_T2c_comb['Instruments_max'] = 'Max_G24'

df_L2d1_comb = df_L2d1.rename(columns={'Area_mean_Licor':'Area','Max_Licor':'Max'})
df_L2d2_comb = df_L2d2.rename(columns={'Area_mean_Licor':'Area','Max_Licor':'Max'})
df_L2d1_comb['Instruments_max'] = 'Max_Licor'
df_L2d2_comb['Instruments_max'] = 'Max_Licor'


# Rotterdam ------------------
df_R_comb = total_peaks_R.copy(deep=True)
max_columns = df_R_comb.filter(like='Max').copy(deep=True)
df_R_comb.drop(list(max_columns.columns),axis=1,inplace=True) #must be directly after the filter line, sicne later additional columns are added which should not be droped from df_R
max_columns['Loc'] = df_R_comb['Loc']
max_columns['Release_rate'] = df_R_comb['Release_rate']
max_columns['Loc_tuple'] = max_columns.apply(combine_columns, axis=1)
max_columns.drop(['Loc','Release_rate'],axis=1,inplace=True)
df_R_comb.reset_index(inplace=True,drop=False)
df_R_comb = pd.melt(df_R_comb, id_vars=['Loc','Release_rate','Longitude', 'Latitude','Datetime','Peak','Mean_speed'],
                    value_vars=['Area_mean_aeris', 'Area_mean_G23', 'Area_mean_G43','Area_mean_miro','Area_mean_aero'], 
                    var_name='Instruments_area', value_name='Area')
max_columns = pd.melt(max_columns, id_vars=['Loc_tuple'], value_vars=['Max_aeris', 'Max_G23', 'Max_G43', 'Max_miro', 'Max_aero'], 
                      var_name='Instruments_max', value_name='Max')
df_R_comb = pd.concat([df_R_comb, max_columns], axis=1)
df_R_comb.dropna(subset=['Area'], inplace=True)
df_R_comb.drop(['Instruments_area'],axis=1,inplace=True)

# Claculate log
df_R_comb['ln(Area)'] = np.log(df_R_comb['Area'])
df_R_comb['ln(Max)'] = np.log(df_R_comb['Max'])

df_R_rr_count = df_R_comb.groupby(['Release_rate']).size().reset_index(name='count')

# Rotterdam UU and TNO separate -----------------------
# first extract right instruments
R1 = total_peaks_R[:'2022-09-06 11:05:00'].copy(deep=True)
R2 = total_peaks_R['2022-09-06 11:05:00':].copy(deep=True)
df_R_comb_UU = R1[R1['Max_G23'].notna()].copy(deep=True) #only in the morning
df_R_comb_TNO = R1[R1['Max_miro'].notna()].copy(deep=True)
df_R_comb_TNO = pd.concat([df_R_comb_TNO,R2]) # in the morning + afternoon

# TNO
max_columns = df_R_comb_TNO.filter(like='Max').copy(deep=True)
df_R_comb_TNO.drop(list(max_columns.columns),axis=1,inplace=True) #must be directly after the filter line, sicne later additional columns are added which should not be droped from df_R
max_columns['Loc'] = df_R_comb_TNO['Loc']
max_columns['Release_rate'] = df_R_comb_TNO['Release_rate']
max_columns['Loc_tuple'] = max_columns.apply(combine_columns, axis=1)
max_columns.drop(['Loc','Release_rate'],axis=1,inplace=True)
df_R_comb_TNO = pd.melt(df_R_comb_TNO, id_vars=['Loc','Release_rate','Longitude', 'Latitude','Mean_speed'],
                    value_vars=['Area_mean_G43', 'Area_mean_aeris','Area_mean_miro','Area_mean_aero'], 
                    var_name='Instruments_area', value_name='Area')
max_columns = pd.melt(max_columns, id_vars=['Loc_tuple'], value_vars=['Max_G43','Max_aeris', 'Max_miro', 'Max_aero'], 
                      var_name='Instruments_max', value_name='Max')
df_R_comb_TNO = pd.concat([df_R_comb_TNO, max_columns], axis=1)
df_R_comb_TNO.dropna(subset=['Area'], inplace=True)
df_R_comb_TNO.drop(['Instruments_area'],axis=1,inplace=True)

df_R_comb_TNO['ln(Area)'] = np.log(df_R_comb_TNO['Area']) # Claculate log
df_R_comb_TNO['ln(Max)'] = np.log(df_R_comb_TNO['Max'])

df_R_TNO_rr_count = df_R_comb_TNO.groupby(['Release_rate']).size().reset_index(name='count')

# UU
max_columns = df_R_comb_UU.filter(like='Max').copy(deep=True)
df_R_comb_UU.drop(list(max_columns.columns),axis=1,inplace=True) #must be directly after the filter line, sicne later additional columns are added which should not be droped from df_R
max_columns['Loc'] = df_R_comb_UU['Loc']
max_columns['Release_rate'] = df_R_comb_UU['Release_rate']
max_columns['Loc_tuple'] = max_columns.apply(combine_columns, axis=1)
max_columns.drop(['Loc','Release_rate'],axis=1,inplace=True)
df_R_comb_UU = pd.melt(df_R_comb_UU, id_vars=['Loc','Release_rate','Longitude', 'Latitude','Mean_speed'],
                    value_vars=['Area_mean_aeris', 'Area_mean_G23', 'Area_mean_G43'], 
                    var_name='Instruments_area', value_name='Area')
max_columns = pd.melt(max_columns, id_vars=['Loc_tuple'], value_vars=['Max_aeris', 'Max_G23', 'Max_G43'], 
                      var_name='Instruments_max', value_name='Max')
df_R_comb_UU = pd.concat([df_R_comb_UU, max_columns], axis=1)
df_R_comb_UU.dropna(subset=['Area'], inplace=True)
df_R_comb_UU.drop(['Instruments_area'],axis=1,inplace=True)

df_R_comb_UU['ln(Area)'] = np.log(df_R_comb_UU['Area']) # Claculate log
df_R_comb_UU['ln(Max)'] = np.log(df_R_comb_UU['Max'])

df_R_UU_rr_count = df_R_comb_UU.groupby(['Release_rate']).size().reset_index(name='count')



# Utrecht ------------------
df_U1_comb = df_U1.copy(deep=True)
max_columns = df_U1_comb.filter(like='Max').copy(deep=True)
df_U1_comb.drop(list(max_columns.columns),axis=1,inplace=True) #must be directly after the filter line, sicne later additional columns are added which should not be droped from df_R
max_columns['Loc'] = df_U1_comb['Loc']
max_columns['Release_rate'] = df_U1_comb['Release_rate']
max_columns['Loc_tuple'] = max_columns.apply(combine_columns, axis=1)
max_columns.drop(['Loc','Release_rate'],axis=1,inplace=True)
df_U1_comb.reset_index(inplace=True,drop=False)
df_U1_comb = pd.melt(df_U1_comb, id_vars=['Loc','Release_rate','Latitude','Longitude','Mean_speed','Datetime'],
                    value_vars=['Area_mean_G23', 'Area_mean_G43'], 
                    var_name='Instruments_area', value_name='Area')
max_columns = pd.melt(max_columns, id_vars=['Loc_tuple'], value_vars=['Max_G23', 'Max_G43'], 
                      var_name='Instruments_max', value_name='Max')
df_U1_comb = pd.concat([df_U1_comb, max_columns], axis=1)
df_U1_comb.drop(['Instruments_area'],axis=1,inplace=True)

# Calculate log
df_U1_comb['ln(Area)'] = np.log(df_U1_comb['Area'])
df_U1_comb['ln(Max)'] = np.log(df_U1_comb['Max'])

df_U1_rr_count = df_U1_comb.groupby(['Release_rate']).size().reset_index(name='count')

# Utrecht III ------------------
df_U2_comb = df_U2.copy(deep=True)
max_columns = df_U2_comb.filter(like='Max').copy(deep=True)
df_U2_comb.drop(list(max_columns.columns),axis=1,inplace=True) #must be directly after the filter line, sicne later additional columns are added which should not be droped from df_R
max_columns['Loc'] = df_U2_comb['Loc']
max_columns['Release_rate'] = df_U2_comb['Release_rate']
max_columns['Loc_tuple'] = max_columns.apply(combine_columns, axis=1)
max_columns.drop(['Loc','Release_rate'],axis=1,inplace=True)
df_U2_comb.reset_index(inplace=True,drop=False)
df_U2_comb = pd.melt(df_U2_comb, id_vars=['Loc','Release_rate','Latitude','Longitude','Mean_speed','Datetime'],
                    value_vars=['Area_mean_G23', 'Area_mean_aeris'], 
                    var_name='Instruments_area', value_name='Area')
max_columns = pd.melt(max_columns, id_vars=['Loc_tuple'], value_vars=['Max_G23', 'Max_aeris'], 
                      var_name='Instruments_max', value_name='Max')
df_U2_comb = pd.concat([df_U2_comb, max_columns], axis=1)
df_U2_comb.drop(['Instruments_area'],axis=1,inplace=True)

# Calculate log
df_U2_comb['ln(Area)'] = np.log(df_U2_comb['Area'])
df_U2_comb['ln(Max)'] = np.log(df_U2_comb['Max'])

df_U2_rr_count = df_U2_comb.groupby(['Release_rate']).size().reset_index(name='count')


# London ------------------

# Day2 ---
df_L1d2_comb = df_L1d2.copy(deep=True)
max_columns = df_L1d2_comb.filter(like='Max').copy(deep=True)
df_L1d2_comb.drop(list(max_columns.columns),axis=1,inplace=True) #must be directly after the filter line, sicne later additional columns are added which should not be droped from df_R
max_columns['Release_rate'] = df_L1d2_comb['Release_rate']
df_L1d2_comb.reset_index(inplace=True,drop=False)
df_L1d2_comb = pd.melt(df_L1d2_comb, id_vars=['Release_rate','Peak','Latitude','Longitude','Mean_speed','Datetime'],
                    value_vars=['Area_mean_G23', 'Area_mean_LGR'], 
                    var_name='Instruments_area', value_name='Area')
max_columns = pd.melt(max_columns, id_vars=['Release_rate'], value_vars=['Max_G23', 'Max_LGR'], 
                      var_name='Instruments_max', value_name='Max')
max_columns.drop(['Release_rate'],axis=1,inplace=True)
df_L1d2_comb = pd.concat([df_L1d2_comb, max_columns], axis=1)
df_L1d2_comb.drop(['Instruments_area'],axis=1,inplace=True)

# Day3 ---
df_L1d3_comb = df_L1d3.copy(deep=True)
max_columns = df_L1d3_comb.filter(like='Max').copy(deep=True)
df_L1d3_comb.drop(list(max_columns.columns),axis=1,inplace=True) #must be directly after the filter line, sicne later additional columns are added which should not be droped from df_R
max_columns['Release_rate'] = df_L1d3_comb['Release_rate']
df_L1d3_comb.reset_index(inplace=True,drop=False)
df_L1d3_comb = pd.melt(df_L1d3_comb, id_vars=['Release_rate','Peak','Latitude','Longitude','Mean_speed','Datetime'],
                    value_vars=['Area_mean_G23', 'Area_mean_Licor'], 
                    var_name='Instruments_area', value_name='Area')
max_columns = pd.melt(max_columns, id_vars=['Release_rate'], value_vars=['Max_G23', 'Max_Licor'], 
                      var_name='Instruments_max', value_name='Max')
max_columns.drop(['Release_rate'],axis=1,inplace=True)
df_L1d3_comb = pd.concat([df_L1d3_comb, max_columns], axis=1)
df_L1d3_comb.drop(['Instruments_area'],axis=1,inplace=True)

# Day4 ---
df_L1d4_comb = df_L1d4.copy(deep=True)
max_columns = df_L1d4_comb.filter(like='Max').copy(deep=True)
df_L1d4_comb.drop(list(max_columns.columns),axis=1,inplace=True) #must be directly after the filter line, sicne later additional columns are added which should not be droped from df_R
max_columns['Release_rate'] = df_L1d4_comb['Release_rate']
df_L1d4_comb.reset_index(inplace=True,drop=False)
df_L1d4_comb = pd.melt(df_L1d4_comb, id_vars=['Release_rate','Peak','Latitude','Longitude','Mean_speed','Datetime'],
                    value_vars=['Area_mean_G23', 'Area_mean_Licor'], 
                    var_name='Instruments_area', value_name='Area')
max_columns = pd.melt(max_columns, id_vars=['Release_rate'], value_vars=['Max_G23', 'Max_Licor'], 
                      var_name='Instruments_max', value_name='Max')
max_columns.drop(['Release_rate'],axis=1,inplace=True)
df_L1d4_comb = pd.concat([df_L1d4_comb, max_columns], axis=1)
df_L1d4_comb.drop(['Instruments_area'],axis=1,inplace=True)

df_L1d5_comb = df_L1d5.copy(deep=True)
df_L1d5_comb['Instruments_max'] = ['Max_G23'] * len(df_L1d5_comb)
df_L1d5_comb = df_L1d5_comb.rename(columns={'Area_mean_G23':'Area','Max_G23':'Max'})

# Calculate log
df_L1d2_comb['ln(Area)'] = np.log(df_L1d2_comb['Area'])
df_L1d2_comb['ln(Max)'] = np.log(df_L1d2_comb['Max'])

df_L1d3_comb['ln(Area)'] = np.log(df_L1d3_comb['Area'])
df_L1d3_comb['ln(Max)'] = np.log(df_L1d3_comb['Max'])

df_L1d4_comb['ln(Area)'] = np.log(df_L1d4_comb['Area'])
df_L1d4_comb['ln(Max)'] = np.log(df_L1d4_comb['Max'])

df_L1d5_comb['ln(Area)'] = np.log(df_L1d5_comb['Area'])
df_L1d5_comb['ln(Max)'] = np.log(df_L1d5_comb['Max'])


df_L1d2_rr_count = df_L1d2_comb.groupby(['Release_rate']).size().reset_index(name='count')
df_L1d3_rr_count = df_L1d3_comb.groupby(['Release_rate']).size().reset_index(name='count')
df_L1d4_rr_count = df_L1d4_comb.groupby(['Release_rate']).size().reset_index(name='count')
df_L1d5_rr_count = df_L1d5_comb.groupby(['Release_rate']).size().reset_index(name='count')

# London II ------------------

# Day1 ---

df_L2d1_comb['ln(Area)'] = np.log(df_L2d1_comb['Area'])
df_L2d1_comb['ln(Max)'] = np.log(df_L2d1_comb['Max'])

df_L2d2_comb['ln(Area)'] = np.log(df_L2d2_comb['Area'])
df_L2d2_comb['ln(Max)'] = np.log(df_L2d2_comb['Max'])

df_L2d1_rr_count = df_L2d1_comb.groupby(['Release_rate']).size().reset_index(name='count')
df_L2d2_rr_count = df_L2d2_comb.groupby(['Release_rate']).size().reset_index(name='count')


# Toronto ------------------

df_T1b_comb['ln(Area)'] = np.log(df_T1b_comb['Area'])
df_T1b_comb['ln(Max)'] = np.log(df_T1b_comb['Max'])

df_T1c_comb['ln(Area)'] = np.log(df_T1c_comb['Area'])
df_T1c_comb['ln(Max)'] = np.log(df_T1c_comb['Max'])

df_T2c_comb['ln(Area)'] = np.log(df_T2c_comb['Area'])
df_T2c_comb['ln(Max)'] = np.log(df_T2c_comb['Max'])

df_T1b_rr_count = df_T1b_comb.groupby(['Release_rate']).size().reset_index(name='count')
df_T1c_rr_count = df_T1c_comb.groupby(['Release_rate']).size().reset_index(name='count')
df_T2c_rr_count = df_T2c_comb.groupby(['Release_rate']).size().reset_index(name='count')




# Add distance to source ------------------
# also sets the Datetime as index again (was reseted in a previous step)
add_distance_to_df(df_R_comb,'Rotterdam')
add_distance_to_df(df_U1_comb,'Utrecht I')
add_distance_to_df(df_U2_comb,'Utrecht II')
add_distance_to_df(df_L1d2_comb,'London I')
add_distance_to_df(df_L1d3_comb,'London I')
add_distance_to_df(df_L1d4_comb,'London I')
add_distance_to_df(df_L1d5_comb,'London I')
add_distance_to_df(df_T1b_comb,'Toronto',Day=1)
add_distance_to_df(df_T1c_comb,'Toronto',Day=1)
add_distance_to_df(df_T2c_comb,'Toronto',Day=2)
add_distance_to_df(df_L2d1_comb,'London II')
add_distance_to_df(df_L2d2_comb,'London II')


df_R_comb['City']    = 'Rotterdam'
df_U1_comb['City']   = 'Utrecht I'
df_U2_comb['City']   = 'Utrecht II'
df_L1d2_comb['City'] = 'London I-Day2'
df_L1d3_comb['City'] = 'London I-Day3'
df_L1d4_comb['City'] = 'London I-Day4'
df_L1d5_comb['City'] = 'London I-Day5'
df_T1b_comb['City']  = 'Toronto-1b'
df_T1c_comb['City']  = 'Toronto-1c'
df_T2c_comb['City']  = 'Toronto-2c'
df_L2d1_comb['City'] = 'London II-Day1'
df_L2d2_comb['City'] = 'London II-Day2'

df_L1d2_comb['Loc']  = 1
df_L1d3_comb['Loc']  = 1
df_L1d4_comb['Loc']  = 1
df_L1d5_comb['Loc']  = 1
df_T1b_comb['Loc']   = 1
df_T1c_comb['Loc']   = 1
df_T2c_comb['Loc']   = 1
df_L2d1_comb['Loc']  = 1
df_L2d2_comb['Loc']  = 1


# Merge all cities into one df------------------
# reset index now and restore it later to avoid index related issues during pd.concat
df_RU2T2L4L2 = pd.concat(
    [
        df_R_comb.copy().reset_index(drop=False),
        df_U1_comb.copy().reset_index(drop=False),
        df_U2_comb.copy().dropna(subset=['Max']).reset_index(drop=False),
        df_T1b_comb.copy().reset_index(drop=False),
        df_T1c_comb.copy().reset_index(drop=False),
        df_T2c_comb.copy().reset_index(drop=False),
        df_L1d2_comb.copy().reset_index(drop=False),
        df_L1d3_comb.copy().reset_index(drop=False),
        df_L1d4_comb.copy().reset_index(drop=False),
        df_L1d5_comb.copy().reset_index(drop=False),
        df_L2d1_comb.copy().reset_index(drop=False),
        df_L2d2_comb.copy().reset_index(drop=False),
    ],
    axis=0,
    ignore_index=True,
    sort=False
)
df_RU2T2L4L2 = df_RU2T2L4L2.set_index(df_RU2T2L4L2.columns[0], drop=True) # set Datetime column as index
df_RU2T2L4L2_rr_count = df_RU2T2L4L2.groupby(['Release_rate']).size().reset_index(name='count')



# save_to_csv=True
if save_to_csv:
    # Save the DataFrame as a CSV file
    df_RU2T2L4L2.to_csv(path_finaldata  / 'RU2T2L4L2_TOTAL_PEAKS_comb.csv')
    df_RU2T2L4L2_rr_count.to_csv(path_finaldata / 'RU2T2L4L2_count.csv')
    # release height =4 in London NOT included
    
    df_R_comb.to_csv(path_finaldata     / 'R_comb_final.csv')
    df_U1_comb.to_csv(path_finaldata    / 'U1_comb_final.csv')
    df_U2_comb.to_csv(path_finaldata    / 'U2_comb_final.csv')
    df_L1d2_comb.to_csv(path_finaldata  / 'L1d2_comb_final.csv')
    df_L1d3_comb.to_csv(path_finaldata  / 'L1d3_comb_final.csv')
    df_L1d4_comb.to_csv(path_finaldata  / 'L1d4_comb_final.csv')
    df_L1d5_comb.to_csv(path_finaldata  / 'L1d5_comb_final.csv')
    df_T1b_comb.to_csv(path_finaldata   / 'T1b_comb_final.csv')
    df_T1c_comb.to_csv(path_finaldata   / 'T1c_comb_final.csv')
    df_T2c_comb.to_csv(path_finaldata   / 'T2c_comb_final.csv')
    df_L2d1_comb.to_csv(path_finaldata  / 'L2d1_comb_final.csv')
    df_L2d2_comb.to_csv(path_finaldata  / 'L2d2_comb_final.csv')
    
    df_R_rr_count.to_csv(path_finaldata     / 'R_comb_count.csv')
    df_U1_rr_count.to_csv(path_finaldata    / 'U1_comb_count.csv')
    df_U2_rr_count.to_csv(path_finaldata    / 'U2_comb_count.csv')
    df_L1d2_rr_count.to_csv(path_finaldata  / 'L1d2_comb_count.csv')
    df_L1d3_rr_count.to_csv(path_finaldata  / 'L1d3_comb_count.csv')
    df_L1d4_rr_count.to_csv(path_finaldata  / 'L1d4_comb_count.csv')
    df_L1d5_rr_count.to_csv(path_finaldata  / 'L1d5_comb_count.csv')
    df_T1b_rr_count.to_csv(path_finaldata   / 'T1b_comb_count.csv')
    df_T1c_rr_count.to_csv(path_finaldata   / 'T1c_comb_count.csv')
    df_T2c_rr_count.to_csv(path_finaldata   / 'T2c_comb_count.csv')
    df_L2d1_rr_count.to_csv(path_finaldata  / 'L2d1_comb_count.csv')
    df_L2d2_rr_count.to_csv(path_finaldata  / 'L2d2_comb_count.csv')



#%% End
