# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:27:11 2024

@author: Judith Tettenborn (j.a.f.tettenborn@uu.nl) & Daan Stroeken


This script creates the main figures (Plot 1 & Plot 2) included in the publication 
Tettenborn et al. (2025).

Plot1
Visualizes a comparison of peak maximum and peak area between different instruments.

Plot2
Visualizes all peak measurements (peak area/peak maximum versus release rate), 
including linear regression fits.

Plot3
Visulaize peak measurments as timeseries as a lollipop plot

STATS
Some statistical calculations on the peak length, driving speed and distance to source.


"""


#%% Load Packages & Data

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
# import statsmodels.api as sm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# CRE code functions:
from preprocessing.read_in_data import *
from peak_analysis.find_analyze_peaks import *
from plotting.general_plots import *
from helper_functions.utils import *
from stats_analysis.stats_functions import *

from helper_functions.constants import (
    dict_color_instr,
    dict_color_city,
    dict_spec_instr,
    dict_instr_names
    )

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
    L1_vars_d5_G23,
    L2_vars_d1_Licor,
    L2_vars_d2_Licor
    )

# READ IN DATA
  
path_finaldata  = path_base / 'data' / 'final' 
path_res        = path_base / 'results/'
path_fig        = path_base / 'results' / 'Figures/'
path_fig_plot1  = path_fig / 'Plot1_MasterComparison_Instruments/'
path_fig_plot2  = path_fig / 'Plot2_LinReg/'
path_fig_plot3  = path_fig / 'Plot3_Lollipop/'



# Utrecht
total_peaks_U1 = pd.read_csv(path_finaldata / 'U1_TOTAL_PEAKS.csv', index_col='Datetime', parse_dates=['Datetime']) 
total_peaks_U2 = pd.read_csv(path_finaldata / 'U2_TOTAL_PEAKS.csv', index_col='Datetime', parse_dates=['Datetime'])  
 
df_U1_comb     = pd.read_csv(path_finaldata / 'U1_comb_final.csv', index_col='Datetime', parse_dates=['Datetime']) 

# Rotterdam
total_peaks_R = pd.read_csv(path_finaldata / 'R_TOTAL_PEAKS.csv', index_col='Datetime', parse_dates=['Datetime'])  
total_peaks_R = total_peaks_R[total_peaks_R['Release_rate'] != 0] 

df_R_comb =  pd.read_csv(path_finaldata / 'R_comb_final.csv', index_col='Datetime', parse_dates=['Datetime']) 
df_R_comb = df_R_comb[df_R_comb['Release_rate'] != 0]


# Tornto
total_peaks_T1b = pd.read_csv(path_finaldata / 'T_TOTAL_PEAKS_1b.csv', index_col='Datetime', parse_dates=['Datetime'])  
total_peaks_T1c = pd.read_csv(path_finaldata / 'T_TOTAL_PEAKS_1c.csv', index_col='Datetime', parse_dates=['Datetime'])  
total_peaks_T2c = pd.read_csv(path_finaldata / 'T_TOTAL_PEAKS_2c.csv', index_col='Datetime', parse_dates=['Datetime']) 

df_T1b_comb     = pd.read_csv(path_finaldata / 'T1b_comb_final.csv', index_col='Datetime', parse_dates=['Datetime']) 
df_T1c_comb     = pd.read_csv(path_finaldata / 'T1c_comb_final.csv', index_col='Datetime', parse_dates=['Datetime']) 
df_T2c_comb     = pd.read_csv(path_finaldata / 'T2c_comb_final.csv', index_col='Datetime', parse_dates=['Datetime']) 

# London
total_peaks_L1d2 = pd.read_csv(path_finaldata / 'L1_TOTAL_PEAKS_Day2.csv', index_col='Datetime', parse_dates=['Datetime'])  
total_peaks_L1d3 = pd.read_csv(path_finaldata / 'L1_TOTAL_PEAKS_Day3.csv', index_col='Datetime', parse_dates=['Datetime'])  
total_peaks_L1d5 = pd.read_csv(path_finaldata / 'L1_TOTAL_PEAKS_Day5.csv', index_col='Datetime', parse_dates=['Datetime']) 


# London II
total_peaks_L2d1 = pd.read_csv(path_finaldata / 'L2_TOTAL_PEAKS_Day1.csv', index_col='Datetime', parse_dates=['Datetime'])  
total_peaks_L2d2 = pd.read_csv(path_finaldata / 'L2_TOTAL_PEAKS_Day2.csv', index_col='Datetime', parse_dates=['Datetime'])  
total_peaks_L2d2 = total_peaks_L2d2[total_peaks_L2d2['Release_rate'] > 0.2] # only one peak at 0.2 L/min -> delete


# ALL
total_peaks_all_alldist = pd.read_csv(path_finaldata / 'RU2T2L4L2_TOTAL_PEAKS.csv', index_col='Datetime', parse_dates=['Datetime'])  
df_all_alldist          = pd.read_csv(path_finaldata / 'RU2T2L4L2_TOTAL_PEAKS_comb.csv', index_col='Datetime', parse_dates=['Datetime']) 
total_peaks_all         = total_peaks_all_alldist[total_peaks_all_alldist['Distance_to_source']<75].copy()
df_all                  = df_all_alldist[df_all_alldist['Distance_to_source']<75].copy()






#%% P1: Comparison Instrument PLOT

'''
Figure R+U
Comparison of (a )peak maximum and (b) spatial peak area from different instruments 
deployed in Rotterdam and Utrecht I (subscript’_U’),shown are the data points 
and a linear regression fit with intercept 0 for each instrument.The results 
from the G2301, Mira Ultra, MGA10 and TILDAS devices are plotted on the y-axis
and the results from the G4302 instrument on the x-axis. The black dotted line
represents the 1:1 line. (For the G2301 analyzer, peaks exceeding a maximum of
20 ppm are marked with an ’x’ and excluded from the fitting process.)

Figure L
Comparison of (a )peak maximum and (b) spatial peak area from different instruments
in London I (Day1 and Day2). Regression fits with intercept 0 are applied to the
data for each instrument. The results from the uMEA and LI-7810 analyzers are
plotted on the y-axis and the results from the G2301-m instrument on the x-axis.
The black dotted line represents the 1:1 line. (Peaks exceeding a maximum of 20 ppm
are marked with an ’x’ and were excluded from the fitting process.)
     '''

#%%% FINAL FIGURE: R+U - 1Plot




fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,8))

# Full range
x1 = np.arange(0,650)
x2 = np.arange(0,2250)

Ar_items = [None]*10
Max_items = [None]*10
titles = []
titles2 = []
i = 0

# Utrecht ====================================================================================================
spec = 'G23'
name = 'G2301_U'
title = 'G2301_U'
titles.append(title)
titles2.append(title)
color_U = 'indianred'

#-------
# 1. Fit only to data which have a maximum <20ppm ===================
cond3 = total_peaks_U1[f'Max_G23']<20
plotpeaks = total_peaks_U1[cond3].copy()

# Area -----------------------------------------------------
x = plotpeaks['Area_mean_G43']
Ar_items[i] = ax2.scatter(x,plotpeaks[f'Area_mean_{spec}'],color=color_U)    
# ax_1.scatter(outliers['Area_G43_mean'], outliers[f'Area_mean_{spec}'],facecolors='none',edgecolors='grey', label= 'outlier')

popt, pcov  = curve_fit(scalar_multiplication, x, plotpeaks[f'Area_mean_{spec}'],p0=1.0,maxfev=200000)
a1      = round(popt[0],3)
#a_          = round(a,3)
r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Area_mean_{spec}'])
correlation = round(plotpeaks['Area_mean_G43'].corr(plotpeaks[f'Area_mean_{spec}']),2)

Ar_items[i+1], = ax2.plot(x2,popt[0]*x2,color=color_U,label=f'{title}: y = {a1}' r' $\cdot$ x ' f'($R^2$ = {r2})') #, rho = {correlation})

# Max -----------------------------------------------------
x = plotpeaks['Max_G43']
Max_items[i] = ax1.scatter(x,plotpeaks[f'Max_{spec}'],color=color_U) #,label=title

popt, pcov  = curve_fit(scalar_multiplication, x, plotpeaks[f'Max_{spec}'],p0=1.0,maxfev=20000)
a1      = round(popt[0],3)
r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Max_{spec}'])
correlation = round(plotpeaks['Max_G43'].corr(plotpeaks[f'Max_{spec}']),2)

Max_items[i+1], = ax1.plot(x1,popt[0]*x1,color=color_U,label=f'{title}: y = {a1}' r' $\cdot$ x ' f'($R^2$ = {r2})') #, rho = {correlation})

# 2. Fit to all G2301 data ==========================================

cond3 = total_peaks_U1[f'Max_G23']>20
peaks_G23_larger20 = total_peaks_U1[cond3]


# Display peaks >20ppm -------------------------------------------
Ar_items[i] = ax2.scatter(peaks_G23_larger20['Area_mean_G43'],peaks_G23_larger20[f'Area_mean_G23'],color=color_U,marker = 'x')    
Max_items[i] = ax1.scatter(peaks_G23_larger20['Max_G43'],peaks_G23_larger20[f'Max_G23'],color=color_U, marker = 'x') #,label=title


# Rotterdam ====================================================================================================


for spec_vars in [R_vars_aeris,R_vars_G23,R_vars_miro,R_vars_aerodyne]:
   
    spec = spec_vars['spec']
    print(spec)
    title = spec_vars['title']
    name = spec_vars['name']
    titles.append(title)
    titles2.append(title)
    cond1 = total_peaks_R[f'Max_{spec}'].notna()
    cond2 = total_peaks_R['Area_mean_G43'].notna()
    
    # FINAL: Fit to all G23 + separate fit to only <20ppm
    if spec == 'G23':
        
        # 1. Fit only to data which have a maximum <20ppm ===================
        cond3 = total_peaks_R[f'Max_G23']<20
        plotpeaks = total_peaks_R[cond1 & cond2 & cond3].copy()
        
        # Area -----------------------------------------------------
        x = plotpeaks['Area_mean_G43']
        Ar_items[i] = ax2.scatter(x,plotpeaks[f'Area_mean_{spec}'],color=dict_color_instr[name])    
        # ax_1.scatter(outliers['Area_G43_mean'], outliers[f'Area_mean_{spec}'],facecolors='none',edgecolors='grey', label= 'outlier')
    
        popt, pcov  = curve_fit(scalar_multiplication, x, plotpeaks[f'Area_mean_{spec}'],p0=1.0,maxfev=200000)
        a1      = round(popt[0],3)
        #a_          = round(a,3)
        r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Area_mean_{spec}'])
        correlation = round(plotpeaks['Area_mean_G43'].corr(plotpeaks[f'Area_mean_{spec}']),2)
        
        Ar_items[i+1], = ax2.plot(x2,popt[0]*x2,color=dict_color_instr[name],label=f'{title}: y = {a1}' r' $\cdot$ x ' f'($R^2$ = {r2})') #, rho = {correlation}
        
        # Max -----------------------------------------------------
        x = plotpeaks['Max_G43']
        Max_items[i] = ax1.scatter(x,plotpeaks[f'Max_{spec}'],color=dict_color_instr[name]) #,label=title
      
        popt, pcov  = curve_fit(scalar_multiplication, x, plotpeaks[f'Max_{spec}'],p0=1.0,maxfev=20000)
        a1      = round(popt[0],3)
        r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Max_{spec}'])
        correlation = round(plotpeaks['Max_G43'].corr(plotpeaks[f'Max_{spec}']),2)
        
        Max_items[i+1], = ax1.plot(x1,popt[0]*x1,color=dict_color_instr[name],label=f'{title}: y = {a1}' r' $\cdot$ x ' f'($R^2$ = {r2})')
    
        # 2. Fit to all G2301 data ==========================================
        
        cond3 = total_peaks_R[f'Max_G23']>20
        peaks_G23_larger20 = total_peaks_R[cond3].copy()
        plotpeaks = total_peaks_R[cond1 & cond2].copy()
        
        # Display peaks >20ppm -------------------------------------------
        Ar_items[i] = ax2.scatter(peaks_G23_larger20['Area_mean_G43'],peaks_G23_larger20[f'Area_mean_G23'],color=dict_color_instr[name],marker = 'x')    
        Max_items[i] = ax1.scatter(peaks_G23_larger20['Max_G43'],peaks_G23_larger20[f'Max_G23'],color=dict_color_instr[name], marker = 'x') #,label=title
        
    
        i += 2
        
    else:
        plotpeaks = total_peaks_R[cond1 & cond2].copy()
        
        # Area -----------------------------------------------------
        x = plotpeaks['Area_mean_G43']
        Ar_items[i] = ax2.scatter(x,plotpeaks[f'Area_mean_{spec}'],color=dict_color_instr[name])    
        # ax_1.scatter(outliers['Area_G43_mean'], outliers[f'Area_mean_{spec}'],facecolors='none',edgecolors='grey', label= 'outlier')
    
        popt, pcov  = curve_fit(scalar_multiplication, x, plotpeaks[f'Area_mean_{spec}'],p0=1.0,maxfev=200000)
        a1      = round(popt[0],3)
        #a_          = round(a,3)
        r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Area_mean_{spec}'])
        correlation = round(plotpeaks['Area_mean_G43'].corr(plotpeaks[f'Area_mean_{spec}']),2)
        
        Ar_items[i+1], = ax2.plot(x2,popt[0]*x2,color=dict_color_instr[name],label=f'{title}: y = {a1}' r' $\cdot$ x ' f'($R^2$ = {r2})') #, rho = {correlation}
        
        # Max -----------------------------------------------------
        x = plotpeaks['Max_G43']
        Max_items[i] = ax1.scatter(x,plotpeaks[f'Max_{spec}'],color=dict_color_instr[name]) #,label=title
      
        popt, pcov  = curve_fit(scalar_multiplication, x, plotpeaks[f'Max_{spec}'],p0=1.0,maxfev=20000)
        a1      = round(popt[0],3)
        r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Max_{spec}'])
        correlation = round(plotpeaks['Max_G43'].corr(plotpeaks[f'Max_{spec}']),2)
        
        Max_items[i+1], = ax1.plot(x1,popt[0]*x1,color=dict_color_instr[name],label=f'{title}: y = {a1}' r' $\cdot$ x ' f'($R^2$ = {r2})')
    
        i += 2
        
# ================================================================================================================


line1, = ax2.plot(x2,scalar_multiplication(x2,1),"k:")
titles.append('1:1')
line2, = ax1.plot(x1,scalar_multiplication(x1,1),"k:")
# ax1.grid(True)
# ax2.grid(True)

ax2.set_ylabel(f'Area other instruments [ppm*m]',fontsize=20)
ax2.set_xlabel('Area G4302 [ppm*m]',fontsize=20)
ax2.set_title('Area',fontweight='bold',fontsize=24)
ax1.set_ylabel(f'Max other instruments [ppm]',fontsize=20)
ax1.set_xlabel('Max G4302 [ppm]',fontsize=20)
ax1.set_title('Max',fontweight='bold',fontsize=24)

#plt.suptitle('Instrument Comparison Plot',fontsize=22)

ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)
ax2.tick_params(axis='x', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)

plt.subplots_adjust(wspace=0.4)

# Create a legend for the first subplot (ax1)
legend_ax1 = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), fancybox=True, shadow=True, ncol=1,fontsize=16)
# Create a legend for the second subplot (ax2)
legend_ax2 = ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), fancybox=True, shadow=True, ncol=1,fontsize=16)


save_plots = False
if save_plots:
    fig.savefig(path_fig_plot1+f'RandU_mastercomparison.pdf',bbox_inches='tight') #_Zoom
    fig.savefig(path_fig_plot1+f'RandU_mastercomparison.png',bbox_inches='tight') #_Zoom
    fig.savefig(path_fig_plot1+f'RandU_mastercomparison.svg',bbox_inches='tight') #_Zoom

plt.show()


#%%% R+U - 2Plot


# Define x-axis ranges for the plots
x1 = np.arange(0, 650)
x2 = np.arange(0, 2250)

Ar_items = [None] * 10
Max_items = [None] * 10
titles = []
titles2 = []
i = 0

# Color settings for Utrecht instrument
color_U = 'indianred'

# --- Plot 1: Area Comparison ---
fig_area, ax_area = plt.subplots(figsize=(8,10))

# Utrecht instrument: G23
spec = 'G23'
title = 'G2301_U'
titles.append(title)
titles2.append(title)

# Fit only data with max < 20 ppm
cond3 = total_peaks_U1[f'Max_G23'] < 20
plotpeaks = total_peaks_U1[cond3].copy()

# Scatter plot for Area
x = plotpeaks['Area_mean_G43']
Ar_items[i] = ax_area.scatter(x, plotpeaks[f'Area_mean_{spec}'], color=color_U)

# Curve fitting for Area
popt, _ = curve_fit(scalar_multiplication, x, plotpeaks[f'Area_mean_{spec}'], p0=1.0, maxfev=200000)
a1 = round(popt[0], 3)
r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Area_mean_{spec}'])
ax_area.plot(x2, popt[0] * x2, color=color_U, label=f'{title}: y = {a1} $\cdot$ x ($R^2$ = {r2})')

# Plot points where max > 20 ppm
cond3 = total_peaks_U1[f'Max_G23'] > 20
peaks_G23_larger20 = total_peaks_U1[cond3]
ax_area.scatter(peaks_G23_larger20['Area_mean_G43'], peaks_G23_larger20[f'Area_mean_G23'], color=color_U, marker='x')

# Other instruments (Rotterdam)
for spec_vars in [R_vars_aeris, R_vars_G23, R_vars_miro, R_vars_aerodyne]:
    # Scatter and fit for Area
    plotpeaks = total_peaks_R[(total_peaks_R[f'Max_{spec_vars["spec"]}'].notna()) & (total_peaks_R['Area_mean_G43'].notna())].copy()
    x = plotpeaks['Area_mean_G43']
    ax_area.scatter(x, plotpeaks[f'Area_mean_{spec_vars["spec"]}'], color=dict_color_instr[spec_vars["name"]])
    
    # Curve fitting
    popt, _ = curve_fit(scalar_multiplication, x, plotpeaks[f'Area_mean_{spec_vars["spec"]}'], p0=1.0, maxfev=200000)
    a1 = round(popt[0], 3)
    r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Area_mean_{spec_vars["spec"]}'])
    ax_area.plot(x2, popt[0] * x2, color=dict_color_instr[spec_vars["name"]], label=f'{spec_vars["title"]}: y = {a1} $\cdot$ x ($R^2$ = {r2})')
    i += 2

# Plot the 1:1 line for Area
ax_area.plot(x2, scalar_multiplication(x2, 1), "k:") #, label="1:1"

# Labels, title, and legend for Area
ax_area.set_ylabel(r'$\mathrm{\left[CH_4\right]_{area}}$ other instruments [ppm*m]', fontsize=22)
ax_area.set_xlabel(r'$\mathrm{\left[CH_4\right]_{area}}$ G4302 [ppm*m]', fontsize=22)
ax_area.tick_params(axis='x', labelsize=20)
ax_area.tick_params(axis='y', labelsize=20)
ax_area.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=1, fontsize=18)



# --- Plot 2: Max Comparison ---------------------------------------------------------------------------

fig_max, ax_max = plt.subplots(figsize=(8, 10))

# Utrecht instrument: G23
spec = 'G23'
name = 'G2301_U'
title = 'G2301_U'

# Fit only data with max < 20 ppm
cond3 = total_peaks_U1[f'Max_G23'] < 20
plotpeaks = total_peaks_U1[cond3].copy()

# Scatter plot for Max
x = plotpeaks['Max_G43']
Max_items[i] = ax_max.scatter(x, plotpeaks[f'Max_{spec}'], color=color_U)

# Curve fitting for Max
popt, _ = curve_fit(scalar_multiplication, x, plotpeaks[f'Max_{spec}'], p0=1.0, maxfev=20000)
a1 = round(popt[0], 3)
r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Max_{spec}'])
ax_max.plot(x1, popt[0] * x1, color=color_U, label=f'{title}: y = {a1} $\cdot$ x ($R^2$ = {r2})')

# Plot points where max > 20 ppm
ax_max.scatter(peaks_G23_larger20['Max_G43'], peaks_G23_larger20[f'Max_G23'], color=color_U, marker='x')

# Other instruments (Rotterdam)
for spec_vars in [R_vars_aeris, R_vars_G23, R_vars_miro, R_vars_aerodyne]:
    # Scatter and fit for Max
    plotpeaks = total_peaks_R[(total_peaks_R[f'Max_{spec_vars["spec"]}'].notna()) & (total_peaks_R['Max_G43'].notna())].copy()
    x = plotpeaks['Max_G43']
    ax_max.scatter(x, plotpeaks[f'Max_{spec_vars["spec"]}'], color=dict_color_instr[spec_vars["name"]])
    
    # Curve fitting
    popt, _ = curve_fit(scalar_multiplication, x, plotpeaks[f'Max_{spec_vars["spec"]}'], p0=1.0, maxfev=20000)
    a1 = round(popt[0], 3)
    r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Max_{spec_vars["spec"]}'])
    ax_max.plot(x1, popt[0] * x1, color=dict_color_instr[spec_vars["name"]], label=f'{spec_vars["title"]}: y = {a1} $\cdot$ x ($R^2$ = {r2})')
    i += 2

# Plot the 1:1 line for Max
ax_max.plot(x1, scalar_multiplication(x1, 1), "k:") #, label="1:1"

# Labels, title, and legend for Max
ax_max.set_ylabel(r'$\mathrm{\left[CH_4\right]_{max}}$ other instruments [ppm]', fontsize=22)
ax_max.set_xlabel(r'$\mathrm{\left[CH_4\right]_{max}}$ G4302 [ppm]', fontsize=22)
ax_max.tick_params(axis='x', labelsize=20)
ax_max.tick_params(axis='y', labelsize=20)
ax_max.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=1, fontsize=18)

# Save plots if desired
save_plots = False
if save_plots:
    fig_area.savefig(path_fig_plot1 + 'RandU_mastercomparison_Area.pdf', dpi=1000, bbox_inches='tight')
    fig_area.savefig(path_fig_plot1 + 'RandU_mastercomparison_Area.png', bbox_inches='tight')
    fig_area.savefig(path_fig_plot1 + 'RandU_mastercomparison_Area.svg', bbox_inches='tight')

    fig_max.savefig(path_fig_plot1 + 'RandU_mastercomparison_Max.pdf', dpi=1000, bbox_inches='tight')
    fig_max.savefig(path_fig_plot1 + 'RandU_mastercomparison_Max.png', bbox_inches='tight')
    fig_max.savefig(path_fig_plot1 + 'RandU_mastercomparison_Max.svg', bbox_inches='tight')

plt.show()







#%%% R+U Zoom




fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,8))

# Full range
x1 = np.arange(0,650)
x2 = np.arange(0,2250)

Ar_items = [None]*10
Max_items = [None]*10
titles = []
titles2 = []
i = 0

# Utrecht ====================================================================================================
spec = 'G23'
name = 'G2301_U'
title = 'G2301_U'
titles.append(title)
titles2.append(title)
color_U = 'indianred'

#-------
# 1. Fit only to data which have a maximum <20ppm ===================
cond3 = total_peaks_U1[f'Max_G23']<20
plotpeaks = total_peaks_U1[cond3].copy()

# Area -----------------------------------------------------
x = plotpeaks['Area_mean_G43']
Ar_items[i] = ax2.scatter(x,plotpeaks[f'Area_mean_{spec}'],color=color_U)    
# ax_1.scatter(outliers['Area_G43_mean'], outliers[f'Area_mean_{spec}'],facecolors='none',edgecolors='grey', label= 'outlier')

popt, pcov  = curve_fit(scalar_multiplication, x, plotpeaks[f'Area_mean_{spec}'],p0=1.0,maxfev=200000)
a1      = round(popt[0],3)
#a_          = round(a,3)
r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Area_mean_{spec}'])
correlation = round(plotpeaks['Area_mean_G43'].corr(plotpeaks[f'Area_mean_{spec}']),2)

Ar_items[i+1], = ax2.plot(x2,popt[0]*x2,color=color_U,label=f'{title}: y = {a1}' r' $\cdot$ x ' f'($R^2$ = {r2})') #, rho = {correlation})

# Max -----------------------------------------------------
x = plotpeaks['Max_G43']
Max_items[i] = ax1.scatter(x,plotpeaks[f'Max_{spec}'],color=color_U) #,label=title

popt, pcov  = curve_fit(scalar_multiplication, x, plotpeaks[f'Max_{spec}'],p0=1.0,maxfev=20000)
a1      = round(popt[0],3)
r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Max_{spec}'])
correlation = round(plotpeaks['Max_G43'].corr(plotpeaks[f'Max_{spec}']),2)

Max_items[i+1], = ax1.plot(x1,popt[0]*x1,color=color_U,label=f'{title}: y = {a1}' r' $\cdot$ x ' f'($R^2$ = {r2})') #, rho = {correlation})

# 2. Fit to all G2301 data ==========================================

cond3 = total_peaks_U1[f'Max_G23']>20
peaks_G23_larger20 = total_peaks_U1[cond3]


# Display peaks >20ppm -------------------------------------------
Ar_items[i] = ax2.scatter(peaks_G23_larger20['Area_mean_G43'],peaks_G23_larger20[f'Area_mean_G23'],color=color_U,marker = 'x')    
Max_items[i] = ax1.scatter(peaks_G23_larger20['Max_G43'],peaks_G23_larger20[f'Max_G23'],color=color_U, marker = 'x') #,label=title


# Rotterdam ====================================================================================================


for spec_vars in [R_vars_aeris,R_vars_G23,R_vars_miro,R_vars_aerodyne]:
   
    spec = spec_vars['spec']
    print(spec)
    title = spec_vars['title']
    name = spec_vars['name']
    titles.append(title)
    titles2.append(title)
    cond1 = total_peaks_R[f'Max_{spec}'].notna()
    cond2 = total_peaks_R['Area_mean_G43'].notna()
    
    # FINAL: Fit to all G23 + separate fit to only <20ppm
    if spec == 'G23':
        
        # 1. Fit only to data which have a maximum <20ppm ===================
        cond3 = total_peaks_R[f'Max_G23']<20
        plotpeaks = total_peaks_R[cond1 & cond2 & cond3].copy()
        
        # Area -----------------------------------------------------
        x = plotpeaks['Area_mean_G43']
        Ar_items[i] = ax2.scatter(x,plotpeaks[f'Area_mean_{spec}'],color=dict_color_instr[name])    
        # ax_1.scatter(outliers['Area_G43_mean'], outliers[f'Area_mean_{spec}'],facecolors='none',edgecolors='grey', label= 'outlier')
    
        popt, pcov  = curve_fit(scalar_multiplication, x, plotpeaks[f'Area_mean_{spec}'],p0=1.0,maxfev=200000)
        a1      = round(popt[0],3)
        #a_          = round(a,3)
        r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Area_mean_{spec}'])
        correlation = round(plotpeaks['Area_mean_G43'].corr(plotpeaks[f'Area_mean_{spec}']),2)
        
        Ar_items[i+1], = ax2.plot(x2,popt[0]*x2,color=dict_color_instr[name],label=f'{title}: y = {a1}' r' $\cdot$ x ' f'($R^2$ = {r2})') #, rho = {correlation}
        
        # Max -----------------------------------------------------
        x = plotpeaks['Max_G43']
        Max_items[i] = ax1.scatter(x,plotpeaks[f'Max_{spec}'],color=dict_color_instr[name]) #,label=title
      
        popt, pcov  = curve_fit(scalar_multiplication, x, plotpeaks[f'Max_{spec}'],p0=1.0,maxfev=20000)
        a1      = round(popt[0],3)
        r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Max_{spec}'])
        correlation = round(plotpeaks['Max_G43'].corr(plotpeaks[f'Max_{spec}']),2)
        
        Max_items[i+1], = ax1.plot(x1,popt[0]*x1,color=dict_color_instr[name],label=f'{title}: y = {a1}' r' $\cdot$ x ' f'($R^2$ = {r2})')
    
        # 2. Fit to all G2301 data ==========================================
        
        cond3 = total_peaks_R[f'Max_G23']>20
        peaks_G23_larger20 = total_peaks_R[cond3].copy()
        plotpeaks = total_peaks_R[cond1 & cond2].copy()
        
        # Display peaks >20ppm -------------------------------------------
        Ar_items[i] = ax2.scatter(peaks_G23_larger20['Area_mean_G43'],peaks_G23_larger20[f'Area_mean_G23'],color=dict_color_instr[name],marker = 'x')    
        Max_items[i] = ax1.scatter(peaks_G23_larger20['Max_G43'],peaks_G23_larger20[f'Max_G23'],color=dict_color_instr[name], marker = 'x') #,label=title
        
    
        i += 2
        
    else:
        plotpeaks = total_peaks_R[cond1 & cond2].copy()
        
        # Area -----------------------------------------------------
        x = plotpeaks['Area_mean_G43']
        Ar_items[i] = ax2.scatter(x,plotpeaks[f'Area_mean_{spec}'],color=dict_color_instr[name])    
        # ax_1.scatter(outliers['Area_G43_mean'], outliers[f'Area_mean_{spec}'],facecolors='none',edgecolors='grey', label= 'outlier')
    
        popt, pcov  = curve_fit(scalar_multiplication, x, plotpeaks[f'Area_mean_{spec}'],p0=1.0,maxfev=200000)
        a1      = round(popt[0],3)
        #a_          = round(a,3)
        r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Area_mean_{spec}'])
        correlation = round(plotpeaks['Area_mean_G43'].corr(plotpeaks[f'Area_mean_{spec}']),2)
        
        Ar_items[i+1], = ax2.plot(x2,popt[0]*x2,color=dict_color_instr[name],label=f'{title}: y = {a1}' r' $\cdot$ x ' f'($R^2$ = {r2})') #, rho = {correlation}
        
        # Max -----------------------------------------------------
        x = plotpeaks['Max_G43']
        Max_items[i] = ax1.scatter(x,plotpeaks[f'Max_{spec}'],color=dict_color_instr[name]) #,label=title
      
        popt, pcov  = curve_fit(scalar_multiplication, x, plotpeaks[f'Max_{spec}'],p0=1.0,maxfev=20000)
        a1      = round(popt[0],3)
        r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Max_{spec}'])
        correlation = round(plotpeaks['Max_G43'].corr(plotpeaks[f'Max_{spec}']),2)
        
        Max_items[i+1], = ax1.plot(x1,popt[0]*x1,color=dict_color_instr[name],label=f'{title}: y = {a1}' r' $\cdot$ x ' f'($R^2$ = {r2})')
    
        i += 2
        
# ================================================================================================================


line1, = ax2.plot(x2,scalar_multiplication(x2,1),"k:")
titles.append('1:1')
line2, = ax1.plot(x1,scalar_multiplication(x1,1),"k:")
# ax1.grid(True)
# ax2.grid(True)

ax2.set_ylabel(f'Area other instruments [ppm*m]',fontsize=20)
ax2.set_xlabel('Area G4302 [ppm*m]',fontsize=20)
ax2.set_title('Area',fontweight='bold',fontsize=24)
ax1.set_ylabel(f'Max other instruments [ppm]',fontsize=20)
ax1.set_xlabel('Max G4302 [ppm]',fontsize=20)
ax1.set_title('Max',fontweight='bold',fontsize=24)

#plt.suptitle('Instrument Comparison Plot',fontsize=22)

ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)
ax2.tick_params(axis='x', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)

plt.subplots_adjust(wspace=0.4)

# Create a legend for the first subplot (ax1)
legend_ax1 = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), fancybox=True, shadow=True, ncol=1,fontsize=16)
# Create a legend for the second subplot (ax2)
legend_ax2 = ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), fancybox=True, shadow=True, ncol=1,fontsize=16)

# Zoom 1
ax1.set_xlim(0,60)
ax1.set_ylim(0,60)
ax2.set_xlim(0,250)
ax2.set_ylim(0,250)

#save_plots = True
if save_plots:
    fig.savefig(path_fig_plot1+f'RandU_mastercomparison_zoom.pdf',bbox_inches='tight') #_Zoom
    fig.savefig(path_fig_plot1+f'RandU_mastercomparison_zoom.png',bbox_inches='tight') #_Zoom
    fig.savefig(path_fig_plot1+f'RandU_mastercomparison_zoom.svg',bbox_inches='tight') #_Zoom

plt.show()










#%%% FINAL FIGURE: L: x=G23 - d2+d3 - 1Plot

# exclude outliers for which Max_LGR>20ppm
# reasoning: since measurement range of G2301 only 0-20ppm -> unexcpected behaviour for peak max > 20 ppm

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,12))

# Full range
x1 = np.arange(0,25)
x2 = np.arange(0,300)

Ar_items = [None]*10
Max_items = [None]*10
titles = []
titles2 = []
i = 0

#####################################################################################################
# Day 2 
#####################################################################################################
spec = 'LGR'
# name = 'G23_L'
title = 'uMEA(G2301)_L'
titles.append(title)
titles2.append(title)

#-------
# 1. Fit only to data which have a maximum <20ppm =================================================================

#cond3 = ~total_peaks_L1d2['Peak'].isin([1,2,92, 93, 108,111])
cond1 = total_peaks_L1d2['Distance_to_source']<75
cond2 = total_peaks_L1d2[f'Max_{spec}']<20
plotpeaks = total_peaks_L1d2[cond2].copy()

# Area -----------------------------------------------------
x = plotpeaks['Area_mean_G23']
Ar_items[i] = ax2.scatter(x,plotpeaks[f'Area_mean_{spec}'],color=dict_color_instr[spec])    
# ax_1.scatter(outliers['Area_G43_mean'], outliers[f'Area_LGR_mean'],facecolors='none',edgecolors='grey', label= 'outlier')

popt, pcov  = curve_fit(scalar_multiplication, x, plotpeaks[f'Area_mean_{spec}'],p0=1.0,maxfev=200000)
a1      = round(popt[0],3)
#a_          = round(a,3)
r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Area_mean_{spec}'])
correlation = round(plotpeaks['Area_mean_G23'].corr(plotpeaks[f'Area_mean_{spec}']),2)

# Ar_items[i+1], = ax2.plot(x2,popt[0]*x2,color=dict_color_instr[spec],label=f'{title} (excl. outlier): y = {a1} $\cdot$ x ($R^2$ = {r2})') #$\rho$
Ar_items[i+1], = ax2.plot(x2,popt[0]*x2,color=dict_color_instr[spec],label=f'{title}: y = {a1} $\cdot$ x ($R^2$ = {r2})') #$\rho$

# Max -----------------------------------------------------
x = plotpeaks['Max_G23']
Max_items[i] = ax1.scatter(x,plotpeaks[f'Max_{spec}'],color=dict_color_instr[spec]) #,label=title

popt, pcov  = curve_fit(scalar_multiplication, x, plotpeaks[f'Max_{spec}'],p0=1.0,maxfev=20000)
a1      = round(popt[0],3)
r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Max_{spec}'])
correlation = round(plotpeaks['Max_G23'].corr(plotpeaks[f'Max_{spec}']),2)

# Max_items[i+1], = ax1.plot(x1,popt[0]*x1,color=dict_color_instr[spec],label=f'{title} (excl. outlier): y = {a1} $\cdot$ x ($R^2$ = {r2})')
Max_items[i+1], = ax1.plot(x1,popt[0]*x1,color=dict_color_instr[spec],label=f'{title}: y = {a1} $\cdot$ x ($R^2$ = {r2})')



# # 2. Fit to all G2301 data ====================================================================================================
# #cond3 = total_peaks_L1d2['Peak'].isin([1,2,92, 93, 108,111])
# cond2 = total_peaks_L1d2[f'Max_{spec}']>20
# peaks_G23_larger20 = total_peaks_L1d2[cond2]
# plotpeaks_all = total_peaks_L1d2.copy()

# Display peaks >20ppm -------------------------------------------
Ar_items[i] = ax2.scatter(peaks_G23_larger20['Area_mean_G23'],peaks_G23_larger20[f'Area_mean_{spec}'],color=color_U,marker = 'x')    
Max_items[i] = ax1.scatter(peaks_G23_larger20['Max_G23'],peaks_G23_larger20[f'Max_{spec}'],color=color_U, marker = 'x') #,label=title

# x = plotpeaks_all['Area_mean_G23']
# #Ar_items[i] = ax2.scatter(x,plotpeaks[f'Area_LGR_mean'],color=color_U)    
# # ax_1.scatter(outliers['Area_G43_mean'], outliers[f'Area_LGR_mean'],facecolors='none',edgecolors='grey', label= 'outlier')

# popt, pcov  = curve_fit(scalar_multiplication, x, plotpeaks_all[f'Area_mean_{spec}'],p0=1.0,maxfev=200000)
# a1      = round(popt[0],3)
# #a_          = round(a,3)
# r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Area_mean_{spec}'])
# correlation = round(plotpeaks['Area_mean_G23'].corr(plotpeaks[f'Area_mean_{spec}']),2)

# Ar_items[i+1], = ax2.plot(x2,popt[0]*x2,color=dict_color_instr[spec],linestyle='dashed',label=f'{title} (all): y = {a1} $\cdot$ x ($R^2$ = {r2})') #$\rho$

# # Max -----------------------------------------------------
# x = plotpeaks_all['Max_G23']
# #Max_items[i] = ax1.scatter(x,plotpeaks[f'Max_LGR'],color=color_U) #,label=title

# popt, pcov  = curve_fit(scalar_multiplication, x, plotpeaks_all[f'Max_{spec}'],p0=1.0,maxfev=20000)
# a1      = round(popt[0],3)
# r2 = r_squared(scalar_multiplication, popt, x, plotpeaks_all[f'Max_{spec}'])
# correlation = round(plotpeaks_all['Max_G23'].corr(plotpeaks_all[f'Max_{spec}']),2)

# Max_items[i+1], = ax1.plot(x1,popt[0]*x1,color=dict_color_instr[spec],linestyle='dashed',label=f'{title} (all): y = {a1} $\cdot$ x ($R^2$ = {r2})')




#####################################################################################################
# Day 3 
#####################################################################################################
spec = 'Licor'
# name = 'G23_L'
title = 'LI-7810(G2301)_L'
titles.append(title)
titles2.append(title)

#-------
# 1. Fit only to data which have a maximum <20ppm =================================================================

#cond3 = ~total_peaks_L1d2['Peak'].isin([1,2,92, 93, 108,111])
cond1 = total_peaks_L1d3['Distance_to_source']<75
cond2 = total_peaks_L1d3[f'Max_{spec}']<20
plotpeaks = total_peaks_L1d3[cond2].copy()

# Area -----------------------------------------------------
x = plotpeaks['Area_mean_G23']
Ar_items[i] = ax2.scatter(x,plotpeaks[f'Area_mean_{spec}'],color=dict_color_instr[spec])    
# ax_1.scatter(outliers['Area_G43_mean'], outliers[f'Area_LGR_mean'],facecolors='none',edgecolors='grey', label= 'outlier')

popt, pcov  = curve_fit(scalar_multiplication, x, plotpeaks[f'Area_mean_{spec}'],p0=1.0,maxfev=200000)
a1      = round(popt[0],3)
#a_          = round(a,3)
r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Area_mean_{spec}'])
correlation = round(plotpeaks['Area_mean_G23'].corr(plotpeaks[f'Area_mean_{spec}']),2)

Ar_items[i+1], = ax2.plot(x2,popt[0]*x2,color=dict_color_instr[spec],label=f'{title}: y = {a1} $\cdot$ x ($R^2$ = {r2})') #$\rho$

# Max -----------------------------------------------------
x = plotpeaks['Max_G23']
Max_items[i] = ax1.scatter(x,plotpeaks[f'Max_{spec}'],color=dict_color_instr[spec]) #,label=title

popt, pcov  = curve_fit(scalar_multiplication, x, plotpeaks[f'Max_{spec}'],p0=1.0,maxfev=20000)
a1      = round(popt[0],3)
r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Max_{spec}'])
correlation = round(plotpeaks['Max_G23'].corr(plotpeaks[f'Max_{spec}']),2)

Max_items[i+1], = ax1.plot(x1,popt[0]*x1,color=dict_color_instr[spec],label=f'{title}: y = {a1} $\cdot$ x ($R^2$ = {r2})')







#####################################################################################################
#####################################################################################################

line1, = ax2.plot(x2,scalar_multiplication(x2,1),"k:")
titles.append('1:1')
line2, = ax1.plot(x1,scalar_multiplication(x1,1),"k:")
# ax1.grid(True)
# ax2.grid(True)

ax2.set_ylabel(r'$\mathrm{\left[CH_4\right]_{area}}$ other instrument [ppm*m]',fontsize=20)
ax2.set_xlabel(r'$\mathrm{\left[CH_4\right]_{area}}$ G2301 [ppm*m]',fontsize=20)
# ax2.set_title('Area',fontweight='bold',fontsize=24)
ax1.set_ylabel(r'$\mathrm{\left[CH_4\right]_{max}}$ other instrument [ppm]',fontsize=20)
ax1.set_xlabel(r'$\mathrm{\left[CH_4\right]_{max}}$ G2301 [ppm]',fontsize=20)
# ax1.set_title('Max',fontweight='bold',fontsize=24)

#plt.suptitle('Instrument Comparison Plot',fontsize=22)

ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)
ax2.tick_params(axis='x', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)

plt.subplots_adjust(wspace=0.4)

# Create a legend for the first subplot (ax1)
legend_ax1 = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=1,fontsize=16)
# Create a legend for the second subplot (ax2)
legend_ax2 = ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=1,fontsize=16)


save_plots = False
if save_plots:
    fig.savefig(path_fig_plot1+f'Ld23_mastercomparison_xG23.pdf',bbox_inches='tight') #_Zoom
    fig.savefig(path_fig_plot1+f'Ld23_mastercomparison_xG23.png',bbox_inches='tight') #_Zoom
    fig.savefig(path_fig_plot1+f'Ld23_mastercomparison_xG23.svg',bbox_inches='tight') #_Zoom

plt.show()

#%%% L: x=G23 - d2+d3 - 2Plots separately

# --- First Plot: Max CH4 Concentration Comparison ---
fig1, ax1 = plt.subplots(figsize=(10, 12))

x1 = np.arange(0, 25)

cond2 = total_peaks_L1d2['Max_LGR']>20
peaks_G23_larger20 = total_peaks_L1d2[cond2]

# Day 2 - Max CH4 Concentration
spec = 'LGR'
title = 'uMEA(G2301)_L'
cond2 = total_peaks_L1d2[f'Max_{spec}'] < 20
plotpeaks = total_peaks_L1d2[cond2].copy()

# Scatter and fit line for Day 2
x = plotpeaks['Max_G23']
ax1.scatter(x, plotpeaks[f'Max_{spec}'], color=dict_color_instr[spec])
popt, _ = curve_fit(scalar_multiplication, x, plotpeaks[f'Max_{spec}'], p0=1.0, maxfev=20000)
a1 = round(popt[0], 3)
r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Max_{spec}'])
ax1.plot(x1, popt[0] * x1, color=dict_color_instr[spec], label=f'{title}: y = {a1} $\cdot$ x ($R^2$ = {r2})')

# Display peaks > 20 ppm
ax1.scatter(peaks_G23_larger20['Max_G23'], peaks_G23_larger20[f'Max_{spec}'], color=dict_color_instr[spec], marker='x')

# Day 3 - Max CH4 Concentration
spec = 'Licor'
title = 'LI-7810(G2301)_L'
cond2 = total_peaks_L1d3[f'Max_{spec}'] < 20
plotpeaks = total_peaks_L1d3[cond2].copy()
x = plotpeaks['Max_G23']
ax1.scatter(x, plotpeaks[f'Max_{spec}'], color=dict_color_instr[spec])
popt, _ = curve_fit(scalar_multiplication, x, plotpeaks[f'Max_{spec}'], p0=1.0, maxfev=20000)
a1 = round(popt[0], 3)
r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Max_{spec}'])
ax1.plot(x1, popt[0] * x1, color=dict_color_instr[spec], label=f'{title}: y = {a1} $\cdot$ x ($R^2$ = {r2})')

line1, = ax1.plot(x1,scalar_multiplication(x1,1),"k:",linewidth=2)

# Add labels and legend
ax1.set_ylabel(r'$\mathrm{\left[CH_4\right]_{max}}$ other instrument [ppm]', fontsize=22)
ax1.set_xlabel(r'$\mathrm{\left[CH_4\right]_{max}}$ G2301 [ppm]', fontsize=22)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, fontsize=18)
ax1.tick_params(axis='both', labelsize=22)
# fig1.savefig(path_fig_plot1 + 'Ld235_mastercomparison_xG23_Max.png', bbox_inches='tight')
# fig1.savefig(path_fig_plot1 + 'Ld235_mastercomparison_xG23_Max.pdf', dpi=1000,  bbox_inches='tight')
# fig1.savefig(path_fig_plot1 + 'Ld235_mastercomparison_xG23_Max.svg', bbox_inches='tight')

# --- Second Plot: CH4 Area Comparison ---
fig2, ax2 = plt.subplots(figsize=(10, 12))

x2 = np.arange(0, 300)

# Day 2 - CH4 Area
spec = 'LGR'
title = 'uMEA(G2301)_L'
plotpeaks = total_peaks_L1d2[total_peaks_L1d2[f'Max_{spec}'] < 20].copy()
x = plotpeaks['Area_mean_G23']
ax2.scatter(x, plotpeaks[f'Area_mean_{spec}'], color=dict_color_instr[spec])
popt, _ = curve_fit(scalar_multiplication, x, plotpeaks[f'Area_mean_{spec}'], p0=1.0, maxfev=200000)
a1 = round(popt[0], 3)
r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Area_mean_{spec}'])
ax2.plot(x2, popt[0] * x2, color=dict_color_instr[spec], label=f'{title}: y = {a1} $\cdot$ x ($R^2$ = {r2})')

# Display peaks > 20 ppm
ax2.scatter(peaks_G23_larger20['Area_mean_G23'], peaks_G23_larger20[f'Area_mean_{spec}'], color=dict_color_instr[spec], marker='x')


# Day 3 - CH4 Area
spec = 'Licor'
title = 'LI-7810(G2301)_L'
plotpeaks = total_peaks_L1d3[total_peaks_L1d3[f'Max_{spec}'] < 20].copy()
x = plotpeaks['Area_mean_G23']
ax2.scatter(x, plotpeaks[f'Area_mean_{spec}'], color=dict_color_instr[spec])
popt, _ = curve_fit(scalar_multiplication, x, plotpeaks[f'Area_mean_{spec}'], p0=1.0, maxfev=200000)
a1 = round(popt[0], 3)
r2 = r_squared(scalar_multiplication, popt, x, plotpeaks[f'Area_mean_{spec}'])
ax2.plot(x2, popt[0] * x2, color=dict_color_instr[spec], label=f'{title}: y = {a1} $\cdot$ x ($R^2$ = {r2})')

line2, = ax2.plot(x2,scalar_multiplication(x2,1),"k:",linewidth=2)
titles.append('1:1')

# Add labels and legend
ax2.set_ylabel(r'$\mathrm{\left[CH_4\right]_{area}}$ other instrument [ppm*m]', fontsize=22)
ax2.set_xlabel(r'$\mathrm{\left[CH_4\right]_{area}}$ G2301 [ppm*m]', fontsize=22)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, fontsize=18)
ax2.tick_params(axis='both', labelsize=22)
# fig2.savefig(path_fig_plot1 + 'Ld23_mastercomparison_xG23_Area.png', bbox_inches='tight')
# fig2.savefig(path_fig_plot1 + 'Ld23_mastercomparison_xG23_Area.pdf', dpi=1000, bbox_inches='tight')
# fig2.savefig(path_fig_plot1 + 'Ld23_mastercomparison_xG23_Area.svg', bbox_inches='tight')

plt.show()









#%% P2: Lin. Reg.

'''
Correlation of the natural logarithm of (a) the peak maximum enhancement and (b)
spatial peak area with the natural logarithm of the release rates for all 
controlled release experiments reported in this manuscript (except London I Day2).
Black markers indicate mean values per release rate and city, unfilled markers 
indicate potential outliers. The second x-axis indicates release rates deployed, 
the red lines are linear regressions to all data. The Weller equation (grey line 
in (a)) is displayed as a comparison, and the light (dark) gray area indicates 
peaks below 110% (102%) of background level.
    '''

#%%% FINAL FIGURE
# RU2T2L2L2 - extra marker

confidence_interval = False # set to TRUE to add confidence intervals, takes a bit of time
save_plots=False


fig, ax = plt.subplots(1,2, figsize=(18,10))

ax1 = ax[0]
ax2 = ax[1]
ax1.grid(True)
ax2.grid(True)

# Add Shadings to visualize peak threshold -----------------------------------------

x = np.array([-2.5,5.2])
ax1.fill_between(x,-4,np.log(0.2),color='lightgray',alpha=0.4)

x = np.array([-2.5,5.2])
ax1.fill_between(x,-4,np.log(0.04),color='lightgray',alpha=0.9)

# ----------------------------------------------------------------------------


# Create empty lists to collect data for legends
legend_handles1 = [] # inner legend, displaying cities
legend_handles2 = []
legend_handles1b = [] # outer legend, displaying lin. reg. fit
legend_handles2b = []


all_area_R = []
all_max_R = []
all_area_U1 = []
all_max_U1 = []
all_area_U2 = []
all_max_U2 = []
all_area_T1b = []
all_max_T1b = []
all_area_T1c = []
all_max_T1c = []
all_area_T2c = []
all_max_T2c = []
all_area_L1d2 = []
all_max_L1d2 = []
all_area_L1d5 = []
all_max_L1d5 = []
all_area_L2d1 = []
all_max_L2d1 = []
all_area_L2d2 = []
all_max_L2d2 = []


# Create empty lists to collect data for legends
legend_handles1 = []
legend_handles2 = []


log_peaks_R = total_peaks_R.copy(deep=True)
log_peaks_R = log_peaks_R[log_peaks_R['Loc'] == 1]
log_peaks_U1 = total_peaks_U1.copy(deep=True)
log_peaks_U2 = total_peaks_U2.copy(deep=True)
log_peaks_T1b = total_peaks_T1b.copy(deep=True)
log_peaks_T1c = total_peaks_T1c.copy(deep=True)
log_peaks_T2c = total_peaks_T2c.copy(deep=True)
log_peaks_L1d2 = total_peaks_L1d2.copy(deep=True)
log_peaks_L1d5 = total_peaks_L1d5.copy(deep=True)
log_peaks_L2d1 = total_peaks_L2d1.copy(deep=True)
log_peaks_L2d2 = total_peaks_L2d2.copy(deep=True)

max_dist = 75
log_peaks_R = log_peaks_R[((log_peaks_R['Release_rate'] != 0)) & (log_peaks_R['Distance_to_source'] < max_dist)] 
log_peaks_U1 = log_peaks_U1[((log_peaks_U1['Release_rate'] != 0)) & (log_peaks_U1['Distance_to_source'] < max_dist)] 
log_peaks_U2 = log_peaks_U2[((log_peaks_U2['Release_rate'] != 0)) & (log_peaks_U2['Distance_to_source'] < max_dist)]  
log_peaks_T1b = log_peaks_T1b[((log_peaks_T1b['Release_rate'] != 0)) & (log_peaks_T1b['Distance_to_source'] < max_dist)] 
log_peaks_T1c = log_peaks_T1c[((log_peaks_T1c['Release_rate'] != 0)) & (log_peaks_T1c['Distance_to_source'] < max_dist)] 
log_peaks_T2c = log_peaks_T2c[((log_peaks_T2c['Release_rate'] != 0)) & (log_peaks_T2c['Distance_to_source'] < max_dist)] 
log_peaks_L1d2 = log_peaks_L1d2[((log_peaks_L1d2['Release_rate'] != 0) & (log_peaks_L1d2['Distance_to_source'] < max_dist))]
log_peaks_L1d5 = log_peaks_L1d5[((log_peaks_L1d5['Release_rate'] != 0) & (log_peaks_L1d5['Distance_to_source'] < max_dist))]
log_peaks_L2d1 = log_peaks_L2d1[((log_peaks_L2d1['Release_rate'] != 0) & (log_peaks_L2d1['Distance_to_source'] < max_dist))]
log_peaks_L2d2 = log_peaks_L2d2[((log_peaks_L2d2['Release_rate'] != 0) & (log_peaks_L2d2['Distance_to_source'] < max_dist))] 


legend_handles_notincluded = []

x_R,all_max_R,all_area_R,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_R,all_max_R,all_area_R,ax,legend_handles1,legend_handles2,'.',65,False, R_vars_aeris,R_vars_G43,R_vars_G23,R_vars_miro,R_vars_aerodyne)

x_U1,all_max_U1,all_area_U1,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_U1,all_max_U1,all_area_U1,ax,legend_handles1,legend_handles2,'v',47,False, U1_vars_G43,U1_vars_G23)

x_U2,all_max_U2,all_area_U2,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_U2,all_max_U2,all_area_U2,ax,legend_handles1,legend_handles2,'d',47,False, U2_vars_aeris,U2_vars_G23)
  
x_L1d2,all_max_L1d2,all_area_L1d2,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_L1d2,all_max_L1d2,all_area_L1d2,ax,legend_handles1,legend_handles2,'p',50,True,L1_vars_d2_LGR,L1_vars_d2_G23) #London Day2
   
x_L1d5,all_max_L1d5,all_area_L1d5,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_L1d5,all_max_L1d5,all_area_L1d5,ax,legend_handles1,legend_handles2,'*',80,True,L1_vars_d5_G23) #London Day3

x_L2d1,all_max_L2d1,all_area_L2d1,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_L2d1,all_max_L2d1,all_area_L2d1,ax,legend_handles1,legend_handles2,'^',47,True,L2_vars_d1_Licor)

x_L2d2,all_max_L2d2,all_area_L2d2,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_L2d2,all_max_L2d2,all_area_L2d2,ax,legend_handles1,legend_handles2,'<',47,True,L2_vars_d2_Licor)

x_T1,all_max_T1b,all_area_T1b,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_T1b,all_max_T1b,all_area_T1b,ax,legend_handles1,legend_handles2,'D',55,True,T_vars_1b_LGR)

x_T1,all_max_T1c,all_area_T1c,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_T1c,all_max_T1c,all_area_T1c,ax,legend_handles1,legend_handles2,'>',55,True,T_vars_1c_G24) # marker size none

x_T2,all_max_T2c,all_area_T2c,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_T2c,all_max_T2c,all_area_T2c,ax,legend_handles1,legend_handles2,'X',55,True,T_vars_2c_G24) # marker size none


means_area_R, means_max_R, median_area_R, median_max_R = mean_and_median_log(all_area_R,all_max_R)
means_area_U1, means_max_U1, median_area_U1, median_max_U1 = mean_and_median_log(all_area_U1,all_max_U1)
means_area_U2, means_max_U2, median_area_U2, median_max_U2 = mean_and_median_log(all_area_U2,all_max_U2)
means_area_L1d2, means_max_L1d2, median_area_L1d2, median_max_L1d2 = mean_and_median_log(all_area_L1d2,all_max_L1d2)
means_area_L1d5, means_max_L1d5, median_area_L1d5, median_max_L1d5 = mean_and_median_log(all_area_L1d5,all_max_L1d5)
means_area_T1b, means_max_T1b, median_area_T1b, median_max_T1b = mean_and_median_log(all_area_T1b,all_max_T1b)
means_area_T1c, means_max_T1c, median_area_T1c, median_max_T1c = mean_and_median_log(all_area_T1c,all_max_T1c)
means_area_T2c, means_max_T2c, median_area_T2c, median_max_T2c = mean_and_median_log(all_area_T2c,all_max_T2c)

means_area_L2d1, means_max_L2d1, median_area_L2d1, median_max_L2d1 = mean_and_median_log(all_area_L2d1,all_max_L2d1)
means_area_L2d2, means_max_L2d2, median_area_L2d2, median_max_L2d2 = mean_and_median_log(all_area_L2d2,all_max_L2d2)



# Different marker for mean
ax1.scatter(means_max_R.index,means_max_R.Max,marker = '.',s=110, c= 'lightgrey', edgecolor='black') #,label = 'mean' #484848
ax1.scatter(means_max_U1.index,means_max_U1.Max,marker = 'v',s=70, c= 'lightgrey', edgecolor='black')
ax1.scatter(means_max_U2.index,means_max_U2.Max,marker = 'd',s=70, c= 'lightgrey', edgecolor='black')
ax1.scatter(means_max_L1d2.index,means_max_L1d2.Max,marker = 'p',s=100, c= 'lightgrey', edgecolor='black')
ax1.scatter(means_max_L1d5.index,means_max_L1d5.Max,marker = '*',s=120,c= 'lightgrey', edgecolor='black')
ax1.scatter(means_max_T1b.index,means_max_T1b.Max,marker = 'D',s=40,c= 'lightgrey', edgecolor='black')
ax1.scatter(means_max_T1c.index,means_max_T1c.Max,marker = '>',s=45,c= 'lightgrey', edgecolor='black')
ax1.scatter(means_max_T2c.index,means_max_T2c.Max,marker = 'X',s=60,c= 'lightgrey', edgecolor='black')
ax1.scatter(means_max_L2d1.index,means_max_L2d1.Max,marker = '^',s=100,c= 'lightgrey', edgecolor='black')
ax1.scatter(means_max_L2d2.index,means_max_L2d2.Max,marker = '<',s=100,c= 'lightgrey', edgecolor='black')


ax2.scatter(means_area_R.index,means_area_R.Area,marker = '.',s=110, c= 'lightgrey', edgecolor='black') #,label = 'mean'
ax2.scatter(means_area_U1.index,means_area_U1.Area,marker = 'v',s=70, c= 'lightgrey', edgecolor='black')
ax2.scatter(means_area_U2.index,means_area_U2.Area,marker = 'd',s=70, c= 'lightgrey', edgecolor='black')
ax2.scatter(means_area_L1d2.index,means_area_L1d2.Area,marker = 'p',s=100, c= 'lightgrey', edgecolor='black')
ax2.scatter(means_area_L1d5.index,means_area_L1d5.Area,marker = '*',s=120,c= 'lightgrey', edgecolor='black')
ax2.scatter(means_area_T1b.index,means_area_T1b.Area,marker = 'D',s=40,c= 'lightgrey', edgecolor='black')
ax2.scatter(means_area_T1c.index,means_area_T1c.Area,marker = '>',s=45,c= 'lightgrey', edgecolor='black')
ax2.scatter(means_area_T2c.index,means_area_T2c.Area,marker = 'X',s=60,c= 'lightgrey', edgecolor='black')
ax2.scatter(means_area_L2d1.index,means_area_L2d1.Area,marker = '^',s=100,c= 'lightgrey', edgecolor='black')
ax2.scatter(means_area_L2d2.index,means_area_L2d2.Area,marker = '<',s=100,c= 'lightgrey', edgecolor='black')

legend_handles1.append(ax1.scatter([], [], marker='.',s=110, c= 'lightgrey', edgecolor='black', label='mean'))
legend_handles2.append(ax2.scatter([], [], marker='.',s=110, c= 'lightgrey', edgecolor='black', label='mean'))
  


plt.subplots_adjust(wspace=0.35)

# 2nd axes --------------------------------------------------------------------
# Add 2nd x and y axis to display non logarithmis calues for peak max and area
# and actual release rates
# Plot left (PEAK MAX), y axis
ax1y = ax1.twinx()
ticks_y = [-2,0,2,4,6]
labels_y = np.round(np.exp(ticks_y),1)
ax1y.set_yticks(ticks_y) # Set the ticks and labels for the second y-axis
ax1y.set_yticklabels(labels_y)
ax1y.set_yticklabels([f"{int(label)}" if label >= 1 else f"{label:.1f}" for label in labels_y])
ax1y.set_ylabel(r'$\mathrm{\left[CH_4\right]_{max}}$ [ppm]',fontsize=16)
ax1.set_ylim(-4,7)
ax1y.set_ylim(-4,7)

# Plot left (PEAK MAX), x axis
ax1x = ax1.twiny()
# Manually position the second x-axis below the original x-axis
ax1x.spines['bottom'].set_position(('outward', 75))  # You can adjust the position value (40) to control the offset
# ticks_x = [0, 1, 2,3]
# labels_x = np.round(np.exp(ticks_x),2)
# Display all release rates actually used on the 2nd non log x axis
labels_x = [0.15,0.3,0.5,1,2.2,3,5,10,15,20,40,80,120] # actual release rates in L/min
ticks_x = np.round(np.log(labels_x),2) # transform to log to plot it at right position on the axis
ax1x.set_xticks(ticks_x) # Set the ticks and labels for the second y-axis
ax1x.set_xticklabels(labels_x)
# Set the location of the ticks to be at the bottom
ax1x.xaxis.set_ticks_position('bottom')
ax1x.set_xlabel(r'Release Rate [$\mathrm{Lmin^{-1}}$]',fontsize=16) #,labelpad=-750
ax1x.xaxis.set_label_coords(0.5, -0.208)
ax1.set_xlim(-2.5,5.2) #-0.5,3.5
ax1x.set_xlim(-2.5,5.2)

# Plot right (AREA), y axis
ax2y = ax2.twinx()
ticks_y = [0,1,2,3,4,5,6,7,8]
labels_y = np.round(np.exp(ticks_y),0)# Set the ticks and labels for the second y-axis
ax2y.set_yticks(ticks_y)
ax2y.set_yticklabels(labels_y)
ax2y.set_yticklabels([f"{int(label)}" for label in labels_y]) # Show no digits (round to integer)
ax2y.set_ylabel(r'$\mathrm{\left[CH_4\right]_{area}}$ [ppm*m]',fontsize=16)
ax2.set_ylim(-1,8)
ax2y.set_ylim(-1,8)

# Plot right (AREA), x axis
ax2x = ax2.twiny()
# Manually position the second x-axis below the original x-axis
ax2x.spines['bottom'].set_position(('outward', 75))  # You can adjust the position value (40) to control the offset
# Display all release rates actually used on the 2nd non log x axis
labels_x = [0.15,0.3,0.5,1,2.2,3,5,10,15,20,40,80,120] # actual release rates in L/min
ticks_x = np.round(np.log(labels_x),2) # transform to log to plot it at right position on the axis
ax2x.set_xticks(ticks_x) # Set the ticks and labels for the second y-axis
ax2x.set_xticklabels(labels_x)
# Set the location of the ticks to be at the bottom
ax2x.xaxis.set_ticks_position('bottom')
ax2x.set_xlabel(r'Release Rate [$\mathrm{Lmin^{-1}}$]',fontsize=16)
ax2x.xaxis.set_label_coords(0.5, -0.208)
ax2.set_xlim(-2.5,5.2) #-0.5,3.5
ax2x.set_xlim(-2.5,5.2)


# ----------------------------------------------------------------------------


ax1.tick_params(axis='x', labelsize=18)
ax1.tick_params(axis='y', labelsize=18)
ax2.tick_params(axis='x', labelsize=18)
ax2.tick_params(axis='y', labelsize=18)

ax1x.tick_params(axis='x', labelsize=13)
ax1y.tick_params(axis='y', labelsize=14)
ax2x.tick_params(axis='x', labelsize=13)
ax2y.tick_params(axis='y', labelsize=14)


fig.canvas.draw() # !!! necessary so that ax1.get_xticklabels() in the following access the most recent labels


# =============================================================================


df_all_max_R = pd.concat(all_max_R, ignore_index=True) 
df_all_area_R = pd.concat(all_area_R, ignore_index=True) 
df_all_max_U1 = pd.concat(all_max_U1, ignore_index=True) 
df_all_area_U1 = pd.concat(all_area_U1, ignore_index=True)
df_all_max_U2 = pd.concat(all_max_U2, ignore_index=True) 
df_all_area_U2 = pd.concat(all_area_U2, ignore_index=True) 
df_all_max_L1d2 = pd.concat(all_max_L1d2, ignore_index=True) 
df_all_area_L1d2 = pd.concat(all_area_L1d2, ignore_index=True) 
df_all_max_L1d5 = pd.concat(all_max_L1d5) 
df_all_area_L1d5 = pd.concat(all_area_L1d5) 
df_all_max_T1b = pd.concat(all_max_T1b) 
df_all_area_T1b = pd.concat(all_area_T1b) 
df_all_max_T1c = pd.concat(all_max_T1c) 
df_all_area_T1c = pd.concat(all_area_T1c)  
df_all_max_T2c = pd.concat(all_max_T2c) 
df_all_area_T2c = pd.concat(all_area_T2c)  
df_all_max_L2d1 = pd.concat(all_max_L2d1, ignore_index=True) #
df_all_area_L2d1 = pd.concat(all_area_L2d1, ignore_index=True) #
df_all_max_L2d2 = pd.concat(all_max_L2d2, ignore_index=True) #
df_all_area_L2d2 = pd.concat(all_area_L2d2, ignore_index=True) #


###############################################################################
# Flag peaks as outlier which Maximum is suspiciously low

# Utrecht I
df_outlier_max_U = df_all_max_U1[((df_all_max_U1['Release_rate_log'] == 1.0986122886681098) & (df_all_max_U1['Max'] < -2.3)) |
                 ((df_all_max_U1['Release_rate_log'] == 0.7793248768009977) & (df_all_max_U1['Max'] < -2.4))]
df_index = df_outlier_max_U.index # Get the index of the filtered rows
df_outlier_area_U = df_all_area_U1.loc[df_index] # Select rows from df_area with the same index as in filtered df_max

ax1.scatter(df_outlier_max_U['Release_rate_log'],df_outlier_max_U['Max'],marker='v', s=47,c='white' ,edgecolor=dict_color_city['Utrecht I']) # +spec_vars['day']
ax2.scatter(df_outlier_area_U['Release_rate_log'],df_outlier_area_U['Area'],marker='v', s=47,c='white' ,edgecolor=dict_color_city['Utrecht I']) # +spec_vars['day']

# Utrecht III
df_outlier_max_U3 = df_all_max_U2[((df_all_max_U2['Release_rate_log'] == 2.70805020110221) & (df_all_max_U2['Max'] < -1.7)) |
                 ((df_all_max_U2['Release_rate_log'] == 1.3862943611198906) & (df_all_max_U2['Max'] < -2)) |
                 ((df_all_max_U2['Release_rate_log'] == 0.7884573603642703) & (df_all_max_U2['Max'] < -2.4))]
df_index = df_outlier_max_U3.index # Get the index of the filtered rows
df_outlier_area_U3 = df_all_area_U2.loc[df_index] # Select rows from df_area with the same index as in filtered df_max

ax1.scatter(df_outlier_max_U3['Release_rate_log'],df_outlier_max_U3['Max'],marker='d', s=47,c='white' ,edgecolor=dict_color_city['Utrecht II']) # +spec_vars['day']
ax2.scatter(df_outlier_area_U3['Release_rate_log'],df_outlier_area_U3['Area'],marker='d', s=47,c='white' ,edgecolor=dict_color_city['Utrecht II']) # +spec_vars['day']

# London II Day 1
df_outlier_max_L2d1 = df_all_max_L2d1[((df_all_max_L2d1['Release_rate_log'] == 2.363680192353857) & (df_all_max_L2d1['Max'] < -1.8)) |
                 ((df_all_max_L2d1['Release_rate_log'] == 1.7298840655099674) & (df_all_max_L2d1['Max'] < -2))]
df_index = df_outlier_max_L2d1.index # Get the index of the filtered rows
df_outlier_area_L2d1 = df_all_area_L2d1.loc[df_index] # Select rows from df_area with the same index as in filtered df_max

ax1.scatter(df_outlier_max_L2d1['Release_rate_log'],df_outlier_max_L2d1['Max'],marker='^', s=47,c='white' ,edgecolor=dict_color_city['London IIDay1']) # +spec_vars['day']
ax2.scatter(df_outlier_area_L2d1['Release_rate_log'],df_outlier_area_L2d1['Area'],marker='^', s=47,c='white' ,edgecolor=dict_color_city['London IIDay1']) # +spec_vars['day']

# Toronto Day 1 bike
df_outlier_max_T1b = df_all_max_T1b[((df_all_max_T1b['Release_rate_log'] == 2.2925347571405443) & (df_all_max_T1b['Max'] < -1.9)) |
                 ((df_all_max_T1b['Release_rate_log'] == 0.9162907318741552) & (df_all_max_T1b['Max'] < -2.3))]
df_index = df_outlier_max_T1b.index # Get the index of the filtered rows
df_outlier_area_T1b = df_all_area_T1b.loc[df_index] # Select rows from df_area with the same index as in filtered df_max

ax1.scatter(df_outlier_max_T1b['Release_rate_log'],df_outlier_max_T1b['Max'],marker='D', s=47,c='white' ,edgecolor=dict_color_city['TorontoDay1-bike']) 
ax2.scatter(df_outlier_area_T1b['Release_rate_log'],df_outlier_area_T1b['Area'],marker='D', s=47,c='white' ,edgecolor=dict_color_city['TorontoDay1-bike']) 

# Toronto Day 1 car
df_outlier_max_T1c = df_all_max_T1c[((df_all_max_T1c['Release_rate_log'] == 0.9162907318741552) & (df_all_max_T1c['Max'] < -2.3))]
df_index = df_outlier_max_T1c.index # Get the index of the filtered rows
df_outlier_area_T1c = df_all_area_T1c.loc[df_index] # Select rows from df_area with the same index as in filtered df_max

ax1.scatter(df_outlier_max_T1c['Release_rate_log'],df_outlier_max_T1c['Max'],marker='>', s=47,c='white' ,edgecolor=dict_color_city['TorontoDay1-car']) 
ax2.scatter(df_outlier_area_T1c['Release_rate_log'],df_outlier_area_T1c['Area'],marker='>', s=47,c='white' ,edgecolor=dict_color_city['TorontoDay1-car']) 


# Toronto Day 2 car
df_outlier_max_T2c = df_all_max_T2c[((df_all_max_T2c['Release_rate_log'] == 2.2925347571405443) & (df_all_max_T2c['Max'] < -1.9)) |
                 ((df_all_max_T2c['Release_rate_log'] == 1.6094379124341005) & (df_all_max_T2c['Max'] < -2))]
df_index = df_outlier_max_T2c.index # Get the index of the filtered rows
df_outlier_area_T2c = df_all_area_T2c.loc[df_index] # Select rows from df_area with the same index as in filtered df_max

ax1.scatter(df_outlier_max_T2c['Release_rate_log'],df_outlier_max_T2c['Max'],marker='X', s=47,c='white' ,edgecolor=dict_color_city['TorontoDay2-car']) 
ax2.scatter(df_outlier_area_T2c['Release_rate_log'],df_outlier_area_T2c['Area'],marker='X', s=47,c='white' ,edgecolor=dict_color_city['TorontoDay2-car']) 

# Rotterdam
df_outlier_max_R = df_all_max_R[((df_all_max_R['Release_rate_log'] == 4.382026634673881) & (df_all_max_R['Max'] < -0.7))]
df_index = df_outlier_max_R.index # Get the index of the filtered rows
df_outlier_area_R = df_all_area_R.loc[df_index] # Select rows from df_area with the same index as in filtered df_max

ax1.scatter(df_outlier_max_R['Release_rate_log'],df_outlier_max_R['Max'],marker='.', s=65,c='white' ,edgecolor=dict_color_city['Rotterdam']) 
ax2.scatter(df_outlier_area_R['Release_rate_log'],df_outlier_area_R['Area'],marker='.', s=65,c='white' ,edgecolor=dict_color_city['Rotterdam']) 

# London Day2
df_outlier_max_Ld2 = df_all_max_L1d2[((df_all_max_L1d2['Release_rate_log'] == 4.248495242049359) & (df_all_max_L1d2['Max'] < -0.9))]
df_index = df_outlier_max_Ld2.index # Get the index of the filtered rows
df_outlier_area_Ld2 = df_all_area_L1d2.loc[df_index] # Select rows from df_area with the same index as in filtered df_max

ax1.scatter(df_outlier_max_Ld2['Release_rate_log'],df_outlier_max_Ld2['Max'],marker='p', s=50,c='white' ,edgecolor=dict_color_city['London IDay2']) 
ax2.scatter(df_outlier_area_Ld2['Release_rate_log'],df_outlier_area_Ld2['Area'],marker='p', s=50,c='white' ,edgecolor=dict_color_city['London IDay2']) 

###############################################################################


# ALL DATA ------------------------------------------------------

# RU2T2L2L2
df_all_max = pd.concat([df_all_max_R, df_all_max_U1,df_all_max_U2,df_all_max_L1d2,df_all_max_L1d5, 
                        df_all_max_T1b,df_all_max_T1c,df_all_max_T2c,df_all_max_L2d1,df_all_max_L2d2], ignore_index=True)
df_all_area = pd.concat([df_all_area_R, df_all_area_U1,df_all_area_U2,df_all_area_L1d2,df_all_area_L1d5, 
                        df_all_area_T1b,df_all_area_T1c,df_all_area_T2c,df_all_area_L2d1,df_all_area_L2d2], ignore_index=True) 


x = df_all_max.Release_rate_log

p = scipy.stats.linregress(df_all_max.Release_rate_log,df_all_max.Max)
slope = round(p[0],3)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
legend14, = ax1.plot(x,p[0]*x+p[1], linestyle='dashdot',color='red',linewidth=3.5, label=f'Maximum eq.:      ln(y) = {slope} ln(x) - {np.abs(b)} ($R^2$ = {r2})')
legend12, = ax1.plot(x,0.817*x-0.988,linewidth=3.5,label=f'Weller (2019):      ln(y) = 0.817 ln(x) - 0.988',color='grey') #crimson
print('RU2T2L2L2: slope= '+str(slope)+' , y axis= '+str(b))
legend_handles1b.append(legend14)
legend_handles1b.append(legend12)


p = scipy.stats.linregress(df_all_area.Release_rate_log,df_all_area.Area)
slope = round(p[0],3)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
legend15, = ax2.plot(x,p[0]*x+p[1], linestyle='dashdot',color='red',linewidth=3.5, label=f'Area eq.:              ln(y) = {slope} ln(x) + {b} ($R^2$ = {r2})')
legend_handles2b.append(legend15)
print('RU2T2L2L2: slope= '+str(slope)+' , y axis= '+str(b))



    
# Add confidence intervalls ----------------------------------------------------------------------------------

# Fit on all data:
if confidence_interval:
    add_confidenceinterval_to_plot(df_all_max.Release_rate_log,df_all_max.Max,10000,ax1,legend_handles1,conf_level_in_std=2)
    add_confidenceinterval_to_plot(df_all_area.Release_rate_log,df_all_area.Area,10000,ax2,legend_handles2b,conf_level_in_std=2)

ax1.grid(True)
ax2.grid(True)

# LEGEND --------------------------------------------------------------------------------------------------------------

inner_legend = ax1.legend(handles=legend_handles1, loc='upper left', fontsize=14)

# Create a legend for the first subplot (ax1)
legend_ax1 = ax1.legend(handles=legend_handles1b, loc='upper center', bbox_to_anchor=(0.5, -0.22), fancybox=True, shadow=True, ncol=1,fontsize=16)

# Create a legend for the second subplot (ax2)
legend_ax2 = ax2.legend(handles=legend_handles2b,loc='upper center', bbox_to_anchor=(0.5, -0.22), fancybox=True, shadow=True, ncol=1,fontsize=16)

ax1.add_artist(inner_legend)  # Add the inner legend back to the plot
ax1.add_artist(legend_ax1)



# TITLE ------------------------------------------------------------------------------------------------------------------

#plt.suptitle('Linear fit of nat. log transformed data \nRotterdam, Utrecht, Toronto and London',fontsize=14)

# ax1.set_title('Max',fontsize=22,fontweight='bold')
# ax2.set_title('Area',fontsize=22,fontweight='bold')
ax1.set_ylabel(r'ln($\mathrm{\left[CH_4\right]_{max}}\ \left[ \frac{\mathrm{ppm}}{1\ \mathrm{ppm}} \right]$)',fontsize=22,fontweight='bold') #,fontweight='bold'
ax2.set_ylabel(r'ln($\mathrm{\left[CH_4\right]_{area}}\ \left[ \frac{\mathrm{ppm*m}}{1\ \mathrm{ppm*m}} \right]$)',fontsize=22,fontweight='bold')
ax1.set_xlabel(r'ln(Emission Rate $\left[ \frac{\mathrm{Lmin^{-1}}}{1\ \mathrm{Lmin^{-1}}} \right]$)',fontsize=22,fontweight='bold',labelpad=5)
ax2.set_xlabel(r'ln(Emission Rate $\left[ \frac{\mathrm{Lmin^{-1}}}{1\ \mathrm{Lmin^{-1}}} \right]$)',fontsize=22,fontweight='bold',labelpad=5)

# save_plots=False
if save_plots:
    
    plt.savefig(path_fig_plot2+'All_Compare_Cities/RU2T2L2L2_AandPMvsRR_3reg_withlowerlegend.png',bbox_inches='tight')
    plt.savefig(path_fig_plot2+'All_Compare_Cities/RU2T2L2L2_AandPMvsRR_3reg_withlowerlegend.pdf',bbox_inches='tight')
    plt.savefig(path_fig_plot2+'All_Compare_Cities/RU2T2L2L2_AandPMvsRR_3reg_withlowerlegend.svg',bbox_inches='tight')
    
    
plt.show()



#%%% RU2T2L2L2 No Fit

confidence_interval = False # set to TRUE to add confidence intervals, takes a bit of time
save_plots=False

fig, ax = plt.subplots(1,2, figsize=(18,10))

ax1 = ax[0]
ax2 = ax[1]
ax1.grid(True)
ax2.grid(True)

i=0

# Create empty lists to collect data for legends
legend_handles1 = [] # inner legend, displaying cities
legend_handles2 = []
legend_handles1b = [] # outer legend, displaying lin. reg. fit
legend_handles2b = []


all_area_R = []
all_max_R = []
all_area_U1 = []
all_max_U1 = []
all_area_U2 = []
all_max_U2 = []
all_area_T1b = []
all_max_T1b = []
all_area_T1c = []
all_max_T1c = []
all_area_T2c = []
all_max_T2c = []
all_area_L1d2 = []
all_max_L1d2 = []
all_area_L1d5 = []
all_max_L1d5 = []
all_area_L2d1 = []
all_max_L2d1 = []
all_area_L2d2 = []
all_max_L2d2 = []


# Create empty lists to collect data for legends
legend_handles1 = []
legend_handles2 = []


log_peaks_R = total_peaks_R.copy(deep=True)
log_peaks_U1 = total_peaks_U1.copy(deep=True)
log_peaks_U2 = total_peaks_U2.copy(deep=True)
log_peaks_T1b = total_peaks_T1b.copy(deep=True)
log_peaks_T1c = total_peaks_T1c.copy(deep=True)
log_peaks_T2c = total_peaks_T2c.copy(deep=True)
log_peaks_L1d2 = total_peaks_L1d2.copy(deep=True)
log_peaks_L1d5 = total_peaks_L1d5.copy(deep=True)
log_peaks_L2d1 = total_peaks_L2d1.copy(deep=True)
log_peaks_L2d2 = total_peaks_L2d2.copy(deep=True)

max_dist = 75
log_peaks_R = log_peaks_R[((log_peaks_R['Release_rate'] != 0)) & (log_peaks_R['Distance_to_source'] < max_dist)] 
log_peaks_U1 = log_peaks_U1[((log_peaks_U1['Release_rate'] != 0)) & (log_peaks_U1['Distance_to_source'] < max_dist)] 
log_peaks_U2 = log_peaks_U2[((log_peaks_U2['Release_rate'] != 0)) & (log_peaks_U2['Distance_to_source'] < max_dist)]  
log_peaks_T1b = log_peaks_T1b[((log_peaks_T1b['Release_rate'] != 0)) & (log_peaks_T1b['Distance_to_source'] < max_dist)] 
log_peaks_T1c = log_peaks_T1c[((log_peaks_T1c['Release_rate'] != 0)) & (log_peaks_T1c['Distance_to_source'] < max_dist)] 
log_peaks_T2c = log_peaks_T2c[((log_peaks_T2c['Release_rate'] != 0)) & (log_peaks_T2c['Distance_to_source'] < max_dist)] 
log_peaks_L1d2 = log_peaks_L1d2[((log_peaks_L1d2['Release_rate'] != 0) & (log_peaks_L1d2['Distance_to_source'] < max_dist))]
log_peaks_L1d5 = log_peaks_L1d5[((log_peaks_L1d5['Release_rate'] != 0) & (log_peaks_L1d5['Distance_to_source'] < max_dist))]
log_peaks_L2d1 = log_peaks_L2d1[((log_peaks_L2d1['Release_rate'] != 0) & (log_peaks_L2d1['Distance_to_source'] < max_dist))]
log_peaks_L2d2 = log_peaks_L2d2[((log_peaks_L2d2['Release_rate'] != 0) & (log_peaks_L2d2['Distance_to_source'] < max_dist))] 




legend_handles_notincluded = []

x_R,all_max_R,all_area_R,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_R,all_max_R,all_area_R,ax,legend_handles1,legend_handles2,'.',65,False, R_vars_aeris,R_vars_G43,R_vars_G23,R_vars_miro,R_vars_aerodyne)

x_U1,all_max_U1,all_area_U1,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_U1,all_max_U1,all_area_U1,ax,legend_handles1,legend_handles2,'v',47,False, U1_vars_G43,U1_vars_G23)
#
x_U2,all_max_U2,all_area_U2,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_U2,all_max_U2,all_area_U2,ax,legend_handles1,legend_handles2,'d',47,False, U2_vars_aeris,U2_vars_G23)

x_L1d2,all_max_L1d2,all_area_L1d2,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_L1d2,all_max_L1d2,all_area_L1d2,ax,legend_handles1,legend_handles2,'p',50,True,L1_vars_d2_LGR,L1_vars_d2_G23) #London Day2
  
x_L1d5,all_max_L1d5,all_area_L1d5,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_L1d5,all_max_L1d5,all_area_L1d5,ax,legend_handles1,legend_handles2,'*',80,True,L1_vars_d5_G23) #London Day3

x_L2d1,all_max_L2d1,all_area_L2d1,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_L2d1,all_max_L2d1,all_area_L2d1,ax,legend_handles1,legend_handles2,'^',47,True,L2_vars_d1_Licor)

x_L2d2,all_max_L2d2,all_area_L2d2,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_L2d2,all_max_L2d2,all_area_L2d2,ax,legend_handles1,legend_handles2,'<',47,True,L2_vars_d2_Licor)

x_T1,all_max_T1b,all_area_T1b,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_T1b,all_max_T1b,all_area_T1b,ax,legend_handles1,legend_handles2,'D',55,True,T_vars_1b_LGR)

x_T1,all_max_T1c,all_area_T1c,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_T1c,all_max_T1c,all_area_T1c,ax,legend_handles1,legend_handles2,'>',55,True,T_vars_1c_G24) # marker size none

x_T2,all_max_T2c,all_area_T2c,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_T2c,all_max_T2c,all_area_T2c,ax,legend_handles1,legend_handles2,'X',55,True,T_vars_2c_G24) # marker size none


means_area_R, means_max_R, median_area_R, median_max_R = mean_and_median_log(all_area_R,all_max_R)
means_area_U1, means_max_U1, median_area_U1, median_max_U1 = mean_and_median_log(all_area_U1,all_max_U1)
means_area_U2, means_max_U2, median_area_U2, median_max_U2 = mean_and_median_log(all_area_U2,all_max_U2)
means_area_L1d2, means_max_L1d2, median_area_L1d2, median_max_L1d2 = mean_and_median_log(all_area_L1d2,all_max_L1d2)
means_area_L1d5, means_max_L1d5, median_area_L1d5, median_max_L1d5 = mean_and_median_log(all_area_L1d5,all_max_L1d5)
means_area_T1b, means_max_T1b, median_area_T1b, median_max_T1b = mean_and_median_log(all_area_T1b,all_max_T1b)
means_area_T1c, means_max_T1c, median_area_T1c, median_max_T1c = mean_and_median_log(all_area_T1c,all_max_T1c)
means_area_T2c, means_max_T2c, median_area_T2c, median_max_T2c = mean_and_median_log(all_area_T2c,all_max_T2c)
means_area_L2d1, means_max_L2d1, median_area_L2d1, median_max_L2d1 = mean_and_median_log(all_area_L2d1,all_max_L2d1)
means_area_L2d2, means_max_L2d2, median_area_L2d2, median_max_L2d2 = mean_and_median_log(all_area_L2d2,all_max_L2d2)


# Different marker for mean
ax1.scatter(means_max_R.index,means_max_R.Max,marker = '.',s=95, c= '#484848', edgecolor='black') #,label = 'mean'
ax1.scatter(means_max_U1.index,means_max_U1.Max,marker = 'v',s=70, c= '#484848', edgecolor='black')
ax1.scatter(means_max_U2.index,means_max_U2.Max,marker = 'd',s=70, c= '#484848', edgecolor='black')
ax1.scatter(means_max_L1d2.index,means_max_L1d2.Max,marker = 'p',s=100, c= '#484848', edgecolor='black')
ax1.scatter(means_max_L1d5.index,means_max_L1d5.Max,marker = '*',s=100,c= '#484848', edgecolor='black')
ax1.scatter(means_max_T1b.index,means_max_T1b.Max,marker = 'D',s=40,c= '#484848', edgecolor='black')
ax1.scatter(means_max_T1c.index,means_max_T1c.Max,marker = '>',s=40,c= '#484848', edgecolor='black')
ax1.scatter(means_max_T2c.index,means_max_T2c.Max,marker = 'X',s=40,c= '#484848', edgecolor='black')
ax1.scatter(means_max_L2d1.index,means_max_L2d1.Max,marker = '^',s=100,c= '#484848', edgecolor='black')
ax1.scatter(means_max_L2d2.index,means_max_L2d2.Max,marker = '<',s=100,c= '#484848', edgecolor='black')


ax2.scatter(means_area_R.index,means_area_R.Area,marker = '.',s=95, c= '#484848', edgecolor='black') #,label = 'mean'
ax2.scatter(means_area_U1.index,means_area_U1.Area,marker = 'v',s=70, c= '#484848', edgecolor='black')
ax2.scatter(means_area_U2.index,means_area_U2.Area,marker = 'd',s=70, c= '#484848', edgecolor='black')
ax2.scatter(means_area_L1d2.index,means_area_L1d2.Area,marker = 'p',s=100, c= '#484848', edgecolor='black')
ax2.scatter(means_area_L1d5.index,means_area_L1d5.Area,marker = '*',s=100,c= '#484848', edgecolor='black')
ax2.scatter(means_area_T1b.index,means_area_T1b.Area,marker = 'D',s=40,c= '#484848', edgecolor='black')
ax2.scatter(means_area_T1c.index,means_area_T1c.Area,marker = '>',s=40,c= '#484848', edgecolor='black')
ax2.scatter(means_area_T2c.index,means_area_T2c.Area,marker = 'X',s=40,c= '#484848', edgecolor='black')
ax2.scatter(means_area_L2d1.index,means_area_L2d1.Area,marker = '^',s=100,c= '#484848', edgecolor='black')
ax2.scatter(means_area_L2d2.index,means_area_L2d2.Area,marker = '<',s=100,c= '#484848', edgecolor='black')

legend_handles1.append(ax1.scatter([], [], marker='.', color='black', label='mean'))
legend_handles2.append(ax2.scatter([], [], marker='.', color='black', label='mean'))
  


plt.subplots_adjust(wspace=0.35)

# 2nd axes --------------------------------------------------------------------
# Add 2nd x and y axis to display non logarithmis calues for peak max and area
# and actual release rates
# Plot left (PEAK MAX), y axis
ax1y = ax1.twinx()
ticks_y = [-2,0,2,4,6]
labels_y = np.round(np.exp(ticks_y),1)
ax1y.set_yticks(ticks_y) # Set the ticks and labels for the second y-axis
ax1y.set_yticklabels(labels_y)
ax1y.set_yticklabels([f"{int(label)}" if label >= 1 else f"{label:.1f}" for label in labels_y])
ax1y.set_ylabel('Peak Maximum [ppm]',fontsize=16)
ax1.set_ylim(-4,7)
ax1y.set_ylim(-4,7)

# Plot left (PEAK MAX), x axis
ax1x = ax1.twiny()
# Manually position the second x-axis below the original x-axis
ax1x.spines['bottom'].set_position(('outward', 75))  # You can adjust the position value (40) to control the offset
# ticks_x = [0, 1, 2,3]
# labels_x = np.round(np.exp(ticks_x),2)
# Display all release rates actually used on the 2nd non log x axis
labels_x = [0.15,0.3,0.5,1,2.2,3,5,10,15,20,40,80,120] # actual release rates in L/min
ticks_x = np.round(np.log(labels_x),2) # transform to log to plot it at right position on the axis
ax1x.set_xticks(ticks_x) # Set the ticks and labels for the second y-axis
ax1x.set_xticklabels(labels_x)
# Set the location of the ticks to be at the bottom
ax1x.xaxis.set_ticks_position('bottom')
ax1x.set_xlabel('Release Rate [L/min]',fontsize=16) #,labelpad=-750
ax1x.xaxis.set_label_coords(0.5, -0.208)
ax1.set_xlim(-2.2,5.2) #-0.5,3.5
ax1x.set_xlim(-2.2,5.2)

# Plot right (AREA), y axis
ax2y = ax2.twinx()
ticks_y = [0,1,2,3,4,5,6,7,8]
labels_y = np.round(np.exp(ticks_y),0)# Set the ticks and labels for the second y-axis
ax2y.set_yticks(ticks_y)
ax2y.set_yticklabels(labels_y)
ax2y.set_yticklabels([f"{int(label)}" for label in labels_y]) # Show no digits (round to integer)
ax2y.set_ylabel('Peak Area [ppm*m]',fontsize=16)
ax2.set_ylim(-1,8)
ax2y.set_ylim(-1,8)

# Plot right (AREA), x axis
ax2x = ax2.twiny()
# Manually position the second x-axis below the original x-axis
ax2x.spines['bottom'].set_position(('outward', 75))  # You can adjust the position value (40) to control the offset
# Display all release rates actually used on the 2nd non log x axis
labels_x = [0.15,0.3,0.5,1,2.2,3,5,10,15,20,40,80,120] # actual release rates in L/min
ticks_x = np.round(np.log(labels_x),2) # transform to log to plot it at right position on the axis
ax2x.set_xticks(ticks_x) # Set the ticks and labels for the second y-axis
ax2x.set_xticklabels(labels_x)
# Set the location of the ticks to be at the bottom
ax2x.xaxis.set_ticks_position('bottom')
ax2x.set_xlabel('Release Rate [L/min]',fontsize=16)
ax2x.xaxis.set_label_coords(0.5, -0.208)
ax2.set_xlim(-2.2,5.2) #-0.5,3.5
ax2x.set_xlim(-2.2,5.2)
# ----------------------------------------------------------------------------



ax1.tick_params(axis='x', labelsize=18)
ax1.tick_params(axis='y', labelsize=18)
ax2.tick_params(axis='x', labelsize=18)
ax2.tick_params(axis='y', labelsize=18)

ax1x.tick_params(axis='x', labelsize=13)
ax1y.tick_params(axis='y', labelsize=14)
ax2x.tick_params(axis='x', labelsize=13)
ax2y.tick_params(axis='y', labelsize=14)


fig.canvas.draw() # !!! necessary so that ax1.get_xticklabels() in the following access the most recent labels


# =============================================================================


df_all_max_R = pd.concat(all_max_R, ignore_index=True) 
df_all_area_R = pd.concat(all_area_R, ignore_index=True) 
df_all_max_U1 = pd.concat(all_max_U1, ignore_index=True) 
df_all_area_U1 = pd.concat(all_area_U1, ignore_index=True)
df_all_max_U2 = pd.concat(all_max_U2, ignore_index=True) 
df_all_area_U2 = pd.concat(all_area_U2, ignore_index=True) 
df_all_max_L1d2 = pd.concat(all_max_L1d2, ignore_index=True) 
df_all_area_L1d2 = pd.concat(all_area_L1d2, ignore_index=True) 
df_all_max_L1d5 = pd.concat(all_max_L1d5) 
df_all_area_L1d5 = pd.concat(all_area_L1d5) 
df_all_max_T1b = pd.concat(all_max_T1b) 
df_all_area_T1b = pd.concat(all_area_T1b) 
df_all_max_T1c = pd.concat(all_max_T1c) 
df_all_area_T1c = pd.concat(all_area_T1c)  
df_all_max_T2c = pd.concat(all_max_T2c) 
df_all_area_T2c = pd.concat(all_area_T2c)  
df_all_max_L2d1 = pd.concat(all_max_L2d1, ignore_index=True) #
df_all_area_L2d1 = pd.concat(all_area_L2d1, ignore_index=True) #
df_all_max_L2d2 = pd.concat(all_max_L2d2, ignore_index=True) #
df_all_area_L2d2 = pd.concat(all_area_L2d2, ignore_index=True) #


# RU2T2L3L2
df_all_max = pd.concat([df_all_max_R,df_all_max_U1,df_all_max_U2,df_all_max_L1d2,df_all_max_L1d5,
                        df_all_max_T1b,df_all_max_T1c,df_all_max_T2c,df_all_max_L2d1,df_all_max_L2d2], ignore_index=True) #,df_all_max_L1d3
df_all_area = pd.concat([df_all_area_R,df_all_area_U1,df_all_area_U2,df_all_area_L1d2,df_all_area_L1d5,
                        df_all_area_T1b,df_all_area_T1c,df_all_area_T2c,df_all_area_L2d1,df_all_area_L2d2], ignore_index=True) #df_all_area_L1d3,




# ALL DATA ------------------------------------------------------

# x = df_all_max.Release_rate_log

# p = scipy.stats.linregress(df_all_max.Release_rate_log,df_all_max.Max)
# slope = round(p[0],3)
# b = round(p[1],2)
# rval  = round(p[2],2)
# r2 = round(rval**2,2)
# legend14, = ax1.plot(x,p[0]*x+p[1], linestyle='dashdot',color='red',linewidth=3.5, label=f'Maximum eq.:      ln(y) = {slope} ln(x) + {b} ($R^2$ = {r2})')
# legend12, = ax1.plot(x,0.817*x-0.988,linewidth=3.5,label=f'Weller (2019):      ln(y) = 0.817 ln(x) - 0.988',color='grey') #crimson
# print('RU2T2L3L2: slope= '+str(slope)+' , y axis= '+str(b))
# legend_handles1b.append(legend14)
# legend_handles1b.append(legend12)


# p = scipy.stats.linregress(df_all_area.Release_rate_log,df_all_area.Area)
# slope = round(p[0],3)
# b = round(p[1],2)
# rval  = round(p[2],2)
# r2 = round(rval**2,2)
# legend15, = ax2.plot(x,p[0]*x+p[1], linestyle='dashdot',color='red',linewidth=3.5, label=f'Area eq.:              ln(y) = {slope} ln(x) + {b} ($R^2$ = {r2})')
# legend_handles2b.append(legend15)
# print('RU2T2L3L2: slope= '+str(slope)+' , y axis= '+str(b))



 
    
# Add confidence intervalls ----------------------------------------------------------------------------------

# Fit on all data:
if confidence_interval:
    add_confidenceinterval_to_plot(df_all_max.Release_rate_log,df_all_max.Max,10000,ax1,legend_handles1b,conf_level_in_std=2)
    add_confidenceinterval_to_plot(df_all_area.Release_rate_log,df_all_area.Area,10000,ax2,legend_handles2b,conf_level_in_std=2)

ax1.grid(True)
ax2.grid(True)

# LEGEND --------------------------------------------------------------------------------------------------------------

inner_legend = ax1.legend(handles=legend_handles1, loc='upper left', fontsize=14)

# Create a legend for the first subplot (ax1)
legend_ax1 = ax1.legend(handles=legend_handles1b, loc='upper center', bbox_to_anchor=(0.5, -0.22), fancybox=True, shadow=True, ncol=1,fontsize=16)

# Create a legend for the second subplot (ax2)
legend_ax2 = ax2.legend(handles=legend_handles2b,loc='upper center', bbox_to_anchor=(0.5, -0.22), fancybox=True, shadow=True, ncol=1,fontsize=16)

ax1.add_artist(inner_legend)  # Add the inner legend back to the plot
ax1.add_artist(legend_ax1)




# TITLE ------------------------------------------------------------------------------------------------------------------

#plt.suptitle('Linear fit of nat. log transformed data \nRotterdam, Utrecht, Toronto and London',fontsize=14)

ax1.set_title('Max',fontsize=22,fontweight='bold')
ax2.set_title('Area',fontsize=22,fontweight='bold')
ax1.set_ylabel(r'ln(Peak Maximum $\left[ \frac{\mathrm{ppm}}{\mathrm{10^{-6}}} \right]$)',fontsize=22,fontweight='bold') #,fontweight='bold'
ax2.set_ylabel(r'ln(Peak Area $\left[ \frac{\mathrm{ppm*m}}{\mathrm{10^{-6}*m}} \right]$)',fontsize=22,fontweight='bold')
ax1.set_xlabel(r'ln(Release Rate $\left[ \frac{\mathrm{L/min}}{1\ \mathrm{L/min}} \right]$)',fontsize=22,fontweight='bold',labelpad=5)
ax2.set_xlabel(r'ln(Release Rate $\left[ \frac{\mathrm{L/min}}{1\ \mathrm{L/min}} \right]$)',fontsize=22,fontweight='bold',labelpad=5)

# save_plots=False
if save_plots:
    plt.savefig(path_fig_plot2+'All_Compare_Cities/RU2T2L2L2.pdf',bbox_inches='tight')
    plt.savefig(path_fig_plot2+'All_Compare_Cities/RU2T2L2L2.svg',bbox_inches='tight')
    
    

plt.show()





#%%% RU2T2L2L2 - All fits

confidence_interval = False # set to TRUE to add confidence intervals, takes a bit of time
save_plots=False

fig, ax = plt.subplots(1,2, figsize=(18,10))

ax1 = ax[0]
ax2 = ax[1]
ax1.grid(True)
ax2.grid(True)

# Add Shadings to visualize peak threshold -----------------------------------------

# x = np.array([-2.2,5.2])
# ax2.fill_between(x,-1,np.log(0.2),color='lightgray',alpha=0.8)

x = np.array([-2.5,5.2])
ax1.fill_between(x,-4,np.log(0.2),color='lightgray',alpha=0.4)

x = np.array([-2.5,5.2])
ax1.fill_between(x,-4,np.log(0.04),color='lightgray',alpha=0.9)

# ----------------------------------------------------------------------------


# Create empty lists to collect data for legends
legend_handles1 = [] # inner legend, displaying cities
legend_handles2 = []
legend_handles1b = [] # outer legend, displaying lin. reg. fit
legend_handles2b = []


all_area_R = []
all_max_R = []
all_area_U1 = []
all_max_U1 = []
all_area_U2 = []
all_max_U2 = []
all_area_T1b = []
all_max_T1b = []
all_area_T1c = []
all_max_T1c = []
all_area_T2c = []
all_max_T2c = []
all_area_L1d2 = []
all_max_L1d2 = []
all_area_L1d5 = []
all_max_L1d5 = []
all_area_L2d1 = []
all_max_L2d1 = []
all_area_L2d2 = []
all_max_L2d2 = []


# Create empty lists to collect data for legends
legend_handles1 = []
legend_handles2 = []


log_peaks_R = total_peaks_R.copy(deep=True)
log_peaks_R = log_peaks_R[log_peaks_R['Loc'] == 1]
log_peaks_U1 = total_peaks_U1.copy(deep=True)
log_peaks_U2 = total_peaks_U2.copy(deep=True)
log_peaks_T1b = total_peaks_T1b.copy(deep=True)
log_peaks_T1c = total_peaks_T1c.copy(deep=True)
log_peaks_T2c = total_peaks_T2c.copy(deep=True)
log_peaks_L1d2 = total_peaks_L1d2.copy(deep=True)
log_peaks_L1d5 = total_peaks_L1d5.copy(deep=True)
log_peaks_L2d1 = total_peaks_L2d1.copy(deep=True)
log_peaks_L2d2 = total_peaks_L2d2.copy(deep=True)

max_dist = 75
log_peaks_R = log_peaks_R[((log_peaks_R['Release_rate'] != 0)) & (log_peaks_R['Distance_to_source'] < max_dist)] 
log_peaks_U1 = log_peaks_U1[((log_peaks_U1['Release_rate'] != 0)) & (log_peaks_U1['Distance_to_source'] < max_dist)] 
log_peaks_U2 = log_peaks_U2[((log_peaks_U2['Release_rate'] != 0)) & (log_peaks_U2['Distance_to_source'] < max_dist)]  
log_peaks_T1b = log_peaks_T1b[((log_peaks_T1b['Release_rate'] != 0)) & (log_peaks_T1b['Distance_to_source'] < max_dist)] 
log_peaks_T1c = log_peaks_T1c[((log_peaks_T1c['Release_rate'] != 0)) & (log_peaks_T1c['Distance_to_source'] < max_dist)] 
log_peaks_T2c = log_peaks_T2c[((log_peaks_T2c['Release_rate'] != 0)) & (log_peaks_T2c['Distance_to_source'] < max_dist)] 
log_peaks_L1d2 = log_peaks_L1d2[((log_peaks_L1d2['Release_rate'] != 0) & (log_peaks_L1d2['Distance_to_source'] < max_dist))]
log_peaks_L1d5 = log_peaks_L1d5[((log_peaks_L1d5['Release_rate'] != 0) & (log_peaks_L1d5['Distance_to_source'] < max_dist))]
log_peaks_L2d1 = log_peaks_L2d1[((log_peaks_L2d1['Release_rate'] != 0) & (log_peaks_L2d1['Distance_to_source'] < max_dist))]
log_peaks_L2d2 = log_peaks_L2d2[((log_peaks_L2d2['Release_rate'] != 0) & (log_peaks_L2d2['Distance_to_source'] < max_dist))] 


legend_handles_notincluded = []

x_R,all_max_R,all_area_R,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_R,all_max_R,all_area_R,ax,legend_handles1,legend_handles2,'.',65,False, R_vars_aeris,R_vars_G43,R_vars_G23,R_vars_miro,R_vars_aerodyne)

x_U1,all_max_U1,all_area_U1,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_U1,all_max_U1,all_area_U1,ax,legend_handles1,legend_handles2,'v',47,False, U1_vars_G43,U1_vars_G23)

x_U2,all_max_U2,all_area_U2,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_U2,all_max_U2,all_area_U2,ax,legend_handles1,legend_handles2,'d',47,False, U2_vars_aeris,U2_vars_G23)
  
x_L1d2,all_max_L1d2,all_area_L1d2,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_L1d2,all_max_L1d2,all_area_L1d2,ax,legend_handles1,legend_handles2,'p',50,True,L1_vars_d2_LGR,L1_vars_d2_G23) #London Day2
  
x_L1d5,all_max_L1d5,all_area_L1d5,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_L1d5,all_max_L1d5,all_area_L1d5,ax,legend_handles1,legend_handles2,'*',80,True,L1_vars_d5_G23) #London Day3

x_L2d1,all_max_L2d1,all_area_L2d1,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_L2d1,all_max_L2d1,all_area_L2d1,ax,legend_handles1,legend_handles2,'^',47,True,L2_vars_d1_Licor)

x_L2d2,all_max_L2d2,all_area_L2d2,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_L2d2,all_max_L2d2,all_area_L2d2,ax,legend_handles1,legend_handles2,'<',47,True,L2_vars_d2_Licor)

x_T1,all_max_T1b,all_area_T1b,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_T1b,all_max_T1b,all_area_T1b,ax,legend_handles1,legend_handles2,'D',55,True,T_vars_1b_LGR)

x_T1,all_max_T1c,all_area_T1c,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_T1c,all_max_T1c,all_area_T1c,ax,legend_handles1,legend_handles2,'>',55,True,T_vars_1c_G24) # marker size none

x_T2,all_max_T2c,all_area_T2c,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_T2c,all_max_T2c,all_area_T2c,ax,legend_handles1,legend_handles2,'X',55,True,T_vars_2c_G24) # marker size none


means_area_R, means_max_R, median_area_R, median_max_R = mean_and_median_log(all_area_R,all_max_R)
means_area_U1, means_max_U1, median_area_U1, median_max_U1 = mean_and_median_log(all_area_U1,all_max_U1)
means_area_U2, means_max_U2, median_area_U2, median_max_U2 = mean_and_median_log(all_area_U2,all_max_U2)
means_area_L1d2, means_max_L1d2, median_area_L1d2, median_max_L1d2 = mean_and_median_log(all_area_L1d2,all_max_L1d2)
means_area_L1d5, means_max_L1d5, median_area_L1d5, median_max_L1d5 = mean_and_median_log(all_area_L1d5,all_max_L1d5)
means_area_T1b, means_max_T1b, median_area_T1b, median_max_T1b = mean_and_median_log(all_area_T1b,all_max_T1b)
means_area_T1c, means_max_T1c, median_area_T1c, median_max_T1c = mean_and_median_log(all_area_T1c,all_max_T1c)
means_area_T2c, means_max_T2c, median_area_T2c, median_max_T2c = mean_and_median_log(all_area_T2c,all_max_T2c)
means_area_L2d1, means_max_L2d1, median_area_L2d1, median_max_L2d1 = mean_and_median_log(all_area_L2d1,all_max_L2d1)
means_area_L2d2, means_max_L2d2, median_area_L2d2, median_max_L2d2 = mean_and_median_log(all_area_L2d2,all_max_L2d2)



# Different marker for mean
ax1.scatter(means_max_R.index,means_max_R.Max,marker = '.',s=110, c= 'lightgrey', edgecolor='black') #,label = 'mean' #484848
ax1.scatter(means_max_U1.index,means_max_U1.Max,marker = 'v',s=70, c= 'lightgrey', edgecolor='black')
ax1.scatter(means_max_U2.index,means_max_U2.Max,marker = 'd',s=70, c= 'lightgrey', edgecolor='black')
ax1.scatter(means_max_L1d2.index,means_max_L1d2.Max,marker = 'p',s=100, c= 'lightgrey', edgecolor='black')
ax1.scatter(means_max_L1d5.index,means_max_L1d5.Max,marker = '*',s=120,c= 'lightgrey', edgecolor='black')
ax1.scatter(means_max_T1b.index,means_max_T1b.Max,marker = 'D',s=40,c= 'lightgrey', edgecolor='black')
ax1.scatter(means_max_T1c.index,means_max_T1c.Max,marker = '>',s=45,c= 'lightgrey', edgecolor='black')
ax1.scatter(means_max_T2c.index,means_max_T2c.Max,marker = 'X',s=60,c= 'lightgrey', edgecolor='black')
ax1.scatter(means_max_L2d1.index,means_max_L2d1.Max,marker = '^',s=100,c= 'lightgrey', edgecolor='black')
ax1.scatter(means_max_L2d2.index,means_max_L2d2.Max,marker = '<',s=100,c= 'lightgrey', edgecolor='black')


ax2.scatter(means_area_R.index,means_area_R.Area,marker = '.',s=110, c= 'lightgrey', edgecolor='black') #,label = 'mean'
ax2.scatter(means_area_U1.index,means_area_U1.Area,marker = 'v',s=70, c= 'lightgrey', edgecolor='black')
ax2.scatter(means_area_U2.index,means_area_U2.Area,marker = 'd',s=70, c= 'lightgrey', edgecolor='black')
ax2.scatter(means_area_L1d2.index,means_area_L1d2.Area,marker = 'p',s=100, c= 'lightgrey', edgecolor='black')
ax2.scatter(means_area_L1d5.index,means_area_L1d5.Area,marker = '*',s=120,c= 'lightgrey', edgecolor='black')
ax2.scatter(means_area_T1b.index,means_area_T1b.Area,marker = 'D',s=40,c= 'lightgrey', edgecolor='black')
ax2.scatter(means_area_T1c.index,means_area_T1c.Area,marker = '>',s=45,c= 'lightgrey', edgecolor='black')
ax2.scatter(means_area_T2c.index,means_area_T2c.Area,marker = 'X',s=60,c= 'lightgrey', edgecolor='black')
ax2.scatter(means_area_L2d1.index,means_area_L2d1.Area,marker = '^',s=100,c= 'lightgrey', edgecolor='black')
ax2.scatter(means_area_L2d2.index,means_area_L2d2.Area,marker = '<',s=100,c= 'lightgrey', edgecolor='black')

legend_handles1.append(ax1.scatter([], [], marker='.',s=110, c= 'lightgrey', edgecolor='black', label='mean'))
legend_handles2.append(ax2.scatter([], [], marker='.',s=110, c= 'lightgrey', edgecolor='black', label='mean'))
  


plt.subplots_adjust(wspace=0.35)

# 2nd axes --------------------------------------------------------------------
# Add 2nd x and y axis to display non logarithmis calues for peak max and area
# and actual release rates
# Plot left (PEAK MAX), y axis
ax1y = ax1.twinx()
ticks_y = [-2,0,2,4,6]
labels_y = np.round(np.exp(ticks_y),1)
ax1y.set_yticks(ticks_y) # Set the ticks and labels for the second y-axis
ax1y.set_yticklabels(labels_y)
ax1y.set_yticklabels([f"{int(label)}" if label >= 1 else f"{label:.1f}" for label in labels_y])
ax1y.set_ylabel('Peak Maximum [ppm]',fontsize=16)
ax1.set_ylim(-4,7)
ax1y.set_ylim(-4,7)

# Plot left (PEAK MAX), x axis
ax1x = ax1.twiny()
# Manually position the second x-axis below the original x-axis
ax1x.spines['bottom'].set_position(('outward', 75))  # You can adjust the position value (40) to control the offset
# ticks_x = [0, 1, 2,3]
# labels_x = np.round(np.exp(ticks_x),2)
# Display all release rates actually used on the 2nd non log x axis
labels_x = [0.15,0.3,0.5,1,2.2,3,5,10,15,20,40,80,120] # actual release rates in L/min
ticks_x = np.round(np.log(labels_x),2) # transform to log to plot it at right position on the axis
ax1x.set_xticks(ticks_x) # Set the ticks and labels for the second y-axis
ax1x.set_xticklabels(labels_x)
# Set the location of the ticks to be at the bottom
ax1x.xaxis.set_ticks_position('bottom')
ax1x.set_xlabel('Release Rate [L/min]',fontsize=16) #,labelpad=-750
ax1x.xaxis.set_label_coords(0.5, -0.208)
ax1.set_xlim(-2.5,5.2) #-0.5,3.5
ax1x.set_xlim(-2.5,5.2)

# Plot right (AREA), y axis
ax2y = ax2.twinx()
ticks_y = [0,1,2,3,4,5,6,7,8]
labels_y = np.round(np.exp(ticks_y),0)# Set the ticks and labels for the second y-axis
ax2y.set_yticks(ticks_y)
ax2y.set_yticklabels(labels_y)
ax2y.set_yticklabels([f"{int(label)}" for label in labels_y]) # Show no digits (round to integer)
ax2y.set_ylabel('Peak Area [ppm*m]',fontsize=16)
ax2.set_ylim(-1,8)
ax2y.set_ylim(-1,8)

# Plot right (AREA), x axis
ax2x = ax2.twiny()
# Manually position the second x-axis below the original x-axis
ax2x.spines['bottom'].set_position(('outward', 75))  # You can adjust the position value (40) to control the offset
# Display all release rates actually used on the 2nd non log x axis
labels_x = [0.15,0.3,0.5,1,2.2,3,5,10,15,20,40,80,120] # actual release rates in L/min
ticks_x = np.round(np.log(labels_x),2) # transform to log to plot it at right position on the axis
ax2x.set_xticks(ticks_x) # Set the ticks and labels for the second y-axis
ax2x.set_xticklabels(labels_x)
# Set the location of the ticks to be at the bottom
ax2x.xaxis.set_ticks_position('bottom')
ax2x.set_xlabel('Release Rate [L/min]',fontsize=16)
ax2x.xaxis.set_label_coords(0.5, -0.208)
ax2.set_xlim(-2.5,5.2) #-0.5,3.5
ax2x.set_xlim(-2.5,5.2)




# ----------------------------------------------------------------------------



ax1.tick_params(axis='x', labelsize=18)
ax1.tick_params(axis='y', labelsize=18)
ax2.tick_params(axis='x', labelsize=18)
ax2.tick_params(axis='y', labelsize=18)

ax1x.tick_params(axis='x', labelsize=13)
ax1y.tick_params(axis='y', labelsize=14)
ax2x.tick_params(axis='x', labelsize=13)
ax2y.tick_params(axis='y', labelsize=14)


fig.canvas.draw() # !!! necessary so that ax1.get_xticklabels() in the following access the most recent labels


# =============================================================================


df_all_max_R = pd.concat(all_max_R, ignore_index=True) 
df_all_area_R = pd.concat(all_area_R, ignore_index=True) 
df_all_max_U1 = pd.concat(all_max_U1, ignore_index=True) 
df_all_area_U1 = pd.concat(all_area_U1, ignore_index=True)
#
df_all_max_U2 = pd.concat(all_max_U2, ignore_index=True) 
df_all_area_U2 = pd.concat(all_area_U2, ignore_index=True) 
df_all_max_L1d2 = pd.concat(all_max_L1d2, ignore_index=True) 
df_all_area_L1d2 = pd.concat(all_area_L1d2, ignore_index=True) 
df_all_max_L1d5 = pd.concat(all_max_L1d5) 
df_all_area_L1d5 = pd.concat(all_area_L1d5) 
df_all_max_T1b = pd.concat(all_max_T1b) 
df_all_area_T1b = pd.concat(all_area_T1b) 
df_all_max_T1c = pd.concat(all_max_T1c) 
df_all_area_T1c = pd.concat(all_area_T1c)  
df_all_max_T2c = pd.concat(all_max_T2c) 
df_all_area_T2c = pd.concat(all_area_T2c)  
df_all_max_L2d1 = pd.concat(all_max_L2d1, ignore_index=True) #
df_all_area_L2d1 = pd.concat(all_area_L2d1, ignore_index=True) #
df_all_max_L2d2 = pd.concat(all_max_L2d2, ignore_index=True) #
df_all_area_L2d2 = pd.concat(all_area_L2d2, ignore_index=True) #


# ALL DATA ------------------------------------------------------

# RU2T2L2L2
df_all_max = pd.concat([df_all_max_R, df_all_max_U1,df_all_max_U2,df_all_max_L1d2,df_all_max_L1d5,
                        df_all_max_T1b,df_all_max_T1c,df_all_max_T2c,df_all_max_L2d1,df_all_max_L2d2], ignore_index=True) 
df_all_area = pd.concat([df_all_area_R, df_all_area_U1,df_all_area_U2,df_all_area_L1d2,df_all_area_L1d5, 
                        df_all_area_T1b,df_all_area_T1c,df_all_area_T2c,df_all_area_L2d1,df_all_area_L2d2], ignore_index=True) 

linewidth = 3
x = df_all_max.Release_rate_log

p = scipy.stats.linregress(df_all_max.Release_rate_log,df_all_max.Max)
slope = round(p[0],3)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
legend14, = ax1.plot(x,p[0]*x+p[1], linestyle='dashdot',color='red',linewidth=3.5, label=f'Maximum eq.:      ln(y) = {slope} ln(x) - {np.abs(b)} ($R^2$ = {r2})')
legend12, = ax1.plot(x,0.817*x-0.988,linewidth=3.5,label=f'Weller (2019):      ln(y) = 0.817 ln(x) - 0.988',color='grey') #crimson
print('RU2T2L2L2: slope= '+str(slope)+' , y axis= '+str(b))
legend_handles1b.append(legend14)
legend_handles1b.append(legend12)


p = scipy.stats.linregress(df_all_area.Release_rate_log,df_all_area.Area)
slope = round(p[0],3)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
legend15, = ax2.plot(x,p[0]*x+p[1], linestyle='dashdot',color='red',linewidth=3.5, label=f'Area eq.:              ln(y) = {slope} ln(x) + {b} ($R^2$ = {r2})')
legend_handles2b.append(legend15)
print('RU2T2L2L2: slope= '+str(slope)+' , y axis= '+str(b))

# Rotterdam ------------------------------------
spec_vars = R_vars_G23
p = scipy.stats.linregress(df_all_max_R.Release_rate_log,df_all_max_R.Max)
slope = round(p[0],4)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
#ax1.plot(x_R,p[0]*x_R+p[1],  label=f'Pearsons: {rval}; $R^2$: {r2} \nRotterdam:          ln(y) = {slope} ln(x) - {abs(b)}', color='black')
legend1, = ax1.plot(x_R,p[0]*x_R+p[1],linewidth=3.5, linestyle='dashdot', color=dict_color_city[spec_vars['city']],  label=f'Rotterdam:          ln(y) = {slope} ln(x) + {abs(b)}, $R^2$: {r2}')
print('Rotterdam: slope= '+str(slope)+' , y axis= '+str(b)) ##c27902

p = scipy.stats.linregress(df_all_area_R.Release_rate_log,df_all_area_R.Area)
slope = round(p[0],4)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
legend2, = ax2.plot(x_R,p[0]*x_R+p[1],linewidth=3.6,linestyle='dashdot', color=dict_color_city[spec_vars['city']],  label=f'Rotterdam:          ln(y) = {slope} ln(x) + {b}, $R^2$: {r2}')
print('Rotterdam: slope= '+str(slope)+' , y axis= '+str(b))

# Torotno Day 1b
spec_vars = T_vars_1b_LGR
p = scipy.stats.linregress(df_all_max_T1b.Release_rate_log,df_all_max_T1b.Max)
slope = round(p[0],4)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = rval**2
legend3, = ax1.plot(x_T1,p[0]*x_T1+p[1], linestyle='dashed',linewidth=linewidth,color=dict_color_city[spec_vars['city']+spec_vars['day']], label=f'Toronto 1b:          ln(y) = {slope} ln(x) - {abs(b)}, $R^2$: {r2}')
print('Toronto 1b: slope= '+str(slope)+' , y axis= '+str(b))

p = scipy.stats.linregress(df_all_area_T1b.Release_rate_log,df_all_area_T1b.Area)
slope = round(p[0],4)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = rval**2
legend4, = ax2.plot(x_T1,p[0]*x_T1+p[1], linestyle='dashed',linewidth=linewidth,color=dict_color_city[spec_vars['city']+spec_vars['day']],  label=f'Toronto 1b:          ln(y) = {slope} ln(x) + {b}, $R^2$: {r2}')
print('Toronto 1b: slope= '+str(slope)+' , y axis= '+str(b))

# Day 1c
spec_vars = T_vars_1c_G24
p = scipy.stats.linregress(df_all_max_T1c.Release_rate_log,df_all_max_T1c.Max)
slope = round(p[0],4)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = rval**2
legend5, = ax1.plot(x_T1,p[0]*x_T1+p[1], linestyle='dashed',linewidth=linewidth,color=dict_color_city[spec_vars['city']+spec_vars['day']], label=f'Toronto 1c:          ln(y) = {slope} ln(x) - {abs(b)}, $R^2$: {r2}')
print('Toronto 1c: slope= '+str(slope)+' , y axis= '+str(b)) #(0,(5,10))

p = scipy.stats.linregress(df_all_area_T1c.Release_rate_log,df_all_area_T1c.Area)
slope = round(p[0],4)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = rval**2
legend6, = ax2.plot(x_T1,p[0]*x_T1+p[1], linestyle='dashed',linewidth=linewidth,color=dict_color_city[spec_vars['city']+spec_vars['day']],label=f'Toronto 1c:          ln(y) = {slope} ln(x) + {b}, $R^2$: {r2}')
print('Toronto 1c: slope= '+str(slope)+' , y axis= '+str(b))

# Day 2c
spec_vars = T_vars_2c_G24
p = scipy.stats.linregress(df_all_max_T2c.Release_rate_log,df_all_max_T2c.Max)
slope = round(p[0],4)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
legend7, = ax1.plot(x_T2,p[0]*x_T2+p[1], linestyle='dashed',linewidth=linewidth,color=dict_color_city[spec_vars['city']+spec_vars['day']],label=f'Toronto 2c:          ln(y) = {slope} ln(x) - {abs(b)}, $R^2$: {r2}')
print('Toronto 2c: slope= '+str(slope)+' , y axis= '+str(b)) #(0,(3,10,1,10,1,10))

p = scipy.stats.linregress(df_all_area_T2c.Release_rate_log,df_all_area_T2c.Area)
slope = round(p[0],4)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
legend8, = ax2.plot(x_T2,p[0]*x_T2+p[1], linestyle='dashed',linewidth=linewidth,color=dict_color_city[spec_vars['city']+spec_vars['day']],label=f'Toronto 2c:          ln(y) = {slope} ln(x) + {b}, $R^2$: {r2}')
print('Toronto 2c: slope= '+str(slope)+' , y axis= '+str(b))


# Utrecht I --------------------------------------
spec_vars = U1_vars_G43
p = scipy.stats.linregress(df_all_max_U1.Release_rate_log,df_all_max_U1.Max)
slope = round(p[0],4)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
legend9, = ax1.plot(x_U1,p[0]*x_U1+p[1], linestyle=':',linewidth=linewidth,color=dict_color_city[spec_vars['city']], label=f'Utrecht I:               ln(y) = {slope} ln(x) - {abs(b)}, $R^2$: {r2}') #, color='black'
print('Utrecht I: slope= '+str(slope)+' , y axis= '+str(b))

p = scipy.stats.linregress(df_all_area_U1.Release_rate_log,df_all_area_U1.Area)
slope = round(p[0],4)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
legend10, = ax2.plot(x_U1,p[0]*x_U1+p[1], linestyle=':',linewidth=linewidth,color=dict_color_city[spec_vars['city']], label=f'Utrecht I:               ln(y) = {slope} ln(x) + {b}, $R^2$: {r2}')
print('Utrecht I: slope= '+str(slope)+' , y axis= '+str(b))

# Utrecht II --------------------------------------
spec_vars = U2_vars_G23
p = scipy.stats.linregress(df_all_max_U2.Release_rate_log,df_all_max_U2.Max)
slope = round(p[0],4)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
legend9, = ax1.plot(x_U2,p[0]*x_U2+p[1], linestyle=':',linewidth=linewidth,color=dict_color_city[spec_vars['city']], label=f'Utrecht II:               ln(y) = {slope} ln(x) - {abs(b)}, $R^2$: {r2}') #, color='black'
print('Utrecht: slope= '+str(slope)+' , y axis= '+str(b))

p = scipy.stats.linregress(df_all_area_U2.Release_rate_log,df_all_area_U2.Area)
slope = round(p[0],4)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
legend10, = ax2.plot(x_U2,p[0]*x_U2+p[1], linestyle=':',linewidth=linewidth,color=dict_color_city[spec_vars['city']], label=f'Utrecht II:               ln(y) = {slope} ln(x) + {b}, $R^2$: {r2}')
print('Utrecht: slope= '+str(slope)+' , y axis= '+str(b))


# London I --------------------------------------
spec_vars = L1_vars_d2_LGR
p = scipy.stats.linregress(df_all_max_L1d2.Release_rate_log,df_all_max_L1d2.Max)
slope = round(p[0],4)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
legend11, = ax1.plot(x_L1d2,p[0]*x_L1d2+p[1], linestyle='dashdot',linewidth=linewidth,color=dict_color_city[spec_vars['city']+spec_vars['day']], label=f'London I:               ln(y) = {slope} ln(x) - {abs(b)}, $R^2$: {r2}')
print('London I: slope= '+str(slope)+' , y axis= '+str(b))

p = scipy.stats.linregress(df_all_area_L1d2.Release_rate_log,df_all_area_L1d2.Area)
slope = round(p[0],4)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
legend13, = ax2.plot(x_L1d2,p[0]*x_L1d2+p[1], linestyle='dashdot',linewidth=linewidth,color=dict_color_city[spec_vars['city']+spec_vars['day']], label=f'London I:               ln(y) = {slope} ln(x) + {b}, $R^2$: {r2}')
print('London I: slope= '+str(slope)+' , y axis= '+str(b))


# London II Day 1 --------------------------------------
spec_vars = L2_vars_d1_Licor
p = scipy.stats.linregress(df_all_max_L2d1.Release_rate_log,df_all_max_L2d1.Max)
slope = round(p[0],4)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
legend11, = ax1.plot(x_L2d1,p[0]*x_L2d1+p[1], linestyle='dashdot',linewidth=linewidth,color=dict_color_city[spec_vars['city']+spec_vars['day']], label=f'London II Day1:               ln(y) = {slope} ln(x) - {abs(b)}, $R^2$: {r2}')
print('London II Day1: slope= '+str(slope)+' , y axis= '+str(b))

p = scipy.stats.linregress(df_all_area_L2d1.Release_rate_log,df_all_area_L2d1.Area)
slope = round(p[0],4)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
legend13, = ax2.plot(x_L2d1,p[0]*x_L2d1+p[1], linestyle='dashdot',linewidth=linewidth,color=dict_color_city[spec_vars['city']+spec_vars['day']], label=f'London II Day1:               ln(y) = {slope} ln(x) + {b}, $R^2$: {r2}')
print('London II Day1: slope= '+str(slope)+' , y axis= '+str(b))

# London II Day 2 --------------------------------------
spec_vars = L2_vars_d2_Licor
p = scipy.stats.linregress(df_all_max_L2d2.Release_rate_log,df_all_max_L2d2.Max)
slope = round(p[0],4)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
legend11, = ax1.plot(x_L2d2,p[0]*x_L2d2+p[1], linestyle='dashdot',linewidth=linewidth,color=dict_color_city[spec_vars['city']+spec_vars['day']], label=f'London II Day2:               ln(y) = {slope} ln(x) - {abs(b)}, $R^2$: {r2}')
print('London II Day2: slope= '+str(slope)+' , y axis= '+str(b))

p = scipy.stats.linregress(df_all_area_L2d2.Release_rate_log,df_all_area_L2d2.Area)
slope = round(p[0],4)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
legend13, = ax2.plot(x_L2d2,p[0]*x_L2d2+p[1], linestyle='dashdot',linewidth=linewidth,color=dict_color_city[spec_vars['city']+spec_vars['day']], label=f'London II Day2:               ln(y) = {slope} ln(x) + {b}, $R^2$: {r2}')
print('London II Day2: slope= '+str(slope)+' , y axis= '+str(b))

    
# Add confidence intervalls ----------------------------------------------------------------------------------

# Fit on all data:
if confidence_interval:
    add_confidenceinterval_to_plot(df_all_max.Release_rate_log,df_all_max.Max,10000,ax1,legend_handles1b,conf_level_in_std=2)
    add_confidenceinterval_to_plot(df_all_area.Release_rate_log,df_all_area.Area,10000,ax2,legend_handles2b,conf_level_in_std=2)

ax1.grid(True)
ax2.grid(True)

# LEGEND --------------------------------------------------------------------------------------------------------------

inner_legend = ax1.legend(handles=legend_handles1, loc='upper left', fontsize=14)

# Create a legend for the first subplot (ax1)
legend_ax1 = ax1.legend(handles=legend_handles1b, loc='upper center', bbox_to_anchor=(0.5, -0.22), fancybox=True, shadow=True, ncol=1,fontsize=16)

# Create a legend for the second subplot (ax2)
legend_ax2 = ax2.legend(handles=legend_handles2b,loc='upper center', bbox_to_anchor=(0.5, -0.22), fancybox=True, shadow=True, ncol=1,fontsize=16)

ax1.add_artist(inner_legend)  # Add the inner legend back to the plot
ax1.add_artist(legend_ax1)



# TITLE ------------------------------------------------------------------------------------------------------------------

#plt.suptitle('Linear fit of nat. log transformed data \nRotterdam, Utrecht, Toronto and London',fontsize=14)

ax1.set_title('Max',fontsize=22,fontweight='bold')
ax2.set_title('Area',fontsize=22,fontweight='bold')
ax1.set_ylabel(r'ln(Peak Maximum $\left[ \frac{\mathrm{ppm}}{\mathrm{10^{-6}}} \right]$)',fontsize=22,fontweight='bold') #,fontweight='bold'
ax2.set_ylabel(r'ln(Peak Area $\left[ \frac{\mathrm{ppm*m}}{\mathrm{10^{-6}*m}} \right]$)',fontsize=22,fontweight='bold')
ax1.set_xlabel(r'ln(Release Rate $\left[ \frac{\mathrm{L/min}}{1\ \mathrm{L/min}} \right]$)',fontsize=22,fontweight='bold',labelpad=5)
ax2.set_xlabel(r'ln(Release Rate $\left[ \frac{\mathrm{L/min}}{1\ \mathrm{L/min}} \right]$)',fontsize=22,fontweight='bold',labelpad=5)

# save_plots=False
if save_plots:
    
    # plt.savefig(path_fig_plot2+'All_Compare_Cities/RU2T2L2L2_AandPMvsRR_3reg_Rloc1.pdf',bbox_inches='tight')
    # plt.savefig(path_fig_plot2+'All_Compare_Cities/RU2T2L2L2_AandPMvsRR_3reg_Rloc1.svg',bbox_inches='tight')
    
    plt.savefig(path_fig_plot2+'All_Compare_Cities/RU2T2L2L2_AandPMvsRR_3reg_Rloc1_test.pdf',bbox_inches='tight')
    plt.savefig(path_fig_plot2+'All_Compare_Cities/RU2T2L2L2_AandPMvsRR_3reg_Rloc1_test.svg',bbox_inches='tight')
    
    
plt.show()











#%%% U2 locations separate

confidence_intervals = False # set to True to add confidence intervals

fig, ax = plt.subplots(1,2, figsize=(18,10))

ax1 = ax[0]
ax2 = ax[1]
ax1.grid(True)
ax2.grid(True)

# Add Shadings to visualize peak threshold -----------------------------------------

# x = np.array([-2.2,5.2])
# ax2.fill_between(x,-1,np.log(0.2),color='lightgray',alpha=0.8)

x = np.array([-2.2,5.2])
ax1.fill_between(x,-4,np.log(0.2),color='lightgray',alpha=0.4)

x = np.array([-2.2,5.2])
ax1.fill_between(x,-4,np.log(0.04),color='lightgray',alpha=0.9)

# ----------------------------------------------------------------------------


# Create empty lists to collect data for legends
legend_handles1 = [] # inner legend, displaying cities
legend_handles2 = []
legend_handles1b = [] # outer legend, displaying lin. reg. fit
legend_handles2b = []


all_area_U2_loc1 = []
all_max_U2_loc1 = []
all_area_U2_loc2 = []
all_max_U2_loc2 = []


# Create empty lists to collect data for legends
legend_handles1 = []
legend_handles2 = []



log_peaks_U2 = total_peaks_U2.copy(deep=True)
log_peaks_U2_loc1 = log_peaks_U2[((log_peaks_U2['Release_rate'] != 0) & (log_peaks_U2['Loc'] != 2) & (log_peaks_U2['Loc'] != 20))] # & (log_peaks_U2['Distance_to_source'] <75)
log_peaks_U2_loc2 = log_peaks_U2[((log_peaks_U2['Release_rate'] != 0) & ((log_peaks_U2['Loc'] == 2) | (log_peaks_U2['Loc'] == 20)))] # & (log_peaks_U2['Distance_to_source'] <75)



legend_handles_notincluded = []

x_U2_loc1,all_max_U2_loc1,all_area_U2_loc1,legend_handles1,legend_handles2 = plot2_linreg_plotscatter_color(log_peaks_U2_loc1,all_max_U2_loc1,all_area_U2_loc1,ax,legend_handles1,legend_handles2,'d',47,False,'darkorchid','Utrecht_III loc1', U2_vars_aeris,U2_vars_G23)
x_U2_loc2,all_max_U2_loc2,all_area_U2_loc2,legend_handles1,legend_handles2 = plot2_linreg_plotscatter_color(log_peaks_U2_loc2,all_max_U2_loc2,all_area_U2_loc2,ax,legend_handles1,legend_handles2,'d',47,False,'indigo','Utrecht_III loc2', U2_vars_aeris,U2_vars_G23)
  
means_area_U2_loc1, means_max_U2_loc1, median_area_U2_loc1, median_max_U2_loc1 = mean_and_median_log(all_area_U2_loc1,all_max_U2_loc1)
means_area_U2_loc2, means_max_U2_loc2, median_area_U2_loc2, median_max_U2_loc2 = mean_and_median_log(all_area_U2_loc2,all_max_U2_loc2)


# Different marker for mean
ax1.scatter(means_max_U2_loc1.index,means_max_U2_loc1.Max,marker = 'd',s=70, c= 'lightgray', edgecolor='black')
ax1.scatter(means_max_U2_loc2.index,means_max_U2_loc2.Max,marker = 'd',s=70, c= 'white', edgecolor='black')
ax2.scatter(means_area_U2_loc1.index,means_area_U2_loc1.Area,marker = 'd',s=70, c= 'lightgray', edgecolor='black')
ax2.scatter(means_area_U2_loc2.index,means_area_U2_loc2.Area,marker = 'd',s=70, c= 'white', edgecolor='black')

legend_handles1.append(ax1.scatter([], [], marker='.', color='black', label='mean'))
legend_handles2.append(ax2.scatter([], [], marker='.', color='black', label='mean'))
  


plt.subplots_adjust(wspace=0.35)

# 2nd axes --------------------------------------------------------------------
# Add 2nd x and y axis to display non logarithmis calues for peak max and area
# and actual release rates
# Plot left (PEAK MAX), y axis
ax1y = ax1.twinx()
ticks_y = [-2,0,2,4,6]
labels_y = np.round(np.exp(ticks_y),1)
ax1y.set_yticks(ticks_y) # Set the ticks and labels for the second y-axis
ax1y.set_yticklabels(labels_y)
ax1y.set_yticklabels([f"{int(label)}" if label >= 1 else f"{label:.1f}" for label in labels_y])
ax1y.set_ylabel('Peak Maximum [ppm]',fontsize=16)
ax1.set_ylim(-4,7)
ax1y.set_ylim(-4,7)

# Plot left (PEAK MAX), x axis
ax1x = ax1.twiny()
# Manually position the second x-axis below the original x-axis
ax1x.spines['bottom'].set_position(('outward', 75))  # You can adjust the position value (40) to control the offset
# ticks_x = [0, 1, 2,3]
# labels_x = np.round(np.exp(ticks_x),2)
# Display all release rates actually used on the 2nd non log x axis
labels_x = [0.15,0.3,0.5,1,2.2,3,5,10,15,20,40,80,120] # actual release rates in L/min
ticks_x = np.round(np.log(labels_x),2) # transform to log to plot it at right position on the axis
ax1x.set_xticks(ticks_x) # Set the ticks and labels for the second y-axis
ax1x.set_xticklabels(labels_x)
# Set the location of the ticks to be at the bottom
ax1x.xaxis.set_ticks_position('bottom')
ax1x.set_xlabel('Release Rate [L/min]',fontsize=16) #,labelpad=-750
ax1x.xaxis.set_label_coords(0.5, -0.208)
ax1.set_xlim(-2.2,5.2) #-0.5,3.5
ax1x.set_xlim(-2.2,5.2)

# Plot right (AREA), y axis
ax2y = ax2.twinx()
ticks_y = [0,1,2,3,4,5,6,7,8]
labels_y = np.round(np.exp(ticks_y),0)# Set the ticks and labels for the second y-axis
ax2y.set_yticks(ticks_y)
ax2y.set_yticklabels(labels_y)
ax2y.set_yticklabels([f"{int(label)}" for label in labels_y]) # Show no digits (round to integer)
ax2y.set_ylabel('Peak Area [ppm*m]',fontsize=16)
ax2.set_ylim(-1,8)
ax2y.set_ylim(-1,8)

# Plot right (AREA), x axis
ax2x = ax2.twiny()
# Manually position the second x-axis below the original x-axis
ax2x.spines['bottom'].set_position(('outward', 75))  # You can adjust the position value (40) to control the offset
# Display all release rates actually used on the 2nd non log x axis
labels_x = [0.15,0.3,0.5,1,2.2,3,5,10,15,20,40,80,120] # actual release rates in L/min
ticks_x = np.round(np.log(labels_x),2) # transform to log to plot it at right position on the axis
ax2x.set_xticks(ticks_x) # Set the ticks and labels for the second y-axis
ax2x.set_xticklabels(labels_x)
# Set the location of the ticks to be at the bottom
ax2x.xaxis.set_ticks_position('bottom')
ax2x.set_xlabel('Release Rate [L/min]',fontsize=16)
ax2x.xaxis.set_label_coords(0.5, -0.208)
ax2.set_xlim(-2.2,5.2) #-0.5,3.5
ax2x.set_xlim(-2.2,5.2)



# ----------------------------------------------------------------------------


ax1.tick_params(axis='x', labelsize=18)
ax1.tick_params(axis='y', labelsize=18)
ax2.tick_params(axis='x', labelsize=18)
ax2.tick_params(axis='y', labelsize=18)

ax1x.tick_params(axis='x', labelsize=13)
ax1y.tick_params(axis='y', labelsize=14)
ax2x.tick_params(axis='x', labelsize=13)
ax2y.tick_params(axis='y', labelsize=14)


fig.canvas.draw() # !!! necessary so that ax1.get_xticklabels() in the following access the most recent labels


# =============================================================================


df_all_max_U2_loc1 = pd.concat(all_max_U2_loc1, ignore_index=True) 
df_all_area_U2_loc1 = pd.concat(all_area_U2_loc1, ignore_index=True) 
df_all_max_U2_loc2 = pd.concat(all_max_U2_loc2, ignore_index=True) 
df_all_area_U2_loc2 = pd.concat(all_area_U2_loc2, ignore_index=True) 



# ALL DATA ------------------------------------------------------

# U II
df_all_max = pd.concat([df_all_max_U2_loc1,df_all_max_U2_loc2], ignore_index=True) #,df_all_max_L1d3
df_all_area = pd.concat([df_all_area_U2_loc1,df_all_area_U2_loc2], ignore_index=True) #df_all_area_L1d3,


x = df_all_max.Release_rate_log

p = scipy.stats.linregress(df_all_max.Release_rate_log,df_all_max.Max)
slope = round(p[0],3)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
legend14, = ax1.plot(x,p[0]*x+p[1], linestyle='dashdot',color='red',linewidth=3.5, label=f'Maximum eq.:      ln(y) = {slope} ln(x) + {b} ($R^2$ = {r2})')
legend12, = ax1.plot(x,0.817*x-0.988,linewidth=3.5,label=f'Weller (2019):      ln(y) = 0.817 ln(x) - 0.988',color='grey') #crimson
print('RU2T2L2L2: slope= '+str(slope)+' , y axis= '+str(b))
legend_handles1b.append(legend14)
legend_handles1b.append(legend12)


p = scipy.stats.linregress(df_all_area.Release_rate_log,df_all_area.Area)
slope = round(p[0],3)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
legend15, = ax2.plot(x,p[0]*x+p[1], linestyle='dashdot',color='red',linewidth=3.5, label=f'Area eq.:              ln(y) = {slope} ln(x) + {b} ($R^2$ = {r2})')
legend_handles2b.append(legend15)
print('RU2T2L2L2: slope= '+str(slope)+' , y axis= '+str(b))



    
# Add confidence intervalls ----------------------------------------------------------------------------------

# Fit on all data:
if confidence_interval:
    add_confidenceinterval_to_plot(df_all_max.Release_rate_log,df_all_max.Max,10000,ax1,legend_handles1b,conf_level_in_std=2)
    add_confidenceinterval_to_plot(df_all_area.Release_rate_log,df_all_area.Area,10000,ax2,legend_handles2b,conf_level_in_std=2)

ax1.grid(True)
ax2.grid(True)

# LEGEND --------------------------------------------------------------------------------------------------------------

inner_legend = ax1.legend(handles=legend_handles1, loc='upper left', fontsize=14)

# Create a legend for the first subplot (ax1)
legend_ax1 = ax1.legend(handles=legend_handles1b, loc='upper center', bbox_to_anchor=(0.5, -0.22), fancybox=True, shadow=True, ncol=1,fontsize=16)

# Create a legend for the second subplot (ax2)
legend_ax2 = ax2.legend(handles=legend_handles2b,loc='upper center', bbox_to_anchor=(0.5, -0.22), fancybox=True, shadow=True, ncol=1,fontsize=16)

ax1.add_artist(inner_legend)  # Add the inner legend back to the plot
ax1.add_artist(legend_ax1)




# TITLE ------------------------------------------------------------------------------------------------------------------

#plt.suptitle('Linear fit of nat. log transformed data \nRotterdam, Utrecht, Toronto and London',fontsize=14)

ax1.set_title('Max',fontsize=22,fontweight='bold')
ax2.set_title('Area',fontsize=22,fontweight='bold')
ax1.set_ylabel(r'ln(Peak Maximum $\left[ \frac{\mathrm{ppm}}{\mathrm{10^{-6}}} \right]$)',fontsize=22,fontweight='bold') #,fontweight='bold'
ax2.set_ylabel(r'ln(Peak Area $\left[ \frac{\mathrm{ppm*m}}{\mathrm{10^{-6}*m}} \right]$)',fontsize=22,fontweight='bold')
ax1.set_xlabel(r'ln(Release Rate $\left[ \frac{\mathrm{L/min}}{1\ \mathrm{L/min}} \right]$)',fontsize=22,fontweight='bold',labelpad=5)
ax2.set_xlabel(r'ln(Release Rate $\left[ \frac{\mathrm{L/min}}{1\ \mathrm{L/min}} \right]$)',fontsize=22,fontweight='bold',labelpad=5)

save_plots=False
if save_plots:
    plt.savefig(path_fig_plot2+'All_Compare_Cities/UII_AandPMvsRR_3reg_locsseperate.pdf',bbox_inches='tight')
    plt.savefig(path_fig_plot2+'All_Compare_Cities/UII_AandPMvsRR_3reg_locsseperate.svg',bbox_inches='tight')
    
plt.show()


#%%% RU2T2L2L2 - G2301 only

confidence_interval = False # set to TRUE to add confidence intervals

fig, ax = plt.subplots(1,2, figsize=(18,10))

ax1 = ax[0]
ax2 = ax[1]
ax1.grid(True)
ax2.grid(True)

# Add Shadings to visualize peak threshold -----------------------------------------

# x = np.array([-2.2,5.2])
# ax2.fill_between(x,-1,np.log(0.2),color='lightgray',alpha=0.8)

x = np.array([-2.2,5.2])
ax1.fill_between(x,-4,np.log(0.2),color='lightgray',alpha=0.4)

x = np.array([-2.2,5.2])
ax1.fill_between(x,-4,np.log(0.04),color='lightgray',alpha=0.9)

# ----------------------------------------------------------------------------


# Create empty lists to collect data for legends
legend_handles1 = [] # inner legend, displaying cities
legend_handles2 = []
legend_handles1b = [] # outer legend, displaying lin. reg. fit
legend_handles2b = []


all_area_R = []
all_max_R = []
all_area_U1 = []
all_max_U1 = []
all_area_U2 = []
all_max_U2 = []
all_area_T1b = []
all_max_T1b = []
all_area_T1c = []
all_max_T1c = []
all_area_T2c = []
all_max_T2c = []
all_area_L1d2 = []
all_max_L1d2 = []
all_area_L1d5 = []
all_max_L1d5 = []
all_area_L2d1 = []
all_max_L2d1 = []
all_area_L2d2 = []
all_max_L2d2 = []


# Create empty lists to collect data for legends
legend_handles1 = []
legend_handles2 = []


log_peaks_R = total_peaks_R.copy(deep=True)
log_peaks_U1 = total_peaks_U1.copy(deep=True)
log_peaks_U2 = total_peaks_U2.copy(deep=True)
log_peaks_T1b = total_peaks_T1b.copy(deep=True)
log_peaks_T1c = total_peaks_T1c.copy(deep=True)
log_peaks_T2c = total_peaks_T2c.copy(deep=True)
log_peaks_L1d2 = total_peaks_L1d2.copy(deep=True)
log_peaks_L1d5 = total_peaks_L1d5.copy(deep=True)
log_peaks_L2d1 = total_peaks_L2d1.copy(deep=True)
log_peaks_L2d2 = total_peaks_L2d2.copy(deep=True)
log_peaks_T1b = log_peaks_T1b[log_peaks_T1b['Release_rate'] != 0]
log_peaks_T1c = log_peaks_T1c[log_peaks_T1c['Release_rate'] != 0]
log_peaks_T2c = log_peaks_T2c[log_peaks_T2c['Release_rate'] != 0]
log_peaks_L1d2 = log_peaks_L1d2[((log_peaks_L1d2['Release_rate'] != 0))]
log_peaks_L1d5 = log_peaks_L1d5[((log_peaks_L1d5['Release_rate'] != 0))]
log_peaks_L1d2 = log_peaks_L1d2[((log_peaks_L1d2['Release_rate'] != 0) & (log_peaks_L1d2['Distance_to_source'] <75))]
log_peaks_L1d5 = log_peaks_L1d5[((log_peaks_L1d5['Release_rate'] != 0) & (log_peaks_L1d5['Distance_to_source'] <75))]
log_peaks_L2d1 = log_peaks_L2d1[((log_peaks_L2d1['Release_rate'] != 0) & (log_peaks_L2d1['Distance_to_source'] <75))]
log_peaks_L2d2 = log_peaks_L2d2[((log_peaks_L2d2['Release_rate'] != 0) )] #& (log_peaks_L2d2['Distance_to_source'] <75)
log_peaks_U2 = log_peaks_U2[((log_peaks_U2['Release_rate'] != 0))] # & (log_peaks_U2['Distance_to_source'] <75)




legend_handles_notincluded = []

x_R,all_max_R,all_area_R,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_R,all_max_R,all_area_R,ax,legend_handles1,legend_handles2,'.',65,False, R_vars_G23)

x_U1,all_max_U1,all_area_U1,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_U1,all_max_U1,all_area_U1,ax,legend_handles1,legend_handles2,'v',47,False,U1_vars_G23)

x_U2,all_max_U2,all_area_U2,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_U2,all_max_U2,all_area_U2,ax,legend_handles1,legend_handles2,'d',47,False,U2_vars_G23)
 
x_L1d2,all_max_L1d2,all_area_L1d2,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_L1d2,all_max_L1d2,all_area_L1d2,ax,legend_handles1,legend_handles2,'p',50,True,L1_vars_d2_G23) #London Day2
  
x_L1d5,all_max_L1d5,all_area_L1d5,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_L1d5,all_max_L1d5,all_area_L1d5,ax,legend_handles1,legend_handles2,'*',80,True,L1_vars_d5_G23) 

# x_L2d1,all_max_L2d1,all_area_L2d1,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_L2d1,all_max_L2d1,all_area_L2d1,ax,legend_handles1,legend_handles2,'^',47,True,L2_vars_d1_Licor)

# x_L2d2,all_max_L2d2,all_area_L2d2,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_L2d2,all_max_L2d2,all_area_L2d2,ax,legend_handles1,legend_handles2,'^',47,True,L2_vars_d2_Licor)

# x_T1,all_max_T1b,all_area_T1b,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_T1b,all_max_T1b,all_area_T1b,ax,legend_handles1,legend_handles2,'D',55,True,T_vars_1b_LGR)

# x_T1,all_max_T1c,all_area_T1c,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_T1c,all_max_T1c,all_area_T1c,ax,legend_handles1,legend_handles2,'>',55,True,T_vars_1c_G24) # marker size none

# x_T2,all_max_T2c,all_area_T2c,legend_handles1,legend_handles2 = plot2_linreg_plotscatter(log_peaks_T2c,all_max_T2c,all_area_T2c,ax,legend_handles1,legend_handles2,'X',55,True,T_vars_2c_G24) # marker size none


means_area_R, means_max_R, median_area_R, median_max_R = mean_and_median_log(all_area_R,all_max_R)
means_area_U1, means_max_U1, median_area_U1, median_max_U1 = mean_and_median_log(all_area_U1,all_max_U1)
means_area_U2, means_max_U2, median_area_U2, median_max_U2 = mean_and_median_log(all_area_U2,all_max_U2)
means_area_L1d2, means_max_L1d2, median_area_L1d2, median_max_L1d2 = mean_and_median_log(all_area_L1d2,all_max_L1d2)
means_area_L1d5, means_max_L1d5, median_area_L1d5, median_max_L1d5 = mean_and_median_log(all_area_L1d5,all_max_L1d5)
# means_area_T1b, means_max_T1b, median_area_T1b, median_max_T1b = mean_and_median_log(all_area_T1b,all_max_T1b)
# means_area_T1c, means_max_T1c, median_area_T1c, median_max_T1c = mean_and_median_log(all_area_T1c,all_max_T1c)
# means_area_T2c, means_max_T2c, median_area_T2c, median_max_T2c = mean_and_median_log(all_area_T2c,all_max_T2c)
# means_area_L2d1, means_max_L2d1, median_area_L2d1, median_max_L2d1 = mean_and_median_log(all_area_L2d1,all_max_L2d1)
# means_area_L2d2, means_max_L2d2, median_area_L2d2, median_max_L2d2 = mean_and_median_log(all_area_L2d2,all_max_L2d2)



# Different marker for mean
ax1.scatter(means_max_R.index,means_max_R.Max,marker = '.',s=95, c= '#484848', edgecolor='black') #,label = 'mean'
ax1.scatter(means_max_U1.index,means_max_U1.Max,marker = 'v',s=70, c= '#484848', edgecolor='black')
ax1.scatter(means_max_U2.index,means_max_U2.Max,marker = 'd',s=70, c= '#484848', edgecolor='black')
ax1.scatter(means_max_L1d2.index,means_max_L1d2.Max,marker = 'p',s=100, c= '#484848', edgecolor='black')
ax1.scatter(means_max_L1d5.index,means_max_L1d5.Max,marker = '*',s=100,c= '#484848', edgecolor='black')
# ax1.scatter(means_max_T1b.index,means_max_T1b.Max,marker = 'D',s=40,c= '#484848', edgecolor='black')
# ax1.scatter(means_max_T1c.index,means_max_T1c.Max,marker = '>',s=40,c= '#484848', edgecolor='black')
# ax1.scatter(means_max_T2c.index,means_max_T2c.Max,marker = 'X',s=40,c= '#484848', edgecolor='black')
# ax1.scatter(means_max_L2d1.index,means_max_L2d1.Max,marker = '^',s=100,c= '#484848', edgecolor='black')
# ax1.scatter(means_max_L2d2.index,means_max_L2d2.Max,marker = '^',s=100,c= '#484848', edgecolor='black')


ax2.scatter(means_area_R.index,means_area_R.Area,marker = '.',s=95, c= '#484848', edgecolor='black') #,label = 'mean'
ax2.scatter(means_area_U1.index,means_area_U1.Area,marker = 'v',s=70, c= '#484848', edgecolor='black')
ax2.scatter(means_area_U2.index,means_area_U2.Area,marker = 'd',s=70, c= '#484848', edgecolor='black')
ax2.scatter(means_area_L1d2.index,means_area_L1d2.Area,marker = 'p',s=100, c= '#484848', edgecolor='black')
ax2.scatter(means_area_L1d5.index,means_area_L1d5.Area,marker = '*',s=100,c= '#484848', edgecolor='black')
# ax2.scatter(means_area_T1b.index,means_area_T1b.Area,marker = 'D',s=40,c= '#484848', edgecolor='black')
# ax2.scatter(means_area_T1c.index,means_area_T1c.Area,marker = '>',s=40,c= '#484848', edgecolor='black')
# ax2.scatter(means_area_T2c.index,means_area_T2c.Area,marker = 'X',s=40,c= '#484848', edgecolor='black')
# ax2.scatter(means_area_L2d1.index,means_area_L2d1.Area,marker = '^',s=100,c= '#484848', edgecolor='black')
# ax2.scatter(means_area_L2d2.index,means_area_L2d2.Area,marker = '^',s=100,c= '#484848', edgecolor='black')

legend_handles1.append(ax1.scatter([], [], marker='.', color='black', label='mean'))
legend_handles2.append(ax2.scatter([], [], marker='.', color='black', label='mean'))
  


plt.subplots_adjust(wspace=0.35)

# 2nd axes --------------------------------------------------------------------
# Add 2nd x and y axis to display non logarithmis calues for peak max and area
# and actual release rates
# Plot left (PEAK MAX), y axis
ax1y = ax1.twinx()
ticks_y = [-2,0,2,4,6]
labels_y = np.round(np.exp(ticks_y),1)
ax1y.set_yticks(ticks_y) # Set the ticks and labels for the second y-axis
ax1y.set_yticklabels(labels_y)
ax1y.set_yticklabels([f"{int(label)}" if label >= 1 else f"{label:.1f}" for label in labels_y])
ax1y.set_ylabel('Peak Maximum [ppm]',fontsize=16)
ax1.set_ylim(-4,7)
ax1y.set_ylim(-4,7)

# Plot left (PEAK MAX), x axis
ax1x = ax1.twiny()
# Manually position the second x-axis below the original x-axis
ax1x.spines['bottom'].set_position(('outward', 75))  # You can adjust the position value (40) to control the offset
# ticks_x = [0, 1, 2,3]
# labels_x = np.round(np.exp(ticks_x),2)
# Display all release rates actually used on the 2nd non log x axis
labels_x = [0.15,0.3,0.5,1,2.2,3,5,10,15,20,40,80,120] # actual release rates in L/min
ticks_x = np.round(np.log(labels_x),2) # transform to log to plot it at right position on the axis
ax1x.set_xticks(ticks_x) # Set the ticks and labels for the second y-axis
ax1x.set_xticklabels(labels_x)
# Set the location of the ticks to be at the bottom
ax1x.xaxis.set_ticks_position('bottom')
ax1x.set_xlabel('Release Rate [L/min]',fontsize=16) #,labelpad=-750
ax1x.xaxis.set_label_coords(0.5, -0.208)
ax1.set_xlim(-2.2,5.2) #-0.5,3.5
ax1x.set_xlim(-2.2,5.2)

# Plot right (AREA), y axis
ax2y = ax2.twinx()
ticks_y = [0,1,2,3,4,5,6,7,8]
labels_y = np.round(np.exp(ticks_y),0)# Set the ticks and labels for the second y-axis
ax2y.set_yticks(ticks_y)
ax2y.set_yticklabels(labels_y)
ax2y.set_yticklabels([f"{int(label)}" for label in labels_y]) # Show no digits (round to integer)
ax2y.set_ylabel('Peak Area [ppm*m]',fontsize=16)
ax2.set_ylim(-1,8)
ax2y.set_ylim(-1,8)

# Plot right (AREA), x axis
ax2x = ax2.twiny()
# Manually position the second x-axis below the original x-axis
ax2x.spines['bottom'].set_position(('outward', 75))  # You can adjust the position value (40) to control the offset
# Display all release rates actually used on the 2nd non log x axis
labels_x = [0.15,0.3,0.5,1,2.2,3,5,10,15,20,40,80,120] # actual release rates in L/min
ticks_x = np.round(np.log(labels_x),2) # transform to log to plot it at right position on the axis
ax2x.set_xticks(ticks_x) # Set the ticks and labels for the second y-axis
ax2x.set_xticklabels(labels_x)
# Set the location of the ticks to be at the bottom
ax2x.xaxis.set_ticks_position('bottom')
ax2x.set_xlabel('Release Rate [L/min]',fontsize=16)
ax2x.xaxis.set_label_coords(0.5, -0.208)
ax2.set_xlim(-2.2,5.2) #-0.5,3.5
ax2x.set_xlim(-2.2,5.2)


# ----------------------------------------------------------------------------

ax1.tick_params(axis='x', labelsize=18)
ax1.tick_params(axis='y', labelsize=18)
ax2.tick_params(axis='x', labelsize=18)
ax2.tick_params(axis='y', labelsize=18)

ax1x.tick_params(axis='x', labelsize=13)
ax1y.tick_params(axis='y', labelsize=14)
ax2x.tick_params(axis='x', labelsize=13)
ax2y.tick_params(axis='y', labelsize=14)


fig.canvas.draw() # !!! necessary so that ax1.get_xticklabels() in the following access the most recent labels


# =============================================================================


df_all_max_R = pd.concat(all_max_R, ignore_index=True) 
df_all_area_R = pd.concat(all_area_R, ignore_index=True) 
df_all_max_U1 = pd.concat(all_max_U1, ignore_index=True) 
df_all_area_U1 = pd.concat(all_area_U1, ignore_index=True)
df_all_max_U2 = pd.concat(all_max_U2, ignore_index=True) 
df_all_area_U2 = pd.concat(all_area_U2, ignore_index=True) 
df_all_max_L1d2 = pd.concat(all_max_L1d2, ignore_index=True) 
df_all_area_L1d2 = pd.concat(all_area_L1d2, ignore_index=True) 
df_all_max_L1d5 = pd.concat(all_max_L1d5) 
df_all_area_L1d5 = pd.concat(all_area_L1d5) 
# df_all_max_T1b = pd.concat(all_max_T1b) 
# df_all_area_T1b = pd.concat(all_area_T1b) 
# df_all_max_T1c = pd.concat(all_max_T1c) 
# df_all_area_T1c = pd.concat(all_area_T1c)  
# df_all_max_T2c = pd.concat(all_max_T2c) 
# df_all_area_T2c = pd.concat(all_area_T2c)  
# df_all_max_L2d1 = pd.concat(all_max_L2d1, ignore_index=True) #
# df_all_area_L2d1 = pd.concat(all_area_L2d1, ignore_index=True) #
# df_all_max_L2d2 = pd.concat(all_max_L2d2, ignore_index=True) #
# df_all_area_L2d2 = pd.concat(all_area_L2d2, ignore_index=True) #


# RU2T2L2L2
df_all_max = pd.concat([df_all_max_R,df_all_max_U1,df_all_max_U2,df_all_max_L1d2,df_all_max_L1d5], ignore_index=True) 
df_all_area = pd.concat([df_all_area_R,df_all_area_U1,df_all_area_U2,df_all_area_L1d2,df_all_area_L1d5], ignore_index=True) 




# ALL DATA ------------------------------------------------------

x = df_all_max.Release_rate_log

p = scipy.stats.linregress(df_all_max.Release_rate_log,df_all_max.Max)
slope = round(p[0],3)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
legend14, = ax1.plot(x,p[0]*x+p[1], linestyle='dashdot',color='red',linewidth=3.5, label=f'Maximum eq.:      ln(y) = {slope} ln(x) + {b} ($R^2$ = {r2})')
legend12, = ax1.plot(x,0.817*x-0.988,linewidth=3.5,label=f'Weller (2019):      ln(y) = 0.817 ln(x) - 0.988',color='grey') #crimson
print('RU2L2: slope= '+str(slope)+' , y axis= '+str(b))
legend_handles1b.append(legend14)
legend_handles1b.append(legend12)


p = scipy.stats.linregress(df_all_area.Release_rate_log,df_all_area.Area)
slope = round(p[0],3)
b = round(p[1],2)
rval  = round(p[2],2)
r2 = round(rval**2,2)
legend15, = ax2.plot(x,p[0]*x+p[1], linestyle='dashdot',color='red',linewidth=3.5, label=f'Area eq.:              ln(y) = {slope} ln(x) + {b} ($R^2$ = {r2})')
legend_handles2b.append(legend15)
print('RU2L2: slope= '+str(slope)+' , y axis= '+str(b))




    
# Add confidence intervalls ----------------------------------------------------------------------------------

# Fit on all data:
if confidence_interval:
    add_confidenceinterval_to_plot(df_all_max.Release_rate_log,df_all_max.Max,10000,ax1,legend_handles1b,conf_level_in_std=2)
    add_confidenceinterval_to_plot(df_all_area.Release_rate_log,df_all_area.Area,10000,ax2,legend_handles2b,conf_level_in_std=2)

ax1.grid(True)
ax2.grid(True)

# LEGEND --------------------------------------------------------------------------------------------------------------

inner_legend = ax1.legend(handles=legend_handles1, loc='upper left', fontsize=14)

# Create a legend for the first subplot (ax1)
legend_ax1 = ax1.legend(handles=legend_handles1b, loc='upper center', bbox_to_anchor=(0.5, -0.22), fancybox=True, shadow=True, ncol=1,fontsize=16)

# Create a legend for the second subplot (ax2)
legend_ax2 = ax2.legend(handles=legend_handles2b,loc='upper center', bbox_to_anchor=(0.5, -0.22), fancybox=True, shadow=True, ncol=1,fontsize=16)

ax1.add_artist(inner_legend)  # Add the inner legend back to the plot
ax1.add_artist(legend_ax1)




# TITLE ------------------------------------------------------------------------------------------------------------------

#plt.suptitle('Linear fit of nat. log transformed data \nRotterdam, Utrecht, Toronto and London',fontsize=14)

ax1.set_title('Max',fontsize=22,fontweight='bold')
ax2.set_title('Area',fontsize=22,fontweight='bold')
ax1.set_ylabel(r'ln(Peak Maximum $\left[ \frac{\mathrm{ppm}}{\mathrm{10^{-6}}} \right]$)',fontsize=22,fontweight='bold') #,fontweight='bold'
ax2.set_ylabel(r'ln(Peak Area $\left[ \frac{\mathrm{ppm*m}}{\mathrm{10^{-6}*m}} \right]$)',fontsize=22,fontweight='bold')
ax1.set_xlabel(r'ln(Release Rate $\left[ \frac{\mathrm{L/min}}{1\ \mathrm{L/min}} \right]$)',fontsize=22,fontweight='bold',labelpad=5)
ax2.set_xlabel(r'ln(Release Rate $\left[ \frac{\mathrm{L/min}}{1\ \mathrm{L/min}} \right]$)',fontsize=22,fontweight='bold',labelpad=5)

save_plots=False
if save_plots:
    plt.savefig(path_fig_plot2+'All_Compare_Cities/RU2L2_AandPMvsRR_3reg_G2301.pdf',bbox_inches='tight')
    plt.savefig(path_fig_plot2+'All_Compare_Cities/RU2L2_AandPMvsRR_3reg_G2301.svg',bbox_inches='tight')
    

plt.show()




#%% P3: Lollipop Plots
  

# rr = 0.15 L/min
plot_lollipop(df_R_comb,0.15,[-2.35,4.5,-2.35,4.5],path_save=path_fig_plot3)
plot_lollipop_bothinsameplot(df_R_comb,0.15,[-2.5,4.8],path_save=path_fig_plot3)
a_std, a_relstd, a_mean = std_larger_for_area_or_max(df_R_comb,0.15)

# rr = 3.33 L/min
plot_lollipop(df_R_comb,3.33,[-1.2,6.5,-1.2,6.5],path_save=path_fig_plot3)
plot_lollipop_bothinsameplot(df_R_comb,3.33,[-1.2,6.5],path_save=path_fig_plot3)
a_std, a_relstd, a_mean = std_larger_for_area_or_max(df_R_comb,3.33)

# rr = 1 L/min
plot_lollipop(df_R_comb,1,[-2,4.6,-2,4.6],path_save=path_fig_plot3)
plot_lollipop_bothinsameplot(df_R_comb,1,[-2,4.6],path_save=path_fig_plot3)

# rr = 5 L/min
plot_lollipop(df_R_comb,5,[-1.2,5.8,-1.2,5.8],path_save=path_fig_plot3)
plot_lollipop_bothinsameplot(df_R_comb,5,[-1.2,5.8],path_save=path_fig_plot3)

# rr = 10 L/min
plot_lollipop(df_R_comb,10,[-1.2,5.8,-1.2,5.8],path_save=path_fig_plot3)
plot_lollipop_bothinsameplot(df_R_comb,10,[-1.2,5.8],path_save=path_fig_plot3)

# rr = 20 L/min - MORNING
plot_lollipop(df_R_comb.loc[:'2022-09-06 09:24:00'],20,[-1.5,7,-1.5,7], daytime='Morning', path_save=path_fig_plot3)
plot_lollipop_bothinsameplot(df_R_comb.loc[:'2022-09-06 09:24:00'],20,[-1.5,7], daytime='Morning',path_save=path_fig_plot3)

# rr = 20 L/min - AFTERNOON
plot_lollipop(df_R_comb.loc['2022-09-06 11:04:00':],20,[-1.5,7,-1.5,7], daytime='Afternoon', path_save=path_fig_plot3)
plot_lollipop_bothinsameplot(df_R_comb.loc['2022-09-06 11:04:00':],20,[-1.5,7], daytime='Afternoon',path_save=path_fig_plot3)

# rr = 40 L/min - MORNING
plot_lollipop(df_R_comb.loc[:'2022-09-06 09:55:00'],40,[-1.5,7,-1.5,7], daytime='Morning',path_save=path_fig_plot3)
plot_lollipop_bothinsameplot(df_R_comb.loc[:'2022-09-06 09:55:00'],40,[-1.5,7], daytime='Morning',path_save=path_fig_plot3)

# rr = 40 L/min - AFTERNOON
plot_lollipop(df_R_comb.loc['2022-09-06 11:47:00':],40,[-1.5,7,-1.5,7], daytime='Afternoon',path_save=path_fig_plot3)
plot_lollipop_bothinsameplot(df_R_comb.loc['2022-09-06 11:47:00':],40,[-1.5,7], daytime='Afternoon',path_save=path_fig_plot3)

# rr = 80 L/min
plot_lollipop(df_R_comb,80,[-2,8,-2,8],path_save=path_fig_plot3)
plot_lollipop_bothinsameplot(df_R_comb,80,[-2,8],path_save=path_fig_plot3)

# rr = 120 L/min
plot_lollipop(df_R_comb,120,[-2,8,-2,8],path_save=path_fig_plot3)
plot_lollipop_bothinsameplot(df_R_comb,120,[-2,8],path_save=path_fig_plot3)






#%% STATS

#%%% Length peaks

def calc_length_peak(df_city,name,output_file=None):
    df = df_city.copy(deep=True)
    # Convert index to DatetimeIndex if it's not already
    df.index = pd.to_datetime(df.index)
    df['Peakstart_QC'] = pd.to_datetime(df['Peakstart_QC'], errors='coerce')
    df['Peakend_QC'] = pd.to_datetime(df['Peakend_QC'], errors='coerce')
    df['Peaklength_QC'] = (df['Peakend_QC'] - df['Peakstart_QC']).dt.total_seconds()
    df['Peaklength_1stpart'] = (df.index - df['Peakstart_QC']).dt.total_seconds()
    df['Peaklength_2ndpart'] = (df['Peakend_QC'] - df.index).dt.total_seconds()
    
    # Calculate mean, median, 10th and 90th percentile of 'Peaklength_QC'
    mean_peak_length = df['Peaklength_QC'].mean()
    median_peak_length = df['Peaklength_QC'].median()
    percentile_10 = df['Peaklength_QC'].quantile(0.1)
    percentile_90 = df['Peaklength_QC'].quantile(0.9)
    max_peak_length = df['Peaklength_QC'].max()
    max_1stpart = df['Peaklength_1stpart'].max()
    max_2ndpart = df['Peaklength_2ndpart'].max()

    print('-------------------------------------------------')
    print(name)
    print("Mean Peak Length:", round(mean_peak_length))
    print("Median Peak Length:", round(median_peak_length))
    print("10th Percentile:", round(percentile_10))
    print("90th Percentile:", round(percentile_90))
    print("Max Peak Length:", round(max_peak_length))
    print("Max Length 1st Part:", round(max_1stpart))
    print("Max Length 2nd Part:", round(max_2ndpart))
    print("95th Percentile 1st Part:", round(df['Peaklength_1stpart'].quantile(0.95)))
    print("95th Percentile 2nd Part:", round(df['Peaklength_2ndpart'].quantile(0.95)))

    # Assuming df is your DataFrame
    max_index = df['Peaklength_QC'].idxmax()
    
    print("Index with maximum Peaklength_QC:", max_index)
    
    if output_file:
        with open(output_file, 'a') as f:
            f.write('-------------------------------------------------\n')
            f.write(name + '\n')
            f.write("Mean Peak Length: {}\n".format(round(mean_peak_length)))
            f.write("Median Peak Length: {}\n".format(round(median_peak_length)))
            f.write("10th Percentile: {}\n".format(round(percentile_10)))
            f.write("90th Percentile: {}\n".format(round(percentile_90)))
            f.write("Max Peak Length: {}\n".format(round(max_peak_length)))
            f.write("Max Length 1st Part: {}\n".format(round(max_1stpart)))
            f.write("Max Length 2nd Part: {}\n".format(round(max_2ndpart)))
            f.write("95th Percentile 1st Part: {}\n".format(round(df['Peaklength_1stpart'].quantile(0.95))))
            f.write("95th Percentile 2nd Part: {}\n".format(round(df['Peaklength_2ndpart'].quantile(0.95))))
        
# output_file = path_res / "Statistics/Statistics_on_peaklengths.txt"  # Specify the output file path
output_file = None
calc_length_peak(total_peaks_R, 'Rotterdam', output_file)
calc_length_peak(total_peaks_U1, 'Utrecht', output_file)
calc_length_peak(total_peaks_L1d2, 'London D2', output_file)
calc_length_peak(total_peaks_L1d5,'London D5', output_file)
calc_length_peak(total_peaks_T1b,'Toronto 1b', output_file)
calc_length_peak(total_peaks_T1c,'Toronto 1c', output_file)
calc_length_peak(total_peaks_T2c,'Toronto 2c', output_file)


#%%% Distance and Speed

df_stats = df_all.copy()
df_stats = df_stats[df_stats['Distance_to_source']<75]
df_all_mean = df_stats.groupby(['City']).mean()
df_all_median = df_stats.groupby(['City']).median()
df_all_min = df_stats.groupby(['City']).min()
df_all_max = df_stats.groupby(['City']).max()
df_all_std = df_stats.groupby(['City']).std()

print('Distance') #-------------------------------------
print('Mean')
print(df_all_mean['Distance_to_source'])
print('Median')
print(df_all_median['Distance_to_source'])
print('Minimum')
print(df_all_min['Distance_to_source'])
print('Maximum')
print(df_all_max['Distance_to_source'])

print('Speed') #----------------------------------------
print('Mean')
print(df_all_mean['Mean_speed'])
print('Median')
print(df_all_median['Mean_speed'])
print('Minimum')
print(df_all_min['Mean_speed'])
print('Maximum')
print(df_all_max['Mean_speed'])
print('Std')
print(df_all_std['Mean_speed'])








#%% End Script






