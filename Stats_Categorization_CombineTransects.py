# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:16:26 2024

@author: Judith Tettenborn (j.a.f.tettenborn@uu.nl)


Uses the output of the Monte Carlo analysis:

One excel file per release (e.g. the 20 Lmin-1 release in Rotterdam) with N sheets 
(N=maximal number of transects obtained for this specific release -> varies)

Each sheet has the columns: 
rE_Weller - emission rate estimation using the Weller eq. given the peak maximum
rE_ALL_max - emission rate estimation using the Maximum eq. given the peak maximum
rE_ALL_area - emission rate estimation using the Area eq. given the spatial peak area
ln(Max) - natural logarithm of the peak maximum
ln(Area) - natural logarithm of the spatial peak area

Each row in the column ln(Area) in the sheet N is the average taken over N ln(Area) samples 
randomly drawn in the Monte Carlo analysis (i.e. for N=3, three ln(Area) values
for this release rate were drawn and an average calculated). Similarly, the rE_ALL_area
values are the emission rate estimation with the Area eq. based on this average.


Similar to the script "Stats_Categorization", the emission rate estimations are
classified into emission categories:
<0.5 Lmin−1 - Very low
0.5−6 Lmin−1 - Low
6−40 Lmin−1 - Medium
>40 Lmin−1 - High


The Sankey diagrams visualize the categorization success when taking into account
N transects with N = 1, 3 and 6.


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
import time
import matplotlib.pyplot as plt
from pysankey import sankey

from helper_functions.constants import (
    dict_color_category
    )

from helper_functions.utils import *
from stats_analysis.stats_functions import *



#%% LOAD DATA

path_statsdata   = path_base / 'data' / 'final' / 'analysis_montecarlo/'
if not (path_base / 'results' / 'Figures' / 'STATS' / 'Categorization_CombineTransects').is_dir():
       (path_base / 'results' / 'Figures' / 'STATS' / 'Categorization_CombineTransects').mkdir(parents=True)
path_savefig     = path_base / 'results' / 'Figures' / 'STATS' / 'Categorization_CombineTransects/'



        
'''
N>10 - all releases are included for which at least 10 measurements exist
N>20 - all releases are included for which at least 20 measurements exist

negative curves: Only distributions (of area/maximum measurements) with a small 
to medium offset. Name refers to the shape of the curve of the figure from the Monte Carlo analysis 
(Figure 5 in the publication). 

positive curves: Only distributions with a large offset -> due to the large offset 
of those distributions from the regression line, the emission rate estimations are 
that much off, that using additional measurements is not improving the estimation.
This can happen due to very different env. conditions (built environment or weather conditions).

    '''


       
##################################################################################################
### FINAL - used in Tettenborn et al. (2025) #####################################################

# Only negative curve - #N >= 10 -------------------------------------------------------------

filenames = ["MonteCarlo_Results_London_I-Day2_loc1.0_rr35.0_MC2000.xlsx","MonteCarlo_Results_London_I-Day2_loc1.0_rr70.0_MC2000.xlsx","MonteCarlo_Results_London_I-Day5_loc1.0_rr70.0_MC2000.xlsx",
              "MonteCarlo_Results_Rotterdam_loc1.0_rr5.0_MC2000.xlsx","MonteCarlo_Results_Rotterdam_loc1.0_rr10.0_MC2000.xlsx",
              "MonteCarlo_Results_Rotterdam_loc1.0_rr20.0_MC2000.xlsx","MonteCarlo_Results_Rotterdam_loc1.0_rr40.0_MC2000.xlsx","MonteCarlo_Results_Rotterdam_loc1.0_rr80.0_MC2000.xlsx",
              "MonteCarlo_Results_Toronto-2c_loc1.0_rr0.5_MC2000.xlsx","MonteCarlo_Results_Toronto-2c_loc1.0_rr1.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_I_loc1.0_rr2.18_MC2000.xlsx",
              "MonteCarlo_Results_Utrecht_I_loc1.0_rr3.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_I_loc2.0_rr15.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc1.0_rr4.0_MC2000.xlsx",
              "MonteCarlo_Results_Utrecht_II_loc1.0_rr15.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc1.0_rr20.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc1.0_rr100.0_MC2000.xlsx",
              "MonteCarlo_Results_Utrecht_II_loc2.0_rr0.15_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc2.0_rr0.5_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc2.0_rr1.0_MC2000.xlsx",
              "MonteCarlo_Results_Utrecht_II_loc2.0_rr2.2_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc2.0_rr2.5_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc2.0_rr4.0_MC2000.xlsx",
              "MonteCarlo_Results_Utrecht_II_loc2.0_rr20.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc3.0_rr4.0_MC2000.xlsx"] 
release_rates_readin = [35,70,70,5,10,20,40,80,0.5,1,2.18,3,15,4,15,20,100,0.15,0.5,1,2.2,2.5,4,20,4] 
city_names = ['London_I-Day2','London_I-Day2','London_I-Day5',
              'Rotterdam','Rotterdam','Rotterdam','Rotterdam','Rotterdam','Toronto-2c','Toronto-2c','Utrecht','Utrecht','Utrecht','Utrecht_II','Utrecht_II','Utrecht_II','Utrecht_II','Utrecht_II','Utrecht_II',
              'Utrecht_II','Utrecht_II','Utrecht_II','Utrecht_II','Utrecht_II','Utrecht_II']
locs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0,2.0, 2.0, 2.0, 2.0, 2.0, 3.0]

##################################################################################################
##################################################################################################
        
  
    
# TESTS TESTS TESTS TESTS TESTS TESTS TESTS TESTS TESTS TESTS TESTS TESTS TESTS TESTS TESTS TESTS
# All - #N >= 10 =============================================================================

# filenames = ["MonteCarlo_Results_London_I-Day2_loc1.0_rr35.0_MC2000.xlsx","MonteCarlo_Results_London_I-Day2_loc1.0_rr70.0_MC2000.xlsx","MonteCarlo_Results_London_I-Day5_loc1.0_rr70.0_MC2000.xlsx",
#               "MonteCarlo_Results_London_II-Day1_loc1.0_rr0.99_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr5.64_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr10.63_MC2000.xlsx",
#               "MonteCarlo_Results_London_II-Day1_loc1.0_rr30.58_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr30.6_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr50.52_MC2000.xlsx",
#               "MonteCarlo_Results_London_II-Day1_loc1.0_rr70.48_MC2000.xlsx","MonteCarlo_Results_London_II-Day2_loc1.0_rr0.49_MC2000.xlsx","MonteCarlo_Results_London_II-Day2_loc1.0_rr0.99_MC2000.xlsx",
#               "MonteCarlo_Results_Rotterdam_loc1.0_rr5.0_MC2000.xlsx","MonteCarlo_Results_Rotterdam_loc1.0_rr10.0_MC2000.xlsx",
#               "MonteCarlo_Results_Rotterdam_loc1.0_rr20.0_MC2000.xlsx","MonteCarlo_Results_Rotterdam_loc1.0_rr40.0_MC2000.xlsx","MonteCarlo_Results_Rotterdam_loc1.0_rr80.0_MC2000.xlsx",
#               "MonteCarlo_Results_Toronto-2c_loc1.0_rr0.5_MC2000.xlsx","MonteCarlo_Results_Toronto-2c_loc1.0_rr1.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_I_loc1.0_rr2.18_MC2000.xlsx",
#               "MonteCarlo_Results_Utrecht_I_loc1.0_rr3.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_I_loc2.0_rr15.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc1.0_rr4.0_MC2000.xlsx",
#               "MonteCarlo_Results_Utrecht_II_loc1.0_rr15.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc1.0_rr20.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc1.0_rr100.0_MC2000.xlsx",
#               "MonteCarlo_Results_Utrecht_II_loc2.0_rr0.15_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc2.0_rr0.5_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc2.0_rr1.0_MC2000.xlsx",
#               "MonteCarlo_Results_Utrecht_II_loc2.0_rr2.2_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc2.0_rr2.5_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc2.0_rr4.0_MC2000.xlsx",
#               "MonteCarlo_Results_Utrecht_II_loc2.0_rr20.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc2.0_rr80.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc3.0_rr4.0_MC2000.xlsx"] 
# release_rates_readin = [35,70,70,0.99,5.64,10.63,30.58,30.6,50.52,70.48,0.49,0.99,5,10,20,40,80,0.5,1,2.18,3,15,4,15,20,100,0.15,0.5,1,2.2,2.5,4,20,80,4] 
# city_names = ['London_I-Day2','London_I-Day2','London_I-Day5','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day2','London_II-Day2',
#               'Rotterdam','Rotterdam','Rotterdam','Rotterdam','Rotterdam','Toronto-2c','Toronto-2c','Utrecht','Utrecht','Utrecht','Utrecht_II','Utrecht_II','Utrecht_II','Utrecht_II','Utrecht_II','Utrecht_II',
#               'Utrecht_II','Utrecht_II','Utrecht_II','Utrecht_II','Utrecht_II','Utrecht_II','Utrecht_II',]
# locs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0]



# Only positive curve - #N >= 10 -------------------------------------------------------------

# filenames = ["MonteCarlo_Results_London_II-Day1_loc1.0_rr0.99_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr5.64_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr10.63_MC2000.xlsx",
#               "MonteCarlo_Results_London_II-Day1_loc1.0_rr30.6_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr30.58_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr50.52_MC2000.xlsx",
#               "MonteCarlo_Results_London_II-Day1_loc1.0_rr70.48_MC2000.xlsx","MonteCarlo_Results_London_II-Day2_loc1.0_rr0.49_MC2000.xlsx","MonteCarlo_Results_London_II-Day2_loc1.0_rr0.99_MC2000.xlsx",
#               "MonteCarlo_Results_Utrecht_II_loc2.0_rr80.0_MC2000.xlsx"] 
# release_rates_readin = [0.99,5.64,10.63,30.6,30.58,50.52,70.48,0.49,0.99,80] 
# city_names = ['London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day2','London_II-Day2','Utrecht_II']
# locs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]

# ==================================================================================================



# All - #N >= 20 =============================================================================

# filenames = ["MonteCarlo_Results_London_I-Day2_loc1.0_rr70.0_MC2000.xlsx","MonteCarlo_Results_London_I-Day5_loc1.0_rr70.0_MC2000.xlsx",
#               "MonteCarlo_Results_London_II-Day1_loc1.0_rr0.99_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr5.64_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr10.63_MC2000.xlsx",
#               "MonteCarlo_Results_London_II-Day1_loc1.0_rr30.6_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr30.58_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr50.52_MC2000.xlsx",
#               "MonteCarlo_Results_London_II-Day1_loc1.0_rr70.48_MC2000.xlsx","MonteCarlo_Results_London_II-Day2_loc1.0_rr0.49_MC2000.xlsx","MonteCarlo_Results_London_II-Day2_loc1.0_rr0.99_MC2000.xlsx",
#               "MonteCarlo_Results_Rotterdam_loc1.0_rr5.0_MC2000.xlsx","MonteCarlo_Results_Rotterdam_loc1.0_rr10.0_MC2000.xlsx","MonteCarlo_Results_Rotterdam_loc1.0_rr20.0_MC2000.xlsx",
#               "MonteCarlo_Results_Rotterdam_loc1.0_rr40.0_MC2000.xlsx","MonteCarlo_Results_Rotterdam_loc1.0_rr80.0_MC2000.xlsx",
#               "MonteCarlo_Results_Utrecht_I_loc1.0_rr2.18_MC2000.xlsx","MonteCarlo_Results_Utrecht_I_loc1.0_rr3.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_I_loc2.0_rr15.0_MC2000.xlsx",
#               "MonteCarlo_Results_Utrecht_II_loc1.0_rr15.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc1.0_rr20.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc2.0_rr4.0_MC2000.xlsx"] 
# release_rates_readin = [70,70,0.99,5.64,10.63,30.58,30.6,50.52,70.48,0.49,0.99,5,10,20,40,80,2.18,3,15,15,20,4] 
# city_names = ['London_I-Day2','London_I-Day5','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day2','London_II-Day2',
#               'Rotterdam','Rotterdam','Rotterdam','Rotterdam','Rotterdam','Utrecht','Utrecht','Utrecht','Utrecht_II','Utrecht_II','Utrecht_II']
# locs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0]

# Only positive curve - #N >= 20 (= only London II) -------------------------------------------------------------

# filenames = ["MonteCarlo_Results_London_II-Day1_loc1.0_rr0.99_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr5.64_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr10.63_MC2000.xlsx",
#               "MonteCarlo_Results_London_II-Day1_loc1.0_rr30.6_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr30.58_MC2000.xlsx","MonteCarlo_Results_London_II-Day1_loc1.0_rr50.52_MC2000.xlsx",
#               "MonteCarlo_Results_London_II-Day1_loc1.0_rr70.48_MC2000.xlsx","MonteCarlo_Results_London_II-Day2_loc1.0_rr0.49_MC2000.xlsx","MonteCarlo_Results_London_II-Day2_loc1.0_rr0.99_MC2000.xlsx"] 
# release_rates_readin = [0.99,5.64,10.63,30.6,30.58,50.52,70.48,0.49,0.99] 
# city_names = ['London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day1','London_II-Day2','London_II-Day2']
# locs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# Only negative curve - #N >= 10 -------------------------------------------------------------

# filenames = ["MonteCarlo_Results_London_I-Day2_loc1.0_rr70.0_MC2000.xlsx","MonteCarlo_Results_London_I-Day5_loc1.0_rr70.0_MC2000.xlsx",
#               "MonteCarlo_Results_Rotterdam_loc1.0_rr5.0_MC2000.xlsx","MonteCarlo_Results_Rotterdam_loc1.0_rr10.0_MC2000.xlsx",
#               "MonteCarlo_Results_Rotterdam_loc1.0_rr20.0_MC2000.xlsx","MonteCarlo_Results_Rotterdam_loc1.0_rr40.0_MC2000.xlsx","MonteCarlo_Results_Rotterdam_loc1.0_rr80.0_MC2000.xlsx",
#               "MonteCarlo_Results_Utrecht_I_loc1.0_rr2.18_MC2000.xlsx","MonteCarlo_Results_Utrecht_I_loc1.0_rr3.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_I_loc2.0_rr15.0_MC2000.xlsx",
#               "MonteCarlo_Results_Utrecht_II_loc1.0_rr15.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc1.0_rr20.0_MC2000.xlsx","MonteCarlo_Results_Utrecht_II_loc2.0_rr4.0_MC2000.xlsx"] 
# release_rates_readin = [70,70,5,10,20,40,80,2.18,3,15,15,20,4] 
# city_names = ['London_I-Day2','London_I-Day5','Rotterdam','Rotterdam','Rotterdam','Rotterdam','Rotterdam','Utrecht','Utrecht','Utrecht','Utrecht_II','Utrecht_II','Utrecht_II']
# locs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0]


    


#%% Categorization

# add emission categories as strings (e.g. "2-Low")


start_time = time.time()

list_data = []
    
for i in range(len(filenames)):
    filepath = path_statsdata / filenames[i]

    # Load all sheets into a dictionary of DataFrames
    xls = pd.ExcelFile(filepath)
    sheet_names = xls.sheet_names

    # As list
    data = list(pd.read_excel(filepath, sheet_name=sheet_names[:20]).values())
    add_characteristics_cat_str(data,release_rates_readin[i],city_names[i],locs[i]) # add_characteristics_cat_str() calls categorize_release_rates_str()
    
    list_data.append(data)
    print(i) 
    
print("--- %s seconds ---" % round(time.time() - start_time, 2))
    

data_comb_0 = [pd.concat(dataframes, axis=0, ignore_index=True) for dataframes in zip(*list_data)]
data_comb = [df.copy(deep=True) for df in data_comb_0]

est_methods = ['Weller_cat','Max_cat','Area_cat']


    
#%% Sankey Diagramm

#%% N=1 ---------------------------------------------------------

N=1
df = data_comb[N-1].copy() # select values for N=1 (N-1: index starts with 0, so N=1 is 0)

df_perc = mean_cat_success_str(df,est_methods) # calculates the categorization success in percentages

# Filter rows where 'True_cat' and 'Estimation_cat' are equal
filtered_df = df_perc[df_perc['True_cat'] == df_perc['Estimation_cat']]
filtered_df['Weighted_Percentage'] = filtered_df['Area_cat_Count']/filtered_df['Area_cat_Count'].sum()
filtered_df['Weighted_Percentage'] = filtered_df['Weighted_Percentage']*filtered_df['Area_cat_Percentage']
print(filtered_df['Weighted_Percentage'].sum())


#%%% FINAL - N=1: Area - #N>10 - negative curves

fig,ax = plt.subplots(figsize=(9,12))

method='Area_cat_Percentage'

ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    rightLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    fontsize=20,
    ax=ax
)

# Rotate category labels
for text in ax.texts:
    text.set_rotation(90)  # Rotate text to vertical

x_min, x_max = ax1.dataLim.x0, ax1.dataLim.x1    
y_min, y_max = ax1.dataLim.y0, ax1.dataLim.y1

# Version 0 - y axis label on the side ----------------------------------------    
ax1.text(x_min-410,0.5*y_max,'True $r_E$',fontsize=20, rotation=90, verticalalignment='center')
ax1.text(x_max+300,0.5*y_max,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')

# 4
ax1.text(10,49000,f'{int(round(df_perc[method][15],0))}%',fontsize=20,weight='bold')
ax1.text(10,46000,f'{int(round(df_perc[method][14],0))}%',fontsize=16)
ax1.text(10,44900,f'{int(round(df_perc[method][13],0))}%',fontsize=14)
# ax1.text(1,44900,f'{round(df_perc[method][12],0)}%',fontsize=16)
# 3
ax1.text(10,40300,f'{int(round(df_perc[method][11],0))}%',fontsize=16)
ax1.text(10,33300,f'{int(round(df_perc[method][10],0))}%',fontsize=20,weight='bold')
ax1.text(10,28800,f'{int(round(df_perc[method][9],0))}%',fontsize=16)
# ax1.text(10,35000,f'{round(df_perc[method][8],0)}%',fontsize=16)
# 2
ax1.text(10,25900,f'{int(round(df_perc[method][7],0))}%',fontsize=16)
ax1.text(10,22000,f'{int(round(df_perc[method][6],0))}%',fontsize=16)
ax1.text(10,13500,f'{int(round(df_perc[method][5],0))}%',fontsize=20,weight='bold')
ax1.text(10,7200,f'{int(round(df_perc[method][4],0))}%',fontsize=16)
# 1
# ax1.text(1,3650,f'{round(df_perc[method][2],0)}%',fontsize=16) #leave out below 5%
ax1.text(10,4500,f'{int(round(df_perc[method][1],0))}%',fontsize=16)
ax1.text(10,1400,f'{int(round(df_perc[method][0],0))}%',fontsize=20,weight='bold')


plt.show() # to display
# plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area_negcurves.png', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area_negcurves.pdf', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area_negcurves.svg', bbox_inches='tight') 




#%%% N=1: Area - #N>10

fig,ax = plt.subplots(figsize=(10,12))

ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    rightLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    fontsize=20,
    ax=ax
)

# Rotate category labels
for text in ax.texts:
    text.set_rotation(90)  # Rotate text to vertical
    
plt.title(f'N={N}: (#N>10) Area eq.',fontsize=24,fontweight='bold')

method='Area_cat_Percentage'

# ax1.text(-10,680,'True $r_R$',fontsize=20, rotation=90, verticalalignment='center')
# ax1.text(74,680,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')


# 4
ax1.text(10,69000,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')
ax1.text(10,62500,f'{round(df_perc[method][14],0)}%',fontsize=16)
ax1.text(10,60000,f'{round(df_perc[method][13],0)}%',fontsize=16)
# ax1.text(1,16900,f'{round(df_perc[method][12],0)}%',fontsize=16)
# 3
ax1.text(10,55000,f'{round(df_perc[method][11],0)}%',fontsize=16)
ax1.text(10,46500,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(10,39000,f'{round(df_perc[method][9],0)}%',fontsize=16)
# ax1.text(10,35000,f'{round(df_perc[method][8],0)}%',fontsize=16)
# 2
ax1.text(10,34000,f'{round(df_perc[method][7],0)}%',fontsize=16)
ax1.text(10,30000,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(10,20000,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(10,11000,f'{round(df_perc[method][4],0)}%',fontsize=16)
# 1
# ax1.text(1,3650,f'{round(df_perc[method][2],0)}%',fontsize=16) #leave out below 5%
ax1.text(10,6500,f'{round(df_perc[method][1],0)}%',fontsize=16)
ax1.text(10,2000,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')


plt.show() # to display
# plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area.png', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area.pdf', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area.svg', bbox_inches='tight') 


#%%% N=1: Area - #N>10 - positive curves

fig,ax = plt.subplots(figsize=(10,12))

ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    rightLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    fontsize=20,
    ax=ax
)

plt.title(f'N={N}: (#N>10) Area eq. - r_c>0',fontsize=24,fontweight='bold')

method='Area_cat_Percentage'

# ax1.text(-10,680,'True $r_R$',fontsize=20, rotation=90, verticalalignment='center')
# ax1.text(74,680,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')


# 4
ax1.text(10,19700,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')
ax1.text(10,17000,f'{round(df_perc[method][14],0)}%',fontsize=16)
ax1.text(10,15150,f'{round(df_perc[method][13],0)}%',fontsize=14)
# ax1.text(1,44900,f'{round(df_perc[method][12],0)}%',fontsize=16)
# 3
# ax1.text(10,40300,f'{round(df_perc[method][11],0)}%',fontsize=16)
ax1.text(10,13000,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(10,10100,f'{round(df_perc[method][9],0)}%',fontsize=16)
# ax1.text(10,35000,f'{round(df_perc[method][8],0)}%',fontsize=16)
# 2
# ax1.text(10,26000,f'{round(df_perc[method][7],0)}%',fontsize=16)
# ax1.text(10,22000,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(10,7000,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(10,4000,f'{round(df_perc[method][4],0)}%',fontsize=16)
# 1
# ax1.text(1,3650,f'{round(df_perc[method][2],0)}%',fontsize=16) #leave out below 5%
# ax1.text(10,4500,f'{round(df_perc[method][1],0)}%',fontsize=16)
ax1.text(10,500,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')


plt.show() # to display
# plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area_poscurves.png', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area_poscurves.pdf', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area_poscurves.svg', bbox_inches='tight') 





#%%% N=1: Area  - #N>20

fig,ax = plt.subplots(figsize=(10,12))

ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    rightLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    fontsize=20,
    ax=ax
)

plt.title(f'N={N}: (#N>20) Area eq.',fontsize=24,fontweight='bold')

method='Area_cat_Percentage'

# ax1.text(-10,680,'True $r_R$',fontsize=20, rotation=90, verticalalignment='center')
# ax1.text(74,680,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')


# 4
ax1.text(10,43600,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')
ax1.text(10,39000,f'{round(df_perc[method][14],0)}%',fontsize=16)
ax1.text(10,36600,f'{round(df_perc[method][13],0)}%',fontsize=16)
# ax1.text(1,16900,f'{round(df_perc[method][12],0)}%',fontsize=16)
# 3
ax1.text(10,32800,f'{round(df_perc[method][11],0)}%',fontsize=16)
ax1.text(10,26700,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(10,20000,f'{round(df_perc[method][9],0)}%',fontsize=16)
# ax1.text(10,35000,f'{round(df_perc[method][8],0)}%',fontsize=16)
# 2
ax1.text(10,16000,f'{round(df_perc[method][7],0)}%',fontsize=16)
ax1.text(10,14000,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(10,8800,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(10,4200,f'{round(df_perc[method][4],0)}%',fontsize=16)
# 1
# ax1.text(1,3650,f'{round(df_perc[method][2],0)}%',fontsize=16) #leave out below 5%
# ax1.text(10,6500,f'{round(df_perc[method][1],0)}%',fontsize=16)
ax1.text(10,500,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')


plt.show() # to display
# plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area.png', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area.pdf', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area.svg', bbox_inches='tight') 


#%%% N=1: Area - #N>20 - positive curves / London II only

fig,ax = plt.subplots(figsize=(10,12))

ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    rightLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    fontsize=20,
    ax=ax
)

plt.title(f'N={N}: (#N>20) Area eq. - r_c>0',fontsize=24,fontweight='bold')

method='Area_cat_Percentage'

# ax1.text(-10,680,'True $r_R$',fontsize=20, rotation=90, verticalalignment='center')
# ax1.text(74,680,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')

# 4
ax1.text(10,18650,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')
ax1.text(10,17000,f'{round(df_perc[method][14],0)}%',fontsize=16)
ax1.text(10,15100,f'{round(df_perc[method][13],0)}%',fontsize=16)
# ax1.text(1,44900,f'{round(df_perc[method][12],0)}%',fontsize=16)
# 3
# ax1.text(10,40300,f'{round(df_perc[method][11],0)}%',fontsize=16)
ax1.text(10,13000,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(10,10100,f'{round(df_perc[method][9],0)}%',fontsize=16)
# ax1.text(10,35000,f'{round(df_perc[method][8],0)}%',fontsize=16)
# 2
# ax1.text(10,26000,f'{round(df_perc[method][7],0)}%',fontsize=16)
# ax1.text(10,22000,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(10,7000,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(10,4000,f'{round(df_perc[method][4],0)}%',fontsize=16)
# 1
# ax1.text(1,3650,f'{round(df_perc[method][2],0)}%',fontsize=16) #leave out below 5%
# ax1.text(10,4500,f'{round(df_perc[method][1],0)}%',fontsize=16)
ax1.text(10,500,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')


plt.show() # to display
# plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area_poscurves.png', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area_poscurves.pdf', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area_poscurves.svg', bbox_inches='tight') 


#%%% N=1: Area - #N>20 - negative curves

fig,ax = plt.subplots(figsize=(10,12))

ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    rightLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    fontsize=20,
    ax=ax
)

plt.title(f'N={N}: (#N>20) Area eq. - r_c<0',fontsize=24,fontweight='bold')

method='Area_cat_Percentage'

# ax1.text(-10,680,'True $r_R$',fontsize=20, rotation=90, verticalalignment='center')
# ax1.text(74,680,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')


# 4
ax1.text(10,24650,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')
ax1.text(10,21900,f'{round(df_perc[method][14],0)}%',fontsize=16)
ax1.text(10,20900,f'{round(df_perc[method][13],0)}%',fontsize=16)
# ax1.text(1,44900,f'{round(df_perc[method][12],0)}%',fontsize=16)
# 3
ax1.text(10,17800,f'{round(df_perc[method][11],0)}%',fontsize=16)
ax1.text(10,13000,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(10,9500,f'{round(df_perc[method][9],0)}%',fontsize=16)
# ax1.text(10,8100,f'{round(df_perc[method][8],0)}%',fontsize=16)
# 2
ax1.text(10,7400,f'{round(df_perc[method][7],0)}%',fontsize=16)
ax1.text(10,5400,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(10,1800,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(10,10,f'{round(df_perc[method][4],0)}%',fontsize=16)
# 1
# ax1.text(1,3650,f'{round(df_perc[method][2],0)}%',fontsize=16) #leave out below 5%
# ax1.text(10,4500,f'{round(df_perc[method][1],0)}%',fontsize=16)
# ax1.text(10,500,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')


plt.show() # to display
# plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area_negcurves.png', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area_negcurves.pdf', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area_negcurves.svg', bbox_inches='tight') 





#%% N=3 ---------------------------------------------------------

N=3
df = data_comb[N-1].copy()

df_perc = mean_cat_success_str(df,est_methods)


# Filter rows where 'True_cat' and 'Estimation_cat' are equal
filtered_df = df_perc[df_perc['True_cat'] == df_perc['Estimation_cat']]
filtered_df['Weighted_Percentage'] = filtered_df['Area_cat_Count']/filtered_df['Area_cat_Count'].sum()
filtered_df['Weighted_Percentage'] = filtered_df['Weighted_Percentage']*filtered_df['Area_cat_Percentage']
print(filtered_df['Weighted_Percentage'].sum())


#%%% FINAL - N=3: Area - #N>10 - negative curves

fig,ax = plt.subplots(figsize=(9,12))

# plt.title(f'N={N}: (#N>10) Area eq. - r_c<0',fontsize=24,fontweight='bold')

method='Area_cat_Percentage'

ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    rightLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    fontsize=20,
    ax=ax
)

# Rotate category labels
for text in ax.texts:
    text.set_rotation(90)  # Rotate text to vertical

x_min, x_max = ax1.dataLim.x0, ax1.dataLim.x1    
y_min, y_max = ax1.dataLim.y0, ax1.dataLim.y1

# Version 0 - y axis label on the side ----------------------------------------    
ax1.text(x_min-410,0.5*y_max,'True $r_E$',fontsize=20, rotation=90, verticalalignment='center')
ax1.text(x_max+300,0.5*y_max,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')

# 4
ax1.text(10,49000,f'{int(round(df_perc[method][15],0))}%',fontsize=20,weight='bold')
ax1.text(10,45200,f'{int(round(df_perc[method][14],0))}%',fontsize=16)
# ax1.text(10,44900,f'{round(df_perc[method][13],0)}%',fontsize=14)
# ax1.text(1,44900,f'{round(df_perc[method][12],0)}%',fontsize=16)
# 3
ax1.text(10,41100,f'{int(round(df_perc[method][11],0))}%',fontsize=16)
ax1.text(10,33300,f'{int(round(df_perc[method][10],0))}%',fontsize=20,weight='bold')
ax1.text(10,28000,f'{int(round(df_perc[method][9],0))}%',fontsize=16)
# ax1.text(10,35000,f'{round(df_perc[method][8],0)}%',fontsize=16)
# 2
# ax1.text(10,26000,f'{round(df_perc[method][7],0)}%',fontsize=16)
ax1.text(10,22200,f'{int(round(df_perc[method][6],0))}%',fontsize=16)
ax1.text(10,13000,f'{int(round(df_perc[method][5],0))}%',fontsize=20,weight='bold')
# ax1.text(10,7200,f'{round(df_perc[method][4],0)}%',fontsize=16)
# 1
# ax1.text(1,3650,f'{round(df_perc[method][2],0)}%',fontsize=16) #leave out below 5%
ax1.text(10,4500,f'{int(round(df_perc[method][1],0))}%',fontsize=16)
ax1.text(10,1200,f'{int(round(df_perc[method][0],0))}%',fontsize=20,weight='bold')


plt.show() # to display
# plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area_negcurves.png', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area_negcurves.pdf', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area_negcurves.svg', bbox_inches='tight') 



#%%% N=3: Area - #N>10

fig,ax = plt.subplots(figsize=(10,12))
ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1','2','3','4'],
    rightLabels=['1','2','3','4'],
    fontsize=20,
    ax=ax
)

plt.title(f'N={N}: (#N>10) Area eq.',fontsize=24,fontweight='bold')

method='Area_cat_Percentage'

# ax1.text(-10,680,'True $r_R$',fontsize=20, rotation=90, verticalalignment='center')
# ax1.text(74,680,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')


# 4
ax1.text(10,69000,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')
ax1.text(10,62000,f'{round(df_perc[method][14],0)}%',fontsize=16)
# ax1.text(10,60000,f'{round(df_perc[method][13],0)}%',fontsize=16)
# ax1.text(1,16900,f'{round(df_perc[method][12],0)}%',fontsize=16)
# 3
ax1.text(10,56000,f'{round(df_perc[method][11],0)}%',fontsize=16)
ax1.text(10,46500,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(10,38000,f'{round(df_perc[method][9],0)}%',fontsize=16)
# ax1.text(10,35000,f'{round(df_perc[method][8],0)}%',fontsize=16)
# 2
# ax1.text(10,34000,f'{round(df_perc[method][7],0)}%',fontsize=16)
ax1.text(10,31000,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(10,20000,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(10,10500,f'{round(df_perc[method][4],0)}%',fontsize=16)
# 1
# ax1.text(1,3650,f'{round(df_perc[method][2],0)}%',fontsize=16) #leave out below 5%
ax1.text(10,6300,f'{round(df_perc[method][1],0)}%',fontsize=16)
ax1.text(10,1000,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')



plt.show() # to display
plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area.png', bbox_inches='tight') 
plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area.pdf', bbox_inches='tight') 
plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area.svg', bbox_inches='tight') 


#%%% N=3: Area - #N>10 - positive curves

fig,ax = plt.subplots(figsize=(10,12))
ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1','2','3','4'],
    rightLabels=['1','2','3','4'],
    fontsize=20,
    ax=ax
)

plt.title(f'N={N}: (#N>10) Area eq. - r_c>0',fontsize=24,fontweight='bold')

method='Area_cat_Percentage'

# ax1.text(-10,680,'True $r_R$',fontsize=20, rotation=90, verticalalignment='center')
# ax1.text(74,680,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')

# 1
ax1.text(10,500,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')
# ax1.text(1,6500,f'{round(df_perc[method][1],0)}%',fontsize=16)
# ax1.text(1,3650,f'{round(df_perc[method][2],0)}%',fontsize=16) #leave out below 5%
# 2
ax1.text(10,4000,f'{round(df_perc[method][4],0)}%',fontsize=16)
ax1.text(10,7000,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
# ax1.text(10,28000,f'{round(df_perc[method][6],0)}%',fontsize=16)
# ax1.text(10,32000,f'{round(df_perc[method][7],0)}%',fontsize=16)
# 3
# ax1.text(10,35000,f'{round(df_perc[method][8],0)}%',fontsize=16)
ax1.text(10,10100,f'{round(df_perc[method][9],0)}%',fontsize=16)
ax1.text(10,13000,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
# ax1.text(10,15000,f'{round(df_perc[method][11],0)}%',fontsize=16)
# 4
# ax1.text(1,16900,f'{round(df_perc[method][12],0)}%',fontsize=14)
ax1.text(10,15150,f'{round(df_perc[method][13],0)}%',fontsize=14)
ax1.text(10,17000,f'{round(df_perc[method][14],0)}%',fontsize=16)
ax1.text(10,19700,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')


plt.show() # to display
plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area_poscurves.png', bbox_inches='tight') 
plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area_poscurves.pdf', bbox_inches='tight') 
plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area_poscurves.svg', bbox_inches='tight') 





#%%% N=3: Area  - #N>20

fig,ax = plt.subplots(figsize=(10,12))
ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1','2','3','4'],
    rightLabels=['1','2','3','4'],
    fontsize=20,
    ax=ax
)

plt.title(f'N={N}: (#N>20) Area eq.',fontsize=24,fontweight='bold')

method='Area_cat_Percentage'

# ax1.text(-10,680,'True $r_R$',fontsize=20, rotation=90, verticalalignment='center')
# ax1.text(74,680,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')


# 4
ax1.text(10,43700,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')
ax1.text(10,38700,f'{round(df_perc[method][14],0)}%',fontsize=16)
# ax1.text(10,36600,f'{round(df_perc[method][13],0)}%',fontsize=16)
# ax1.text(1,16900,f'{round(df_perc[method][12],0)}%',fontsize=16)
# 3
ax1.text(10,32900,f'{round(df_perc[method][11],0)}%',fontsize=16)
ax1.text(10,25800,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(10,19400,f'{round(df_perc[method][9],0)}%',fontsize=16)
# ax1.text(10,35000,f'{round(df_perc[method][8],0)}%',fontsize=16)
# 2
# ax1.text(10,16000,f'{round(df_perc[method][7],0)}%',fontsize=16)
ax1.text(10,14200,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(10,9000,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(10,4200,f'{round(df_perc[method][4],0)}%',fontsize=16)
# 1
# ax1.text(1,3650,f'{round(df_perc[method][2],0)}%',fontsize=16) #leave out below 5%
# ax1.text(10,6500,f'{round(df_perc[method][1],0)}%',fontsize=16)
ax1.text(10,500,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')


plt.show() # to display
plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area.png', bbox_inches='tight') 
plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area.pdf', bbox_inches='tight') 
plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area.svg', bbox_inches='tight') 


#%%% N=1: Area - #N>20 - positive curves / London II only

fig,ax = plt.subplots(figsize=(10,12))
ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1','2','3','4'],
    rightLabels=['1','2','3','4'],
    fontsize=20,
    ax=ax
)

plt.title(f'N={N}: (#N>20) Area eq. - r_c>0',fontsize=24,fontweight='bold')

method='Area_cat_Percentage'

# ax1.text(-10,680,'True $r_R$',fontsize=20, rotation=90, verticalalignment='center')
# ax1.text(74,680,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')

# 4
ax1.text(10,18700,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')
ax1.text(10,16900,f'{round(df_perc[method][14],0)}%',fontsize=16)
# ax1.text(10,15100,f'{round(df_perc[method][13],0)}%',fontsize=16)
# ax1.text(1,44900,f'{round(df_perc[method][12],0)}%',fontsize=16)
# 3
# ax1.text(10,40300,f'{round(df_perc[method][11],0)}%',fontsize=16)
ax1.text(10,13000,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(10,10100,f'{round(df_perc[method][9],0)}%',fontsize=16)
# ax1.text(10,35000,f'{round(df_perc[method][8],0)}%',fontsize=16)
# 2
# ax1.text(10,26000,f'{round(df_perc[method][7],0)}%',fontsize=16)
# ax1.text(10,22000,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(10,7000,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(10,4000,f'{round(df_perc[method][4],0)}%',fontsize=16)
# 1
# ax1.text(1,3650,f'{round(df_perc[method][2],0)}%',fontsize=16) #leave out below 5%
# ax1.text(10,4500,f'{round(df_perc[method][1],0)}%',fontsize=16)
ax1.text(10,700,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')


plt.show() # to display
# plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area_poscurves.png', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area_poscurves.pdf', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area_poscurves.svg', bbox_inches='tight') 


#%%% N=3: Area - #N>20 - negative curves

fig,ax = plt.subplots(figsize=(10,12))
ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['2','3','4'],
    rightLabels=['1','2','3','4'],
    fontsize=20,
    ax=ax
)

plt.title(f'N={N}: (#N>20) Area eq. - r_c<0',fontsize=24,fontweight='bold')

method='Area_cat_Percentage'

# ax1.text(-10,680,'True $r_R$',fontsize=20, rotation=90, verticalalignment='center')
# ax1.text(74,680,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')


# 4
ax1.text(10,24450,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')
ax1.text(10,21700,f'{round(df_perc[method][14],0)}%',fontsize=16)
# ax1.text(10,20900,f'{round(df_perc[method][13],0)}%',fontsize=16)
# ax1.text(1,44900,f'{round(df_perc[method][12],0)}%',fontsize=16)
# 3
ax1.text(10,18000,f'{round(df_perc[method][11],0)}%',fontsize=16)
ax1.text(10,12800,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(10,8700,f'{round(df_perc[method][9],0)}%',fontsize=16)
# ax1.text(10,8100,f'{round(df_perc[method][8],0)}%',fontsize=16)
# 2
# ax1.text(10,7400,f'{round(df_perc[method][7],0)}%',fontsize=16)
ax1.text(10,5400,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(10,1400,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
# ax1.text(10,10,f'{round(df_perc[method][4],0)}%',fontsize=16)
# 1
# ax1.text(1,3650,f'{round(df_perc[method][2],0)}%',fontsize=16) #leave out below 5%
# ax1.text(10,4500,f'{round(df_perc[method][1],0)}%',fontsize=16)
# ax1.text(10,500,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')


plt.show() # to display
# plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area_negcurves.png', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area_negcurves.pdf', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area_negcurves.svg', bbox_inches='tight') 



#%%% N=3: Weller

fig,ax = plt.subplots(figsize=(10,12))
ax1 = sankey(
    df['True_cat'], df['Weller_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1','2','3','4'],
    rightLabels=['1','2','3','4'],	
    fontsize=20,
    ax=ax
)
plt.title(f'N={N}: Weller eq.',fontsize=24,fontweight='bold')

method='Weller_cat_Percentage'

# 1
ax1.text(1,0,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')
ax1.text(1,1800,f'{round(df_perc[method][1],0)}%',fontsize=16)
ax1.text(1,3770,f'{round(df_perc[method][2],0)}%',fontsize=16) #leave out below 5%
# 2
#ax1.text(1,4850,f'{round(df_perc[method][4],0)}%',fontsize=16)
ax1.text(1,6000,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(1,9000,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(1,10170,f'{round(df_perc[method][7],0)}%',fontsize=16)
# 3
#ax1.text(1,10700,f'{round(df_perc[method][8],0)}%',fontsize=16)
ax1.text(1,10800,f'{round(df_perc[method][9],0)}%',fontsize=16)
ax1.text(1,12400,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(1,15200,f'{round(df_perc[method][11],0)}%',fontsize=16)
# 4
#ax1.text(1,16900,f'{round(df_perc[method][12],0)}%',fontsize=14)
#ax1.text(1,16900,f'{round(df_perc[method][13],0)}%',fontsize=14)
ax1.text(1,17110,f'{round(df_perc[method][14],0)}%',fontsize=16)
ax1.text(1,18100,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')

# plt.savefig(path_savefig / f'Sankey_R_N{N}_Weller.png', bbox_inches='tight')
# plt.savefig(path_savefig / f'Sankey_R_N{N}_Weller.pdf', bbox_inches='tight') 
# plt.savefig(path_savefig / f'Sankey_R_N{N}_Weller.svg', bbox_inches='tight')




#%% N=6 ---------------------------------------------------------

N=6
df = data_comb[N-1].copy()

df_perc = mean_cat_success_str(df,est_methods)


# Filter rows where 'True_cat' and 'Estimation_cat' are equal
filtered_df = df_perc[df_perc['True_cat'] == df_perc['Estimation_cat']]
filtered_df['Weighted_Percentage'] = filtered_df['Area_cat_Count']/filtered_df['Area_cat_Count'].sum()
filtered_df['Weighted_Percentage'] = filtered_df['Weighted_Percentage']*filtered_df['Area_cat_Percentage']
print(filtered_df['Weighted_Percentage'].sum())


#%%% FINAL - N=6: Area - #N>10 - negative curves

fig,ax = plt.subplots(figsize=(9,12))

# plt.title(f'N={N}: (#N>10) Area eq. - r_c<0',fontsize=24,fontweight='bold')

method='Area_cat_Percentage'

ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    rightLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    fontsize=20,
    ax=ax
)

# Rotate category labels
for text in ax.texts:
    text.set_rotation(90)  # Rotate text to vertical

x_min, x_max = ax1.dataLim.x0, ax1.dataLim.x1    
y_min, y_max = ax1.dataLim.y0, ax1.dataLim.y1

# Version 0 - y axis label on the side ----------------------------------------    
ax1.text(x_min-410,0.5*y_max,'True $r_E$',fontsize=20, rotation=90, verticalalignment='center')
ax1.text(x_max+300,0.5*y_max,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')

# 4
ax1.text(10,49000,f'{int(round(df_perc[method][15],0))}%',fontsize=20,weight='bold')
ax1.text(10,45200,f'{int(round(df_perc[method][14],0))}%',fontsize=16)
# ax1.text(10,44900,f'{int(round(df_perc[method][13],0))}%',fontsize=14)
# ax1.text(1,44900,f'{int(round(df_perc[method][12],0))}%',fontsize=16)
# 3
ax1.text(10,41200,f'{int(round(df_perc[method][11],0))}%',fontsize=16)
ax1.text(10,33300,f'{int(round(df_perc[method][10],0))}%',fontsize=20,weight='bold')
ax1.text(10,28000,f'{int(round(df_perc[method][9],0))}%',fontsize=16)
# ax1.text(10,35000,f'{int(round(df_perc[method][8],0)}%',fontsize=16)
# 2
# ax1.text(10,26000,f'{int(round(df_perc[method][7],0))}%',fontsize=16)
ax1.text(10,22400,f'{int(round(df_perc[method][6],0))}%',fontsize=16)
ax1.text(10,13000,f'{int(round(df_perc[method][5],0))}%',fontsize=20,weight='bold')
# ax1.text(10,7200,f'{int(round(df_perc[method][4],0))}%',fontsize=16)
# 1
# ax1.text(1,3650,f'{int(round(df_perc[method][2],0))}%',fontsize=16) #leave out below 5%
ax1.text(10,4500,f'{int(round(df_perc[method][1],0))}%',fontsize=16)
ax1.text(10,1300,f'{int(round(df_perc[method][0],0))}%',fontsize=20,weight='bold')


plt.show() # to display
# plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area_negcurves.png', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area_negcurves.pdf', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area_negcurves.svg', bbox_inches='tight') 



#%%% N=6: Area - #N>10

fig,ax = plt.subplots(figsize=(10,12))
ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1','2','3','4'],
    rightLabels=['1','2','3','4'],
    fontsize=20,
    ax=ax
)

plt.title(f'N={N}: (#N>10) Area eq.',fontsize=24,fontweight='bold')

method='Area_cat_Percentage'

# ax1.text(-10,680,'True $r_R$',fontsize=20, rotation=90, verticalalignment='center')
# ax1.text(74,680,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')


# 4
ax1.text(10,69000,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')
ax1.text(10,62000,f'{round(df_perc[method][14],0)}%',fontsize=16)
# ax1.text(10,60000,f'{round(df_perc[method][13],0)}%',fontsize=16)
# ax1.text(1,16900,f'{round(df_perc[method][12],0)}%',fontsize=16)
# 3
ax1.text(10,56000,f'{round(df_perc[method][11],0)}%',fontsize=16)
ax1.text(10,46500,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(10,38000,f'{round(df_perc[method][9],0)}%',fontsize=16)
# ax1.text(10,35000,f'{round(df_perc[method][8],0)}%',fontsize=16)
# 2
# ax1.text(10,34000,f'{round(df_perc[method][7],0)}%',fontsize=16)
ax1.text(10,31000,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(10,20000,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(10,10500,f'{round(df_perc[method][4],0)}%',fontsize=16)
# 1
# ax1.text(1,3650,f'{round(df_perc[method][2],0)}%',fontsize=16) #leave out below 5%
ax1.text(10,6300,f'{round(df_perc[method][1],0)}%',fontsize=16)
ax1.text(10,2000,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')


plt.show() # to display
# plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area.png', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area.pdf', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area.svg', bbox_inches='tight') 


#%%% N=6: Area - #N>10 - positive curves

fig,ax = plt.subplots(figsize=(10,12))
ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1','2','3','4'],
    rightLabels=['1','2','3','4'],
    fontsize=20,
    ax=ax
)

plt.title(f'N={N}: (#N>10) Area eq. - r_c>0',fontsize=24,fontweight='bold')

method='Area_cat_Percentage'

# ax1.text(-10,680,'True $r_R$',fontsize=20, rotation=90, verticalalignment='center')
# ax1.text(74,680,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')

# 1
ax1.text(10,500,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')
# ax1.text(1,6500,f'{round(df_perc[method][1],0)}%',fontsize=16)
# ax1.text(1,3650,f'{round(df_perc[method][2],0)}%',fontsize=16) #leave out below 5%
# 2
ax1.text(10,4000,f'{round(df_perc[method][4],0)}%',fontsize=16)
ax1.text(10,7000,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
# ax1.text(10,28000,f'{round(df_perc[method][6],0)}%',fontsize=16)
# ax1.text(10,32000,f'{round(df_perc[method][7],0)}%',fontsize=16)
# 3
# ax1.text(10,35000,f'{round(df_perc[method][8],0)}%',fontsize=16)
ax1.text(10,10000,f'{round(df_perc[method][9],0)}%',fontsize=16)
ax1.text(10,13000,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
# ax1.text(10,15000,f'{round(df_perc[method][11],0)}%',fontsize=16)
# 4
# ax1.text(1,16900,f'{round(df_perc[method][12],0)}%',fontsize=14)
# ax1.text(10,15150,f'{round(df_perc[method][13],0)}%',fontsize=14)
ax1.text(10,17000,f'{round(df_perc[method][14],0)}%',fontsize=16)
ax1.text(10,19800,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')


plt.show() # to display
# plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area_poscurves.png', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area_poscurves.pdf', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects10/Sankey_transects10_N{N}_Area_poscurves.svg', bbox_inches='tight') 





#%%% N=6: Area  - #N>20

fig,ax = plt.subplots(figsize=(10,12))
ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1','2','3','4'],
    rightLabels=['1','2','3','4'],
    fontsize=20,
    ax=ax
)
plt.title(f'N={N}: (#N>20) Area eq.',fontsize=24,fontweight='bold')

method='Area_cat_Percentage'

# ax1.text(-10,680,'True $r_R$',fontsize=20, rotation=90, verticalalignment='center')
# ax1.text(74,680,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')


# 4
ax1.text(10,43700,f'{round(int(df_perc[method][15],0))}%',fontsize=20,weight='bold')
ax1.text(10,38700,f'{round(int(df_perc[method][14],0))}%',fontsize=16)
# ax1.text(10,36600,f'{round(int(df_perc[method][13],0))}%',fontsize=16)
# ax1.text(1,16900,f'{round(int(df_perc[method][12],0))}%',fontsize=16)
# 3
ax1.text(10,32900,f'{round(int(df_perc[method][11],0))}%',fontsize=16)
ax1.text(10,25800,f'{round(int(df_perc[method][10],0))}%',fontsize=20,weight='bold')
ax1.text(10,19400,f'{round(int(df_perc[method][9],0))}%',fontsize=16)
# ax1.text(10,35000,f'{round(int(df_perc[method][8],0))}%',fontsize=16)
# 2
# ax1.text(10,16000,f'{round(int(df_perc[method][7],0))}%',fontsize=16)
ax1.text(10,14200,f'{round(int(df_perc[method][6],0))}%',fontsize=16)
ax1.text(10,9000,f'{round(int(df_perc[method][5],0))}%',fontsize=20,weight='bold')
ax1.text(10,4200,f'{round(int(df_perc[method][4],0))}%',fontsize=16)
# 1
# ax1.text(1,3650,f'{round(int(df_perc[method][2],0))}%',fontsize=16) #leave out below 5%
# ax1.text(10,6500,f'{round(int(df_perc[method][1],0))}%',fontsize=16)
ax1.text(10,500,f'{round(int(df_perc[method][0],0))}%',fontsize=20,weight='bold')


plt.show() # to display
plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area.png', bbox_inches='tight') 
plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area.pdf', bbox_inches='tight') 
plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area.svg', bbox_inches='tight') 


#%%% N=6: Area - #N>20 - positive curves / London II only

fig,ax = plt.subplots(figsize=(10,12))
ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1','2','3','4'],
    rightLabels=['1','2','3'],
    fontsize=20,
    ax=ax
)

plt.title(f'N={N}: (#N>20) Area eq. - r_c>0',fontsize=24,fontweight='bold')

method='Area_cat_Percentage'

# ax1.text(-10,680,'True $r_R$',fontsize=20, rotation=90, verticalalignment='center')
# ax1.text(74,680,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')

# 4
# ax1.text(10,18700,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')
ax1.text(10,16900,f'{round(df_perc[method][14],0)}%',fontsize=16)
# ax1.text(10,15100,f'{round(df_perc[method][13],0)}%',fontsize=16)
# ax1.text(1,44900,f'{round(df_perc[method][12],0)}%',fontsize=16)
# 3
# ax1.text(10,40300,f'{round(df_perc[method][11],0)}%',fontsize=16)
ax1.text(10,13000,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(10,10000,f'{round(df_perc[method][9],0)}%',fontsize=16)
# ax1.text(10,35000,f'{round(df_perc[method][8],0)}%',fontsize=16)
# 2
# ax1.text(10,26000,f'{round(df_perc[method][7],0)}%',fontsize=16)
# ax1.text(10,22000,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(10,7000,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(10,4000,f'{round(df_perc[method][4],0)}%',fontsize=16)
# 1
# ax1.text(1,3650,f'{round(df_perc[method][2],0)}%',fontsize=16) #leave out below 5%
# ax1.text(10,4500,f'{round(df_perc[method][1],0)}%',fontsize=16)
ax1.text(10,700,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')


plt.show() # to display
plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area_poscurves.png', bbox_inches='tight') 
plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area_poscurves.pdf', bbox_inches='tight') 
plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area_poscurves.svg', bbox_inches='tight') 


#%%% N=6: Area - #N>20 - negative curves

fig,ax = plt.subplots(figsize=(10,12))
ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['2','3','4'],
    rightLabels=['2','3','4'],
    fontsize=20,
    ax=ax
)

plt.title(f'N={N}: (#N>20) Area eq. - r_c<0',fontsize=24,fontweight='bold')

method='Area_cat_Percentage'

# ax1.text(-10,680,'True $r_R$',fontsize=20, rotation=90, verticalalignment='center')
# ax1.text(74,680,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')


# 4
ax1.text(10,24350,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')
ax1.text(10,21500,f'{round(df_perc[method][14],0)}%',fontsize=16)
# ax1.text(10,20900,f'{round(df_perc[method][13],0)}%',fontsize=16)
# ax1.text(1,44900,f'{round(df_perc[method][12],0)}%',fontsize=16)
# 3
ax1.text(10,18200,f'{round(df_perc[method][11],0)}%',fontsize=16)
ax1.text(10,12800,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(10,8700,f'{round(df_perc[method][9],0)}%',fontsize=16)
# ax1.text(10,8100,f'{round(df_perc[method][8],0)}%',fontsize=16)
# 2
# ax1.text(10,7400,f'{round(df_perc[method][7],0)}%',fontsize=16)
ax1.text(10,5600,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(10,1400,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
# ax1.text(10,10,f'{round(df_perc[method][4],0)}%',fontsize=16)
# 1
# ax1.text(1,3650,f'{round(df_perc[method][2],0)}%',fontsize=16) #leave out below 5%
# ax1.text(10,4500,f'{round(df_perc[method][1],0)}%',fontsize=16)
# ax1.text(10,500,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')


plt.show() # to display
# plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area_negcurves.png', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area_negcurves.pdf', bbox_inches='tight') 
# plt.savefig(path_savefig / f'transects20/Sankey_transects20_N{N}_Area_negcurves.svg', bbox_inches='tight') 



