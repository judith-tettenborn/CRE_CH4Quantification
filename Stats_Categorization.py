# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:04:36 2023

@author: Judith Tettenborn (j.a.f.tettenborn@uu.nl)

This script does two things:
    1. Categorize emission rates (the "true release rate" and the calculated ones)
    into emission categories
    2. Visualize the categorization success using Sankey Plots
    
    
For each valid plume transect (i) in the dataset used to derive the regression model, 
emission rates were estimated utilizing the Weller eq., and the two derived regression 
models ("Area eq." and "Maximum eq."). Subsequently, an estimated emission category 
was assigned to each peak given these inferred emission rates. To remain consistent 
with previous studies, release rates were classified into four different categories 
<0.5 Lmin−1 - Very low
0.5−6 Lmin−1 - Low
6−40 Lmin−1 - Medium
>40 Lmin−1 - High

This approach follows von Fischer et al. (2017), but is extended by a category 
for very small emissions as used in Vogel et al. (2024). For each group of peaks 
belonging to the same category, the percentage of correctly classified peaks was 
calculated, along with the percentages of peaks that were erroneously categorized 
into other categories


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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pysankey import sankey

from helper_functions.constants import (
    dict_color_category
    )

from stats_analysis.stats_functions import *



path_finaldata  = path_base / 'data' / 'final'
path_res        = path_base / 'results/'  
if not (path_res / 'Figures' / 'STATS' / 'Categorization').is_dir():
       (path_res / 'Figures' / 'STATS' / 'Categorization').mkdir(parents=True)
savepath_fig    = path_res  / 'Figures' / 'STATS' / 'Categorization/'

# Load dataset 
total_peaks_all_alldist = pd.read_csv(path_finaldata / 'RU2T2L4L2_TOTAL_PEAKS.csv', index_col='Datetime', parse_dates=['Datetime'])  
data_alldist            = pd.read_csv(path_finaldata / 'RU2T2L4L2_TOTAL_PEAKS_comb.csv', index_col='Datetime', parse_dates=['Datetime']) 

total_peaks_all = total_peaks_all_alldist[total_peaks_all_alldist['Distance_to_source']<75].copy()
data = data_alldist[data_alldist['Distance_to_source']<75].copy()

data = data[((data['City']!='London I-Day3') & (data['City']!='London I-Day4'))]
data = data[~((data['City'] == 'Rotterdam') & (data['Loc'] != 1))]
data_count = data.groupby(['Release_rate','City']).size().reset_index(name='count')
total_peaks_count = total_peaks_all.groupby(['Release_rate']).size().reset_index(name='count')

data['ln(rr)'] = np.log(data['Release_rate'])





#%% Categorization


df0 = data.copy(deep=True)
df0 = calc_rE(df0)
df0 = add_true_rr_cat(df0)

df0 = categorize_release_rates_str(df0)



#%% Sankey Diagramm


#%%% --ALL--

df = df0.copy()

est_methods = ['Weller_cat','Max_cat','Area_cat']

df_perc = df.copy()
df_perc = mean_cat_success_str(df_perc,est_methods)



#%%% ALL: Area

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
ax1.text(-17,0.5*y_max,'True $r_E$',fontsize=20, rotation=90, verticalalignment='center')
ax1.text(105,0.5*y_max,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')

y_min, y_max = ax1.dataLim.y0, ax1.dataLim.y1


# 1
ax1.text(1,19,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')
ax1.text(1,80,f'{round(df_perc[method][1],0)}%',fontsize=16)
#ax1.text(1,105,f'{round(df_perc[method][2],0)}%',fontsize=16) #leave out below 5%
# 2
ax1.text(1,160,f'{round(df_perc[method][4],0)}%',fontsize=16)
ax1.text(1,330,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(1,560,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(1,660,f'{round(df_perc[method][7],0)}%',fontsize=16)
# 3
#ax1.text(1,1005,f'{round(df_perc[method][8],0)}%',fontsize=16)
ax1.text(1,785,f'{round(df_perc[method][9],0)}%',fontsize=16)
ax1.text(1,1010,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(1,1280,f'{round(df_perc[method][11],0)}%',fontsize=16)
# 4
#ax1.text(1,1080,f'{round(df_perc[method][12],0)}%',fontsize=14)
ax1.text(1,1465,f'{round(df_perc[method][13],0)}%',fontsize=14)
ax1.text(1,1525,f'{round(df_perc[method][14],0)}%',fontsize=16)
ax1.text(1,1700,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')


plt.show() # to display
# plt.savefig(savepath_fig+'Sankey_ALL_Area.png', bbox_inches='tight') 
# plt.savefig(savepath_fig+'Sankey_ALL_Area.pdf', bbox_inches='tight') 
# plt.savefig(savepath_fig+'Sankey_ALL_Area.svg', bbox_inches='tight') 



#%%% ALL: Maximum

fig,ax = plt.subplots(figsize=(9,12))


ax1 = sankey(
    df['True_cat'], df['Max_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    rightLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    fontsize=20,
    ax=ax
)

# plt.title(f'Maximum eq.',fontsize=24,fontweight='bold')

method='Max_cat_Percentage'

# Rotate category labels
for text in ax.texts:
    text.set_rotation(90)  # Rotate text to vertical

# Version 0 - y axis label on the side ----------------------------------------    
ax1.text(-17,900,'True $r_E$',fontsize=20, rotation=90, verticalalignment='center')
ax1.text(105,900,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')


# 1
ax1.text(1,19,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')
ax1.text(1,80,f'{round(df_perc[method][1],0)}%',fontsize=16)
#ax1.text(1,105,f'{round(df_perc[method][2],0)}%',fontsize=16) #leave out below 5%
# 2
ax1.text(1,165,f'{round(df_perc[method][4],0)}%',fontsize=16)
ax1.text(1,335,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(1,560,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(1,660,f'{round(df_perc[method][7],0)}%',fontsize=16)
# 3
#ax1.text(1,1005,f'{round(df_perc[method][8],0)}%',fontsize=16)
ax1.text(1,790,f'{round(df_perc[method][9],0)}%',fontsize=16)
ax1.text(1,1010,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(1,1280,f'{round(df_perc[method][11],0)}%',fontsize=16)
# 4
#ax1.text(1,1080,f'{round(df_perc[method][12],0)}%',fontsize=14)
ax1.text(1,1465,f'{round(df_perc[method][13],0)}%',fontsize=14)
ax1.text(1,1525,f'{round(df_perc[method][14],0)}%',fontsize=16)
ax1.text(1,1700,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')


plt.show() # to display
# plt.savefig(savepath_fig+'Sankey_ALL_Max.png', bbox_inches='tight') # to save
# plt.savefig(savepath_fig+'Sankey_ALL_Max.pdf', bbox_inches='tight') # to save
# plt.savefig(savepath_fig+'Sankey_ALL_Max.svg', bbox_inches='tight') # to save

#%%% ALL: Weller

fig,ax = plt.subplots(figsize=(9,12))

# plt.title(f'Weller eq.',fontsize=24,fontweight='bold')

ax1 = sankey(
    df['True_cat'], df['Weller_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    rightLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    fontsize=20,
    ax=ax
)

method='Weller_cat_Percentage'

# Rotate category labels
for text in ax.texts:
    text.set_rotation(90)  # Rotate text to vertical

y_min, y_max = ax1.dataLim.y0, ax1.dataLim.y1

ax1.text(-15,0.5*y_max,'True $r_R$',fontsize=20, rotation=90, verticalalignment='center')
ax1.text(105,0.5*y_max,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')

# 1
ax1.text(1,19,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')
ax1.text(1,80,f'{round(df_perc[method][1],0)}%',fontsize=16)
#ax1.text(1,105,f'{round(df_perc[method][2],0)}%',fontsize=16) #leave out below 5%
# 2
ax1.text(1,180,f'{round(df_perc[method][4],0)}%',fontsize=16)
ax1.text(1,350,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(1,570,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(1,660,f'{round(df_perc[method][7],0)}%',fontsize=16)
# 3
#ax1.text(1,1005,f'{round(df_perc[method][8],0)}%',fontsize=16)
ax1.text(1,800,f'{round(df_perc[method][9],0)}%',fontsize=16)
ax1.text(1,1030,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(1,1290,f'{round(df_perc[method][11],0)}%',fontsize=16)
# 4
#ax1.text(1,1080,f'{round(df_perc[method][12],0)}%',fontsize=14)
ax1.text(1,1465,f'{round(df_perc[method][13],0)}%',fontsize=14)
ax1.text(1,1545,f'{round(df_perc[method][14],0)}%',fontsize=16)
ax1.text(1,1700,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')


plt.show() # to display
# plt.savefig(savepath_fig+'Sankey_ALL_Weller.png', bbox_inches='tight') # to save
# plt.savefig(savepath_fig+'Sankey_ALL_Weller.pdf', bbox_inches='tight') # to save
# plt.savefig(savepath_fig+'Sankey_ALL_Weller.svg', bbox_inches='tight') # to save

#%%% --Rotterdam--

df = df0[df0['City']=='Rotterdam'].copy()

est_methods = ['Weller_cat','Max_cat','Area_cat']

df_perc = df.copy()
df_perc = mean_cat_success_str(df,est_methods)



#%%% R: Area

fig,ax = plt.subplots(figsize=(9,12))

# plt.title(f'Rotterdam',fontsize=20,fontweight='bold')


ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['2-Low', '3-Medium', '4-High'],
    rightLabels=['2-Low', '3-Medium', '4-High'],
    fontsize=20,
    ax=ax
)
method='Area_cat_Percentage'

# Rotate category labels
for text in ax.texts:
    text.set_rotation(90)  # Rotate text to vertical

y_min, y_max = ax1.dataLim.y0, ax1.dataLim.y1

ax1.text(-5,0.5*y_max,'True $r_E$',fontsize=20, rotation=90, verticalalignment='center')
ax1.text(34,0.5*y_max,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')

# 4
ax1.text(0.3,540,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')
ax1.text(0.3,482,f'{round(df_perc[method][14],0)}%',fontsize=18)
ax1.text(0.3,465,f'{round(df_perc[method][13],0)}%',fontsize=18)
# ax1.text(1,340,f'{round(df_perc[method][12],0)}%',fontsize=14)
# 3
ax1.text(0.3,385,f'{round(df_perc[method][11],0)}%',fontsize=18)
ax1.text(0.3,270,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(0.3,175,f'{round(df_perc[method][9],0)}%',fontsize=18)
# ax1.text(0.3,242,f'{round(df_perc[method][8],0)}%',fontsize=18)
# 2
ax1.text(0.3,123,f'{round(df_perc[method][7],0)}%',fontsize=18)
ax1.text(0.3,75,f'{round(df_perc[method][6],0)}%',fontsize=18)
ax1.text(0.3,11,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
# ax1.text(0.3,66,f'{round(df_perc[method][4],0)}%',fontsize=16)

 
# plt.savefig(savepath_fig+'Rotterdam/Sankey_Rloc1_Area.png', bbox_inches='tight') 
# plt.savefig(savepath_fig+'Rotterdam/Sankey_Rloc1_Area.pdf', bbox_inches='tight')
# plt.savefig(savepath_fig+'Rotterdam/Sankey_Rloc1_Area.svg', bbox_inches='tight')

# plt.savefig(savepath_fig+'Rotterdam/Sankey_Rloc1_Area_title.png', bbox_inches='tight') 
# plt.savefig(savepath_fig+'Rotterdam/Sankey_Rloc1_Area_title.pdf', bbox_inches='tight')
# plt.savefig(savepath_fig+'Rotterdam/Sankey_Rloc1_Area_title.svg', bbox_inches='tight')



#%%% R: Weller

fig,ax = plt.subplots(figsize=(9,12))

# plt.title(f'Weller eq.',fontsize=20,fontweight='bold')

method='Weller_cat_Percentage'

ax1 = sankey(
    df['True_cat'], df['Weller_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['2-Low', '3-Medium', '4-High'],
    rightLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    fontsize=20,
    ax=ax
)

# Rotate category labels
for text in ax.texts:
    text.set_rotation(90)  # Rotate text to vertical

y_min, y_max = ax1.dataLim.y0, ax1.dataLim.y1

ax1.text(-5,0.5*y_max,'True $r_R$',fontsize=20, rotation=90, verticalalignment='center')
ax1.text(34,0.5*y_max,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')

# 1
ax1.text(1,0,f'{round(df_perc[method][0],0)}%',fontsize=18,weight='bold')
ax1.text(1,50,f'{round(df_perc[method][1],0)}%',fontsize=14)
#ax1.text(1,130,f'{round(df_perc[method][2],0)}%',fontsize=14)
# 2
#ax1.text(1,170,f'{round(df_perc[method][4],0)}%',fontsize=14)
ax1.text(1,205,f'{round(df_perc[method][5],0)}%',fontsize=18,weight='bold')
ax1.text(1,340,f'{round(df_perc[method][6],0)}%',fontsize=14)
ax1.text(1,425,f'{round(df_perc[method][7],0)}%',fontsize=14)
# 3
#ax1.text(1,470,f'{round(df_perc[method][8],0)}%',fontsize=14)
ax1.text(1,485,f'{round(df_perc[method][9],0)}%',fontsize=14)
ax1.text(1,560,f'{round(df_perc[method][10],0)}%',fontsize=18,weight='bold')
ax1.text(1,680,f'{round(df_perc[method][11],0)}%',fontsize=14)
# 4
#ax1.text(1,773,f'{round(df_perc[method][12],0)}%',fontsize=14)
ax1.text(1,778,f'{round(df_perc[method][13],0)}%',fontsize=14)
ax1.text(1,794,f'{round(df_perc[method][14],0)}%',fontsize=14)
ax1.text(1,855,f'{round(df_perc[method][15],0)}%',fontsize=18,weight='bold')

# plt.savefig(savepath_fig+'Rotterdam/Sankey_R_Weller.png', bbox_inches='tight') # to save
# plt.savefig(savepath_fig+'Rotterdam/Sankey_R_Weller.pdf', bbox_inches='tight') # to save
# plt.savefig(savepath_fig+'Rotterdam/Sankey_R_Weller.svg', bbox_inches='tight') # to save



#%%% --Utrecht II --

df = df0.copy()

df = df[(df['City'] == 'Utrecht II')]

est_methods = ['Weller_cat','Max_cat','Area_cat']

df_perc = df.copy()
df_perc = mean_cat_success_str(df_perc,est_methods)


#%%% U II: Area

fig,ax = plt.subplots(figsize=(9,12))

# plt.title(f'Utrecht II',fontsize=20,fontweight='bold')

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

y_min, y_max = ax1.dataLim.y0, ax1.dataLim.y1
x_min, x_max = ax1.dataLim.x0, ax1.dataLim.x1

ax1.text(x_min-3.5,0.5*y_max,'True $r_E$',fontsize=20, rotation=90, verticalalignment='center')
ax1.text(x_max+2.5,0.5*y_max,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')


# 4
ax1.text(0.3,410,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')
ax1.text(0.3,380,f'{round(df_perc[method][14],0)}%',fontsize=16)
# ax1.text(1,360,f'{round(df_perc[method][13],0)}%',fontsize=14)
# ax1.text(1,340,f'{round(df_perc[method][12],0)}%',fontsize=14)
# 3
ax1.text(0.3,350,f'{round(df_perc[method][11],0)}%',fontsize=16)
ax1.text(0.3,300,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(0.3,255,f'{round(df_perc[method][9],0)}%',fontsize=16)
ax1.text(0.3,242,f'{round(df_perc[method][8],0)}%',fontsize=16)
# 2
ax1.text(0.3,227,f'{round(df_perc[method][7],0)}%',fontsize=16)
ax1.text(0.3,200,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(0.3,120,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(0.3,66,f'{round(df_perc[method][4],0)}%',fontsize=16)
# 1
#ax1.text(1,130,f'{round(df_perc[method][2],0)}%',fontsize=14)
ax1.text(0.3,40,f'{round(df_perc[method][1],0)}%',fontsize=16)
ax1.text(0.3,12,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')


# plt.savefig(savepath_fig+'Utrecht_III/Sankey_U3_Area.png', bbox_inches='tight') # save figure
# plt.savefig(savepath_fig+'Utrecht_III/Sankey_U3_Area.pdf', bbox_inches='tight') 
# plt.savefig(savepath_fig+'Utrecht_III/Sankey_U3_Area.svg', bbox_inches='tight') 

# plt.savefig(savepath_fig+'Utrecht_III/Sankey_U3_Area_title.png', bbox_inches='tight') # save figure
# plt.savefig(savepath_fig+'Utrecht_III/Sankey_U3_Area_title.pdf', bbox_inches='tight') 
# plt.savefig(savepath_fig+'Utrecht_III/Sankey_U3_Area_title.svg', bbox_inches='tight') 



#%%% U II: Weller

fig,ax = plt.subplots(figsize=(9,12))

# plt.title(f'N=1: Weller eq. - Utrecht III',fontsize=20,fontweight='bold')

method='Weller_cat_Percentage'

ax1 = sankey(
    df['True_cat'], df['Weller_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    rightLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    fontsize=20,
    ax=ax
)

# Rotate category labels
for text in ax.texts:
    text.set_rotation(90)  # Rotate text to vertical

y_min, y_max = ax1.dataLim.y0, ax1.dataLim.y1
x_min, x_max = ax1.dataLim.x0, ax1.dataLim.x1

ax1.text(x_min-3.5,0.5*y_max,'True $r_R$',fontsize=20, rotation=90, verticalalignment='center')
ax1.text(x_max+2.5,0.5*y_max,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')


# 1
ax1.text(0.3,15,f'{round(df_perc[method][0],0)}%',fontsize=18,weight='bold')
ax1.text(0.3,45,f'{round(df_perc[method][1],0)}%',fontsize=14)
#ax1.text(1,130,f'{round(df_perc[method][2],0)}%',fontsize=14)
# 2
ax1.text(0.3,75,f'{round(df_perc[method][4],0)}%',fontsize=14)
ax1.text(0.3,150,f'{round(df_perc[method][5],0)}%',fontsize=18,weight='bold')
ax1.text(0.3,220,f'{round(df_perc[method][6],0)}%',fontsize=14)
# ax1.text(0.3,227,f'{round(df_perc[method][7],0)}%',fontsize=14)
# 3
ax1.text(0.3,242,f'{round(df_perc[method][8],0)}%',fontsize=14)
ax1.text(0.3,280,f'{round(df_perc[method][9],0)}%',fontsize=14)
ax1.text(0.3,330,f'{round(df_perc[method][10],0)}%',fontsize=18,weight='bold')
ax1.text(0.3,365,f'{round(df_perc[method][11],0)}%',fontsize=14)
# 4
# ax1.text(1,340,f'{round(df_perc[method][12],0)}%',fontsize=14)
ax1.text(0.3,377,f'{round(df_perc[method][13],0)}%',fontsize=14)
ax1.text(0.3,390,f'{round(df_perc[method][14],0)}%',fontsize=14)
ax1.text(0.3,420,f'{round(df_perc[method][15],0)}%',fontsize=18,weight='bold')

# plt.savefig(savepath_fig+'Utrecht_III/Sankey_U3_Weller.png', bbox_inches='tight') # save figure
# plt.savefig(savepath_fig+'Utrecht_III/Sankey_U3_Weller.pdf', bbox_inches='tight') 
# plt.savefig(savepath_fig+'Utrecht_III/Sankey_U3_Weller.svg', bbox_inches='tight') 


#%%% --Utrecht I--

df = df0[df0['City']=='Utrecht I'].copy()

est_methods = ['Weller_cat','Max_cat','Area_cat']

df_perc = df.copy()
df_perc = mean_cat_success_str(df,est_methods)


# df['True_cat'] = df['True_cat'].astype(str)
# df['Weller_cat'] = df['Weller_cat'].astype(str)
# df['Max_cat'] = df['Max_cat'].astype(str)
# df['Area_cat'] = df['Area_cat'].astype(str)
# df['Maximum2_cat'] = df['Maximum2_cat'].astype(str)
# df['Area2_cat'] = df['Area2_cat'].astype(str)
# df['Max_odr_cat'] = df['Max_odr_cat'].astype(str)
# df['Area_odr_cat'] = df['Area_odr_cat'].astype(str)
# df['AM_cat'] = df['AM_cat'].astype(str)
# df['AMK_cat'] = df['AMK_cat'].astype(str)
# df['GNB_AM_cat'] = df['GNB_AM_cat'].astype(str)
# df['GNB_AMK_cat'] = df['GNB_AMK_cat'].astype(str)


#%%% U I: Area

fig,ax = plt.subplots(figsize=(9,12))

# plt.title(f'Utrecht I',fontsize=20,fontweight='bold')

method='Area_cat_Percentage'

ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['2-Low', '3-Medium'],
    rightLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    fontsize=20,
    ax=ax
)

# Rotate category labels
for text in ax.texts:
    text.set_rotation(90)  # Rotate text to vertical

y_min, y_max = ax1.dataLim.y0, ax1.dataLim.y1
x_min, x_max = ax1.dataLim.x0, ax1.dataLim.x1

ax1.text(x_min-1.8,0.5*y_max,'True $r_E$',fontsize=20, rotation=90, verticalalignment='center')
ax1.text(x_max+1.2,0.5*y_max,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')


# 3
ax1.text(0.2,173,f'{round(df_perc[method][11],0)}%',fontsize=18)
ax1.text(0.2,122,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(0.2,109,f'{round(df_perc[method][9],0)}%',fontsize=18)
# 2
ax1.text(0.2,99,f'{round(df_perc[method][7],0)}%',fontsize=18)
ax1.text(0.2,81,f'{round(df_perc[method][6],0)}%',fontsize=18)
ax1.text(0.2,30,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(0.2,0,f'{round(df_perc[method][4],0)}%',fontsize=18)



# plt.savefig(savepath_fig+'Utrecht/Sankey_U_Area.png', bbox_inches='tight') 
# plt.savefig(savepath_fig+'Utrecht/Sankey_U_Area.pdf', bbox_inches='tight') 
# plt.savefig(savepath_fig+'Utrecht/Sankey_U_Area.svg', bbox_inches='tight') 

# plt.savefig(savepath_fig+'Utrecht/Sankey_U_Area_title.png', bbox_inches='tight') # save figure
# plt.savefig(savepath_fig+'Utrecht/Sankey_U_Area_title.pdf', bbox_inches='tight') 
# plt.savefig(savepath_fig+'Utrecht/Sankey_U_Area_title.svg', bbox_inches='tight') 



#%%% U Different Regression Models

# U Area 2

fig,ax = plt.subplots(figsize=(10,12))
ax1 = sankey(
    df['True_cat'], df['Area2_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['2','3'],
    rightLabels=['2','3','4'],
    fontsize=20,
    ax=ax
)
plt.title(f'Area eq. 2',fontsize=20,fontweight='bold')

method='Area2_cat_Percentage'

ax1.text(-1.5,100,'True $r_R$',fontsize=20, rotation=90, verticalalignment='center')
ax1.text(12,100,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')

# 2
#ax1.text(0.1,17,f'{round(df_perc[method][4],0)}%',fontsize=16)
ax1.text(0.1,30,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(0.1,70,f'{round(df_perc[method][6],0)}%',fontsize=16)
#ax1.text(0.1,81,f'{round(df_perc[method][7],0)}%',fontsize=16)
# 3
ax1.text(0.1,93,f'{round(df_perc[method][9],0)}%',fontsize=16)
ax1.text(0.1,130,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(0.1,190,f'{round(df_perc[method][11],0)}%',fontsize=16)

# plt.savefig(savepath_ols+'Sankey_U_Area2.png', bbox_inches='tight') # to save
# plt.savefig(savepath_ols+'Sankey_U_Area2.pdf', bbox_inches='tight') # to save
# plt.savefig(savepath_ols+'Sankey_U_Area2.svg', bbox_inches='tight') # to save



# U: Area ODR

fig,ax = plt.subplots(figsize=(10,12))
ax1 = sankey(
    df['True_cat'], df['Area_odr_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['2','3'],
    rightLabels=['1','2','3','4'],
    fontsize=20,
    ax=ax
)
plt.title(f'Area eq. ODR',fontsize=20,fontweight='bold')

method='Area_odr_cat_Percentage'

ax1.text(-1.5,100,'True $r_R$', fontsize=20, rotation=90, verticalalignment='center') #
ax1.text(12,100,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')

# 2
ax1.text(0.1,17,f'{round(df_perc[method][4],0)}%',fontsize=16)
ax1.text(0.1,47,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(0.1,70,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(0.1,81,f'{round(df_perc[method][7],0)}%',fontsize=16)
# 3
ax1.text(0.1,90,f'{round(df_perc[method][9],0)}%',fontsize=16)
ax1.text(0.1,96,f'{round(df_perc[method][9],0)}%',fontsize=16)
ax1.text(0.1,108,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(0.1,158,f'{round(df_perc[method][11],0)}%',fontsize=16)



plt.show() # to display
# plt.savefig(savepath+'Leak_Categorization/Sankey_R_Area.png', bbox_inches='tight') # to save
# plt.savefig(savepath+'Leak_Categorization/Sankey_R_Area.pdf', bbox_inches='tight') # to save
# plt.savefig(savepath+'Leak_Categorization/Sankey_R_Area.svg', bbox_inches='tight') # to save

# U: Max ODR

fig,ax = plt.subplots(figsize=(10,12))
ax1 = sankey(
    df['True_cat'], df['Max_odr_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['2','3'],
    rightLabels=['1','2','3','4'],
    fontsize=20,
    ax=ax
)
plt.title(f'Maximum eq. ODR',fontsize=20,fontweight='bold')

method='Max_odr_cat_Percentage'

ax1.text(-1.5,100,'True $r_R$', fontsize=20, rotation=90, verticalalignment='center') #
ax1.text(12,100,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')

# 2
ax1.text(0.1,17,f'{round(df_perc[method][4],0)}%',fontsize=16)
ax1.text(0.1,47,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(0.1,70,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(0.1,81,f'{round(df_perc[method][7],0)}%',fontsize=16)
# 3
ax1.text(0.1,90,f'{round(df_perc[method][9],0)}%',fontsize=16)
ax1.text(0.1,96,f'{round(df_perc[method][9],0)}%',fontsize=16)
ax1.text(0.1,108,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(0.1,158,f'{round(df_perc[method][11],0)}%',fontsize=16)

# U: GNB-AM

fig,ax = plt.subplots(figsize=(10,12))
ax1 = sankey(
    df['True_cat'], df['GNB_AM_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['2','3'],
    rightLabels=['1','2','3','4'],
    fontsize=20,
    ax=ax
)
plt.title(f'GNB AM',fontsize=20,fontweight='bold')

method='GNB_AM_cat_Percentage'

ax1.text(-1.5,100,'True $r_R$', fontsize=20, rotation=90, verticalalignment='center') #
ax1.text(12,100,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')

# 2
ax1.text(0.1,16,f'{round(df_perc[method][4],0)}%',fontsize=16)
ax1.text(0.1,47,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(0.1,75,f'{round(df_perc[method][6],0)}%',fontsize=16)
#ax1.text(0.1,81,f'{round(df_perc[method][7],0)}%',fontsize=16)
# 3
ax1.text(0.1,90,f'{round(df_perc[method][9],0)}%',fontsize=16)
ax1.text(0.1,96,f'{round(df_perc[method][9],0)}%',fontsize=16)
ax1.text(0.1,138,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(0.1,180,f'{round(df_perc[method][11],0)}%',fontsize=16)

# plt.savefig(savepath_gnb+'Sankey_U_GNB_AM.png', bbox_inches='tight') # to save
# plt.savefig(savepath_gnb+'Sankey_U_GNB_AM.pdf', bbox_inches='tight') # to save
# plt.savefig(savepath_gnb+'Sankey_U_GNB_AM.svg', bbox_inches='tight') # to save

# U: GNB-AMK

fig,ax = plt.subplots(figsize=(10,12))
ax1 = sankey(
    df['True_cat'], df['GNB_AMK_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['2','3'],
    rightLabels=['1','2','3','4'],
    fontsize=20,
    ax=ax
)
plt.title(f'GNB AMK',fontsize=20,fontweight='bold')

method='GNB_AMK_cat_Percentage'

ax1.text(-1.5,100,'True $r_R$', fontsize=20, rotation=90, verticalalignment='center') #
ax1.text(12,100,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')

# 2
ax1.text(0.1,16,f'{round(df_perc[method][4],0)}%',fontsize=16)
ax1.text(0.1,47,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(0.1,72,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(0.1,83,f'{round(df_perc[method][7],0)}%',fontsize=16)
# 3
ax1.text(0.1,90,f'{round(df_perc[method][9],0)}%',fontsize=16)
ax1.text(0.1,98,f'{round(df_perc[method][9],0)}%',fontsize=16)
ax1.text(0.1,125,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(0.1,178,f'{round(df_perc[method][11],0)}%',fontsize=16)

# plt.savefig(savepath_gnb+'Sankey_U_GNB_AMK.png', bbox_inches='tight') # to save
# plt.savefig(savepath_gnb+'Sankey_U_GNB_AMK.pdf', bbox_inches='tight') # to save
# plt.savefig(savepath_gnb+'Sankey_U_GNB_AMK.svg', bbox_inches='tight') # to save


# U: AM

# leftLabels=['Very Low','Low','Medium','High'],
# rightLabels=['Very Low','Low','Medium','High'],

fig,ax = plt.subplots(figsize=(10,12))
ax1 = sankey(
    df['True_cat'], df['AM_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['2','3'],
    rightLabels=['2','3','4'],
    fontsize=20,
    ax=ax
)
# fig.suptitle(r'Utrecht',ha='center',y=0.95,fontsize=18)
# plt.title('Area', fontsize=18,ha='center', y=1.01, fontweight="bold")
#plt.title(f'Utrecht\nMax eq.',fontsize=20)
plt.title(f'AM model',fontsize=20,fontweight='bold')

method='AM_cat_Percentage'

ax1.text(-1.5,100,'True $r_R$',fontsize=20, rotation=90, verticalalignment='center')
ax1.text(12,100,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')

# 2
#ax1.text(0.1,17,f'{round(U1_cat_countandpercent[method][4],0)}%',fontsize=16)
ax1.text(0.1,27,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(0.1,69,f'{round(df_perc[method][6],0)}%',fontsize=16)
#ax1.text(0.1,80,f'{round(U1_cat_countandpercent[method][7],0)}%',fontsize=16)
# 3
#ax1.text(0.1,90,f'{round(U1_cat_countandpercent[method][8],0)}%',fontsize=16)
ax1.text(0.1,95,f'{round(df_perc[method][9],0)}%',fontsize=16)
ax1.text(0.1,135,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(0.1,190,f'{round(df_perc[method][11],0)}%',fontsize=16)

# plt.savefig(savepath_multiOLS+'Sankey_U_AM.png', bbox_inches='tight') 
# plt.savefig(savepath_multiOLS+'Sankey_U_AM.pdf', bbox_inches='tight') 
# plt.savefig(savepath_multiOLS+'Sankey_U_AM.svg', bbox_inches='tight')





#%%% ---London I Day2---

df = df0[df0['City']=='London I-Day2'].copy()

est_methods = ['Weller_cat','Max_cat','Area_cat']

df_perc = df.copy()
df_perc = mean_cat_success_str(df,est_methods)



#%%% L Day2: Area

fig,ax = plt.subplots(figsize=(9,12))

# plt.title(f'London I-Day2',fontsize=20,fontweight='bold')

method='Area_cat_Percentage'

ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['3-Medium', '4-High'],
    rightLabels=['2-Low', '3-Medium', '4-High'],
    fontsize=20,
    ax=ax
)

# Rotate category labels
for text in ax.texts:
    text.set_rotation(90)  # Rotate text to vertical

y_min, y_max = ax1.dataLim.y0, ax1.dataLim.y1
x_min, x_max = ax1.dataLim.x0, ax1.dataLim.x1

ax1.text(x_min-0.8,0.5*y_max,'True $r_E$',fontsize=20, rotation=90, verticalalignment='center')
ax1.text(x_max+0.55,0.5*y_max,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')

# 4
ax1.text(0.05,75,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')
ax1.text(0.05,41,f'{round(df_perc[method][14],0)}%',fontsize=18)
ax1.text(0.05,28.5,f'{round(df_perc[method][13],0)}%',fontsize=18)
# ax1.text(1,340,f'{round(df_perc[method][12],0)}%',fontsize=14)
# 3
ax1.text(0.05,23.6,f'{round(df_perc[method][11],0)}%',fontsize=18)
ax1.text(0.05,11,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(0.05,0,f'{round(df_perc[method][9],0)}%',fontsize=18)
# ax1.text(0.1,0,f'{round(df_perc[method][8],0)}%',fontsize=14)


# plt.savefig(savepath_fig+'London I/Sankey_Ld2_Area.png', bbox_inches='tight') 
# plt.savefig(savepath_fig+'London I/Sankey_Ld2_Area.pdf', bbox_inches='tight') 
# plt.savefig(savepath_fig+'London I/Sankey_Ld2_Area.svg', bbox_inches='tight') 

# plt.savefig(savepath_fig+'London I/Sankey_Ld2_Area_title.png', bbox_inches='tight') # save figure
# plt.savefig(savepath_fig+'London I/Sankey_Ld2_Area_title.pdf', bbox_inches='tight') 
# plt.savefig(savepath_fig+'London I/Sankey_Ld2_Area_title.svg', bbox_inches='tight') 



#%%% ---London II ---

df = df0[(df0['City']=='London II-Day1') | (df0['City']=='London II-Day2')].copy()

# df = df[df['Distance_to_source']<50]

est_methods = ['Weller_cat','Max_cat','Area_cat']

df_perc = df.copy()
df_perc = mean_cat_success_str(df,est_methods)



#%%% LII Day1: Area

fig,ax = plt.subplots(figsize=(9,12))

# plt.title(f'London II',fontsize=20,fontweight='bold')

method='Area_cat_Percentage'

ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    rightLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    fontsize=20,
    ax=ax
)
# plt.title(f'Area eq.',fontsize=24,fontweight='bold')

# Rotate category labels
for text in ax.texts:
    text.set_rotation(90)  # Rotate text to vertical
    
y_min, y_max = ax1.dataLim.y0, ax1.dataLim.y1
x_min, x_max = ax1.dataLim.x0, ax1.dataLim.x1

ax1.text(x_min-2.2,0.5*y_max,'True $r_E$',fontsize=20, rotation=90, verticalalignment='center')
ax1.text(x_max+1.5,0.5*y_max,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')

# 4
ax1.text(0.2,296,f'{round(df_perc[method][15],0)}%',fontsize=18,weight='bold')
ax1.text(0.2,269,f'{round(df_perc[method][14],0)}%',fontsize=16)
ax1.text(0.2,241,f'{round(df_perc[method][13],0)}%',fontsize=16)
# ax1.text(1,200,f'{round(df_perc[method][12],0)}%',fontsize=16)
# 3
# ax1.text(0.2,195,f'{round(df_perc[method][11],0)}%',fontsize=16)
ax1.text(0.2,205,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(0.2,155,f'{round(df_perc[method][9],0)}%',fontsize=16)
# ax1.text(0.1,109,f'{round(df_perc[method][8],0)}%',fontsize=16)
# 2
# ax1.text(0.1,99,f'{round(df_perc[method][7],0)}%',fontsize=16)
# ax1.text(0.1,81,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(0.2,108,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(0.2,65,f'{round(df_perc[method][4],0)}%',fontsize=16)
# 1
#ax1.text(1,130,f'{round(df_perc[method][2],0)}%',fontsize=14)
# ax1.text(1,90,f'{round(df_perc[method][1],0)}%',fontsize=14)
ax1.text(0.2,13,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')


# plt.savefig(savepath_fig+'London II/Sankey_L2_Area.png', bbox_inches='tight') # save figure
# plt.savefig(savepath_fig+'London II/Sankey_L2_Area.pdf', bbox_inches='tight') 
# plt.savefig(savepath_fig+'London II/Sankey_L2_Area.svg', bbox_inches='tight') 

# plt.savefig(savepath_fig+'London II/Sankey_L2_Area_50m.png', bbox_inches='tight') # save figure
# plt.savefig(savepath_fig+'London II/Sankey_L2_Area_50m.pdf', bbox_inches='tight') 
# plt.savefig(savepath_fig+'London II/Sankey_L2_Area_50m.svg', bbox_inches='tight') 


#%%% ---London II Day 1 ---

df = df0[(df0['City']=='London II-Day1')].copy()

# df = df[df['Distance_to_source']<50]

est_methods = ['Weller_cat','Max_cat','Area_cat']

df_perc = df.copy()
df_perc = mean_cat_success_str(df,est_methods)



#%%% LII Day1: Area

fig,ax = plt.subplots(figsize=(9,12))

# plt.title(f'London II-Day1',fontsize=20,fontweight='bold')

method='Area_cat_Percentage'

ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['2-Low', '3-Medium', '4-High'],
    rightLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    fontsize=20,
    ax=ax
)
# plt.title(f'Area eq.',fontsize=24,fontweight='bold')

# Rotate category labels
for text in ax.texts:
    text.set_rotation(90)  # Rotate text to vertical
    
y_min, y_max = ax1.dataLim.y0, ax1.dataLim.y1
x_min, x_max = ax1.dataLim.x0, ax1.dataLim.x1

ax1.text(x_min-1.55,0.5*y_max,'True $r_E$',fontsize=20, rotation=90, verticalalignment='center')
ax1.text(x_max+1.2,0.5*y_max,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')


# 4
ax1.text(0.2,213,f'{round(df_perc[method][15],0)}%',fontsize=20,weight='bold')
ax1.text(0.2,185,f'{round(df_perc[method][14],0)}%',fontsize=18)
ax1.text(0.2,158,f'{round(df_perc[method][13],0)}%',fontsize=18)
# ax1.text(1,200,f'{round(df_perc[method][12],0)}%',fontsize=16)
# 3
# ax1.text(0.2,195,f'{round(df_perc[method][11],0)}%',fontsize=16)
ax1.text(0.2,125,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(0.2,72,f'{round(df_perc[method][9],0)}%',fontsize=18)
# ax1.text(0.1,109,f'{round(df_perc[method][8],0)}%',fontsize=16)
# 2
# ax1.text(0.1,99,f'{round(df_perc[method][7],0)}%',fontsize=16)
# ax1.text(0.1,81,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(0.2,32,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(0.2,8,f'{round(df_perc[method][4],0)}%',fontsize=18)
# 1
#ax1.text(1,130,f'{round(df_perc[method][2],0)}%',fontsize=14)
# ax1.text(1,90,f'{round(df_perc[method][1],0)}%',fontsize=14)
# ax1.text(0.2,13,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')

# plt.savefig(savepath_fig+'London II/Sankey_L2d1_Area.png', bbox_inches='tight') 
# plt.savefig(savepath_fig+'London II/Sankey_L2d1_Area.pdf', bbox_inches='tight') 
# plt.savefig(savepath_fig+'London II/Sankey_L2d1_Area.svg', bbox_inches='tight') 

# plt.savefig(savepath_fig+'London II/Sankey_L2d1_Area_title.png', bbox_inches='tight') # save figure
# plt.savefig(savepath_fig+'London II/Sankey_L2d1_Area_title.pdf', bbox_inches='tight') 
# plt.savefig(savepath_fig+'London II/Sankey_L2d1_Area_title.svg', bbox_inches='tight') 

#%%% ---London II Day 2 ---

df = df0[(df0['City']=='London II-Day2')].copy()

# df = df[df['Distance_to_source']<50]

est_methods = ['Weller_cat','Max_cat','Area_cat']

df_perc = df.copy()
df_perc = mean_cat_success_str(df,est_methods)


#%%% LII Day1: Area

fig,ax = plt.subplots(figsize=(9,12))

# plt.title(f'London II-Day2',fontsize=20,fontweight='bold')

method='Area_cat_Percentage'

ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1-VL', '2-Low'],
    rightLabels=['1-VL', '2-Low'],
    fontsize=20,
    ax=ax
)
# plt.title(f'Area eq.',fontsize=24,fontweight='bold')

# Rotate category labels
for text in ax.texts:
    text.set_rotation(90)  # Rotate text to vertical
    
y_min, y_max = ax1.dataLim.y0, ax1.dataLim.y1
x_min, x_max = ax1.dataLim.x0, ax1.dataLim.x1

ax1.text(x_min-0.52,0.5*y_max,'True $r_E$',fontsize=20, rotation=90, verticalalignment='center')
ax1.text(x_max+0.4,0.5*y_max,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')


# 2
# ax1.text(0.1,99,f'{round(df_perc[method][7],0)}%',fontsize=16)
# ax1.text(0.1,81,f'{round(df_perc[method][6],0)}%',fontsize=16)
ax1.text(0.1,71,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(0.1,53,f'{round(df_perc[method][4],0)}%',fontsize=18)
# 1
#ax1.text(1,130,f'{round(df_perc[method][2],0)}%',fontsize=14)
# ax1.text(1,90,f'{round(df_perc[method][1],0)}%',fontsize=14)
ax1.text(0.1,16,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')

# plt.savefig(savepath_fig+'London II/Sankey_L2d2_Area.png', bbox_inches='tight') 
# plt.savefig(savepath_fig+'London II/Sankey_L2d2_Area.pdf', bbox_inches='tight') 
# plt.savefig(savepath_fig+'London II/Sankey_L2d2_Area.svg', bbox_inches='tight') 

# plt.savefig(savepath_fig+'London II/Sankey_L2d2_Area_title.png', bbox_inches='tight') # save figure
# plt.savefig(savepath_fig+'London II/Sankey_L2d2_Area_title.pdf', bbox_inches='tight') 
# plt.savefig(savepath_fig+'London II/Sankey_L2d2_Area_title.svg', bbox_inches='tight') 

#%%% --Toronto --

df = df0[(df0['City']=='Toronto-1b') | (df0['City']=='Toronto-1c') | (df0['City']=='Toronto-2c')].copy()

est_methods = ['Weller_cat','Max_cat','Area_cat']

df_perc = df.copy()
df_perc = mean_cat_success_str(df,est_methods)


#%%% T1b: Area

fig,ax = plt.subplots(figsize=(9,12))

# plt.title(f'Toronto',fontsize=20,fontweight='bold')

method='Area_cat_Percentage'

ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1-VL', '2-Low', '3-Medium'],
    rightLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    fontsize=20,
    ax=ax
)
# plt.title(f'Area eq.',fontsize=24,fontweight='bold')

# Rotate category labels
for text in ax.texts:
    text.set_rotation(90)  # Rotate text to vertical
    
y_min, y_max = ax1.dataLim.y0, ax1.dataLim.y1
x_min, x_max = ax1.dataLim.x0, ax1.dataLim.x1

ax1.text(x_min-0.65,0.5*y_max,'True $r_E$',fontsize=20, rotation=90, verticalalignment='center')
ax1.text(x_max+0.48,0.5*y_max,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')
# 3
ax1.text(0.05,90,f'{round(df_perc[method][11],0)}%',fontsize=18)
ax1.text(0.05,80,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(0.05,68.4,f'{round(df_perc[method][9],0)}%',fontsize=18)
ax1.text(0.05,65,f'{round(df_perc[method][8],0)}%',fontsize=16)
# 2
# ax1.text(0.1,99,f'{round(df_perc[method][7],0)}%',fontsize=16)
ax1.text(0.05,59,f'{round(df_perc[method][6],0)}%',fontsize=18)
ax1.text(0.05,44,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(0.05,27,f'{round(df_perc[method][4],0)}%',fontsize=18)
# 1
# ax1.text(1,130,f'{round(df_perc[method][3],0)}%',fontsize=14)
# ax1.text(1,130,f'{round(df_perc[method][2],0)}%',fontsize=14)
ax1.text(0.05,19.4,f'{round(df_perc[method][1],0)}%',fontsize=18)
ax1.text(0.05,8,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')



# plt.savefig(savepath_fig+'Toronto/Sankey_T_Area.png', bbox_inches='tight') # save figure
# plt.savefig(savepath_fig+'Toronto/Sankey_T_Area.pdf', bbox_inches='tight') 
# plt.savefig(savepath_fig+'Toronto/Sankey_T_Area.svg', bbox_inches='tight') 


#%%% --Toronto Day1--

df = df0[(df0['City']=='Toronto-1b') | (df0['City']=='Toronto-1c')].copy()

est_methods = ['Weller_cat','Max_cat','Area_cat']

df_perc = df.copy()
df_perc = mean_cat_success_str(df,est_methods)


#%%% T1c: Area

fig,ax = plt.subplots(figsize=(9,12))

# plt.title(f'Toronto Day1',fontsize=20,fontweight='bold')

method='Area_cat_Percentage'

ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['2-Low', '3-Medium'],
    rightLabels=['1-VL', '2-Low', '3-Medium', '4-High'],
    fontsize=20,
    ax=ax
)
# plt.title(f'Area eq.',fontsize=24,fontweight='bold')

# Rotate category labels
for text in ax.texts:
    text.set_rotation(90)  # Rotate text to vertical
    
y_min, y_max = ax1.dataLim.y0, ax1.dataLim.y1
x_min, x_max = ax1.dataLim.x0, ax1.dataLim.x1

ax1.text(x_min-0.3,0.5*y_max,'True $r_E$',fontsize=20, rotation=90, verticalalignment='center')
ax1.text(x_max+0.2,0.5*y_max,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')

# 3
ax1.text(0.05,39.7,f'{round(df_perc[method][11],0)}%',fontsize=18)
ax1.text(0.05,32,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(0.05,22.7,f'{round(df_perc[method][9],0)}%',fontsize=18)
# ax1.text(0.05,15,f'{round(df_perc[method][8],0)}%',fontsize=16)
# 2
# ax1.text(0.1,99,f'{round(df_perc[method][7],0)}%',fontsize=16)
ax1.text(0.05,17.5,f'{round(df_perc[method][6],0)}%',fontsize=18)
ax1.text(0.05,7.5,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(0.05,0,f'{round(df_perc[method][4],0)}%',fontsize=18)


# plt.savefig(savepath_fig+'Toronto/Sankey_T1_Area.png', bbox_inches='tight') 
# plt.savefig(savepath_fig+'Toronto/Sankey_T1_Area.pdf', bbox_inches='tight') 
# plt.savefig(savepath_fig+'Toronto/Sankey_T1_Area.svg', bbox_inches='tight') 

# plt.savefig(savepath_fig+'Toronto/Sankey_T1_Area_title.png', bbox_inches='tight') # save figure
# plt.savefig(savepath_fig+'Toronto/Sankey_T1_Area_title.pdf', bbox_inches='tight') 
# plt.savefig(savepath_fig+'Toronto/Sankey_T1_Area_title.svg', bbox_inches='tight') 


#%%% --Toronto Day2--

df = df0[df0['City']=='Toronto-2c'].copy()

est_methods = ['Weller_cat','Max_cat','Area_cat']

df_perc = df.copy()
df_perc = mean_cat_success_str(df,est_methods)




#%%% T2c: Area

fig,ax = plt.subplots(figsize=(9,12))

# plt.title(f'Toronto Day2',fontsize=20,fontweight='bold')

method='Area_cat_Percentage'

ax1 = sankey(
    df['True_cat'], df['Area_cat'], aspect=20, colorDict=dict_color_category,
    leftLabels=['1-VL', '2-Low', '3-Medium'],
    rightLabels=['1-VL', '2-Low', '3-Medium'],
    fontsize=20,
    ax=ax
)
# plt.title(f'Area eq.',fontsize=24,fontweight='bold')

# Rotate category labels
for text in ax.texts:
    text.set_rotation(90)  # Rotate text to vertical
    
y_min, y_max = ax1.dataLim.y0, ax1.dataLim.y1
x_min, x_max = ax1.dataLim.x0, ax1.dataLim.x1

ax1.text(x_min-0.352,0.5*y_max,'True $r_E$',fontsize=20, rotation=90, verticalalignment='center')
ax1.text(x_max+0.235,0.5*y_max,'Estimated $r_E$',fontsize=20, rotation=90, verticalalignment='center')
# 3
# ax1.text(0.1,55,f'{round(df_perc[method][11],0)}%',fontsize=18)
ax1.text(0.05,47,f'{round(df_perc[method][10],0)}%',fontsize=20,weight='bold')
ax1.text(0.05,44.2,f'{round(df_perc[method][9],0)}%',fontsize=18)
ax1.text(0.05,42.9,f'{round(df_perc[method][8],0)}%',fontsize=16)
# 2
# ax1.text(0.1,99,f'{round(df_perc[method][7],0)}%',fontsize=16)
ax1.text(0.05,40.8,f'{round(df_perc[method][6],0)}%',fontsize=18)
ax1.text(0.05,35,f'{round(df_perc[method][5],0)}%',fontsize=20,weight='bold')
ax1.text(0.05,26.3,f'{round(df_perc[method][4],0)}%',fontsize=18)
# 1
# ax1.text(1,130,f'{round(df_perc[method][3],0)}%',fontsize=14)
# ax1.text(1,130,f'{round(df_perc[method][2],0)}%',fontsize=14)
ax1.text(0.05,20,f'{round(df_perc[method][1],0)}%',fontsize=18)
ax1.text(0.05,9,f'{round(df_perc[method][0],0)}%',fontsize=20,weight='bold')


# plt.savefig(savepath_fig+'Toronto/Sankey_T2c_Area.png', bbox_inches='tight') 
# plt.savefig(savepath_fig+'Toronto/Sankey_T2c_Area.pdf', bbox_inches='tight') 
# plt.savefig(savepath_fig+'Toronto/Sankey_T2c_Area.svg', bbox_inches='tight') 

# plt.savefig(savepath_fig+'Toronto/Sankey_T2c_Area_title.png', bbox_inches='tight') # save figure
# plt.savefig(savepath_fig+'Toronto/Sankey_T2c_Area_title.pdf', bbox_inches='tight') 
# plt.savefig(savepath_fig+'Toronto/Sankey_T2c_Area_title.svg', bbox_inches='tight') 











































