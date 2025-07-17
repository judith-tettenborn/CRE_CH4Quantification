# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:16:11 2024

@author: Judith Tettenborn (j.a.f.tettenborn@uu.nl)
based on work of Daan Stroeken

Main script for (pre-)processing data.
- reads in data and does some preprocessing
- using scipy.signal.find_peaks CH4 enhancements are detected
- the timestamps, enhancements, etc. of the peaks found are saved into an excel
  file (one file per (sub)-experiment with different excel sheets per instrument)
- overviewplots of the timeseries and peaks detected can pe created (optionally, 
  one per instrument)
- plots of each detected peak can be created for the following quality check step 
  (optional, up to several hundreds per instrument)

"""

# Modify Python Path Programmatically -> To include the directory containing the src folder

from pathlib import Path
import sys

# path_base = Path('C:/Users/.../CRE_CH4Quantification/') # insert the the project path here
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
import time

import matplotlib.pyplot as plt
import tilemapbase
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.ticker import FuncFormatter

from preprocessing.read_in_data import *
from peak_analysis.find_analyze_peaks import *
from plotting.general_plots import *
from helper_functions.utils import *

from helper_functions.constants import (
    dict_color_instr,
    HEIGHT,
    BG_QUANTILE,
    DIST_G23,
    WIDTH_G23,
    DIST_G24,
    WIDTH_G24,
    DIST_LGR,
    WIDTH_LGR,
    DIST_LICOR,
    WIDTH_LICOR,
    DIST_G43,
    WIDTH_G43,
    DIST_AERIS,
    WIDTH_AERIS,
    DIST_MIRO,
    WIDTH_MIRO,
    DIST_AERO,
    WIDTH_AERO
    )


# path Controlled Release Experiment Utrecht
path_dataU1     = path_base / 'data' / 'raw' / 'Utrecht_I_2022'
path_dataU2     = path_base / 'data' / 'raw' / 'Utrecht_II_2024'
path_dataT      = path_base / 'data' / 'raw' / 'Toronto_2021'
path_dataL1     = path_base / 'data' / 'raw' / 'London_I_2019' / 'Pic_uMEA_LiCOR_data'
path_dataL2     = path_base / 'data' / 'raw' / 'London_II_2024'
path_dataR      = path_base / 'data' / 'raw' / 'Rotterdam_2022'


if not (path_base / 'results').is_dir(): # output: processed data
    (path_base / 'results').mkdir(parents=True)
path_res = path_base / 'results'

if not (path_base / 'data' / 'processed').is_dir(): # output: processed data
    (path_base / 'data' / 'processed').mkdir(parents=True)
path_processeddata = path_base / 'data' / 'processed'



overviewplot        = True     # Plot the find_peaks result
writexlsx           = True     # save peaks into excel file
indiv_peak_plots    = True     # plot individual peaks for QC? ATTENTION: when True this will create several hundreds of figures PER instrument



###############################################################################
#%% Utrecht I
###############################################################################


#%%% Load & Preprocess data

# create the folder if it they do not exist
if not (path_res / 'Figures' / 'Utrecht_I_2022').is_dir(): # output: figures
    (path_res / 'Figures' / 'Utrecht_I_2022').mkdir(parents=True)
path_fig = path_res / 'Figures' / 'Utrecht_I_2022/'


morning_start   = '2022-11-25 12:06:00'
morning_end     = '2022-11-25 14:18:00'

U1_G2301_gps,U1_G4302_gps = read_and_preprocess_G23andG43_U1(path_dataU1, path_processeddata, writexlsx=writexlsx)

# Set the name attribute for each dataframe
U1_G2301_gps.name = 'G2301'
U1_G4302_gps.name = 'G4302'

#%%% Find peaks

 
if writexlsx:
    writer  = pd.ExcelWriter(path_processeddata / "U1_CH4peaks.xlsx", engine = 'xlsxwriter')
else:
    writer = None

U1_G23_peaks = process_peak_data(U1_G2301_gps[morning_start:morning_end], 'G23', height=HEIGHT, distance=DIST_G23, width=WIDTH_G23, 
                                   overviewplot=overviewplot, savepath = path_fig / 'U1_overviewplot_G2301')

U1_G43_peaks = process_peak_data(U1_G4302_gps[morning_start:morning_end], 'G43', height=HEIGHT, distance=DIST_G43, width=WIDTH_G43, 
                                   overviewplot=overviewplot, savepath = path_fig / 'U1_overviewplot_G4302')

   
if writexlsx:
    U1_G23_peaks.to_excel(writer, sheet_name='G2301')
    U1_G43_peaks.to_excel(writer, sheet_name='G4302')
    writer.book.close()
 
    
# figure peaks + timeseries
fig, ax1 = plt.subplots(figsize=(18,10))
ax2 = ax1.twinx()
ax1.plot(U1_G2301_gps.index,U1_G2301_gps['CH4_ele_G23'], alpha=1, label='G2301', linewidth=2, color=dict_color_instr['G2301'])
ax1.plot(U1_G4302_gps.index,U1_G4302_gps['CH4_ele_G43'], alpha=1, label='G4302', linewidth=2, color=dict_color_instr['Aeris'])
ax1.scatter(U1_G23_peaks.index,U1_G23_peaks['CH4_ele_G23'], alpha=1, label='Peaks G2301', color='red')
ax1.scatter(U1_G43_peaks.index,U1_G43_peaks['CH4_ele_G43'], alpha=1, label='Peaks G4302', color='red')
ax2.scatter(U1_G2301_gps.index,U1_G2301_gps['Speed [m/s]'], alpha=.2, label='Speed m/s', color='orange')
ax3 = ax1.twinx() # Create the third y-axis, sharing the same x-axis
ax3.spines["right"].set_position(("outward", 60))  # Offset the third y-axis to avoid overlap
ax3.plot(U1_G2301_gps.index,U1_G2301_gps['Latitude'], alpha=.8, label='Latitude', color='lightgrey')
ax3.set_ylim(52.085,52.12)
ax1.set_xlim(pd.to_datetime('2022-11-25 11:55:00'),pd.to_datetime('2022-11-25 14:25:00'))
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
ax1.legend()
plt.title('Utrecht I')

#%%% Peak Plots

# Define other necessary variables
coord_extent = [5.1633, 5.166, 52.0873, 52.0888]
release_loc1 = (5.1647191, 52.0874256) 
release_loc2 = (5.164405555555556, 52.0885444)
column_names = {'G2301': 'CH4_ele_G23', 'G4302': 'CH4_ele_G43'}

if indiv_peak_plots:
    # create the folder if it does not exist
    if not (path_fig / 'U1_Peakplots').is_dir():
        (path_fig / 'U1_Peakplots').mkdir(parents=True)
path_save = path_fig / 'U1_Peakplots'

# Remove rows with NaN values in column 'latitude'
gps = U1_G4302_gps.dropna(subset=['Latitude']).copy(deep=True)
# Call the function with necessary arguments
plot_indivpeaks_bevorQC(U1_G43_peaks, gps, path_save, coord_extent, release_loc1, release_loc2, indiv_peak_plots, column_names, U1_G2301_gps, U1_G4302_gps)





###############################################################################
#%% Utrecht II
###############################################################################


#%%% Load & Preprocess data

if not (path_res / 'Figures' / 'Utrecht_II_2024').is_dir(): # output: figures
    (path_res / 'Figures' / 'Utrecht_II_2024').mkdir(parents=True)
path_fig = path_res / 'Figures' / 'Utrecht_II_2024/'

U2_G2301_gps,U2_aeris_gps = read_and_preprocess_U2(path_dataU2, path_processeddata,writexlsx=writexlsx)

# Set the name attribute for each dataframe
U2_G2301_gps.name = 'G2301'
U2_aeris_gps.name = 'Aeris'



#%%% Find peaks


if writexlsx:
    writer  = pd.ExcelWriter(path_processeddata /"U2_CH4peaks.xlsx", engine = 'xlsxwriter')
else:
    writer = None


U2_G2301_peaks = process_peak_data(U2_G2301_gps, 'G23', height=HEIGHT, distance=DIST_G23, width=WIDTH_G23,
                                       overviewplot=overviewplot, savepath = path_fig /'U2_overviewplot_G2301')
U2_aeris_peaks = process_peak_data(U2_aeris_gps, 'aeris', height=HEIGHT, distance=DIST_AERIS, width=WIDTH_AERIS,
                                       overviewplot=overviewplot, savepath = path_fig /'U2_overviewplot_Aeris')
   
if writexlsx:
    U2_G2301_peaks.to_excel(writer, sheet_name='G2301')
    U2_aeris_peaks.to_excel(writer, sheet_name='Aeris')
    writer.book.close()
 
    
# 
fig, ax1 = plt.subplots(figsize=(18,10))
ax2 = ax1.twinx()
ax1.plot(U2_G2301_gps.index,U2_G2301_gps['CH4_ele_G23'], alpha=1, label='G2301', linewidth=2, color=dict_color_instr['G2301'])
ax1.plot(U2_aeris_gps.index,U2_aeris_gps['CH4_ele_aeris'], alpha=1, label='Aeris', linewidth=2, color=dict_color_instr['Aeris'])
ax1.scatter(U2_G2301_peaks.index,U2_G2301_peaks['CH4_ele_G23'], alpha=1, label='Peaks G2301', color='red')
ax1.scatter(U2_aeris_peaks.index,U2_aeris_peaks['CH4_ele_aeris'], alpha=1, label='Peaks Aeris', color='red')
ax2.scatter(U2_G2301_gps.index,U2_G2301_gps['Speed [m/s]'], alpha=.2, label='Speed m/s', color='orange')
ax3 = ax1.twinx() # Create the third y-axis, sharing the same x-axis
ax3.spines["right"].set_position(("outward", 60))  # Offset the third y-axis to avoid overlap
ax3.plot(U2_G2301_gps.index,U2_G2301_gps['Latitude'], alpha=.8, label='Latitude', color='lightgrey')
ax3.set_ylim(52.085,52.12)
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
ax1.legend()
plt.title('Utrecht II')


#%%% Peak Plots

# Define other necessary variables
coord_extent = [5.1633, 5.166, 52.0873, 52.0888]
release_loc1 = (5.164652777777778, 52.0874472)
release_loc1_2 = (5.16506388888889, 52.0875333) 
release_loc2 = (5.164452777777778, 52.0885333)
column_names = {'G2301': 'CH4_ele_G23', 'Aeris': 'CH4_ele_aeris'}


# create the folder if it does not exist
if not (path_fig / 'U2_Peakplots').is_dir():
    (path_fig / 'U2_Peakplots').mkdir(parents=True)
path_save = path_fig / 'U2_Peakplots/'
    
# Remove rows with NaN values in column 'latitude'
gps = U2_G2301_gps.dropna(subset=['Latitude']).copy(deep=True)
# Call the function with necessary arguments
plot_indivpeaks_bevorQC(U2_G2301_peaks, gps, path_save, coord_extent, release_loc1_2, release_loc2, indiv_peak_plots, column_names, U2_G2301_gps, U2_aeris_gps)




#%%% Tests 


#%%%% Check CH4 data

# CH4 + speed ------------------------------------------------------------------
fig,ax = plt.subplots()
ax2 = ax.twinx()
ax2.plot(U2_G2301_gps.index,U2_G2301_gps['Speed [m/s]'],alpha=0.2,color='grey', label='Speed')
ax.plot(U2_G2301_gps.index,U2_G2301_gps['CH4_ele_G23'], label='G2301')
#ax.plot(U2_aeris_gps.index,U2_aeris_gps['CH4_ele_aeris'], label='Mira Ultra Aeris')

plt.xlabel('time')
ax.set_ylabel('CH4 elevation')
ax2.set_ylabel('Speed [m/s]')
plt.title('Utrecht II')
plt.legend()

# # CH4 + latitude ------------------------------------------------------------------
# fig,ax = plt.subplots()
# ax2 = ax.twinx()
# ax2.plot(U2_G2301_gps.index,U2_G2301_gps['Latitude'],alpha=0.2,color='grey', label='lat')
# ax.plot(U2_G2301_gps.index,U2_G2301_gps['CH4_ele_G23'], label='G2301')
# #ax.plot(U2_aeris_gps.index,U2_aeris_gps['CH4_ele_aeris'], label='Mira Ultra Aeris')

# ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
# ax2.set_ylim(52.085,52.138)
# plt.xlabel('time')
# ax.set_ylabel('CH4 elevation')
# ax2.set_ylabel('Latitude')
# plt.title('Utrecht II - gps t5+60s')
# plt.legend()


#%%%% P 2Fig: Release rates per loc

# Loc 1 -----------------------------------------------------------------------

timestamps = {
    '4 Lm a': pd.to_datetime('2024-06-11 10:48:00'),
    '0 Lm a': pd.to_datetime('2024-06-11 11:22:00'),
    '2.4 Lm': pd.to_datetime('2024-06-11 11:38:00'),
    '10 Lm': pd.to_datetime('2024-06-11 11:48:00'),
    '80 Lm': pd.to_datetime('2024-06-11 12:18:00'),
    '20 Lm': pd.to_datetime('2024-06-11 12:32:00'),
    'lunch': pd.to_datetime('2024-06-11 13:01:00'),
    '100 Lm': pd.to_datetime('2024-06-11 14:05:00'),
    '40 Lm': pd.to_datetime('2024-06-11 14:21:00'),
    '0 Lm b': pd.to_datetime('2024-06-11 14:23:00'),
    '15 Lm': pd.to_datetime('2024-06-11 14:28:00'),
    '0 Lm c': pd.to_datetime('2024-06-11 15:23:00'),
    '4 Lm b': pd.to_datetime('2024-06-11 16:19:00'),
    '0.15 Lm': pd.to_datetime('2024-06-11 17:07:00'),
    '1 Lm': pd.to_datetime('2024-06-11 17:44:00'),
    'end': pd.to_datetime('2024-06-11 17:56:00')
}

# Define colors for each segment
colors = ['#ffcccc', '#8c8c8c', '#ccccff', '#ffffcc', '#ffccff', '#ccffff', '#8c8c8c', '#ffd6ff', '#ff9f1c' , '#8c8c8c', '#c8b6ff', '#8c8c8c', '#80ffdb', '#bbd0ff', '#e7c6ff' ]

# Create the plot
fig, ax = plt.subplots(figsize=(18,10))
ax2 = ax.twinx()

# Plot the data
ax2.plot(U2_G2301_gps.index, U2_G2301_gps['Speed [m/s]'], alpha=0.2, color='grey', label='Speed')
ax.plot(U2_G2301_gps.index, U2_G2301_gps['CH4_ele_G23'], label='G2301')
ax.plot(U2_aeris_gps.index, U2_aeris_gps['CH4_ele_aeris'], label='Mira Ultra Aeris')

# Set labels and title
plt.xlabel('Time')
ax.set_ylabel('CH4 Elevation')
ax2.set_ylabel('Speed [m/s]')
plt.title('release rates at Loc 1')

# Add background colors for segments
segments = list(timestamps.keys())
for i in range(len(segments) - 1):
    start = timestamps[segments[i]]
    print(start)
    end = timestamps[segments[i + 1]]
    color = colors[i % len(colors)]
    ax.axvspan(start, end, color=color, alpha=0.3, label=segments[i])

# Create custom legend
handles, labels = ax.get_legend_handles_labels()
segment_handles = [plt.Line2D([0], [0], color=color, lw=4, alpha=0.3) for color in colors[:len(segments) - 1]]
segment_labels = segments[:-1]
plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.1, 1))

# Show the plot
plt.tight_layout()
plt.show()
# fig.savefig(path_fig+f'U2_timeseries_rR_loc1.pdf',bbox_inches='tight') 
# fig.savefig(path_fig+f'U2_timeseries_rR_loc1.png',bbox_inches='tight') 


# Loc 2 -----------------------------------------------------------------------
timestamps = {
    '2.4 Lm a'    : pd.to_datetime('2024-06-11 10:50:00'),
    '2.5 Lm'    : pd.to_datetime('2024-06-11 11:10:00'),
    '2.6 Lm'    : pd.to_datetime('2024-06-11 11:29:00'),
    '4 Lm'      : pd.to_datetime('2024-06-11 11:31:00'),
    '0.5 Lm'     : pd.to_datetime('2024-06-11 12:09:00'),
    '0.15 Lm'    : pd.to_datetime('2024-06-11 12:37:00'),
    'lunch'         : pd.to_datetime('2024-06-11 13:01:00'),
    '0.3 Lm'     : pd.to_datetime('2024-06-11 14:02:00'),
    '2 Lm'      : pd.to_datetime('2024-06-11 14:40:00'),
    '1 Lm'      : pd.to_datetime('2024-06-11 15:13:00'),
    '0 Lm'      : pd.to_datetime('2024-06-11 15:52:00'),
    '2.4 Lm b'     : pd.to_datetime('2024-06-11 17:00:00'),
    '60 Lm'     : pd.to_datetime('2024-06-11 17:30:00'),
    '20 Lm'     : pd.to_datetime('2024-06-11 17:38:00'),
    '80 Lm'     : pd.to_datetime('2024-06-11 18:06:00'),
    'end'      : pd.to_datetime('2024-06-11 18:25:00')
    }



# Define colors for each segment
colors = ['#ffcccc', '#f72585', '#ccccff', '#ffffcc', '#ffccff', '#ccffff', '#8c8c8c', '#ffd6ff', '#ff9f1c' , '#c8b6ff', '#8c8c8c', '#00a6fb','#bbd0ff', '#80ffdb',  '#e7c6ff' ]

# Create the plot
fig, ax = plt.subplots(figsize=(18,10))
ax2 = ax.twinx()

# Plot the data
ax2.plot(U2_G2301_gps.index, U2_G2301_gps['Speed [m/s]'], alpha=0.2, color='grey', label='Speed')
ax.plot(U2_G2301_gps.index, U2_G2301_gps['CH4_ele_G23'], label='G2301')
ax.plot(U2_aeris_gps.index, U2_aeris_gps['CH4_ele_aeris'], label='Mira Ultra Aeris')

# Set labels and title
plt.xlabel('Time')
ax.set_ylabel('CH4 Elevation')
ax2.set_ylabel('Speed [m/s]')
plt.title('release rates at Loc 2')

# Add background colors for segments
segments = list(timestamps.keys())
for i in range(len(segments) - 1):
    start = timestamps[segments[i]]
    print(start)
    end = timestamps[segments[i + 1]]
    color = colors[i % len(colors)]
    ax.axvspan(start, end, color=color, alpha=0.3, label=segments[i])

# Create custom legend
handles, labels = ax.get_legend_handles_labels()
segment_handles = [plt.Line2D([0], [0], color=color, lw=4, alpha=0.3) for color in colors[:len(segments) - 1]]
segment_labels = segments[:-1]
plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.1, 1))

# Show the plot
plt.tight_layout()
plt.show()
# fig.savefig(path_fig+f'U2_timeseries_rR_loc2.pdf',bbox_inches='tight') 
# fig.savefig(path_fig+f'U2_timeseries_rR_loc2.png',bbox_inches='tight') 

#%%%% P 1Fig: Release rates per loc


timestamps1 = {
    '4 Lm a': pd.to_datetime('2024-06-11 10:48:00'), #ffcccc
    '0 Lm a': pd.to_datetime('2024-06-11 11:22:00'), #8c8c8c
    '2.4 Lm': pd.to_datetime('2024-06-11 11:38:00'), #ccccff
    '10 Lm': pd.to_datetime('2024-06-11 11:48:00'),  #ffffcc
    '80 Lm': pd.to_datetime('2024-06-11 12:18:00'),  #ffccff
    '20 Lm': pd.to_datetime('2024-06-11 12:32:00'),  #ccffff
    'lunch': pd.to_datetime('2024-06-11 13:01:00'),  #8c8c8c
    '100 Lm': pd.to_datetime('2024-06-11 14:05:00'), #ffd6ff
    '40 Lm': pd.to_datetime('2024-06-11 14:21:00'),  #ff9f1c
    '0 Lm b': pd.to_datetime('2024-06-11 14:23:00'), #8c8c8c
    '15 Lm': pd.to_datetime('2024-06-11 14:28:00'),  #c8b6ff
    '0 Lm c': pd.to_datetime('2024-06-11 15:23:00'), #8c8c8c
    '4 Lm b': pd.to_datetime('2024-06-11 16:19:00'), #ffcccc
    '0.15 Lm': pd.to_datetime('2024-06-11 17:07:00'),#bbd0ff
    '1 Lm': pd.to_datetime('2024-06-11 17:44:00'),   #e7c6ff
    'end': pd.to_datetime('2024-06-11 17:56:00')     #
}

timestamps2 = {
    '2.4 Lm a'    : pd.to_datetime('2024-06-11 10:50:00'),  #ccccff
    '2.5 Lm'    : pd.to_datetime('2024-06-11 11:10:00'),    #f72585
    '2.6 Lm'    : pd.to_datetime('2024-06-11 11:29:00'),    #ffffcc
    '4 Lm'      : pd.to_datetime('2024-06-11 11:31:00'),    #ffcccc
    '0.5 Lm'     : pd.to_datetime('2024-06-11 12:09:00'),   #ffccff
    '0.15 Lm'    : pd.to_datetime('2024-06-11 12:37:00'),   #bbd0ff
    'lunch'         : pd.to_datetime('2024-06-11 13:01:00'),#8c8c8c
    '0.3 Lm'     : pd.to_datetime('2024-06-11 14:02:00'),   #ffccff
    '2 Lm'      : pd.to_datetime('2024-06-11 14:40:00'),    #ffd6ff
    '1 Lm'      : pd.to_datetime('2024-06-11 15:13:00'),    #e7c6ff
    '0 Lm'      : pd.to_datetime('2024-06-11 15:52:00'),    #8c8c8c
    '2.4 Lm b'     : pd.to_datetime('2024-06-11 17:00:00'), #ccccff
    '60 Lm'     : pd.to_datetime('2024-06-11 17:30:00'),    #ffbf69
    '20 Lm'     : pd.to_datetime('2024-06-11 17:38:00'),    #ccffff
    '80 Lm'     : pd.to_datetime('2024-06-11 18:06:00'),    #7400b8
    'end'      : pd.to_datetime('2024-06-11 18:25:00')
    }



# Define colors for each segment
colors1 = ['#ffcccc', '#8c8c8c', '#ccccff', '#ffffcc', '#ffccff', '#ccffff', '#8c8c8c', '#ffd6ff', '#ff9f1c' , '#8c8c8c', '#c8b6ff', '#8c8c8c', '#ffcccc', '#bbd0ff', '#e7c6ff' ]
colors2 = ['#ccccff', '#f72585', '#ffffcc', '#ffcccc', '#ffccff', '#bbd0ff', '#8c8c8c', '#ffccff', '#ffd6ff' , '#e7c6ff', '#8c8c8c', '#ccccff','#ffbf69', '#ccffff',  '#7400b8' ]


# PLOT =======================================================================
# Create the plot
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(18,10))
ax11 = ax1.twinx()
ax22 = ax2.twinx()

# Loc 1 -----------------------------------------------------------------------

# Plot the data
ax11.plot(U2_G2301_gps.index, U2_G2301_gps['Speed [m/s]'], alpha=0.2, color='grey', label='Speed')
ax1.plot(U2_G2301_gps.index, U2_G2301_gps['CH4_ele_G23'], label='G2301')
ax1.plot(U2_aeris_gps.index, U2_aeris_gps['CH4_ele_aeris'], label='Mira Ultra Aeris')

# Set labels and title
ax1.set_xlabel('Time')
ax1.set_ylabel('CH4 Elevation')
ax11.set_ylabel('Speed [m/s]')
plt.title('release rates')

# Add background colors for segments
segments1 = list(timestamps1.keys())
for i in range(len(segments1) - 1):
    start = timestamps1[segments1[i]]
    print(start)
    end = timestamps1[segments1[i + 1]]
    color = colors1[i % len(colors1)]
    ax1.axvspan(start, end, color=color, alpha=0.3, label=segments1[i])

# Create custom legend
handles1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(handles1, labels1, loc='upper left', bbox_to_anchor=(1.1, 1))


# Loc 2 -----------------------------------------------------------------------


# Plot the data
ax22.plot(U2_G2301_gps.index, U2_G2301_gps['Speed [m/s]'], alpha=0.2, color='grey', label='Speed')
ax2.plot(U2_G2301_gps.index, U2_G2301_gps['CH4_ele_G23'], label='G2301')
ax2.plot(U2_aeris_gps.index, U2_aeris_gps['CH4_ele_aeris'], label='Mira Ultra Aeris')

# Set labels and title
ax2.set_xlabel('Time')
ax2.set_ylabel('CH4 Elevation')
ax22.set_ylabel('Speed [m/s]')
#plt.title('release rates at Loc 2')

# Add background colors for segments
segments2 = list(timestamps2.keys())
for i in range(len(segments2) - 1):
    start = timestamps2[segments2[i]]
    print(start)
    end = timestamps2[segments2[i + 1]]
    color = colors2[i % len(colors2)]
    ax2.axvspan(start, end, color=color, alpha=0.3, label=segments2[i])

# Create custom legend
handles2, labels2 = ax2.get_legend_handles_labels()
# segment_handles = [plt.Line2D([0], [0], color=color, lw=4, alpha=0.3) for color in colors[:len(segments) - 1]]
# segment_labels = segments[:-1]
ax2.legend(handles2, labels2, loc='upper left', bbox_to_anchor=(1.1, 1))

# Show the plot
plt.tight_layout()
plt.show()
# fig.savefig(path_fig+f'U2_timeseries_rR_bothlocs.pdf',bbox_inches='tight') 
# fig.savefig(path_fig+f'U2_timeseries_rR_bothlocs.png',bbox_inches='tight') 

#%%%% Plots Check GPS


df_phone1_1 = gpx_to_df(path_dataU2 /'GPS/Phone1/20240611-1057_1200_Phone1_1.gpx')
df_phone1_2 = gpx_to_df(path_dataU2 /'GPS/Phone1/20240611-1201_1304_Phone1_2.gpx')
df_phone1_3 = gpx_to_df(path_dataU2 /'GPS/Phone1/20240611-1352_1426_Phone1_3.gpx')
df_phone1_4 = gpx_to_df(path_dataU2 /'GPS/Phone1/20240611-1426_1558_Phone1_4.gpx')
df_phone1_5 = gpx_to_df(path_dataU2 /'GPS/Phone1/20240611-1620_1816_Phone1_5.gpx')

gps_phone2 = gpx_to_df(path_dataU2 /'GPS/Phone2/20240611-1620_1844_Phone2_final.gpx')

df_phone1_1.index = df_phone1_1.index.tz_convert(None)
df_phone1_2.index = df_phone1_2.index.tz_convert(None)
df_phone1_3.index = df_phone1_3.index.tz_convert(None)
df_phone1_4.index = df_phone1_4.index.tz_convert(None)
df_phone1_5.index = df_phone1_5.index.tz_convert(None)
gps_phone2.index = gps_phone2.index.tz_convert(None)

# Time Periode 1 ----------------------
fig,ax = plt.subplots()
plt.plot(df_phone1_1.index,df_phone1_1['speed'], label='phone 1')
plt.xlabel('time')
plt.ylabel('latitude')
plt.title('Time Periode 1')
plt.legend()

# Time Periode 2 ----------------------
fig,ax = plt.subplots()
plt.scatter(df_phone1_2.index,df_phone1_2['latitude'], label='phone 1')
plt.xlabel('time')
plt.ylabel('latitude')
plt.title('Time Periode 2')
plt.legend()

# Time Periode 3 ----------------------
fig,ax = plt.subplots()
plt.plot(df_phone1_3.index,df_phone1_3['latitude'], label='phone 1')
plt.xlabel('time')
plt.ylabel('latitude')
plt.title('Time Periode 3')
plt.legend()

# Time Periode 4 ----------------------
fig,ax = plt.subplots()
plt.plot(df_phone1_4.index,df_phone1_4['latitude'], label='phone 1')
plt.xlabel('time')
plt.ylabel('latitude')
plt.title('Time Periode 4')
plt.legend()

# Time Periode 5 ----------------------
fig,ax = plt.subplots()
#plt.plot(df_phone1_5.index,df_phone1_5['latitude'], label='phone 1')
plt.scatter(gps_phone2.index,gps_phone2['latitude'], label='phone 2')
plt.xlabel('time')
plt.ylabel('latitude')
plt.title('Time Periode 5')
plt.legend()

fig,ax = plt.subplots()
plt.plot(df_phone1_5.index,df_phone1_5['longitude'], label='phone 1')
plt.plot(gps_phone2.index,gps_phone2['longitude'], label='phone 2')
plt.xlabel('time')
plt.ylabel('longitude')
plt.title('Time Periode 5')
plt.legend()

fig,ax = plt.subplots()
plt.plot(df_phone1_5.index,df_phone1_5['speed'], label='phone 1')
plt.plot(gps_phone2.index,gps_phone2['speed'], label='phone 2')
plt.xlabel('time')
plt.ylabel('speed [m/s]')
plt.title('Time Periode 5')
plt.legend()


#%%% Overview peak locations


U2_G2301_gps,U2_aeris_gps = read_and_preprocess_U2_no_shift(path_dataU2, path_processeddata,writexlsx=True)

# Set the name attribute for each dataframe
U2_G2301_gps.name = 'G2301'

U2_G2301_peaks = process_peak_data_new(U2_G2301_gps, 'G23', height=HEIGHT, distance=DIST_G23, width=WIDTH_G23,
                                       overviewplot=overviewplot, savepath = path_fig /'U2_overviewplot_G2301')


#%%%% G2301


# Remove rows with NaN values in column 'latitude'
gps = U2_G2301_peaks.dropna(subset=['Latitude']).copy(deep=True)


# Plot location of all peaks in one plot
peak_loc_all = []
for index, row in U2_G2301_peaks.iterrows():
    time        = pd.to_datetime(index)#.round('1s')
    
    #if time in gps.index:
    lon         = U2_G2301_gps.loc[time]['Longitude']
    lat         = U2_G2301_gps.loc[time]['Latitude']
    coords      = (lon,lat)
    peak_loc    = coords
    peak_loc_all.append(peak_loc)



tilemapbase.start_logging()
tilemapbase.init(create=True)
t = tilemapbase.tiles.build_OSM()
extent = tilemapbase.Extent.from_lonlat(
    5.1633, 5.166, 52.0873, 52.0888) #5.1633, 5.16587, 52.0873, 52.0888
extent = extent.to_aspect(1.0)
plotter = tilemapbase.Plotter(extent, t, width=600)

fig, ax1 = plt.subplots(figsize = (12,12))
#ax1.xaxis.set_visible(False)
#ax1.yaxis.set_visible(False)
plotter.plot(ax1, t)
# Loop over the list of tuples and unpack latitude and longitude values
for lat, lon in peak_loc_all:
    x, y        = tilemapbase.project(lat, lon)
    #x,y         = tilemapbase.project(peak_loc_all[i])
    x1,y1       = tilemapbase.project(*release_loc1_2)
    x2,y2       = tilemapbase.project(*release_loc2)
    
    ax1.scatter(x,y, marker = "x", color = 'red' ,s = 30, alpha=0.7)
    ax1.scatter(x1,y1, marker = "x", color = 'black' ,s = 30)
    ax1.scatter(x2,y2, marker = "x", color = 'black' ,s = 30)
        
#plt.savefig(path_fig/"U2_Peakplots/Overview_Peak_Locations_G4.jpg")



#%%%% t4 shift


# t_1 = '2024-06-11 10:57:00':'2024-06-11 12:01:00'
# t_2 = '2024-06-11 12:01:00':'2024-06-11 13:04:00'
# t_3 = '2024-06-11 13:52:00':'2024-06-11 14:26:00'
# t_4 = '2024-06-11 14:31:00':'2024-06-11 15:58:00' 1431-1558
# t_5 = '2024-06-11 16:20:00':'2024-06-11 18:44:00'

# t4_1 1431-1440 = '2024-06-11 14:31:00':'2024-06-11 14:40:00'        #before 14:31 car standing,  loc1 15slpm-loc2 0_3slpm
# t4_2 1440-1513 = '2024-06-11 14:40:00':'2024-06-11 15:13:00'        # loc1 15slpm-loc2 2_2slpm 
# t4_3 1513-1523 = '2024-06-11 15:13:00':'2024-06-11 15:23:00         # loc1 15slpm-loc2 1slpm 
# t4_4 1523-1552 '2024-06-11 15:23:00':'2024-06-11 15:52:00'


# copy dfs
U2_G2301_peaks_test = U2_G2301_peaks['2024-06-11 14:31:00':'2024-06-11 15:58:00'].copy()
U2_G2301_peaks_test.index = pd.to_datetime(U2_G2301_peaks_test.index)
U2_G2301_gps_test = U2_G2301_gps.copy()
# shift gps -> optimal 22 = 66 s
shift = 0 #22 
U2_G2301_gps_test['Longitude_shift'] = np.roll(U2_G2301_gps_test.Longitude, shift)
U2_G2301_gps_test['Latitude_shift'] = np.roll(U2_G2301_gps_test.Latitude, shift)
title_descr=f't4 1431-1558 - shift gps by {shift*3} s'


# Prepare your data
peak_loc_all = []
for index, row in U2_G2301_peaks_test.iterrows(): #['2024-06-11 16:20:00':'2024-06-11 18:44:00']
    time = pd.to_datetime(index)
    lon = U2_G2301_gps_test.loc[time]['Longitude_shift']
    lat = U2_G2301_gps_test.loc[time]['Latitude_shift']
    ch4_value = row['CH4_ele_G23']
    peak_loc_all.append((time, lon, lat, ch4_value))

# Convert times to Unix timestamps
times = [x[0].timestamp() for x in peak_loc_all]
norm = mcolors.Normalize(vmin=min(times), vmax=max(times))
cmap = cm.get_cmap('rainbow')

# Create a colormap
colors = [cmap(norm(time)) for time in times]

# Normalize CH4_ele_G23 values for marker sizes
ch4_values = [x[3] for x in peak_loc_all]
size_norm = mcolors.Normalize(vmin=min(ch4_values), vmax=max(ch4_values))
sizes = [size_norm(ch4) * 800 for ch4 in ch4_values]  # Scale size for better visibility

# Initialize tilemapbase
tilemapbase.start_logging()
tilemapbase.init(create=True)
t = tilemapbase.tiles.build_OSM()
extent = tilemapbase.Extent.from_lonlat(5.1633, 5.166, 52.0873, 52.0888).to_aspect(1.0)
plotter = tilemapbase.Plotter(extent, t, width=600)

fig, ax1 = plt.subplots(figsize=(12, 12))
plotter.plot(ax1, t)

# Plot the points with the colormap
for (time, lat, lon, ch4_value), color, size in zip(peak_loc_all, colors, sizes):
    x, y = tilemapbase.project(lat, lon)
    ax1.scatter(x, y, marker="o", color=color,s=size, alpha=0.7, edgecolor='black')

# Add release locations
x1, y1 = tilemapbase.project(*release_loc1_2)
x2, y2 = tilemapbase.project(*release_loc2)
x3,y3       = tilemapbase.project(*release_loc1)
ax1.scatter(x1, y1, marker="x", color='black', s=100) 
ax1.scatter(x2, y2, marker="x", color='black', s=100)
ax1.scatter(x3,y3, marker = "x", color = 'black' ,s = 100)


# Create ScalarMappable for colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for matplotlib < 3.1

# Add the colorbar with custom formatter
cbar = plt.colorbar(sm, ax=ax1)



# Set the formatter for colorbar
cbar.ax.yaxis.set_major_formatter(FuncFormatter(time_ticks))
cbar.set_label('Time (HH:MM)')


plt.title(f'Utrecht II - {title_descr} - G2301',fontsize=18)
plt.savefig(path_fig/ f'Test_GPS_shift/Overview_peaklocations_{title_descr}.pdf',bbox_inches='tight')
# plt.savefig(path_fig/ f'Test Peak Loc/Shift GPS/Overview_peaklocations_{title_descr}.svg',bbox_inches='tight')


#%%%% t3 shift


# t_1 = '2024-06-11 10:57:00':'2024-06-11 12:01:00'
# t_2 = '2024-06-11 12:01:00':'2024-06-11 13:04:00'
# t_3 = '2024-06-11 13:52:00':'2024-06-11 14:26:00'
# t_4 = '2024-06-11 14:31:00':'2024-06-11 15:58:00' 1431-1558
# t_5 = '2024-06-11 16:20:00':'2024-06-11 18:44:00'

# t3 1402-1423
# t3_1 1402-1421 = '2024-06-11 14:02:00':'2024-06-11 14:21:00'      # loc1 100slpm-loc2 0_3slpm
# t3_2 1421-1423 = '2024-06-11 14:21:00':'2024-06-11 14:23:00'      # loc1 40slpm-loc2 0_3slpm

# copy dfs
U2_G2301_peaks_test = U2_G2301_peaks['2024-06-11 14:02:00':'2024-06-11 14:23:00'].copy()
U2_G2301_peaks_test.index = pd.to_datetime(U2_G2301_peaks_test.index)
U2_G2301_gps_test = U2_G2301_gps.copy()
# shift gps -> optimal: 39
shift = 39          
#U2_G2301_peaks_test.index = U2_G2301_peaks_test.index + pd.Timedelta(seconds=shift)
#U2_G2301_gps_test.index = U2_G2301_gps_test.index + pd.Timedelta(seconds=shift)
U2_G2301_gps_test['Longitude_shift'] = np.roll(U2_G2301_gps_test.Longitude, shift)
U2_G2301_gps_test['Latitude_shift'] = np.roll(U2_G2301_gps_test.Latitude, shift)
title_descr=f't3 1402-1423 - shift gps by {shift*3} s'




# Prepare your data
peak_loc_all = []
for index, row in U2_G2301_peaks_test.iterrows(): #['2024-06-11 16:20:00':'2024-06-11 18:44:00']
    time = pd.to_datetime(index)
    lon = U2_G2301_gps_test.loc[time]['Longitude_shift']
    lat = U2_G2301_gps_test.loc[time]['Latitude_shift']
    ch4_value = row['CH4_ele_G23']
    peak_loc_all.append((time, lon, lat, ch4_value))

# Convert times to Unix timestamps
times = [x[0].timestamp() for x in peak_loc_all]
norm = mcolors.Normalize(vmin=min(times), vmax=max(times))
cmap = cm.get_cmap('rainbow')

# Create a colormap
colors = [cmap(norm(time)) for time in times]

# Normalize CH4_ele_G23 values for marker sizes
ch4_values = [x[3] for x in peak_loc_all]
size_norm = mcolors.Normalize(vmin=min(ch4_values), vmax=max(ch4_values))
sizes = [size_norm(ch4) * 800 for ch4 in ch4_values]  # Scale size for better visibility

# Initialize tilemapbase
tilemapbase.start_logging()
tilemapbase.init(create=True)
t = tilemapbase.tiles.build_OSM()
extent = tilemapbase.Extent.from_lonlat(5.1633, 5.166, 52.0873, 52.0888).to_aspect(1.0)
plotter = tilemapbase.Plotter(extent, t, width=600)



fig, ax1 = plt.subplots(figsize=(12, 12))
plotter.plot(ax1, t)
# Plot the points with the colormap
for (time, lat, lon, ch4_value), color, size in zip(peak_loc_all, colors, sizes):
    x, y = tilemapbase.project(lat, lon)
    ax1.scatter(x, y, marker="o", color=color,s=size, alpha=0.7, edgecolor='black')

# Add release locations
x1, y1 = tilemapbase.project(*release_loc1_2)
x2, y2 = tilemapbase.project(*release_loc2)
x3,y3       = tilemapbase.project(*release_loc1)
ax1.scatter(x1, y1, marker="x", color='black', s=100) 
ax1.scatter(x2, y2, marker="x", color='black', s=100)
ax1.scatter(x3,y3, marker = "x", color = 'black' ,s = 100)

# Create ScalarMappable for colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for matplotlib < 3.1

# Add the colorbar with custom formatter
cbar = plt.colorbar(sm, ax=ax1)

# Set the formatter for colorbar
cbar.ax.yaxis.set_major_formatter(FuncFormatter(time_ticks))
cbar.set_label('Time (HH:MM)')

plt.title(f'Utrecht II - {title_descr} - G2301',fontsize=18)
plt.savefig(path_fig/ f'Test_GPS_shift/Overview_peaklocations_{title_descr}.pdf',bbox_inches='tight')
# plt.savefig(path_fig/ f'Test Peak Loc/Shift GPS/Overview_peaklocations_{title_descr}.svg',bbox_inches='tight')



#%%%% t2 shift


# t1 = '2024-06-11 10:57:00':'2024-06-11 12:01:00'
# t2 1201-1304
# t2 = '2024-06-11 12:01:00':'2024-06-11 13:04:00'
# t3 = '2024-06-11 13:52:00':'2024-06-11 14:26:00'
# t4 = '2024-06-11 14:31:00':'2024-06-11 15:58:00' 1431-1558
# t5 = '2024-06-11 16:20:00':'2024-06-11 18:44:00'

# t2_1 1200-1209 = '2024-06-11 12:00:00':'2024-06-11 12:09:00'      # loc1 10slpm-loc2 4slpm
# t2_2 1209-1218 = '2024-06-11 12:09:00':'2024-06-11 12:18:00'      # loc1 10slpm-loc2 0_5slpm
# t2_3 1218-1232 = '2024-06-11 12:18:00':'2024-06-11 12:32:00'      # loc1 80slpm-loc2 0_5slpm
# t2_4 1232-1250 = '2024-06-11 12:32:00':'2024-06-11 12:50:00'      # loc1 20slpm-loc2 0_5and0_15slpm


# copy dfs
U2_G2301_peaks_test = U2_G2301_peaks['2024-06-11 12:09:00':'2024-06-11 12:50:00'].copy()
U2_G2301_peaks_test.index = pd.to_datetime(U2_G2301_peaks_test.index)
U2_G2301_gps_test = U2_G2301_gps.copy()
# shift gps
shift = 20        
#U2_G2301_peaks_test.index = U2_G2301_peaks_test.index + pd.Timedelta(seconds=shift)
#U2_G2301_gps_test.index = U2_G2301_gps_test.index + pd.Timedelta(seconds=shift)
U2_G2301_gps_test['Longitude_shift'] = np.roll(U2_G2301_gps_test.Longitude, shift)
U2_G2301_gps_test['Latitude_shift'] = np.roll(U2_G2301_gps_test.Latitude, shift)
title_descr=f't2 1209-1250 - shift gps by {shift*3} s'


# Prepare your data
peak_loc_all = []
for index, row in U2_G2301_peaks_test.iterrows(): #['2024-06-11 16:20:00':'2024-06-11 18:44:00']
    time = pd.to_datetime(index)
    lon = U2_G2301_gps_test.loc[time]['Longitude_shift']
    lat = U2_G2301_gps_test.loc[time]['Latitude_shift']
    ch4_value = row['CH4_ele_G23']
    peak_loc_all.append((time, lon, lat, ch4_value))

# Convert times to Unix timestamps
times = [x[0].timestamp() for x in peak_loc_all]
norm = mcolors.Normalize(vmin=min(times), vmax=max(times))
cmap = cm.get_cmap('rainbow')

# Create a colormap
colors = [cmap(norm(time)) for time in times]

# Normalize CH4_ele_G23 values for marker sizes
ch4_values = [x[3] for x in peak_loc_all]
size_norm = mcolors.Normalize(vmin=min(ch4_values), vmax=max(ch4_values))
sizes = [size_norm(ch4) * 800 for ch4 in ch4_values]  # Scale size for better visibility

# Initialize tilemapbase
tilemapbase.start_logging()
tilemapbase.init(create=True)
t = tilemapbase.tiles.build_OSM()
extent = tilemapbase.Extent.from_lonlat(5.1633, 5.166, 52.0873, 52.0888).to_aspect(1.0)
plotter = tilemapbase.Plotter(extent, t, width=600)

fig, ax1 = plt.subplots(figsize=(12, 12))
plotter.plot(ax1, t)

# Plot the points with the colormap
for (time, lat, lon, ch4_value), color, size in zip(peak_loc_all, colors, sizes):
    x, y = tilemapbase.project(lat, lon)
    ax1.scatter(x, y, marker="o", color=color,s=size, alpha=0.7, edgecolor='black')

# Add release locations
x1, y1 = tilemapbase.project(*release_loc1_2)
x2, y2 = tilemapbase.project(*release_loc2)
x3,y3       = tilemapbase.project(*release_loc1)
ax1.scatter(x1, y1, marker="x", color='black', s=100) 
ax1.scatter(x2, y2, marker="x", color='black', s=100)
ax1.scatter(x3,y3, marker = "x", color = 'black' ,s = 100)


# Create ScalarMappable for colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for matplotlib < 3.1

# Add the colorbar with custom formatter
cbar = plt.colorbar(sm, ax=ax1)



# Set the formatter for colorbar
cbar.ax.yaxis.set_major_formatter(FuncFormatter(time_ticks))
cbar.set_label('Time (HH:MM)')


plt.title(f'Utrecht II - {title_descr} - G2301',fontsize=18)
plt.savefig(path_fig/ f'Test_GPS_shift/Overview_peaklocations_{title_descr}.pdf',bbox_inches='tight')
# plt.savefig(path_fig/ f'Test Peak Loc/Shift GPS/Overview_peaklocations_{title_descr}.svg',bbox_inches='tight')



#%%%% t1 shift

# t1 1059-1201
# t1 = '2024-06-11 10:57:00':'2024-06-11 12:01:00'
# t2 = '2024-06-11 12:01:00':'2024-06-11 13:04:00' 
# t3 = '2024-06-11 13:52:00':'2024-06-11 14:26:00'
# t4 = '2024-06-11 14:31:00':'2024-06-11 15:58:00' 1431-1558
# t5 = '2024-06-11 16:20:00':'2024-06-11 18:44:00'


# t1_1 1058-1122 = '2024-06-11 10:58:00':'2024-06-11 11:22:00'          #loc1 4slpm-loc2 2_5slpm
# t1_2 1122-1130 = '2024-06-11 11:22:00':'2024-06-11 11:30:00'          # only loc2 2_5slpm
# t1_3 1131-1138 = '2024-06-11 11:31:00':'2024-06-11 11:38:00'          # only loc2 4slpm
# t1_4 1138-1148 = '2024-06-11 11:38:00':'2024-06-11 11:48:00'          # loc1 4slpm-loc2 4slpm
# t1_5 1148-1200 = '2024-06-11 11:48:00':'2024-06-11 12:00:00'          # loc1 10slpm-loc2 4slpm


# copy dfs
U2_G2301_peaks_test = U2_G2301_peaks['2024-06-11 11:22:00':'2024-06-11 11:30:00'].copy()
U2_G2301_peaks_test.index = pd.to_datetime(U2_G2301_peaks_test.index)
U2_G2301_gps_test = U2_G2301_gps.copy()
# shift gps
shift = 20   # change shift here
U2_G2301_gps_test['Longitude_shift'] = np.roll(U2_G2301_gps_test.Longitude, shift)
U2_G2301_gps_test['Latitude_shift'] = np.roll(U2_G2301_gps_test.Latitude, shift)
title_descr=f't1_2 1122-1130 - shift gps by {shift*3} s' # change description to match timeperiode you inserted


# Prepare your data
peak_loc_all = []
for index, row in U2_G2301_peaks_test.iterrows(): #['2024-06-11 16:20:00':'2024-06-11 18:44:00']
    time = pd.to_datetime(index)
    lon = U2_G2301_gps_test.loc[time]['Longitude_shift']
    lat = U2_G2301_gps_test.loc[time]['Latitude_shift']
    ch4_value = row['CH4_ele_G23']
    peak_loc_all.append((time, lon, lat, ch4_value))

# Convert times to Unix timestamps
times = [x[0].timestamp() for x in peak_loc_all]
norm = mcolors.Normalize(vmin=min(times), vmax=max(times))
cmap = cm.get_cmap('rainbow')

# Create a colormap
colors = [cmap(norm(time)) for time in times]

# Normalize CH4_ele_G23 values for marker sizes
ch4_values = [x[3] for x in peak_loc_all]
size_norm = mcolors.Normalize(vmin=min(ch4_values), vmax=max(ch4_values))
sizes = [size_norm(ch4) * 800 for ch4 in ch4_values]  # Scale size for better visibility

# Initialize tilemapbase
tilemapbase.start_logging()
tilemapbase.init(create=True)
t = tilemapbase.tiles.build_OSM()
extent = tilemapbase.Extent.from_lonlat(5.1633, 5.166, 52.0873, 52.0888).to_aspect(1.0)
plotter = tilemapbase.Plotter(extent, t, width=600)

fig, ax1 = plt.subplots(figsize=(12, 12))
plotter.plot(ax1, t)

# Plot the points with the colormap
for (time, lat, lon, ch4_value), color, size in zip(peak_loc_all, colors, sizes):
    x, y = tilemapbase.project(lat, lon)
    ax1.scatter(x, y, marker="o", color=color,s=size, alpha=0.7, edgecolor='black')

# Add release locations
x1, y1 = tilemapbase.project(*release_loc1_2)
x2, y2 = tilemapbase.project(*release_loc2)
x3,y3       = tilemapbase.project(*release_loc1)
ax1.scatter(x1, y1, marker="x", color='black', s=100) 
ax1.scatter(x2, y2, marker="x", color='black', s=100)
ax1.scatter(x3,y3, marker = "x", color = 'black' ,s = 100)


# Create ScalarMappable for colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for matplotlib < 3.1

# Add the colorbar with custom formatter
cbar = plt.colorbar(sm, ax=ax1)



# Set the formatter for colorbar
cbar.ax.yaxis.set_major_formatter(FuncFormatter(time_ticks))
cbar.set_label('Time (HH:MM)')


plt.title(f'Utrecht II - {title_descr} - G2301',fontsize=18)
plt.savefig(path_fig/ f'Test_GPS_shift/Overview_peaklocations_{title_descr}.pdf',bbox_inches='tight')
# plt.savefig(path_fig/ f'Test_GPS_shift/Overview_peaklocations_{title_descr}.svg',bbox_inches='tight')


#%%%% color scaled, G2301


# t_1 = '2024-06-11 10:57:00':'2024-06-11 12:01:00'
# t_2 = '2024-06-11 12:01:00':'2024-06-11 13:04:00'
# t_3 = '2024-06-11 13:52:00':'2024-06-11 14:26:00'
# t_4 = '2024-06-11 14:26:00':'2024-06-11 15:58:00'
# t_5 = '2024-06-11 16:20:00':'2024-06-11 18:44:00'
# t_100slm = '2024-06-11 14:05:00':'2024-06-11 14:21:00'
# t_15slm = '2024-06-11 14:25:00':'2024-06-11 15:23:00'
# t_loc1end = '2024-06-11 17:56:00':'2024-06-11 18:25:00'
  
 
title_descr = 'all'
title_descr='t5-phone2-gpsplus60s'
title_descr='loc1 0slpm-loc2 80slpm t5 1806-1825'


# t5 = '2024-06-11 16:20:00':'2024-06-11 18:44:00'
# only loc 1 4slpm = '2024-06-11 16:19:00':'2024-06-11 16:44:00'
# loc1 4slpm-loc2 4slpm 1644-1706 = '2024-06-11 16:44:00':'2024-06-11 17:06:00'
# loc 1 4slpm but change loc-loc2 2_5slpm = '2024-06-11 10:58:00':'2024-06-11 11:29:00'

# loc1 4slpm-loc2 2_5slpm t1 1122 = '2024-06-11 10:58:00':'2024-06-11 11:22:00'
# loc1 4slpm-loc2 2_5slpm t1 1115 = '2024-06-11 10:58:00':'2024-06-11 11:15:00'
# only loc2 2_5slpm t1 1122-1130 = '2024-06-11 11:22:00':'2024-06-11 11:30:00'
# only loc2 4slpm t1 1131-1138 = '2024-06-11 11:31:00':'2024-06-11 11:38:00'
# loc1 4and10slpm-loc2 4slpm t1 1138-1209 = '2024-06-11 11:38:00':'2024-06-11 12:09:00'
# loc1 10slpm-loc2 4slpm t1 1148-1209 = '2024-06-11 11:48:00':'2024-06-11 12:09:00'
# loc1 10slpm-loc2 4slpm t1 1148-1200 = '2024-06-11 11:48:00':'2024-06-11 12:00:00'
# loc1 10slpm-loc2 4slpm t2 1200-1209 = '2024-06-11 12:00:00':'2024-06-11 12:09:00'
# loc1 10slpm-loc2 0_5slpm t2 1209-1218 = '2024-06-11 12:09:00':'2024-06-11 12:18:00'
# loc1 80slpm-loc2 0_5slpm t2 1218-1232 = '2024-06-11 12:18:00':'2024-06-11 12:32:00'
# loc1 20slpm-loc2 0_5and0_15slpm t2 1232-1250 = '2024-06-11 12:32:00':'2024-06-11 12:50:00'
# loc1 100slpm-loc2 0_3slpm t3 1402-1421 = '2024-06-11 14:02:00':'2024-06-11 14:05:00'
# loc1 40slpm-loc2 0_3slpm t3 1421-1423 = '2024-06-11 14:21:00':'2024-06-11 14:23:00'
# loc1 15slpm-loc2 0_3slpm t4 1431-1440 = '2024-06-11 14:31:00':'2024-06-11 14:40:00' #before 14:31 car standing
# loc1 15slpm-loc2 2_2slpm t4 1440-1513 = '2024-06-11 14:40:00':'2024-06-11 15:13:00'
# loc1 15slpm-loc2 1slpm t4 1513-1523 = '2024-06-11 15:13:00':'2024-06-11 15:23:00'
# loc1 0slpm-loc2 1slpm t4 1523-1552 = '2024-06-11 15:23:00':'2024-06-11 15:52:00'
# loc1 4slpm-loc2 0slpm t5 1619-1644 = '2024-06-11 16:19:00':'2024-06-11 16:44:00'
# loc1 4slpm-loc2 4slpm t5 1644-1706 = '2024-06-11 16:44:00':'2024-06-11 17:06:00'
# loc1 0_15slpm-loc2 4slpm t5 1706-1730 = '2024-06-11 17:06:00':'2024-06-11 17:30:00'
# loc1 0_15slpm-loc2 20slpm t5 1738-1744 = '2024-06-11 17:38:00':'2024-06-11 17:44:00'
# loc1 1slpm-loc2 20slpm t5 1744-1756 = '2024-06-11 17:44:00':'2024-06-11 17:56:00'
# loc1 0slpm-loc2 20slpm t5 1756-1806 = '2024-06-11 17:56:00':'2024-06-11 18:06:00'
# loc1 0slpm-loc2 80slpm t5 1806-1825 = '2024-06-11 18:06:00':'2024-06-11 18:25:00'


# Prepare your data
peak_loc_all = []
for index, row in U2_G2301_peaks['2024-06-11 18:06:00':'2024-06-11 18:25:00'].iterrows(): #['2024-06-11 16:20:00':'2024-06-11 18:44:00']
    time = pd.to_datetime(index)
    lon = U2_G2301_gps.loc[time]['Longitude']
    lat = U2_G2301_gps.loc[time]['Latitude']
    ch4_value = row['CH4_ele_G23']
    peak_loc_all.append((time, lon, lat, ch4_value))

# Convert times to Unix timestamps
times = [x[0].timestamp() for x in peak_loc_all]
norm = mcolors.Normalize(vmin=min(times), vmax=max(times))
cmap = cm.get_cmap('rainbow')

# Create a colormap
colors = [cmap(norm(time)) for time in times]

# Normalize CH4_ele_G23 values for marker sizes
ch4_values = [x[3] for x in peak_loc_all]
size_norm = mcolors.Normalize(vmin=min(ch4_values), vmax=max(ch4_values))
sizes = [size_norm(ch4) * 800 for ch4 in ch4_values]  # Scale size for better visibility

# Initialize tilemapbase
tilemapbase.start_logging()
tilemapbase.init(create=True)
t = tilemapbase.tiles.build_OSM()
extent = tilemapbase.Extent.from_lonlat(5.1633, 5.166, 52.0873, 52.0888).to_aspect(1.0)
plotter = tilemapbase.Plotter(extent, t, width=600)

fig, ax1 = plt.subplots(figsize=(12, 12))
plotter.plot(ax1, t)

# Plot the points with the colormap
for (time, lat, lon, ch4_value), color, size in zip(peak_loc_all, colors, sizes):
    x, y = tilemapbase.project(lat, lon)
    ax1.scatter(x, y, marker="o", color=color,s=size, alpha=0.7, edgecolor='black')

# Add release locations
x1, y1 = tilemapbase.project(*release_loc1_2)
x2, y2 = tilemapbase.project(*release_loc2)
x3,y3       = tilemapbase.project(*release_loc1)
ax1.scatter(x1, y1, marker="x", color='black', s=60) 
ax1.scatter(x2, y2, marker="x", color='black', s=60)
ax1.scatter(x3,y3, marker = "x", color = 'black' ,s = 60)


# Create ScalarMappable for colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for matplotlib < 3.1

# Add the colorbar with custom formatter
cbar = plt.colorbar(sm, ax=ax1)



# Set the formatter for colorbar
cbar.ax.yaxis.set_major_formatter(FuncFormatter(time_ticks))
cbar.set_label('Time (HH:MM)')


plt.title(f'Utrecht II - {title_descr}')
# plt.savefig(path_fig+f'Test Peak Loc/Overview_peaklocations_{title_descr}_G2301.pdf',bbox_inches='tight')
# plt.savefig(path_fig+f'Test Peak Loc/Overview_peaklocations_{title_descr}_G2301.svg',bbox_inches='tight')





###############################################################################
#%% Toronto
###############################################################################


#%%% Load & Preprocess data

if not (path_res / 'Figures' / 'Toronto_2021').is_dir():  # output: figures
    (path_res / 'Figures' / 'Toronto_2021').mkdir(parents=True)
path_fig = path_res / 'Figures' / 'Toronto_2021'

# Release Coordinates Toronto
release_loc1 = (-79.325254, 43.655007)
release_loc2 = (-79.46952, 43.782970)

T_LGR_1b, T_G2401_1c, T_G2401_2c = read_and_preprocess_BikeandCar_T(path_dataT,path_processeddata,writexlsx=writexlsx)

# Set the name attribute for each dataframe
T_LGR_1b.name   = 'LGR'
T_G2401_1c.name = 'G2401'
T_G2401_2c.name = 'G2401'

#%%% Find peaks


''' --- Day 1 - Bike --- '''
 
if writexlsx:
    writer  = pd.ExcelWriter(path_processeddata / "T_CH4peaks_1b.xlsx", engine = 'xlsxwriter')
else:
    writer = None

T_1b_peaks = process_peak_data(T_LGR_1b, 'LGR', height=HEIGHT, distance=DIST_LGR, width=WIDTH_LGR, 
                             overviewplot=overviewplot, savepath = path_fig /'T1b_overviewplot')
 
if writexlsx: 
    T_1b_peaks.to_excel(writer, sheet_name='Day1-Bike')
    writer.book.close() # Save the Excel file
    
''' --- Day 1 - Car --- '''
    
if writexlsx:
    writer  = pd.ExcelWriter(path_processeddata /"T_CH4peaks_1c.xlsx", engine = 'xlsxwriter')

T_1c_peaks = process_peak_data(T_G2401_1c, 'G24', height=HEIGHT, distance=DIST_G24, width=WIDTH_G24, 
                             overviewplot=overviewplot, savepath = path_fig /'T1c_overviewplot')

 
if writexlsx: 
    T_1c_peaks.to_excel(writer, sheet_name='Day1-Car')
    writer.book.close() # Save the Excel file
    
''' --- Day 2 - Car --- '''
    

if writexlsx:
    writer  = pd.ExcelWriter(path_processeddata /"T_CH4peaks_2c.xlsx", engine = 'xlsxwriter')

T_2c_peaks = process_peak_data(T_G2401_2c, 'G24', height=HEIGHT, distance=DIST_G24, width=WIDTH_G24, 
                             overviewplot=overviewplot, savepath = path_fig /'T2c_overviewplot')

if writexlsx: 
    T_2c_peaks.to_excel(writer, sheet_name='Day2-Car')
    writer.book.close() # Save the Excel file
    
    
#%%% Peak Plots

# Define other necessary variables
coord_extent_1 = [T_LGR_1b['Longitude'].min()-0.002,T_LGR_1b['Longitude'].max()+0.002, T_LGR_1b['Latitude'].min()-0.002, T_LGR_1b['Latitude'].max()+0.002]
coord_extent_2 = [T_G2401_2c['Longitude'].min()-0.002,T_G2401_2c['Longitude'].max()+0.002, T_G2401_2c['Latitude'].min()-0.002, T_G2401_2c['Latitude'].max()+0.002]
column_names_1 = {'LGR': 'CH4_ele_LGR'}
column_names_2 = {'G2401': 'CH4_ele_G24'}
release_loc1 = (-79.325254, 43.655007)
release_loc2 = (-79.46952, 43.782970)



''' --- Day 1 - Bike --- '''


# create the folder if it does not exist
if not (path_fig / 'T_Peakplots' / 'Day1_Bike').is_dir():
    (path_fig / 'T_Peakplots' / 'Day1_Bike').mkdir(parents=True)
path_save = path_fig / 'T_Peakplots' / 'Day1_Bike'

# Remove rows with NaN values in column 'latitude'
gps = T_LGR_1b.dropna(subset=['Latitude']).copy(deep=True)
# Call the plot function to plot individual peaks
plot_indivpeaks_bevorQC(T_1b_peaks,gps, path_save, coord_extent_1, release_loc1, None, indiv_peak_plots, column_names_1, T_LGR_1b)


''' --- Day 1 - Car --- '''


# create the folder if it does not exist
if not (path_fig / 'T_Peakplots' / 'Day1_Car').is_dir():
    (path_fig / 'T_Peakplots' / 'Day1_Car').mkdir(parents=True)
path_save = path_fig / 'T_Peakplots' / 'Day1_Car'

# Remove rows with NaN values in column 'latitude'
gps = T_G2401_1c.dropna(subset=['Latitude']).copy(deep=True)
# Call the plot function to plot individual peaks
plot_indivpeaks_bevorQC(T_1c_peaks,gps, path_save, coord_extent_1, release_loc1, None, indiv_peak_plots, column_names_2, T_G2401_1c)


''' --- Day 2 - Car --- '''


# create the folder if it does not exist
if not (path_fig / 'T_Peakplots' / 'Day1_Bike').is_dir():
    (path_fig / 'T_Peakplots' / 'Day1_Bike').mkdir(parents=True)
path_save = path_fig / 'T_Peakplots' / 'Day2_Car'

# Remove rows with NaN values in column 'latitude'
gps = T_G2401_2c.dropna(subset=['Latitude']).copy(deep=True)
# Call the plot function to plot individual peaks
plot_indivpeaks_bevorQC(T_2c_peaks,gps, path_save, coord_extent_2, release_loc2, None, indiv_peak_plots, column_names_2, T_G2401_2c)


#%%% Timeseries


fig, ax1 = plt.subplots(figsize=(18,10))
ax2 = ax1.twinx()
#ax1.plot(L2_Licor_d1.index,L2_Licor_d1['CH4_Licor'], alpha=.8, label='Licor')
ax1.plot(T_LGR_1b.index,T_LGR_1b['CH4_ele_LGR'], alpha=1, label='LGR', linewidth=2, color='firebrick')
ax1.scatter(T_1b_peaks.index,T_1b_peaks['CH4_ele_LGR'], alpha=1, label='Peaks LGR', color='red')
#ax2.plot(T_LGR_1b.index,T_LGR_1b['Heading'], alpha=.8, label='Heading', color='lightgrey')
ax2.scatter(T_LGR_1b.index,T_LGR_1b['Speed [m/s]'], alpha=.6, label='Speed m/s', color='orange')
ax3 = ax1.twinx() # Create the third y-axis, sharing the same x-axis
ax3.spines["right"].set_position(("outward", 60))  # Offset the third y-axis to avoid overlap
ax3.plot(T_LGR_1b.index + pd.Timedelta(seconds=30),T_LGR_1b['Heading'], alpha=.8, label='Heading', color='lightgrey')
ax3.set_ylim(0,1300)
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
plt.legend()
ax1.set_xlabel('Time')
ax1.set_ylabel('CH4 elevation [ppm]')
ax2.set_ylabel('Speed [m/s]')
ax3.set_ylabel('Heading [degree]')
plt.title('Toronto Day1 bike')



fig, ax1 = plt.subplots(figsize=(18,10))
ax2 = ax1.twinx()
#ax1.plot(L2_Licor_d1.index,L2_Licor_d1['CH4_Licor'], alpha=.8, label='Licor')
ax1.plot(T_G2401_1c.index,T_G2401_1c['CH4_ele_G24'], alpha=1, label='G2401', linewidth=2, color='deeppink')
ax1.scatter(T_1c_peaks.index,T_1c_peaks['CH4_ele_G24'], alpha=1, label='Peaks G2401', color='black')
ax1.scatter(T_G2401_1c.index,T_G2401_1c['Speed [m/s]'], alpha=.6, label='Speed m/s', color='orange')
ax3 = ax1.twinx() # Create the third y-axis, sharing the same x-axis
ax3.spines["right"].set_position(("outward", 60))  # Offset the third y-axis to avoid overlap
ax3.plot(T_G2401_1c.index,T_G2401_1c['Heading'], alpha=.8, label='Heading', color='lightgrey')
ax3.set_ylim(0,1300)
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
plt.legend()
ax1.set_xlabel('Time')
ax1.set_ylabel('CH4 elevation [ppm]')
ax2.set_ylabel('Speed [m/s]')
ax3.set_ylabel('Heading [degree]')
plt.title('Toronto Day1 bike')


fig, ax1 = plt.subplots(figsize=(18,10))
ax2 = ax1.twinx()
#ax1.plot(L2_Licor_d1.index,L2_Licor_d1['CH4_Licor'], alpha=.8, label='Licor')
ax1.plot(T_G2401_2c.index,T_G2401_2c['CH4_ele_G24'], alpha=1, label='G2401', linewidth=2, color='deeppink')
ax1.scatter(T_2c_peaks.index,T_2c_peaks['CH4_ele_G24'], alpha=1, label='Peaks G2401', color='red')
ax1.scatter(T_G2401_2c.index,T_G2401_2c['Speed [m/s]'], alpha=.6, label='Speed m/s', color='orange')
ax3 = ax1.twinx() # Create the third y-axis, sharing the same x-axis
ax3.spines["right"].set_position(("outward", 60))  # Offset the third y-axis to avoid overlap
ax3.plot(T_G2401_2c.index,T_G2401_2c['Heading'], alpha=.8, label='Heading', color='lightgrey')
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
plt.legend()
ax1.set_xlabel('Time')
ax1.set_ylabel('CH4 elevation [ppm]')
ax2.set_ylabel('Speed [m/s]')
ax3.set_ylabel('Heading [degree]')
plt.title('Toronto Day2')





###############################################################################
#%% London I
###############################################################################


"""
A controlled release of methane and ethane was performed across 5 days from the 
9-13/09/2019, of which 1 day was dedicated to the setup and testing and 4 days 
to measurements. Therefore, only the days 2-5 are used in the following. In the
publication the first measurement day was named Day 1 to avoid confusion 
(Day 1 in the publication = Day 2 in the code).

    """


#%%% Load & Preprocess data


if not (path_res / 'Figures' / 'London_I_2019').is_dir(): # output: figures
    (path_res / 'Figures' / 'London_I_2019').mkdir(parents=True)
path_fig = path_res / 'Figures' / 'London_I_2019/'

# Release Coordinates London
release_loc1 = (-0.437888,52.233343)

L1_LGR_d2, L1_G2301_d2, L1_LGR_d3, L1_G2301_d3, L1_Licor_d3, L1_LGR_d4, L1_G2301_d4, L1_Licor_d4, L1_G2301_d5 = read_and_preprocess_L1(path_dataL1,path_processeddata ,writexlsx=writexlsx)

# Set the name attribute for each dataframe
L1_LGR_d2.name = 'LGR'
L1_LGR_d3.name = 'LGR'
L1_LGR_d4.name = 'LGR'
L1_Licor_d3.name = 'Licor'
L1_Licor_d4.name = 'Licor'
L1_G2301_d2.name = 'G2301'
L1_G2301_d3.name = 'G2301'
L1_G2301_d4.name = 'G2301'
L1_G2301_d5.name = 'G2301'

#%%% Find peaks



''' --- Day 2 --- '''
 
if writexlsx:
    writer  = pd.ExcelWriter(path_processeddata /"L1_CH4peaks_day2.xlsx", engine = 'xlsxwriter')
else:
    writer = None

L1_day2_LGR_peaks = process_peak_data(L1_LGR_d2, 'LGR', distance=DIST_LGR, width=WIDTH_LGR,
                                          overviewplot=overviewplot, savepath = path_fig / 'L1_overviewplot_LGR')
    
L1_day2_G23_peaks = process_peak_data(L1_G2301_d2, 'G23', distance=DIST_G23, width=WIDTH_G23,
                                          overviewplot=overviewplot, savepath = path_fig / 'L1_overviewplot_G2301')
if writexlsx:
    L1_day2_LGR_peaks.to_excel(writer, sheet_name='LGR')
    L1_day2_G23_peaks.to_excel(writer, sheet_name='G2301')
    writer.book.close()
    
    
''' --- Day 3 --- '''

if writexlsx:
    writer  = pd.ExcelWriter(path_processeddata /"L1_CH4peaks_day3.xlsx", engine = 'xlsxwriter')
else:
    writer = None

L1_day3_Licor_peaks = process_peak_data(L1_Licor_d3, 'Licor', distance=DIST_LICOR, width=WIDTH_LICOR, 
                                            overviewplot=overviewplot, savepath = path_fig / 'L1_overviewplot_Licor')
    
L1_day3_G23_peaks = process_peak_data(L1_G2301_d3, 'G23', distance=DIST_G23, width=WIDTH_G23, 
                                          overviewplot=overviewplot, savepath = path_fig / 'L1_overviewplot_G2301')

if writexlsx:
    L1_day3_Licor_peaks.to_excel(writer, sheet_name='Licor')
    L1_day3_G23_peaks.to_excel(writer, sheet_name='G2301')
    writer.book.close()
    
    
''' --- Day 4 --- '''

if writexlsx:
    writer  = pd.ExcelWriter(path_processeddata /"L1_CH4peaks_day4.xlsx", engine = 'xlsxwriter')
else:
    writer = None

L1_day4_Licor_peaks = process_peak_data(L1_Licor_d4, 'Licor', distance=DIST_LICOR, width=WIDTH_LICOR, 
                                            overviewplot=overviewplot, savepath = path_fig / 'L1_overviewplot_Licor')
    
L1_day4_G23_peaks = process_peak_data(L1_G2301_d4, 'G23', distance=DIST_G23, width=WIDTH_G23, 
                                          overviewplot=overviewplot, savepath = path_fig / 'L1_overviewplot_G2301')
if writexlsx:
    L1_day4_Licor_peaks.to_excel(writer, sheet_name='Licor')
    L1_day4_G23_peaks.to_excel(writer, sheet_name='G2301')
    writer.book.close()
    
    
''' --- Day 5 --- '''
 
if writexlsx:
    writer  = pd.ExcelWriter(path_processeddata /"L1_CH4peaks_day5.xlsx", engine = 'xlsxwriter')
else:
    writer = None

L1_day5_G23_peaks = process_peak_data(L1_G2301_d5, 'G23', distance=DIST_G23, width=WIDTH_G23, 
                                          overviewplot=overviewplot, savepath = path_fig / 'L1_overviewplot_G2301')
if writexlsx:
    L1_day5_G23_peaks.to_excel(writer, sheet_name='G2301')
    writer.book.close()
    

#%%% Peak Plots

# Define other necessary variables
release_loc1 = (-0.437888,52.233343)
coord_extent_2 = [L1_G2301_d2['Longitude'].min()-0.007,L1_G2301_d2['Longitude'].max()+0.007, L1_G2301_d2['Latitude'].min()-0.007, L1_G2301_d2['Latitude'].max()+0.007] #r_loc: 43.782970N, (-)79.46952W 
coord_extent_3 = [L1_G2301_d3['Longitude'].min()-0.007,L1_G2301_d3['Longitude'].max()+0.007, L1_G2301_d3['Latitude'].min()-0.007, L1_G2301_d3['Latitude'].max()+0.007] #r_loc: 43.782970N, (-)79.46952W 
coord_extent_4 = [L1_G2301_d4['Longitude'].min()-0.007,L1_G2301_d4['Longitude'].max()+0.007, L1_G2301_d4['Latitude'].min()-0.007, L1_G2301_d4['Latitude'].max()+0.007] #r_loc: 43.782970N, (-)79.46952W 
coord_extent_5 = [L1_G2301_d5['Longitude'].min()-0.007,L1_G2301_d5['Longitude'].max()+0.007, L1_G2301_d5['Latitude'].min()-0.007, L1_G2301_d5['Latitude'].max()+0.007] #r_loc: 43.782970N, (-)79.46952W 
column_names_2 = {'LGR': 'CH4_ele_LGR', 'G2301': 'CH4_ele_G23'}
column_names_3 = {'Licor': 'CH4_ele_Licor', 'G2301': 'CH4_ele_G23'}
column_names_5 = {'G2301': 'CH4_ele_G23'}


''' --- Day 2 --- '''

# create the folder if it does not exist
if not (path_fig / 'L1_Peakplots' / 'Day2').is_dir():
    (path_fig / 'L1_Peakplots' / 'Day2').mkdir(parents=True)
path_save = path_fig / 'L1_Peakplots' / 'Day2'

# Remove rows with NaN values in column 'latitude'
gps = L1_LGR_d2.dropna(subset=['Latitude']).copy(deep=True)
# Call the plot function to plot individual peaks
plot_indivpeaks_bevorQC(L1_day2_LGR_peaks, gps, path_save, coord_extent_2, release_loc1, None, indiv_peak_plots, column_names_2, L1_LGR_d2, L1_G2301_d2)


''' --- Day 3 --- '''

# create the folder if it does not exist
if not (path_fig / 'L1_Peakplots' / 'Day3').is_dir():
    (path_fig / 'L1_Peakplots' / 'Day3').mkdir(parents=True)
path_save = path_fig / 'L1_Peakplots' / 'Day3'
    
# Remove rows with NaN values in column 'latitude'
gps = L1_G2301_d3.dropna(subset=['Latitude']).copy(deep=True)
# Call the plot function to plot individual peaks
plot_indivpeaks_bevorQC(L1_day3_G23_peaks, gps, path_save, coord_extent_3, release_loc1, None, indiv_peak_plots, column_names_3, L1_Licor_d3, L1_G2301_d3)


''' --- Day 4 --- '''


# create the folder if it does not exist
if not (path_fig / 'L1_Peakplots' / 'Day4').is_dir():
    (path_fig / 'L1_Peakplots' / 'Day4').mkdir(parents=True)
path_save = path_fig / 'L1_Peakplots' / 'Day4'
    
# Remove rows with NaN values in column 'latitude'
gps = L1_G2301_d4.dropna(subset=['Latitude']).copy(deep=True)
# Call the plot function to plot individual peaks
plot_indivpeaks_bevorQC(L1_day4_G23_peaks, gps, path_save, coord_extent_4, release_loc1, None, indiv_peak_plots, column_names_3, L1_Licor_d4, L1_G2301_d4)


''' --- Day 5 --- '''


# create the folder if it does not exist
if not (path_fig / 'L1_Peakplots' / 'Day5').is_dir():
    (path_fig / 'L1_Peakplots' / 'Day5').mkdir(parents=True)
path_save = path_fig / 'L1_Peakplots' / 'Day5'

# Remove rows with NaN values in column 'latitude'
gps = L1_G2301_d5.dropna(subset=['Latitude']).copy(deep=True)
# Call the plot function to plot individual peaks
plot_indivpeaks_bevorQC(L1_day5_G23_peaks, gps, path_save, coord_extent_5, release_loc1, None, indiv_peak_plots, column_names_5, L1_G2301_d5)





###############################################################################
#%% London II
###############################################################################


#%%% Load & Preprocess data

if not (path_res / 'Figures' / 'London_II_2024').is_dir(): # output: figures
    (path_res / 'Figures' / 'London_II_2024').mkdir(parents=True)
path_fig = path_res / 'Figures' / 'London_II_2024/'



L2_Licor_d1, L2_Licor_d2 = read_and_preprocess_L2(path_dataL2, path_processeddata, writexlsx=writexlsx)

# Set the name attribute for each dataframe
L2_Licor_d1.name = 'Licor'
L2_Licor_d2.name = 'Licor'




#%%% Find Peaks

''' --- Day 1 --- '''

if writexlsx:
    writer  = pd.ExcelWriter(path_processeddata /"L2_CH4peaks_day1.xlsx", engine = 'xlsxwriter')
else:
    writer = None

L2_day1_Licor_peaks = process_peak_data(L2_Licor_d1['2024-05-13 13:17:00':], 'Licor', height=HEIGHT, distance=DIST_LICOR, width=WIDTH_LICOR,
                                            overviewplot=overviewplot, savepath = path_fig /'L2_Day1_overviewplot')
     
if writexlsx:
    L2_day1_Licor_peaks.to_excel(writer, sheet_name='Licor')
    writer.book.close()
    
    
''' --- Day 2 --- '''

if writexlsx:
    writer  = pd.ExcelWriter(path_processeddata /"L2_CH4peaks_day2.xlsx", engine = 'xlsxwriter')
else:
    writer = None

L2_day2_Licor_peaks = process_peak_data(L2_Licor_d2, 'Licor', height=HEIGHT, distance=DIST_LICOR, width=WIDTH_LICOR,
                                            overviewplot=overviewplot, savepath = path_fig /'L2_Day2_overviewplot')
 
if writexlsx:
    L2_day2_Licor_peaks.to_excel(writer, sheet_name='Licor')
    writer.book.close()


#%%% Peak Plots

# Define other necessary variables
release_loc1 = (-0.44161,52.23438) 
coord_extent_1 = [L2_Licor_d1['Longitude'].min()-0.007,L2_Licor_d1['Longitude'].max()+0.007, L2_Licor_d1['Latitude'].min()-0.007, L2_Licor_d1['Latitude'].max()+0.007] #r_loc: 43.782970N, (-)79.46952W 
coord_extent_2 = [L2_Licor_d2['Longitude'].min()-0.007,L2_Licor_d2['Longitude'].max()+0.007, L2_Licor_d2['Latitude'].min()-0.007, L2_Licor_d2['Latitude'].max()+0.007] #r_loc: 43.782970N, (-)79.46952W 
column_names_1 = {'Licor': 'CH4_ele_Licor'}
L2_Licor_d1.name = 'Licor'


''' --- Day 1 --- '''

if not (path_fig / 'L2_Peakplots' / 'Day1').is_dir():
    (path_fig / 'L2_Peakplots' / 'Day1').mkdir(parents=True)
path_save = path_fig / 'L2_Peakplots' / 'Day1'

# Remove rows with NaN values in column 'latitude'
gps = L2_Licor_d1.dropna(subset=['Latitude']).copy(deep=True)
# Call the plot function to plot individual peaks
plot_indivpeaks_bevorQC(L2_day1_Licor_peaks, gps, path_save, coord_extent_1, release_loc1, None, indiv_peak_plots, column_names_1, L2_Licor_d1)


''' --- Day 2 --- ''' 

if not (path_fig / 'L2_Peakplots' / 'Day2').is_dir():
    (path_fig / 'L2_Peakplots' / 'Day2').mkdir(parents=True)
path_save = path_fig / 'L2_Peakplots' / 'Day2'

# Remove rows with NaN values in column 'latitude'
gps = L2_Licor_d2.dropna(subset=['Latitude']).copy(deep=True)
# Call the plot function to plot individual peaks
plot_indivpeaks_bevorQC(L2_day2_Licor_peaks, gps, path_save, coord_extent_2, release_loc1, None, indiv_peak_plots, column_names_1, L2_Licor_d2)



#%%% Plot timeseries


fig, ax1 = plt.subplots(figsize=(18,10))
ax2 = ax1.twinx()
#ax1.plot(L2_Licor_d1.index,L2_Licor_d1['CH4_Licor'], alpha=.8, label='Licor')
ax1.plot(L2_Licor_d1.index,L2_Licor_d1['CH4_ele_Licor'], alpha=1, label='Licor', linewidth=2, color='rebeccapurple')
ax1.scatter(L2_day1_Licor_peaks.index,L2_day1_Licor_peaks['CH4_ele_Licor'], alpha=1, label='Licor', color='red')
ax1.scatter(L2_Licor_d1.index,L2_Licor_d1['Speed [m/s]'], alpha=.6, label='Speed m/s', color='orange')
ax2.plot(L2_Licor_d1.index,L2_Licor_d1['Heading [deg]'], alpha=.8, label='Heading', color='lightgrey')
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
plt.legend()
plt.title('London II Day1')


fig, ax1 = plt.subplots(figsize=(18,10))
ax2 = ax1.twinx()
#ax1.plot(L2_Licor_d1.index,L2_Licor_d1['CH4_Licor'], alpha=.8, label='Licor')
ax1.plot(L2_Licor_d2.index,L2_Licor_d2['CH4_ele_Licor'], alpha=1, label='Licor', linewidth=2, color='rebeccapurple')
ax1.scatter(L2_day2_Licor_peaks.index,L2_day2_Licor_peaks['CH4_ele_Licor'], alpha=1, label='Licor', color='red')
ax1.scatter(L2_Licor_d2.index,L2_Licor_d2['Speed [m/s]'], alpha=.6, label='Speed m/s', color='orange')
ax2.plot(L2_Licor_d2.index,L2_Licor_d2['Heading [deg]'], alpha=.8, label='Heading', color='lightgrey')
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
plt.legend()
plt.title('London II Day2')




###############################################################################
#%% Rotterdam
###############################################################################


#%%% Load & Preprocess data

# create the folder if it they do not exist
if not (path_res / 'Figures' / 'Rotterdam_2022').is_dir(): # output: figures
    (path_res / 'Figures' / 'Rotterdam_2022').mkdir(parents=True)
path_fig = path_res / 'Figures' / 'Rotterdam_2022/'


starttime       = pd.to_datetime('2022-09-06 06:50:00')
endtime         = pd.to_datetime('2022-09-06 12:59:00')

morning_start   = pd.to_datetime('2022-09-06 07:05:00')
morning_end     = pd.to_datetime('2022-09-06 10:44:00')

afternoon_start = pd.to_datetime("2022-09-06 11:05:00")
afternoon_end = pd.to_datetime('2022-09-06 12:26:00')

# writexlsx = False
R_G4302, R_G2301, R_aeris, R_miro, R_aerodyne = read_and_preprocess_R(path_dataR, path_processeddata, BG_QUANTILE, writexlsx=writexlsx)

# Set the name attribute for each dataframe
R_G2301.name = 'G2301'
R_G4302.name = 'G4302'
R_aeris.name = 'Aeris'
R_miro.name = 'Miro'
R_aerodyne.name = 'Aerodyne'


#%%% Find peaks


''' --- Morning - UU --- '''
 
if writexlsx:
    writer  = pd.ExcelWriter(path_processeddata /"R_CH4peaks_morningUU.xlsx", engine = 'xlsxwriter')
else:
    writer = None

R_G2_peaks_m    = process_peak_data(R_G2301[morning_start:morning_end], 'G23', distance=DIST_G23, width=WIDTH_G23,
                                     overviewplot=overviewplot, savepath = path_fig/'R_overviewplot_UUm_G2301')
 
R_G4_peaks_m    = process_peak_data(R_G4302[morning_start:morning_end], 'G43', distance=DIST_G43, width=WIDTH_G43,
                                     overviewplot=overviewplot, savepath = path_fig/'R_overviewplot_UUm_G4302')

R_aeris_peaks_m = process_peak_data(R_aeris[morning_start:morning_end], 'aeris', distance=DIST_AERIS, width=WIDTH_AERIS,
                                        overviewplot=overviewplot, savepath = path_fig/'R_overviewplot_UUm_Aeris')


  
if writexlsx:
    R_G2_peaks_m.to_excel(writer, sheet_name='G2301')
    R_G4_peaks_m.to_excel(writer, sheet_name='G4302')
    R_aeris_peaks_m.to_excel(writer, sheet_name='Aeris')
    writer.book.close()
    
    
''' --- Morning - TNO --- '''
 
if writexlsx:
    writer  = pd.ExcelWriter(path_processeddata /"R_CH4peaks_morningTNO.xlsx", engine = 'xlsxwriter')
else:
    writer = None



R_miro_peaks_m = process_peak_data(R_miro[morning_start:morning_end], 'miro', distance=DIST_MIRO, width=WIDTH_MIRO,
                                       overviewplot=overviewplot, savepath = path_fig/'R_overviewplot_TNOm_Miro')
 
R_aero_peaks_m = process_peak_data(R_aerodyne[morning_start:morning_end], 'aero', distance=DIST_AERO, width=WIDTH_AERO,
                                       overviewplot=overviewplot, savepath = path_fig/'R_overviewplot_TNOm_Aero')


if writexlsx:
    R_aero_peaks_m.to_excel(writer, sheet_name='Aerodyne')
    R_miro_peaks_m.to_excel(writer, sheet_name='Miro')
    writer.book.close()
    
    
''' --- Afternoon - TNO --- '''
 
if writexlsx:
    writer  = pd.ExcelWriter(path_processeddata /"R_CH4peaks_afternoon.xlsx", engine = 'xlsxwriter')
else:
    writer = None

R_G4_peaks_a    = process_peak_data(R_G4302[afternoon_start:afternoon_end], 'G43', distance=DIST_G43, width=WIDTH_G43,
                                     overviewplot=overviewplot, savepath = path_fig/'R_overviewplot_TNOa_G4302')

R_aeris_peaks_a = process_peak_data(R_aeris[afternoon_start:afternoon_end], 'aeris', distance=DIST_AERIS, width=WIDTH_AERIS,
                                        overviewplot=overviewplot, savepath = path_fig/'R_overviewplot_TNOa_Aeris')
 
R_miro_peaks_a  = process_peak_data(R_miro[afternoon_start:afternoon_end], 'miro', distance=DIST_MIRO, width=WIDTH_MIRO,
                                       overviewplot=overviewplot, savepath = path_fig/'R_overviewplot_TNOa_MIRO')
 
R_aero_peaks_a  = process_peak_data(R_aerodyne[afternoon_start:afternoon_end], 'aero', distance=DIST_AERO, width=WIDTH_AERO,
                                       overviewplot=overviewplot, savepath = path_fig/'R_overviewplot_TNOa_AERO')


if writexlsx:
    R_G4_peaks_a.to_excel(writer, sheet_name='G4302')
    R_aeris_peaks_a.to_excel(writer, sheet_name='Aeris')
    R_aero_peaks_a.to_excel(writer, sheet_name='Aerodyne')
    R_miro_peaks_a.to_excel(writer, sheet_name='Miro')
    writer.book.close()
    

 

#%%% Peak Plots


# Define other necessary variables
coord_extent = [4.51832, 4.52830, 51.91921, 51.92288]
release_loc1 = (4.5237450, 51.9201216)
release_loc2 = (4.5224917, 51.9203931) #51.9203931,4.5224917
release_loc3 = (4.523775, 51.921028) # estimated from Daans plot (using google earth)
column_names_mUU = {'G4302': 'CH4_ele_G43','G2301': 'CH4_ele_G23', 'Aeris':'CH4_ele_aeris'}
column_names_mTNO = {'Miro': 'CH4_ele_miro', 'Aerodyne': 'CH4_ele_aero'}
column_names_aTNO = {'Miro': 'CH4_ele_miro', 'Aerodyne': 'CH4_ele_aero', 'G4302': 'CH4_ele_G43', 'Aeris':'CH4_ele_aeris'}
# indiv_peak_plots = True  # or False based on your requirement


''' --- Morning UU --- '''

if not (path_fig / 'R_Peakplots' / 'Morning_UU').is_dir():
    (path_fig / 'R_Peakplots' / 'Morning_UU').mkdir(parents=True)
path_save = path_fig / 'R_Peakplots' / 'Morning_UU'

# Remove rows with NaN values in column 'latitude'
gps = R_G4302.dropna(subset=['Latitude']).copy(deep=True)
# Call the function with necessary arguments
plot_indivpeaks_bevorQC(R_G4_peaks_m, gps, path_save, coord_extent, release_loc1, release_loc2, indiv_peak_plots, column_names_mUU, R_G4302, R_G2301,R_aeris)


''' --- Morning TNO --- '''

if not (path_fig / 'R_Peakplots' / 'Morning_TNO').is_dir():
    (path_fig / 'R_Peakplots' / 'Morning_TNO').mkdir(parents=True)
path_save = path_fig / 'R_Peakplots' / 'Morning_TNO'

# Remove rows with NaN values in column 'latitude'
gps = R_miro.dropna(subset=['Latitude']).copy(deep=True)
# Call the function with necessary arguments
plot_indivpeaks_bevorQC(R_miro_peaks_m, gps, path_save, coord_extent, release_loc1, release_loc2, indiv_peak_plots, column_names_mTNO, R_miro, R_aerodyne)


''' --- Afternoon TNO --- '''

if not (path_fig / 'R_Peakplots' / 'Afternoon_TNO').is_dir():
    (path_fig / 'R_Peakplots' / 'Afternoon_TNO').mkdir(parents=True)
path_save = path_fig / 'R_Peakplots' / 'Afternoon_TNO'

# Remove rows with NaN values in column 'latitude'
gps = R_miro.dropna(subset=['Latitude']).copy(deep=True)
# Call the function with necessary arguments
plot_indivpeaks_bevorQC(R_miro_peaks_a, gps, path_save, coord_extent, release_loc2, release_loc3, indiv_peak_plots, column_names_aTNO, R_miro, R_aerodyne, R_G4302,R_aeris)







