# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 15:11:24 2025

@author: Judith Tettenborn (j.a.f.tettenborn@uu.nl)

"""



#%% Analysis

# threshold for identifying a local maximum as a CH4 peak
HEIGHT = 1.02 # if CH4 mole fraction local max. is 2% over background level
# quantile of the CH4 measurements which is defined as the methane background level
BG_QUANTILE = 0.1 

# Parameters for the peak detection function:
DIST_G23 = 3
WIDTH_G23 = (1,30)
DIST_G24 = 3
WIDTH_G24 = (1,30)
DIST_LGR = 3
WIDTH_LGR = (1,30)
DIST_LICOR = 3
WIDTH_LICOR = (1,30)
DIST_MIRO = 3
WIDTH_MIRO = (1,30)
DIST_AERO = 3
WIDTH_AERO = (1,30)

DIST_G43 = 15
WIDTH_G43 = (1,45)
DIST_AERIS = 15
WIDTH_AERIS = (1,45)









#%% Plots

# Dictionary defining a color for each instrument (for plotting)
dict_color_instr = {'G2301':    '#fb7b50',
                    'G4302':    '#00698b',
                    'Aeris':    '#91e1d7',
                    'Miro':     '#ffc04c',
                    'Aerodyne': '#00b571',
                    'LGR':      'firebrick',
                    'G2401':    'deeppink',
                    'Licor':    'rebeccapurple'
                    }


dict_color_city = {'Rotterdam':         'orange',
                   'Utrecht I':         'orchid',
                   'Utrecht II':        'darkorchid',
                   'TorontoDay1-bike':  'deepskyblue',
                   'TorontoDay1-car':   '#19e8dc',
                   'TorontoDay2-car':   '#2d75b6',
                   'London IDay2':      'mediumseagreen',
                   'London IDay3':      'darkgreen',
                   'London IDay4':      'olive',
                   'London IDay5':      'lime',
                   'London IIDay1':     'brown',
                   'London IIDay2':     'chocolate'
                   }

dict_instr_names = {'Miro':     'MGA10', 
                    'Aerodyne': 'TILDAS', 
                    'G4302':    'G4302',
                    'G2301':    'G2301', 
                    'G2401':    'G2401',
                    'Aeris':    'Mira Ultra',
                    'LGR':      'LGR',
                    'Licor':    'LI-7810'
                    }

dict_spec_instr = {'G2301':     'G23',
                   'G4302':     'G43',
                   'Aeris':     'aeris',
                   'Miro':      'miro',
                   'Aerodyne':  'aero',
                   'LGR':       'LGR',
                   'G2401':     'G24',
                   'Licor':     'Licor'
                   }


