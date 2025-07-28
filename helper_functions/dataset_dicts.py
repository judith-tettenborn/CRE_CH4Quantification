# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 15:10:38 2025

@author: Judith
"""

import pandas as pd
from pathlib import Path
import sys

# path_base = Path('C:/Users/.../CRE_CH4Quantification/') # insert the the project path here
path_base = Path('C:/Users/Judit/Documents/UNI/Utrecht/Hiwi/CRE_CH4Quantification/')
sys.path.append(str(path_base / 'src'))


path_procdata = path_base / 'data' / 'processed'

#%% Datasets

# -----------------------------------------------------------------------------
# Read in datafiles containing methane timeseries
# -----------------------------------------------------------------------------
# Utrecht
U1_G4302    = pd.read_excel(path_procdata / 'U1_G23andG43.xlsx',sheet_name='G4302', index_col='Datetime')  
U1_G2301    = pd.read_excel(path_procdata / 'U1_G23andG43.xlsx',sheet_name='G2301', index_col='Datetime')  

# Utrecht II
U2_G2301    = pd.read_csv(path_procdata / 'U2_G2301.csv', index_col='Datetime', parse_dates=['Datetime'])
U2_aeris    = pd.read_csv(path_procdata / 'U2_aeris.csv', index_col='Datetime', parse_dates=['Datetime']) 

# Rotterdam
R_G4302     = pd.read_csv(path_procdata / 'R_G4302.csv', index_col='Datetime', parse_dates=['Datetime'])  
R_G2301     = pd.read_csv(path_procdata / 'R_G2301.csv', index_col='Datetime', parse_dates=['Datetime'])
R_aeris     = pd.read_csv(path_procdata / 'R_aeris.csv', index_col='Datetime', parse_dates=['Datetime'])  
R_miro      = pd.read_csv(path_procdata / 'R_miro.csv', index_col='Datetime', parse_dates=['Datetime'])
R_aerodyne  = pd.read_csv(path_procdata / 'R_aerodyne.csv', index_col='Datetime', parse_dates=['Datetime'])

# Tornto
T_1b_LGR    = pd.read_csv(path_procdata / 'T_1bike_LGR.csv', index_col='Datetime', parse_dates=['Datetime'])
T_1c_G2401  = pd.read_csv(path_procdata / 'T_1car_G24.csv', index_col='Datetime', parse_dates=['Datetime']) 
T_2c_G2401  = pd.read_csv(path_procdata / 'T_2car_G24.csv', index_col='Datetime', parse_dates=['Datetime']) 

# London I
L1_d2_LGR   = pd.read_excel(path_procdata / 'L1_G23andLGRandLicor.xlsx',sheet_name='D2_LGR', index_col='Datetime')  
L1_d2_G2301 = pd.read_excel(path_procdata / 'L1_G23andLGRandLicor.xlsx',sheet_name='D2_G2301', index_col='Datetime') 
L1_d3_Licor = pd.read_excel(path_procdata / 'L1_G23andLGRandLicor.xlsx',sheet_name='D3_Licor', index_col='Datetime')  
L1_d3_G2301 = pd.read_excel(path_procdata / 'L1_G23andLGRandLicor.xlsx',sheet_name='D3_G2301', index_col='Datetime') 
L1_d5_G2301 = pd.read_excel(path_procdata / 'L1_G23andLGRandLicor.xlsx',sheet_name='D5_G2301', index_col='Datetime')  

# London II
L2_d1_Licor = pd.read_csv(path_procdata / 'L2_day1_Licor.csv', index_col='Datetime', parse_dates=['Datetime']) 
L2_d2_Licor = pd.read_csv(path_procdata / 'L2_day2_Licor.csv', index_col='Datetime', parse_dates=['Datetime']) 


# -----------------------------------------------------------------------------
# Define dictionaries containing informations about the dataset
# -----------------------------------------------------------------------------

# Explanation for the (somewhat confusing and unnecessarily complicated) namings:
# spec: abbreviation used in the variable names in the code
# name: name used for the different instruments which is a mix of actual instrument 
#       names and brand names (due to past dependencies in the code)
# title: proper instrument name as it appears in the publication

# Utrecht I
U1_vars_G43 = {'df': U1_G4302,
                'CH4col':  'CH4_ele_G43', 
                'spec':    'G43',
                'name':    'G4302',
                'title':   'G4302',
                'city':    'Utrecht I',
                'day':     'Day1'
                 }
U1_vars_G23 = {'df': U1_G2301,
                'CH4col':  'CH4_ele_G23',
                'spec':    'G23',
                'name':    'G2301',
                'title':   'G2301',
                'city':    'Utrecht I',
                'day':     'Day1'
                }

# Utrecht II
U2_vars_aeris = {'df': U2_aeris,
                'CH4col':  'CH4_ele_aeris', 
                'spec':    'aeris',
                'name':    'Aeris',
                'title':   'Mira Ultra',
                'city':    'Utrecht II',
                'day':     'Day1'
                 }
U2_vars_G23 = {'df': U2_G2301,
                'CH4col':  'CH4_ele_G23',
                'spec':    'G23',
                'name':    'G2301',
                'title':   'G2301',
                'city':    'Utrecht II',
                'day':     'Day1'
                }

# Rotterdam
R_vars_G43 = {'df': R_G4302,
                'CH4col':  'CH4_ele_G43', 
                'spec':    'G43',
                'name':    'G4302',
                'title':   'G4302',
                'city':    'Rotterdam',
                'day':     'Day1'
                 }
R_vars_G23 = {'df': R_G2301,
                 'CH4col':  'CH4_ele_G23', 
                 'spec':    'G23',
                 'name':    'G2301',
                 'title':   'G2301',
                 'city':    'Rotterdam',
                 'day':     'Day1'
                 }
R_vars_aeris = {'df': R_aeris,
                'CH4col':  'CH4_ele_aeris', 
                'spec':    'aeris',
                'name':    'Aeris',
                'title':   'Mira Ultra',
                'city':    'Rotterdam',
                'day':     'Day1'
                 }
R_vars_miro = {'df': R_miro,
                 'CH4col':  'CH4_ele_miro', 
                 'spec':    'miro',
                 'name':    'Miro',
                 'title':   'MGA10',
                 'city':    'Rotterdam',
                 'day':     'Day1'
                 }
R_vars_aerodyne = {'df': R_aerodyne,
                 'CH4col':  'CH4_ele_aero', 
                 'spec':    'aero',
                 'name':    'Aerodyne',
                 'title':   'TILDAS',
                 'city':    'Rotterdam',
                 'day':     'Day1'
                 }

# Toronto
T_vars_1b_LGR = {'df': T_1b_LGR,
                 'CH4col':  'CH4_ele_LGR', 
                 'spec':    'LGR',
                 'name':    'LGR',
                 'title':   'UGGA',
                 'city':    'Toronto',
                 'day':     'Day1-bike'             
                  }

T_vars_1c_G24 = {'df': T_1c_G2401,
                 'CH4col':  'CH4_ele_G24',
                 'spec':    'G24',
                 'name':    'G2401',
                 'title':   'G2401',
                 'city':    'Toronto',
                 'day':     'Day1-car'             
                  }

T_vars_2c_G24 = {'df': T_2c_G2401,
                 'CH4col':  'CH4_ele_G24',
                 'spec':    'G24',
                 'name':    'G2401',
                 'title':   'G2401',
                 'city':    'Toronto',
                 'day':     'Day2-car'             
                  }

# London I
L1_vars_d2_LGR = {'df': L1_d2_LGR,
                 'CH4col':  'CH4_ele_LGR', 
                 'spec':    'LGR',
                 'name':    'LGR',
                 'title':   'uMEA',
                 'city':    'London I',
                 'day':     'Day2'
                  }
L1_vars_d2_G23 = {'df': L1_d2_G2301,
                 'CH4col':  'CH4_ele_G23',
                 'spec':    'G23',
                 'name':    'G2301',
                 'title':   'G2301',
                 'city':    'London I',
                 'day':     'Day2'
                 }
L1_vars_d3_Licor = {'df': L1_d3_Licor,
                 'CH4col':  'CH4_ele_Licor', 
                 'spec':    'Licor',
                 'name':    'Licor',
                 'title':   'LI-7810',
                 'city':    'London I',
                 'day':     'Day3'
                  }
L1_vars_d3_G23 = {'df': L1_d3_G2301,
                 'CH4col':  'CH4_ele_G23',
                 'spec':    'G23',
                 'name':    'G2301',
                 'title':   'G2301',
                 'city':    'London I',
                 'day':     'Day3'
                 }
L1_vars_d5_G23 = {'df': L1_d5_G2301,
                 'CH4col':  'CH4_ele_G23', 
                 'spec':    'G23',
                 'name':    'G2301',
                 'title':   'G2301',
                 'city':    'London I',
                 'day':     'Day5'
                 }
# London II
L2_vars_d1_Licor = {'df': L2_d1_Licor,
                     'CH4col':  'CH4_ele_Licor', 
                     'spec':    'Licor',
                     'name':    'Licor',
                     'title':   'LI-7810',
                     'city':    'London II',
                     'day':     'Day1'
                      }
L2_vars_d2_Licor = {'df': L2_d2_Licor,
                      'CH4col':  'CH4_ele_Licor', 
                      'spec':    'Licor',
                      'name':    'Licor', 
                      'title':   'LI-7810',
                      'city':    'London II',
                      'day':     'Day2'
                      }




