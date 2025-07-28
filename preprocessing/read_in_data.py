 # -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:34:01 2024

@author: Judith Tettenborn

based on contributions from Daan Stroeken

"""


import pandas as pd
import numpy as np
import os
from datetime import timedelta
from datetime import datetime
from geopy.distance import geodesic
import gpxpy

from helper_functions.utils import *



#%% Utrecht I
# =============================================================================

def read_gps_U1(path_data):
    """
    Reads and processes GPS data from two specific .txt files located in a subdirectory of the given path.

    Parameters:
    -----------
    path_data : str
        Directory path of the release data of this specific CRE experiment.

    Returns:
    --------
    gps : pandas.DataFrame
        Concatenated GPS data from both files with columns:
        'Latitude', 'Longitude', 'Speed [m/s]' and a datetime index.
    """
    
    #   Read GPS data (txt file)   
    gps1 = pd.read_csv(path_data / 'GPS/20221125-140618_GPS.txt',
                                delimiter = ",",
                                #names=['Datetime', 'Latitude', 'Longitude', 'speed [m/s]'],
                                usecols = [1,2,3,7])
    gps2 = pd.read_csv(path_data / 'GPS/20221125-153309_GPS.txt',
                                delimiter = ",",
                                #names=['Datetime', 'Latitude', 'Longitude', 'speed [m/s]'],
                                usecols = [1,2,3,7])
    gps = pd.concat([gps1, gps2], axis=0)
    gps.time = pd.to_datetime(gps.time)
    print(gps.columns)
    gps.rename(columns={'time': 'Datetime', 'latitude': 'Latitude', 'longitude': 'Longitude', 'speed (m/s)': 'Speed [m/s]'}, inplace=True)
    print(gps.columns)
    gps.set_index('Datetime',inplace=True)
    
    return gps


# =============================================================================
#   CH4 analyzer  
# =============================================================================

# --------------- G2301 & G4302  ---------------


def read_G23andG43_U1(path_data):
    dfs_G23 = dat_to_df_dict(path_data / 'G2301_Picarro/')
    dfs_G43 = dat_to_df_dict(path_data / 'G4302_Picarro/')
    
    # Merge all dataframes in the dictionary into a single dataframe
    df_G23 = pd.concat(dfs_G23.values(), ignore_index=True)
    df_G43 = pd.concat(dfs_G43.values(), ignore_index=True)
    
    cols_G23    = ['DATE','TIME','CH4_dry','species','ALARM_STATUS'] #'CO2_dry', 
    cols_G43    = ['DATE','TIME','CH4_dry','ALARM_STATUS'] #'C2H6_dry',
    
    
    G2301 = df_G23[cols_G23].copy()
    G4302 = df_G43[cols_G43].copy()
    
    # Combine date and time into datetime
    G4302.loc[:,'Datetime'] = pd.to_datetime(G4302['DATE'] + ' ' + G4302['TIME'])
    G2301.loc[:,'Datetime'] = pd.to_datetime(G2301['DATE'] + ' ' + G2301['TIME'])
    
    G4302 = G4302.drop(['DATE','TIME'],axis=1)
    G2301 = G2301.drop(['DATE','TIME'],axis=1)
    G2301 = G2301[G2301["species"]==1]
    
    G2301.set_index('Datetime',inplace = True,drop=True)
    G4302.set_index('Datetime',inplace = True,drop=True)
    
    
    G4302['CH4_G43']     = calibrate(G4302['CH4_dry'], 'G43', 'CH4')
    G2301['CH4_G23']     = calibrate(G2301['CH4_dry'], 'G23', 'CH4')
    
    G2301['bg_G23']      = G2301['CH4_G23'].rolling('5min',center=True).quantile(0.10) 
    G2301['CH4_ele_G23'] = (G2301['CH4_G23'] - G2301['bg_G23'])
    
    G4302['bg_G43']      = G4302['CH4_G43'].rolling('5min',center=True).quantile(0.10) 
    G4302['CH4_ele_G43'] = (G4302['CH4_G43'] - G4302['bg_G43'])
    
    G2301.rename(columns={'ALARM_STATUS':'ALARM_STATUS_G23'},inplace=True)
    G4302.rename(columns={'ALARM_STATUS':'ALARM_STATUS_G43'},inplace=True)
    
    
    #  Time correction
    G2301.index = (G2301.index - timedelta(seconds=59)) # match with G4302 (from metadata excel file
    # which contains passing time of the measurement car it seems G4 is correct, but G2 shifted)
    return G2301,G4302


def read_and_preprocess_G23andG43_U1(path_data, path_res, writexlsx=False):
    
    gps = read_gps_U1(path_data)
    G2301,G4302 = read_G23andG43_U1(path_data)
    
    G4302_gps = merge_with_gps(G4302, gps) # in: data_handling.py in helper_functions
    G2301_gps = merge_with_gps(G2301, gps)


    if writexlsx:
        writer  = pd.ExcelWriter(path_res / 'U1_G23andG43.xlsx', engine = 'xlsxwriter')
        G4302_gps.to_excel(writer, sheet_name='G4302')
        G2301_gps.to_excel(writer, sheet_name='G2301')
        writer.book.close()
    
    return G2301_gps,G4302_gps


#%% Utrecht II
# =============================================================================


def gpx_to_df(file_path):
    with open(file_path, 'r') as gpx_file:
        gpx_data = gpxpy.parse(gpx_file)
        
    waypoints_list = []
    for track in gpx_data.tracks:
        for segment in track.segments:
            for point in segment.points:
                waypoints_list.append({
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'time': point.time,
                    'speed': point.speed,
                    # Add more attributes if needed
                })
                
    df = pd.DataFrame(waypoints_list)
    df.loc[:,'Datetime'] = pd.to_datetime(df['time']) + pd.Timedelta(hours=2) #convert to local time
    df = df.drop(['time'],axis=1)
    df.set_index('Datetime',inplace = True,drop=True)
    df.index = df.index.round('1s')
    return df


def read_gps_U2(path_data):
    df_phone1_1 = gpx_to_df(path_data /'GPS/Phone1/20240611-1057_1200_Phone1_1.gpx')
    df_phone1_2 = gpx_to_df(path_data /'GPS/Phone1/20240611-1201_1304_Phone1_2.gpx')
    df_phone1_3 = gpx_to_df(path_data /'GPS/Phone1/20240611-1352_1426_Phone1_3.gpx')
    df_phone1_4 = gpx_to_df(path_data /'GPS/Phone1/20240611-1426_1558_Phone1_4.gpx')
    df_phone1_5 = gpx_to_df(path_data /'GPS/Phone1/20240611-1620_1816_Phone1_5.gpx')
    
    gps_phone2 = gpx_to_df(path_data /'GPS/Phone2/20240611-1620_1844_Phone2_final.gpx')
    
    df_phone1_1.index = df_phone1_1.index.tz_convert(None)
    df_phone1_2.index = df_phone1_2.index.tz_convert(None)
    df_phone1_3.index = df_phone1_3.index.tz_convert(None)
    df_phone1_4.index = df_phone1_4.index.tz_convert(None)
    df_phone1_5.index = df_phone1_5.index.tz_convert(None)
    gps_phone2.index = gps_phone2.index.tz_convert(None)
    
    # GPS has several different offsets (reason unknown) -> correct with shift
    # Shifts were determined by detailed study of methane and location data 
    # and the location of detected peaks
    df_phone1_1.index = (df_phone1_1.index + timedelta(seconds=60))
    df_phone1_2.index = (df_phone1_2.index + timedelta(seconds=60))
    df_phone1_3.index = (df_phone1_3.index + timedelta(seconds=117))
    df_phone1_4.index = (df_phone1_4.index + timedelta(seconds=66))
    gps_phone2.index = (gps_phone2.index + timedelta(seconds=60))
    gps_phone2_1 = gps_phone2.loc['2024-06-11 16:20:26':'2024-06-11 17:38:00'].copy()
    gps_phone2_2 = gps_phone2.loc['2024-06-11 17:38:00':'2024-06-11 18:06:00'].copy()
    gps_phone2_3 = gps_phone2.loc['2024-06-11 18:06:00':'2024-06-11 18:44:31'].copy()
    gps_phone2_2.index = (gps_phone2_2.index + timedelta(seconds=21))
    
    
    # concatenate different gps dfs into one df
    gps_phone1 = pd.concat([df_phone1_1, df_phone1_2,df_phone1_3,df_phone1_4,df_phone1_5], axis=0)
    # phone 1 had a malfunction in between and was not measuring continuously, therefore merging with phone 2 for this timeframe 
    #gps_final = pd.concat([df_phone1_1, df_phone1_2,df_phone1_3,df_phone1_4,gps_phone2], axis=0) 
    gps_final = pd.concat([df_phone1_1, df_phone1_2,df_phone1_3[:'2024-06-11 14:28:04'],df_phone1_4,gps_phone2_1,gps_phone2_2[:'2024-06-11 18:05:59'],gps_phone2_3], axis=0) 
    # Rename columns
    gps_phone1.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude', 'speed': 'Speed [m/s]'}, inplace=True)
    gps_phone2.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude', 'speed': 'Speed [m/s]'}, inplace=True)
    gps_final.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude', 'speed': 'Speed [m/s]'}, inplace=True)

    return gps_final #,df_phone1_1,df_phone1_2,df_phone1_3,df_phone1_4,gps_phone2_1,gps_phone2_2,gps_phone2_3



def read_and_preprocess_U2(path_data, path_res,bg_quantile=None,writexlsx=False):
    
    if not bg_quantile:
        bg_quantile = 0.1 # default value for CH4 background
        
    # =============================================================================
    #   G2301 
    # =============================================================================
    
    
    dfs_G2 = dat_to_df_dict(path_data / 'G2301_Picarro/')
    
    # Merge all dataframes in the dictionary into a single dataframe
    df_G2 = pd.concat(dfs_G2.values(), ignore_index=True)
    
    cols_G23    = ['DATE','TIME','CH4_dry','species','ALARM_STATUS'] #'CO2_dry', 
    G2301 = df_G2[cols_G23].copy()
    
    # Combine date and time into datetime
    G2301.loc[:,'Datetime'] = pd.to_datetime(G2301['DATE'] + ' ' + G2301['TIME'])
    G2301 = G2301.drop(['DATE','TIME'],axis=1)
    G2301 = G2301[G2301["species"]==1]
    G2301.set_index('Datetime',inplace = True,drop=True)
    G2301.rename(columns={'ALARM_STATUS':'ALARM_STATUS_G23'},inplace=True)
    
    # --- Calculate CH4 elevation ------------------
    
    # calibration function obtained at IMAU laboratory in 2022(?)
    G2301['CH4_G23']     = calibrate(G2301['CH4_dry'], 'G23', 'CH4')
    
    G2301['bg_G23']      = G2301['CH4_G23'].rolling('5min',center=True).quantile(bg_quantile) 
    G2301['CH4_ele_G23'] = (G2301['CH4_G23'] - G2301['bg_G23'])
      
    #  --- Time correction ---------------------------
    # 2 Inlet delay tests were made -> use average
    t_delay_G2301 = (15.96+17.04)/2
    G2301.index = (G2301.index - timedelta(seconds=t_delay_G2301))
    
    
    # =============================================================================
    #   Aeris Mira Ultra 
    # =============================================================================
    
    dfs_aeris = txt_to_df_dict(path_data / 'MiraUltra_Aeris/')
    
    # Merge all dataframes in the dictionary into a single dataframe
    df_aeris = pd.concat(dfs_aeris.values(), ignore_index=True)
    cols_aeris    = ['Time Stamp','CH4 (ppm)','Wall Code'] #'CO2_dry', 
    aeris = df_aeris[cols_aeris].copy()
    
    aeris.rename(columns={'Time Stamp':'Datetime', 'CH4 (ppm)':'CH4_aeris'},inplace=True)
    aeris = aeris.set_index('Datetime', drop = True)
    aeris.index = pd.to_datetime(aeris.index)
    
    # --- Calculate CH4 elevation ------------------
    
    aeris['CH4_aeris']     = calibrate(aeris['CH4_aeris'], 'aer', 'CH4')
    
    aeris['bg_aeris'] = aeris['CH4_aeris'].rolling('5min',center=True).quantile(bg_quantile)
    aeris['CH4_ele_aeris'] = aeris['CH4_aeris'] - aeris['bg_aeris']
    
    #  --- Time correction ---------------------------
    # 2 Inlet delay tests were made -> use average
    t_delay_aeris = (11.04+11.62)/2
    aeris.index = (aeris.index - timedelta(seconds=t_delay_aeris)
                               + timedelta(seconds=42)) # to match G2301 data
    
    
    # =============================================================================
    #   Merge with gps 
    # =============================================================================
    
    gps_final = read_gps_U2(path_data)
    
    G2301_gps = G2301.copy()
    df_merged = pd.merge(G2301, gps_final, left_index=True, right_index=True, how='outer')
    df_merged.interpolate(method='linear', inplace=True)
    
    G2301_gps.loc[:,['Longitude', 'Latitude', 'Speed [m/s]']] = df_merged.loc[:,['Longitude', 'Latitude','Speed [m/s]']]
    
    aeris_gps = aeris.copy()
    df_merged = pd.merge(aeris, gps_final, left_index=True, right_index=True, how='outer')
    df_merged.interpolate(method='linear', inplace=True)
    aeris_gps.loc[:,['Longitude', 'Latitude', 'Speed [m/s]']] = df_merged.loc[:,['Longitude', 'Latitude','Speed [m/s]']]
    
    # =============================================================================
    #   Save to CSV
    # =============================================================================
    
    # Print data into csv
    if writexlsx:
        G2301_gps.to_csv(path_res / 'U2_G2301.csv', index=True)
        aeris_gps.to_csv(path_res / 'U2_aeris.csv', index=True)
        
    return G2301_gps, aeris_gps


#################################################################################
# ONLY FOR TESTING
#################################################################################

def read_gps_U2_no_shift(path_data):
    df_phone1_1 = gpx_to_df(path_data /'GPS/Phone1/20240611-1057_1200_Phone1_1.gpx')
    df_phone1_2 = gpx_to_df(path_data /'GPS/Phone1/20240611-1201_1304_Phone1_2.gpx')
    df_phone1_3 = gpx_to_df(path_data /'GPS/Phone1/20240611-1352_1426_Phone1_3.gpx')
    df_phone1_4 = gpx_to_df(path_data /'GPS/Phone1/20240611-1426_1558_Phone1_4.gpx')
    df_phone1_5 = gpx_to_df(path_data /'GPS/Phone1/20240611-1620_1816_Phone1_5.gpx')
    
    gps_phone2 = gpx_to_df(path_data /'GPS/Phone2/20240611-1620_1844_Phone2_final.gpx')
    
    df_phone1_1.index = df_phone1_1.index.tz_convert(None)
    df_phone1_2.index = df_phone1_2.index.tz_convert(None)
    df_phone1_3.index = df_phone1_3.index.tz_convert(None)
    df_phone1_4.index = df_phone1_4.index.tz_convert(None)
    df_phone1_5.index = df_phone1_5.index.tz_convert(None)
    gps_phone2.index = gps_phone2.index.tz_convert(None)
    
    # GPS has several different offsets (reason unknown) -> correct with shift
    # Shifts were determined by detailed study of methane and location data 
    # and the location of detected peaks
    # df_phone1_1.index = (df_phone1_1.index + timedelta(seconds=60))
    # df_phone1_2.index = (df_phone1_2.index + timedelta(seconds=60))
    # df_phone1_3.index = (df_phone1_3.index + timedelta(seconds=117))
    # df_phone1_4.index = (df_phone1_4.index + timedelta(seconds=66))
    # gps_phone2.index = (gps_phone2.index + timedelta(seconds=60))
    gps_phone2_1 = gps_phone2.loc['2024-06-11 16:20:26':'2024-06-11 17:38:00'].copy()
    gps_phone2_2 = gps_phone2.loc['2024-06-11 17:38:00':'2024-06-11 18:06:00'].copy()
    gps_phone2_3 = gps_phone2.loc['2024-06-11 18:06:00':'2024-06-11 18:44:31'].copy()
    # gps_phone2_2.index = (gps_phone2_2.index + timedelta(seconds=21))
    
    
    # concatenate different gps dfs into one df
    gps_phone1 = pd.concat([df_phone1_1, df_phone1_2,df_phone1_3,df_phone1_4,df_phone1_5], axis=0)
    # phone 1 had a malfunction in between and was not measuring continuously, therefore merging with phone 2 for this timeframe 
    #gps_final = pd.concat([df_phone1_1, df_phone1_2,df_phone1_3,df_phone1_4,gps_phone2], axis=0) 
    gps_final = pd.concat([df_phone1_1, df_phone1_2,df_phone1_3[:'2024-06-11 14:28:04'],df_phone1_4,gps_phone2_1,gps_phone2_2[:'2024-06-11 18:05:59'],gps_phone2_3], axis=0) 
    # Rename columns
    gps_phone1.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude', 'speed': 'Speed [m/s]'}, inplace=True)
    gps_phone2.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude', 'speed': 'Speed [m/s]'}, inplace=True)
    gps_final.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude', 'speed': 'Speed [m/s]'}, inplace=True)

    return gps_final 


def read_and_preprocess_U2_no_shift(path_data, path_res,bg_quantile=None,writexlsx=False):
    
    if not bg_quantile:
        bg_quantile = 0.1 # default value for CH4 background
        
    # =============================================================================
    #   G2301 
    # =============================================================================
    
    
    dfs_G2 = dat_to_df_dict(path_data / 'G2301_Picarro/')
    
    # Merge all dataframes in the dictionary into a single dataframe
    df_G2 = pd.concat(dfs_G2.values(), ignore_index=True)
    
    cols_G23    = ['DATE','TIME','CH4_dry','species','ALARM_STATUS'] #'CO2_dry', 
    G2301 = df_G2[cols_G23].copy()
    
    # Combine date and time into datetime
    G2301.loc[:,'Datetime'] = pd.to_datetime(G2301['DATE'] + ' ' + G2301['TIME'])
    G2301 = G2301.drop(['DATE','TIME'],axis=1)
    G2301 = G2301[G2301["species"]==1]
    G2301.set_index('Datetime',inplace = True,drop=True)
    G2301.rename(columns={'ALARM_STATUS':'ALARM_STATUS_G23'},inplace=True)
    
    # --- Calculate CH4 elevation ------------------
    
    # calibration function obtained at IMAU laboratory in 2022(?)
    G2301['CH4_G23']     = calibrate(G2301['CH4_dry'], 'G23', 'CH4')
    
    G2301['bg_G23']      = G2301['CH4_G23'].rolling('5min',center=True).quantile(bg_quantile) 
    G2301['CH4_ele_G23'] = (G2301['CH4_G23'] - G2301['bg_G23'])
      
    #  --- Time correction ---------------------------
    # 2 Inlet delay tests were made -> use average
    t_delay_G2301 = (15.96+17.04)/2
    G2301.index = (G2301.index - timedelta(seconds=t_delay_G2301))
    
    
    # =============================================================================
    #   Aeris Mira Ultra 
    # =============================================================================
    
    dfs_aeris = txt_to_df_dict(path_data / 'MiraUltra_Aeris/')
    
    # Merge all dataframes in the dictionary into a single dataframe
    df_aeris = pd.concat(dfs_aeris.values(), ignore_index=True)
    cols_aeris    = ['Time Stamp','CH4 (ppm)','Wall Code'] #'CO2_dry', 
    aeris = df_aeris[cols_aeris].copy()
    
    aeris.rename(columns={'Time Stamp':'Datetime', 'CH4 (ppm)':'CH4_aeris'},inplace=True)
    aeris = aeris.set_index('Datetime', drop = True)
    aeris.index = pd.to_datetime(aeris.index)
    
    # --- Calculate CH4 elevation ------------------
    
    aeris['CH4_aeris']     = calibrate(aeris['CH4_aeris'], 'aer', 'CH4')
    
    aeris['bg_aeris'] = aeris['CH4_aeris'].rolling('5min',center=True).quantile(bg_quantile)
    aeris['CH4_ele_aeris'] = aeris['CH4_aeris'] - aeris['bg_aeris']
    
    #  --- Time correction ---------------------------
    # 2 Inlet delay tests were made -> use average
    t_delay_aeris = (11.04+11.62)/2
    aeris.index = (aeris.index - timedelta(seconds=t_delay_aeris)
                               + timedelta(seconds=42)) # to match G2301 data
    
    
    # =============================================================================
    #   Merge with gps 
    # =============================================================================
    
    gps_final = read_gps_U2_no_shift(path_data)
    
    print(list(G2301.index[:7]))
    G2301_gps = G2301.copy(deep=True)
    df_merged = pd.merge(G2301, gps_final, left_index=True, right_index=True, how='outer')
    print(list(df_merged.index[:7]))
    df_merged.interpolate(method='linear', inplace=True)
    print(list(df_merged.index[:7]))
    df_merged_unique = df_merged[~df_merged.index.duplicated(keep='first')]
    df_merged_filtered = df_merged_unique.reindex(G2301.index)
    print(list(df_merged_filtered.index[:7]))
    G2301_gps.loc[:,['Longitude', 'Latitude', 'Speed [m/s]']] = df_merged_filtered.loc[:,['Longitude', 'Latitude','Speed [m/s]']]
    print(list(G2301_gps.index[:7]))
    
    aeris_gps = aeris.copy(deep=True)
    df_merged = pd.merge(aeris, gps_final, left_index=True, right_index=True, how='outer')
    df_merged.interpolate(method='linear', inplace=True)
    df_merged_unique = df_merged[~df_merged.index.duplicated(keep='first')]
    df_merged_filtered = df_merged_unique.reindex(aeris.index)
    aeris_gps.loc[:,['Longitude', 'Latitude', 'Speed [m/s]']] = df_merged_filtered.loc[:,['Longitude', 'Latitude','Speed [m/s]']]
    
    # =============================================================================
    #   Save to CSV
    # =============================================================================
    
    # Print data into csv
    if writexlsx:
        G2301_gps.to_csv(path_res / 'U2_G2301.csv', index=True)
        aeris_gps.to_csv(path_res / 'U2_aeris.csv', index=True)
        
    return G2301_gps, aeris_gps




#%% Toronto

# def merge_interpolate_left(df1,df2,col):
#         if df1.index.name == col:
#             combined = pd.merge(df1,df2,on=col,how='left')
#             combined = combined.sort_values(by=col)
#             combined = combined.interpolate(method='linear')
#         else:
#             print('failed')
#             combined = pd.merge(df1,df2,on=col,how='left')
#             combined = combined.sort_values(by=col)
#             combined = combined.interpolate(method='linear')
#         return combined
# merge_interpolate_left should be imported from utils.py

def calc_gps_T(df):
    """
    Calculates speed between consecutive GPS points using geodesic distance.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing GPS data with at least 'Latitude' and 'Longitude' columns,
        and a datetime index with uniform or meaningful time intervals.

    Returns:
    --------
    gps : pandas.DataFrame
        DataFrame with columns: 'Latitude', 'Longitude', and 'Speed [m/s]',
        where speed is computed between consecutive points based on geodesic distance
        and time difference.
    """
    
    gps = df[['Latitude','Longitude']].copy(deep=True)
    
    gps['Speed [m/s]'] = gps.apply(lambda row: 0, axis=1)  # Initialize speed column with 0
    
    for i in range(1, len(gps)):
        # Calculate distance between consecutive rows
        prev_lat = gps.iloc[i-1, 0]  # 0=Latitude
        prev_lon = gps.iloc[i-1, 1]  # 1=Longitude
        curr_lat = gps.iloc[i, 0]
        curr_lon = gps.iloc[i, 1]
        distance = geodesic((prev_lat, prev_lon), (curr_lat, curr_lon)).meters
    
        # Calculate speed using distance and time (assuming time between consecutive rows is constant)
        time_diff = gps.index[i] - gps.index[i-1]
        time_diff_seconds = time_diff.total_seconds()
        speed = distance / time_diff_seconds
    
        # Assign calculated speed to the 'speed' column
        gps.iloc[i, 2] = speed
    
    return gps


def read_and_preprocess_BikeandCar_T(path_dataT, path_res, writexlsx=False):
    
    T_1bike = pd.read_csv(path_dataT / '20211020_bike_UGGA_measurements1.csv', index_col='gps_time', parse_dates=['gps_time'])
    T_1c    = pd.read_csv(path_dataT / '20211020_car_G2401_measurements1.csv', index_col='gps_time', parse_dates=['gps_time'])
    T_2c    = pd.read_csv(path_dataT / '20211024_car_G2401_measurements1.csv', index_col='gps_time', parse_dates=['gps_time'])

    
    # =============================================================================
    #       20.10.21 - Bike
    # =============================================================================
    
    # Release Time
    r1_start    = datetime(2021,10,20,20,13)
    r4_finish   = datetime(2021,10,20,20,49)
    
    starttime   = r1_start - timedelta(minutes=3)
    endtime     = r4_finish + timedelta(minutes=3)
    
    T_1bike.index.names = ['Datetime']
    T_1bike.rename(columns={'ch4': 'CH4_LGR','lat': 'Latitude', 'lon':'Longitude', 'heading':'Heading'}, inplace=True)
    T_1bike = T_1bike[:-2]
    T_1bike = T_1bike.dropna(subset=['Latitude'])
    
    T_1bike = T_1bike.loc[:, ['CH4_LGR', 'Latitude', 'Longitude', 'Heading','wd_corr','ws_corr']]

    
    # --- GPS ----------------------------
    
    gps = calc_gps_T(T_1bike) 
    
    # --- Time correction (inlet delay) ----------------------------
    
    T_1bike = T_1bike.drop(['Latitude', 'Longitude'], axis=1)
    T_1bike.index = (T_1bike.index
                      - timedelta(seconds=30)) # account for inlet delay
    # merge gps with time shifted data (since gps is not affected by inlet no shift needed
    # but since Datetime index was shifted, merging is needed. Interpolation is
    # necessary since the index was not shifted by number of observations (e.g. 30 obs.),
    # but by a time (30s) which can lead to the case that there is no datetime index
    # anymore which fits to the index of the gps)
    T_1bike = merge_interpolate_left(T_1bike, gps,'Datetime')
    
    # --- Calculate CH4 elevation -------------------------------
    
    T_1bike['bg_LGR']      = T_1bike['CH4_LGR'].rolling('5min',center=True).quantile(0.10) 
    T_1bike['CH4_ele_LGR'] = (T_1bike['CH4_LGR'] - T_1bike['bg_LGR'])
    
    # ---  Save to CSV -------------------------------
    
    if writexlsx:
        T_1bike.to_csv(path_res / 'T_1bike_LGR.csv', index=True) # way faster than excel
    
    LGR_1b = T_1bike[starttime:endtime]
    
    
    # =============================================================================
    #       20.10.21 - Car
    # =============================================================================
    
    r1_start    = datetime(2021,10,20,20,11)
    r4_finish   = datetime(2021,10,20,20,49)

    starttime   = r1_start - timedelta(minutes=3)
    endtime     = r4_finish + timedelta(minutes=3)

    G2401_1c    = T_1c[starttime:endtime].copy(deep=True)

    G2401_1c.index.names = ['Datetime']
    G2401_1c.rename(columns={'ch4': 'CH4_G24','lat': 'Latitude', 'lon':'Longitude', 'heading':'Heading'}, inplace=True)

    # Filter the DataFrame to keep only the first occurrence of each number (remove duplicates)
    G2401_1c['FirstOccurrence'] = (G2401_1c['CH4_G24'] != G2401_1c['CH4_G24'].shift(1)).cumsum()
    G2401_1c = G2401_1c.drop_duplicates(subset = ['FirstOccurrence'], keep='first')

    G2401_1c = G2401_1c.loc[:, ['CH4_G24', 'Latitude', 'Longitude', 'Heading','wd_corr','ws_corr']]


    # --- GPS ----------------------------
    
    gps = calc_gps_T(G2401_1c)
        
    # --- Time correction (inlet delay) ----------------------------
    
    G2401_1c = G2401_1c.drop(['Latitude', 'Longitude'], axis=1)

    G2401_1c = pd.concat([G2401_1c, gps],axis=1, ignore_index=False)
    G2401_1c.index = pd.to_datetime(G2401_1c.index)
    

    # --- Calculate CH4 elevation -------------------------------
    
    G2401_1c['bg_G24']      = G2401_1c['CH4_G24'].rolling('5min',center=True).quantile(0.10) 
    G2401_1c['CH4_ele_G24'] = (G2401_1c['CH4_G24'] - G2401_1c['bg_G24'])
    
    # ---  Save to CSV -------------------------------

    if writexlsx:
        G2401_1c.to_csv(path_res / 'T_1car_G24.csv', index=True) # way faster than excel

    
    # =============================================================================
    #       24.10.21 - Car
    # =============================================================================

    # Release Time
    r1_start    = datetime(2021,10,24,13,47)
    r4_finish   = datetime(2021,10,24,15,00)

    starttime   = r1_start - timedelta(minutes=3)
    endtime     = r4_finish + timedelta(minutes=3)

    G2401_2c    = T_2c[starttime:endtime].copy(deep=True)

    G2401_2c.index.names = ['Datetime']
    G2401_2c.rename(columns={'ch4': 'CH4_G24','lat': 'Latitude', 'lon':'Longitude', 'heading':'Heading'}, inplace=True)

    # Filter the DataFrame to keep only the first occurrence of each number (remove duplicates)
    G2401_2c['FirstOccurrence'] = (G2401_2c['CH4_G24'] != G2401_2c['CH4_G24'].shift(1)).cumsum()
    G2401_2c = G2401_2c.drop_duplicates(subset = ['FirstOccurrence'], keep='first')

    G2401_2c = G2401_2c.loc[:, ['CH4_G24', 'Latitude', 'Longitude', 'Heading','wd_corr','ws_corr']]

    # --- GPS ----------------------------
    
    gps = calc_gps_T(G2401_2c)
    
    # --- Time correction (inlet delay) ----------------------------
    
    G2401_2c = G2401_2c.drop(['Latitude', 'Longitude'], axis=1) 
    G2401_2c = pd.concat([G2401_2c, gps],axis=1, ignore_index=False)
    G2401_2c.index = pd.to_datetime(G2401_2c.index) 

    # --- Calculate CH4 elevation -------------------------------
    
    G2401_2c['bg_G24']      = G2401_2c['CH4_G24'].rolling('5min',center=True).quantile(0.10) 
    G2401_2c['CH4_ele_G24'] = (G2401_2c['CH4_G24'] - G2401_2c['bg_G24'])
    
    # ---  Save to CSV -------------------------------

    if writexlsx:
        G2401_2c.to_csv(path_res / 'T_2car_G24.csv', index=True) # way faster than excel

    
    return LGR_1b, G2401_1c, G2401_2c




#%% London I

def read_and_preprocess_L1(path_data, path_res, writexlsx=False):

    
    # =============================================================================
    #       Day 2
    # =============================================================================
    
    datapath   = path_data / 'Day2/'
    
    df_prox  = []
    for file in os.listdir(datapath):
        if file.endswith(".csv"):
            df_prox_release_x = pd.read_csv(datapath / file, sep=",") #, usecols= cols_G23
            df_prox.append(df_prox_release_x)

    df_all        = pd.concat(df_prox).reset_index(drop=True)

    # Combine date and time into datetime
    df_all['Datetime'] = pd.to_datetime(df_all['DATE'] + ' ' + df_all['TIME_cor'], format='%d/%m/%Y %H:%M:%S')
    df_all.set_index('Datetime',inplace = True,drop=True)
    
    # Drop unnecessary columns and rows with Nans
    df_all = df_all.loc[:,['ALARM_STATUS','species','CH4_cal','GPS_ABS_LAT','GPS_ABS_LONG','Heading','Speed_ms-1',
                        'LGRdata','Run']]
    df_all = df_all.drop(pd.NaT)   
    #G2301 = G2301[G2301["species"]==3] # 3 is CH4
    
    # Rename Columns
    df_all.rename(columns={'CH4_cal':'CH4_G23','LGRdata':'CH4_LGR','GPS_ABS_LONG':'Longitude','GPS_ABS_LAT':'Latitude','Speed_ms-1':'Speed [m/s]'},inplace=True)
        
    split_index = df_all.columns.get_loc('CH4_LGR')  # LGR data start

    G2301_d2 = df_all.iloc[:, :split_index].copy(deep=True)  # Split data into two sets, one for G2301, one for LGR measurememnts
    LGR_d2 = df_all.iloc[:, split_index:].copy(deep=True)
    
    
    LGR_d2['Longitude'] = G2301_d2['Longitude']
    LGR_d2['Latitude'] = G2301_d2['Latitude']
    LGR_d2['Speed [m/s]'] = G2301_d2['Speed [m/s]']

    # Prevent having several measurements with same time stamp:
    G2301_d2 = G2301_d2[G2301_d2["species"]==3] 
    # Resetting the index to convert it to a regular column
    LGR_d2.reset_index(inplace=True)
    # Drop duplicate rows based on the index (time in this case)
    LGR_d2.drop_duplicates(subset='Datetime', keep='first', inplace=True)
    # Set the index back to the original index column
    LGR_d2.set_index('Datetime', inplace=True)

    # --- Calculate CH4 elevation -------------------------------
    
    G2301_d2.loc[:, 'bg_G23'] = G2301_d2.loc[:, 'CH4_G23'].rolling('5min', center=True).quantile(0.10)
    G2301_d2.loc[:, 'CH4_ele_G23'] = G2301_d2.loc[:, 'CH4_G23']-G2301_d2.loc[:, 'bg_G23']
    
    LGR_d2.loc[:, 'bg_LGR'] = LGR_d2.loc[:, 'CH4_LGR'].rolling('5min', center=True).quantile(0.10)
    LGR_d2.loc[:, 'CH4_ele_LGR'] = LGR_d2.loc[:, 'CH4_LGR']-LGR_d2.loc[:, 'bg_LGR']

    
    # --- Time correction (inlet delay) ----------------------------

    #print('     Performing time correction')
    # already done by using corrected time in datasheet. It was corrected for
    # Picarro and using the same corrected time aligns (more r less) also the 
    # LGR data
    LGR_d2.index = (LGR_d2.index - timedelta(seconds=2))
    

    # =============================================================================
    #       Day 3
    # =============================================================================

    datapath   = path_data / 'Day3/'

    df_prox  = []
    for file in os.listdir(datapath):
        if file.endswith(".csv"):
            df_prox_release_x = pd.read_csv(datapath / file, sep=",") #, usecols= cols_G23
            df_prox.append(df_prox_release_x)


    df_all        = pd.concat(df_prox).reset_index(drop=True)

    # Combine date and time into datetime
    df_all['Datetime'] = pd.to_datetime(df_all['DATE'] + ' ' + df_all['TIME_cor'], format='%d/%m/%Y %H:%M:%S')
    df_all.set_index('Datetime',inplace = True,drop=True)

    # Drop unnecessary columns and rows with Nans
    df_all = df_all.loc[:,['ALARM_STATUS','species','CH4_cal','GPS_ABS_LAT','GPS_ABS_LONG','Heading','Speed_ms-1',
                         'Run','LGRTime','LGRdata','LGRrun#','LiCOR_ppm','Run#']]
    df_all = df_all.drop(pd.NaT)   

    # Rename Columns
    df_all.rename(columns={'CH4_cal':'CH4_G23','LGRdata':'CH4_LGR','LiCOR_ppm':'CH4_Licor','GPS_ABS_LONG':'Longitude','GPS_ABS_LAT':'Latitude',
                           'Speed_ms-1':'Speed [m/s]','Run':'Run_G23','Run#':'Run_Licor','LGRrun#':'Run_LGR'},inplace=True)
    
    G2301_d3 = df_all.copy(deep=True)
    G2301_d3.drop(columns=['CH4_LGR','CH4_Licor','Run_Licor','Run_LGR','LGRTime'],inplace=True)

    LGR_d3 = df_all.copy(deep=True)
    LGR_d3.drop(columns=['species','CH4_G23','CH4_Licor','Run_Licor','Run_G23'],inplace=True)

    Licor_d3 = df_all.copy(deep=True)
    Licor_d3.drop(columns=['species','CH4_LGR','CH4_G23','Run_G23','Run_LGR','LGRTime'],inplace=True)


    # Prevent having several measurements with same time stamp:
    G2301_d3 = G2301_d3[G2301_d3["species"]==3] 
    # Resetting the index to convert it to a regular column
    LGR_d3.reset_index(inplace=True)
    Licor_d3.reset_index(inplace=True)
    # Drop duplicate rows based on the index (time in this case)
    LGR_d3.drop_duplicates(subset='Datetime', keep='first', inplace=True)
    Licor_d3.drop_duplicates(subset='Datetime', keep='first', inplace=True)
    # Set the index back to the original index column
    LGR_d3.set_index('Datetime', inplace=True)
    Licor_d3.set_index('Datetime', inplace=True)

    # --- Calculate CH4 elevation -------------------------------

    G2301_d3.loc[:, 'bg_G23'] = G2301_d3.loc[:, 'CH4_G23'].rolling('5min', center=True).quantile(0.10)
    G2301_d3.loc[:, 'CH4_ele_G23'] = G2301_d3.loc[:, 'CH4_G23']-G2301_d3.loc[:, 'bg_G23']

    LGR_d3.loc[:, 'bg_LGR'] = LGR_d3.loc[:, 'CH4_LGR'].rolling('5min', center=True).quantile(0.10)
    LGR_d3.loc[:, 'CH4_ele_LGR'] = LGR_d3.loc[:, 'CH4_LGR']-LGR_d3.loc[:, 'bg_LGR']

    Licor_d3.loc[:, 'bg_Licor'] = Licor_d3.loc[:, 'CH4_Licor'].rolling('5min', center=True).quantile(0.10)
    Licor_d3.loc[:, 'CH4_ele_Licor'] = Licor_d3.loc[:, 'CH4_Licor']-Licor_d3.loc[:, 'bg_Licor']
    
    
    # =============================================================================
    #       Day 4
    # =============================================================================

    datapath   = path_data / 'Day4/'

    df_prox  = []
    for file in os.listdir(datapath):
        if file.endswith(".csv"):
            df_prox_release_x = pd.read_csv(datapath / file, sep=",") #, usecols= cols_G23
            df_prox.append(df_prox_release_x)

    df_all        = pd.concat(df_prox).reset_index(drop=True)

    # Combine date and time into datetime
    df_all['Datetime'] = pd.to_datetime(df_all['DATE'] + ' ' + df_all['TIME_cor'], format='%d/%m/%Y %H:%M:%S')
    df_all.set_index('Datetime',inplace = True,drop=True)

    # Drop unnecessary columns and rows with Nans
    df_all = df_all.loc[:,['ALARM_STATUS','species','CH4_cal','GPS_ABS_LAT','GPS_ABS_LONG','Heading','Speed_ms-1',
                         'Run','LGRTime','LGRdata','LGRrun#','LiCOR_ppm','Run#']]
    df_all = df_all.drop(pd.NaT)   

    # Rename Columns
    df_all.rename(columns={'CH4_cal':'CH4_G23','LGRdata':'CH4_LGR','LiCOR_ppm':'CH4_Licor','GPS_ABS_LONG':'Longitude','GPS_ABS_LAT':'Latitude',
                           'Speed_ms-1':'Speed [m/s]','Run':'Run_G23','Run#':'Run_Licor','LGRrun#':'Run_LGR'},inplace=True)
    
    G2301_d4 = df_all.copy(deep=True)
    G2301_d4.drop(columns=['CH4_LGR','CH4_Licor','Run_Licor','Run_LGR','LGRTime'],inplace=True)

    LGR_d4 = df_all.copy(deep=True)
    LGR_d4.drop(columns=['species','CH4_G23','CH4_Licor','Run_Licor','Run_G23'],inplace=True)

    Licor_d4 = df_all.copy(deep=True)
    Licor_d4.drop(columns=['species','CH4_LGR','CH4_G23','Run_G23','Run_LGR','LGRTime'],inplace=True)


    # Prevent having several measurements with same time stamp:
    G2301_d4 = G2301_d4[G2301_d4["species"]==3] 
    # Resetting the index to convert it to a regular column
    LGR_d4.reset_index(inplace=True)
    Licor_d4.reset_index(inplace=True)
    # Drop duplicate rows based on the index (time in this case)
    LGR_d4.drop_duplicates(subset='Datetime', keep='first', inplace=True)
    Licor_d4.drop_duplicates(subset='Datetime', keep='first', inplace=True)
    # If you want to set the index back to the original index column
    LGR_d4.set_index('Datetime', inplace=True)
    Licor_d4.set_index('Datetime', inplace=True)

    # --- Calculate CH4 elevation -------------------------------

    G2301_d4.loc[:, 'bg_G23'] = G2301_d4.loc[:, 'CH4_G23'].rolling('5min', center=True).quantile(0.10)
    G2301_d4.loc[:, 'CH4_ele_G23'] = G2301_d4.loc[:, 'CH4_G23']-G2301_d4.loc[:, 'bg_G23']

    LGR_d4.loc[:, 'bg_LGR'] = LGR_d4.loc[:, 'CH4_LGR'].rolling('5min', center=True).quantile(0.10)
    LGR_d4.loc[:, 'CH4_ele_LGR'] = LGR_d4.loc[:, 'CH4_LGR']-LGR_d4.loc[:, 'bg_LGR']

    Licor_d4.loc[:, 'bg_Licor'] = Licor_d4.loc[:, 'CH4_Licor'].rolling('5min', center=True).quantile(0.10)
    Licor_d4.loc[:, 'CH4_ele_Licor'] = Licor_d4.loc[:, 'CH4_Licor']-Licor_d4.loc[:, 'bg_Licor']
    
    
    # =============================================================================
    #       Day 5
    # =============================================================================
    
    
    datapath   = path_data / 'Day5/'
    
    df_prox  = []
    for file in os.listdir(datapath):
        if file.endswith(".csv"):
            df_prox_release_x = pd.read_csv(datapath / file, sep=",") 
            df_prox.append(df_prox_release_x)


    df_all        = pd.concat(df_prox).reset_index(drop=True)

    # Combine date and time into datetime
    df_all['Datetime'] = pd.to_datetime(df_all['DATE'] + ' ' + df_all['TIME_cor'], format='%d/%m/%Y %H:%M:%S')
    df_all.set_index('Datetime',inplace = True,drop=True)
    
    # Drop unnecessary columns and rows with Nans 
    G2301_d5 = df_all.loc[:,['ALARM_STATUS','species','CH4_cal','GPS_ABS_LAT','GPS_ABS_LONG', 'Heading', 
                           'Distance','Speed_ms-1','Run','Inlet height Height']]
    G2301_d5 = G2301_d5.drop(pd.NaT)
    
    # Rename Columns
    G2301_d5.rename(columns={'CH4_cal':'CH4_G23','GPS_ABS_LONG':'Longitude','GPS_ABS_LAT':'Latitude',
                             'Speed_ms-1':'Speed [m/s]'},inplace=True)
    
    # Prevent having several measurements with same time stamp:
    G2301_d5 = G2301_d5[G2301_d5["species"]==3]
    G2301_d5.index = pd.to_datetime(G2301_d5.index)
    G2301_d5 = G2301_d5.sort_index()
    G2301_d5 = G2301_d5.loc[~G2301_d5.index.duplicated(keep='first')] # drop duplicate rows

    
    # --- Calculate CH4 elevation -------------------------------
    G2301_d5.loc[:, 'bg_G23'] = G2301_d5.loc[:, 'CH4_G23'].rolling('5min', center=True).quantile(0.10)
    G2301_d5.loc[:, 'CH4_ele_G23'] = G2301_d5.loc[:, 'CH4_G23']-G2301_d5.loc[:, 'bg_G23']
    
    
    
    
    # =============================================================================
    #       Save to CSV
    # =============================================================================
    if writexlsx:
        writer  = pd.ExcelWriter(path_res / 'L1_G23andLGRandLicor.xlsx', engine = 'xlsxwriter')
        G2301_d2.to_excel(writer, sheet_name='D2_G2301')
        LGR_d2.to_excel(writer, sheet_name='D2_LGR')
        G2301_d3.to_excel(writer, sheet_name='D3_G2301')
        Licor_d3.to_excel(writer, sheet_name='D3_Licor')
        G2301_d4.to_excel(writer, sheet_name='D4_G2301')
        Licor_d4.to_excel(writer, sheet_name='D4_Licor')
        G2301_d5.to_excel(writer, sheet_name='D5_G2301')
        writer.book.close()
    
    
    return LGR_d2, G2301_d2, LGR_d3, G2301_d3, Licor_d3, LGR_d4, G2301_d4, Licor_d4, G2301_d5





#%% London II

def read_and_preprocess_L2(path_data, path_res, writexlsx=False):

    
    # =============================================================================
    #       Day 1
    # =============================================================================
    
    L2_Licor_d1 = pd.read_csv(path_data / '20240513_Licor_measurements1.csv',
                              delimiter=';')
    
    # Combine date and time into datetime
    L2_Licor_d1['Datetime'] = pd.to_datetime(L2_Licor_d1['DATE'], dayfirst=True) + pd.to_timedelta(L2_Licor_d1['Time_Licor_corrected'].astype(str))
    L2_Licor_d1.set_index('Datetime',inplace = True,drop=True)

    # Drop unnecessary columns and rows with Nans
    L2_Licor_d1 = L2_Licor_d1.loc[:,['CH4_licor/ppm','LAT_licor','LON_licor','Speed m/s','Direction / deg']]

    # Rename Columns
    L2_Licor_d1.rename(columns={'CH4_licor/ppm':'CH4_Licor','LAT_licor':'Latitude','LON_licor':'Longitude',
                           'Speed m/s':'Speed [m/s]','Direction / deg':'Heading [deg]'},inplace=True)


    # --- Calculate CH4 elevation -------------------------------

    L2_Licor_d1.loc[:, 'bg_Licor'] = L2_Licor_d1.loc[:, 'CH4_Licor'].rolling('5min', center=True).quantile(0.10)
    L2_Licor_d1.loc[:, 'CH4_ele_Licor'] = L2_Licor_d1.loc[:, 'CH4_Licor']-L2_Licor_d1.loc[:, 'bg_Licor']

    L2_Licor_d1.name = 'Licor'
    L2_Licor_d1 = delete_duplicate_indices(L2_Licor_d1,'Datetime')

    # --- Save -------------------------------

    if writexlsx:
        L2_Licor_d1.to_csv(path_res / 'L2_day1_Licor.csv', index=True)
      
    # =============================================================================
    #       Day 2
    # =============================================================================
    
    L2_Licor_d2 = pd.read_csv(path_data / '20240514_Licor_measurements1.csv',
                              delimiter=';')

    starttime       = pd.to_datetime('2024-05-14 09:08:00')
    endtime         = pd.to_datetime('2024-05-14 11:24:00')
    
    # Combine date and time into datetime
    L2_Licor_d2['Datetime'] = pd.to_datetime(L2_Licor_d2['DATE'], dayfirst=True) + pd.to_timedelta(L2_Licor_d2['Time_Licor_corrected'].astype(str))
    L2_Licor_d2.set_index('Datetime',inplace = True,drop=True)
    L2_Licor_d2 = L2_Licor_d2.loc[starttime:endtime]

    # Drop unnecessary columns and rows with Nans
    L2_Licor_d2 = L2_Licor_d2.loc[:,['CH4_licor/ppm','LAT_licor','LON_licor','Speed m/s','Direction / deg']]

    # Rename Columns
    L2_Licor_d2.rename(columns={'CH4_licor/ppm':'CH4_Licor','LAT_licor':'Latitude','LON_licor':'Longitude',
                           'Speed m/s':'Speed [m/s]','Direction / deg':'Heading [deg]'},inplace=True)


    # --- Calculate CH4 elevation -------------------------------

    L2_Licor_d2.loc[:, 'bg_Licor'] = L2_Licor_d2.loc[:, 'CH4_Licor'].rolling('5min', center=True).quantile(0.10)
    L2_Licor_d2.loc[:, 'CH4_ele_Licor'] = L2_Licor_d2.loc[:, 'CH4_Licor']-L2_Licor_d2.loc[:, 'bg_Licor']

    L2_Licor_d2.name = 'Licor'
    L2_Licor_d2 = delete_duplicate_indices(L2_Licor_d2,'Datetime')

    # --- Save -------------------------------

    if writexlsx:
        L2_Licor_d2.to_csv(path_res / 'L2_day2_Licor.csv', index=True)  
    
    return L2_Licor_d1, L2_Licor_d2


#%% Rotterdam


def read_and_preprocess_R(path_data, path_res, bg=None, writexlsx=False):
    
    if not bg:
        bg=0.1
        
    starttime       = pd.to_datetime('2022-09-06 06:50:00')
    endtime         = pd.to_datetime('2022-09-06 12:59:00')

    morning_start   = pd.to_datetime('2022-09-06 07:05:00')
    morning_end     = pd.to_datetime('2022-09-06 10:44:00')

    afternoon_start = pd.to_datetime("2022-09-06 11:05:00")
    afternoon_end = pd.to_datetime('2022-09-06 12:26:00')
    
    # All directory paths

    path_G2301   = path_data / 'G2301_Picarro/'
    path_G4302   = path_data / 'G4302_Picarro/'
    path_aeris   = path_data / 'MiraUltra_Aeris/'
    path_miro    = path_data / 'MGA10_Miro/'
    path_aero    = path_data / 'TILDAS_Aerodyne/'
    path_gps     = path_data / 'GPS_UUAQ/'
    
    
    
    
    # =============================================================================
    #       G2301 & G4302 - UU (&TNO) car
    # =============================================================================
    
    # G4302 gets transfered in the afternoon from the UU car to the TNO car
        
    cols_G23    = ['DATE','TIME','CH4_dry_sync',
                   'species','ALARM_STATUS'] #'CO2_sync','CH4_sync','H2O_sync'
    cols_G43    = ['DATE','TIME','CH4_dry','ALARM_STATUS']

    G2301_prox  = []
    for file in os.listdir(path_G2301):
        if file.endswith(".dat"):
            df_prox = pd.read_csv(path_G2301 / file, sep="\s+") #, usecols= cols_G23
            G2301_prox.append(df_prox)

    G4302_prox  = []
    for file in os.listdir(path_G4302):
        if file.endswith(".dat"):
            df_prox = pd.read_csv(path_G4302 / file, sep="\s+", usecols= cols_G43)
            G4302_prox.append(df_prox)

    G2301        = pd.concat(G2301_prox).reset_index(drop=True)
    G4302        = pd.concat(G4302_prox, ignore_index=True, sort=False)

    # Combine date and time into datetime
    G4302['Datetime'] = pd.to_datetime(G4302['DATE'] + ' ' + G4302['TIME'])
    G2301['Datetime'] = pd.to_datetime(G2301['DATE'] + ' ' + G2301['TIME'])

    G4302 = G4302.drop(['DATE','TIME'],axis=1)
    G2301 = G2301.drop(['DATE','TIME'],axis=1)
    G2301 = G2301[G2301["species"]==1]
    
    G2301.rename(columns={'CH4_dry_sync':'CH4_dry'},inplace=True)
    G2301.rename(columns={'ALARM_STATUS':'ALARM_STATUS_G23'},inplace=True)
    G4302.rename(columns={'ALARM_STATUS':'ALARM_STATUS_G43'},inplace=True)

    
    G4302['CH4_G43']     = calibrate(G4302['CH4_dry'], 'G43', 'CH4')
    G2301['CH4_G23']     = calibrate(G2301['CH4_dry'], 'G23', 'CH4')
    
    G2301.set_index('Datetime',inplace = True,drop=True)
    G4302.set_index('Datetime',inplace = True,drop=True)

    
    # --- Calculate CH4 elevation -------------------------------
    
    G2301['bg_G23'] = G2301['CH4_G23'].rolling('5min',center=True).quantile(bg)
    G2301['CH4_ele_G23'] = G2301['CH4_G23'] - G2301['bg_G23']
    
    G4302['bg_G43'] = G4302['CH4_G43'].rolling('5min',center=True).quantile(bg)
    G4302['CH4_ele_G43'] = G4302['CH4_G43'] - G4302['bg_G43']
    
    # --- Time correction (inlet delay) ----------------------------
    
    G4302.index = (G4302.index
                      - timedelta(seconds=16.9)) # match with TNO
    G2301.index = (G2301.index
                      + timedelta(seconds=44)       # clock correction
                      - timedelta(seconds=2)        # inlet delay
                      - timedelta(seconds=5.29))    # matching peaks with G43
    
    
    
    # =============================================================================
    #       Aeris - UU (&TNO) car
    # =============================================================================
    
    # Aeris gets transfered in the afternoon from the UU car to the TNO car
    
    
    aerfile = '220906_072814_MiraUltra_measurements1.txt'
    aeris = pd.read_csv(path_aeris / aerfile, sep=',').add_prefix('aer_')
    aeris.rename(columns={'aer_Time Stamp':'Datetime', 'aer_CH4 (ppm)':'CH4_aeris'},inplace=True)
    aeris = aeris.set_index('Datetime', drop = True)
    aeris.index = pd.to_datetime(aeris.index)

    aeris['CH4_aeris']     = calibrate(aeris['CH4_aeris'], 'aer', 'CH4')


    # --- Calculate CH4 elevation -------------------------------
    
    aeris['bg_aeris'] = aeris['CH4_aeris'].rolling('5min',center=True).quantile(bg)
    aeris['CH4_ele_aeris'] = aeris['CH4_aeris'] - aeris['bg_aeris']
    
     
    # --- Time correction (inlet delay) ----------------------------
    
    aeris.index = (aeris.index
                      - timedelta(hours=1, minutes=1, seconds=8) # clock correction
                      - timedelta(seconds=2) # inlet delay
                      - timedelta(seconds=1.68)) # matching peaks with G43


    # =============================================================================
    #       GPS - UU car
    # =============================================================================
    
    
    gps = pd.read_csv(path_gps / '20220906_GPS.csv',
                      delimiter=';',
                      header = 0,
                      names=['Datetime', 'Latitude', 'Longitude', 'speed']) #usecols = [0,1,2,3]

    gps.Datetime = pd.to_datetime(gps.Datetime)
    gps.set_index('Datetime',inplace=True)
    gps = gps.loc['2022-09-06 07:00:02':]
    gps.loc[:,'speed'] = gps.loc[:,'speed']/3.6
    gps.rename(columns={'speed':'Speed [m/s]'},inplace=True)
    
    # G2301_gps = merge_interpolate(G2301, gps, col='Datetime') # I used this for MA
    # G4302_gps = merge_interpolate(G4302, gps, col='Datetime')
    # aeris_gps = merge_interpolate(aeris, gps, col='Datetime')
    
    G2301_gps = merge_with_gps(G2301, gps) # consistent with Utrecht?
    G4302_gps = merge_with_gps(G4302, gps)
    aeris_gps = merge_with_gps(aeris, gps)
    
    
    # =============================================================================
    #   Miro - TNO car
    # =============================================================================
    
    miro = pd.read_csv(path_miro / '20220906_MGA10_measurements1.csv', index_col=(0),sep=';',
                       usecols=[0,8,19,20,21,22],dtype={'comment':'str','geometry':'str',})
    
    miro.index = pd.to_datetime(miro.index,dayfirst=True)
    miro.rename(columns={'6_CH4':'CH4_miro','speed':'Speed [m/s]','distance':'Distance'},inplace=True)
    miro       = miro.rename_axis('Datetime') # Rename the index column
    miro['CH4_miro']                = miro['CH4_miro']/1000 #convert to ppm
    miro['bg_miro']                 = miro['CH4_miro'].rolling('5min',center=True).quantile(bg)
    miro['CH4_ele_miro']            = miro['CH4_miro'] - miro['bg_miro']
    miro[['Point','Lat_Lon']]       = miro.geometry.str.split(pat=' ',n=1,expand=True) #chenged 2025
    miro[['Longitude','Latitude']]  = miro.Lat_Lon.str.split(pat=' ',n=1,expand=True)    
    miro['Longitude'] = pd.to_numeric(miro['Longitude'].str[1:],errors='coerce')
    miro['Latitude']  = pd.to_numeric(miro['Latitude'].str[:-1],errors='coerce')
    miro.drop(['geometry','Point','Lat_Lon'],axis=1,inplace=True)    
    
    miro['Speed [m/s]'].where(miro['Speed [m/s]']>0.0,np.nan,inplace=True)
    miro.loc[:,'Speed [m/s]'] = (miro.loc[:,'Speed [m/s]'].interpolate()/3.6)
    
    column_name = miro.columns[3]
    miro.drop(column_name,axis=1, inplace=True) # deletes column with text, otherwise not possible to take mean
    miro = miro.groupby(miro.index,axis=0).mean(numeric_only=True)
    
    
    # =============================================================================
    #   Aerodyne - TNO car
    # =============================================================================
    

    aerodyne = pd.read_csv(path_aero / '20220906_Tildas_measurements1.csv', sep = ';',index_col=0,
                           usecols=['datetime', '2_CH4','distance', 'speed',
                                    'comment', 'geometry','wsp_processed', 'wdir_processed'])
    
    aerodyne.index = pd.to_datetime(aerodyne.index,dayfirst=True)
    aerodyne.rename(columns={'2_CH4':'CH4_aero','speed':'Speed [m/s]','distance':'Distance'},inplace=True)
    aerodyne = aerodyne.rename_axis('Datetime') # Rename the index column
    aerodyne['CH4_aero']     = aerodyne['CH4_aero']/1000
    aerodyne['bg_aero']      = aerodyne['CH4_aero'].rolling('5min', center=True).quantile(bg)
    aerodyne['CH4_ele_aero'] = aerodyne['CH4_aero'] - aerodyne['bg_aero']
    
    aerodyne[['Point','Lat_Lon']]       = aerodyne.geometry.str.split(pat=' ',n=1,expand=True)
    aerodyne[['Longitude','Latitude']]  = aerodyne.Lat_Lon.str.split(pat=' ',n=1,expand=True)    
    aerodyne['Longitude']               = pd.to_numeric(aerodyne['Longitude'].str[1:],errors='coerce')
    aerodyne['Latitude']                = pd.to_numeric(aerodyne['Latitude'].str[:-1],errors='coerce')
    aerodyne.drop(['geometry','Point','Lat_Lon'], axis=1,inplace=True)    
    aerodyne.loc[:,'Speed [m/s]']       = (aerodyne.loc[:,'Speed [m/s]'].interpolate())
    aerodyne                            = aerodyne.groupby(aerodyne.index, axis=0).mean(numeric_only=True)
    
    # =============================================================================
    #       Change GPS of G4302&Aeris in Afternoon to TNO GPS
    # =============================================================================
    
    
    # G4302 -------------------------------------------
    
    gps = miro.loc[afternoon_start:afternoon_end, ['Longitude','Latitude','Speed [m/s]']].copy()
    gps.rename(columns={'Longitude': 'Longitude_miro', 'Latitude': 'Latitude_miro','Speed [m/s]': 'Speed_miro'}, inplace=True)
    
    merged_df = pd.merge(G4302_gps.loc[afternoon_start:afternoon_end], gps, left_index=True, right_index=True, how='outer')
    merged_df.interpolate(method='linear', inplace=True)
    merged_df.drop(columns=['Longitude', 'Latitude','Speed [m/s]'], inplace=True)
    merged_df.rename(columns={'Longitude_miro': 'Longitude', 'Latitude_miro': 'Latitude', 'Speed_miro':'Speed [m/s]'}, inplace=True)
    G4302_gps.loc[afternoon_start:afternoon_end,['Longitude', 'Latitude', 'Speed [m/s]']] = merged_df.loc[afternoon_start:afternoon_end,['Longitude', 'Latitude','Speed [m/s]']]
    
    # Aeris -------------------------------------------
    
    gps = miro.loc[afternoon_start:afternoon_end, ['Longitude','Latitude','Speed [m/s]']].copy()
    gps.rename(columns={'Longitude': 'Longitude_miro', 'Latitude': 'Latitude_miro','Speed [m/s]': 'Speed_miro'}, inplace=True)
    
    merged_df = pd.merge(aeris_gps.loc[afternoon_start:afternoon_end], gps, left_index=True, right_index=True, how='outer')
    merged_df.interpolate(method='linear', inplace=True)
    merged_df.drop(columns=['Longitude', 'Latitude','Speed [m/s]'], inplace=True)
    merged_df.rename(columns={'Longitude_miro': 'Longitude', 'Latitude_miro': 'Latitude', 'Speed_miro':'Speed [m/s]'}, inplace=True)
    aeris_gps.loc[afternoon_start:afternoon_end,['Longitude', 'Latitude', 'Speed [m/s]']] = merged_df.loc[afternoon_start:afternoon_end,['Longitude', 'Latitude','Speed [m/s]']]
      

    # =============================================================================
    #   Save to CSV
    # =============================================================================
    
    miro      = miro.loc[starttime - timedelta(minutes=30):endtime + timedelta(minutes=30)]
    aerodyne  = aerodyne.loc[starttime - timedelta(minutes=30):endtime + timedelta(minutes=30)]
    G4302_gps = G4302_gps.loc[starttime - timedelta(minutes=30):endtime + timedelta(minutes=30)]
    G2301_gps = G2301_gps.loc[starttime - timedelta(minutes=30):endtime + timedelta(minutes=30)]
    aeris_gps = aeris_gps.loc[starttime - timedelta(minutes=30):endtime + timedelta(minutes=30)]
    
    # Print data into csv
    if writexlsx:
        G4302_gps.to_csv(path_res / 'R_G4302.csv', index=True)
        G2301_gps.to_csv(path_res / 'R_G2301.csv', index=True)
        aeris_gps.to_csv(path_res / 'R_aeris.csv', index=True)
        miro.to_csv(path_res / 'R_miro.csv', index=True)
        aerodyne.to_csv(path_res / 'R_aerodyne.csv', index=True) 
        
    return G4302_gps, G2301_gps, aeris_gps, miro, aerodyne
    
    


#%% End Script



