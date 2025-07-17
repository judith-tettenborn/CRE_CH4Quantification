Code of the preprint Tettenborn et al. (2025) "Improving Consistency in Methane Emission Quantification from the Natural Gas Distribution System across Measurement Devices"

Script_1_Preprocessing.py
- reads in data
- preprocessing
- finding peaks

Script_2_PreAnalysis.py
- calculates Spatial Peak Area and main features important for later analysis

Script_3_MainAnalysis.py
... not yet uploaded


Functions used in the main scripts can be found in the different folders:
preprocessing/read_in_data -> used in Script 1
helper_functions/
constants -> defines relevant constants that are used in different scripts
dataset_dicts -> loads preprocessed data (output from Script 1) and stores them in different dictionaries together with meta-data - used in script 2 and 3
utils -> functions used in different scripts
