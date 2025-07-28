
# CRE_CH4Quantification

Associated python code of the publication:
Tettenborn, J., Zavala-Araiza, D., Stroeken, D., Maazallahi, H., van der Veen, C., Hensen, A., Velzeboer, I., van den Bulk, P., Vogel, F., Gillespie, L., Ars, S., France, J., Lowry, D., Fisher, R., and Röckmann, T.: **Improving Consistency in Methane Emission Quantification from the Natural Gas Distribution System across Measurement Devices**, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2024-3620, 2025.

This project analyzes methane (CH₄) emissions data and investigates CH₄ measurement characteristics. 

It is based on datasets of mobile measurements of ambient methane collected during a controlled release of methane. Methane was released from a stationary source while a vehicle repeatedly drove past the release point, transecting the resulting methane plume multiple times. 
The objective of this project was to evaluate the capability of mobile platforms to detect methane plumes and to examine the relationship between methane release rates (in L/min) and key plume metrics, such as the maximum measured mole fraction and the integrated methane enhancement during individual plume transects.

The code supports data preprocessing and peak detection. It contains the linear regression model fitted to combined dataset of the various controlled release experiments (CRE). It further contains several (statistical) analysis and result visualizations.

Tettenborn, J., Stroeken, D., & Röckmann, T. (2025). Controlled Release Experiments - CH4 Quantification. In EGUsphere (Version V1). Zenodo. https://doi.org/10.5281/zenodo.16503524


---

## 📁 Project Structure

src/  
├── Script_1_Preprocessing.py   # Load and preprocess data & find CH4 enhancements (peaks)  
├── Script_2_PreAnalysis.py   # Analyse the quality checked peaks  
├── Script_3_MainAnalysis.py   # Analysis Plots  
├── Stats_\*.py   # Various statistical analysis modules  
├── requirements.txt   # Python package dependencies  
│  
├── helper_functions/   # General helper functions and constants  
│ ├── constants.py  
│ ├── dataset_dicts.py  
│ ├── utils.py  
│  
├── peak_analysis/   # Peak finding and analysis functions  
│ └── find_analyze_peaks.py  
│  
├── plotting/   # Plotting routines for figures  
│ └── general_plots.py  
│  
├── preprocessing/   # Data input and preprocessing  
│ └── read_in_data.py  
│  
└── stats_analysis/   # Statistical analysis methods  
│ └── stats_functions.py


The folder structure (of the project including data, results) required for the code to work can be found in: folder_structure.txt


---

## 🛠️ Installation

1. It is recommended to use a virtual environment for installing dependencies. See [this guide](https://docs.python.org/3/tutorial/venv.html) for instructions on creating and managing Python environments.
```bash
# Create and activate a virtual environment (optional but recommended)
conda create -n ch4_env python=3.10.13
conda activate ch4_env
```

2. Get CRE_CH4Quantification code

2.1 Via GitHub:

```bash
# Clone the repository
git clone https://github.com/judith-tettenborn/CRE_CH4Quantification.git
```

2.2 Via Zenodo

Download .zip file

# 📦Dependencies

## Python package dependencies 💻

Main libraries include: `pandas`, `numpy`, `matplotlib`, `scipy`,  `pathlib`


Option 1: Using conda (recommended)
See `environment.yml`

```bash
conda env create -f environment.yml
```


Option 2: Using pip
See `requirements.txt`. 

```bash
# Install dependencies
pip install -r src/requirements.txt
```


## Data dependencies 📊

The raw data used can be found here:
France, J. L.; Lowry, D.; Fisher, R., 2025, "Methane Controlled Release Experiments in Bedford (2019)", https://doi.org/10.34894/Q1NALJ, DataverseNL, V1
France, J. L.; Lowry, D.; Fisher, R., 2025, "Methane Controlled Release Experiments in Bedford (2024)", https://doi.org/10.34894/6WEHES, DataverseNL, V1
Gillespie, L. D.; Ars, S.; Vogel, F., 2025, "Methane Controlled Release Experiment in Toronto (2021)", https://doi.org/10.34894/YVNJN2, DataverseNL, V1
Maazallahi, H.; Hensen, A.; Stroeken, D.; van den Bulk, P.; van der Veen, C.; Velzeboer, I.; Röckmann, T., 2025, "Methane Controlled Release Experiments in Rotterdam (2022)", https://doi.org/10.34894/HD0PTF, DataverseNL, V1
Maazallahi, H.; Stroeken, D.; van der Veen, C.; Röckmann, T., 2025, "Methane Controlled Release Experiment in Utrecht (2022)", https://doi.org/10.34894/TSTKQA, DataverseNL, DRAFT VERSION
Tettenborn, J.; Paglini, R.; Wooley Maisch, C.; Röckmann, T., 2025, "Methane Controlled Release Experiment in Utrecht (2024)", https://doi.org/10.34894/A9IT0K, DataverseNL, V1 

The processed data used can be found here:
Tettenborn, J., 2025, "Methane Controlled Release Experiments - Supplementary Data to Tettenborn et al. (2025)", https://doi.org/10.34894/OP0OFA, DataverseNL, V1

---
## ⚙️ Functionality

#### Script_1_Preprocessing.py
Main script for (pre-)processing data.
- reads in data and does some preprocessing
- using scipy.signal.find_peaks CH4 enhancements are detected
- the timestamps, enhancements, etc. of the peaks found are saved into an excel
  file (one file per (sub)-experiment with different excel sheets per instrument)
- overviewplots of the timeseries and peaks detected can be created (optionally, 
  one per instrument)
- plots of each detected peak can be created for the following quality check step 
  (optional, up to several hundreds per instrument)

#### Script_2_PreAnalysis.py
- calculates spatial peak area and main features important for later analysis
- data are saved in several ways:
    1. Files containing all detected peaks per CRE, with area values calculated. 
    Each instrument has separate columns for Max and Area values.
       total_peaks_T1b -> "T_TOTAL_PEAKS_1b.csv"           - for each experiment
       total_peaks_all -> "RU2T2L4L2_TOTAL_PEAKS.csv"      - all CRE combined
       
    2. Files with all instruments merged: measurements are consolidated into two 
    columns: 'Max' and 'Area'.
       df_R_comb       -> "R_comb_final.csv"               - for each experiment
       df_RU2T2L4L2    -> "RU2T2L4L2_TOTAL_PEAKS_comb.csv" - all CRE combined
- plots of each quality-checked peak can be created (optional, up to several 
  hundreds per CRE)

#### Script_3_MainAnalysis.py 
This script creates the main figures (Plot 1 & Plot 2) included in the publication 
Tettenborn et al. (2025).
**Plot1:** Visualizes a comparison of peak maximum and peak area between different instruments.
**Plot2:** Visualizes all peak measurements (peak area/peak maximum versus release rate), 
including linear regression fits.

#### Stats_Distribution.py
Normality (condition to use linear regression) of the data are investigated, using:
    1. Shapiro-Wilk Test and Lilliefors Test
    2. Histogram Plots
    3. Quantile-Quantile Plots

#### Stats_Categorization.py
1. Categorize emission rates (the "true release rate" and the calculated ones)
    into emission categories
2. Visualize the categorization success using Sankey plots

#### Stats_MonteCarlo.py
Performs a Monte Carlo simulation to assess the variation in estimated emission rates under 
a different number of detections, following the analysis from Luetschwager et al. (2021) (https://doi.org/10.1525/elementa.2020.00143).

#### Stats_Categorization_CombineTransects.py
Uses the output of the Monte Carlo analysis. Similar to the script "Stats_Categorization", the emission rate estimations are classified into emission categories and the categorization success is visualized in Sankey plots.


---
## 🚀 Usage

Run the scripts in the following order:

1. **Script_1_Preprocessing.py**
2. Do the manual quality check on the output files of Script_1
   -> one column per instrument must be added to the file containing 0 (not valid) or 1 (valid)
   -> column Peakstart_QC and Peakend_QC with (if necessary) corrected start and end times of the peak
3. **Script_2_PreAnalysis.py**      
4. **Script_3_MainAnalysis.py**        


---
## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

You are free to use, modify, and distribute this code, including for commercial purposes, provided that:

- Proper attribution is given to the original authors.
- The original license and copyright notice are included.


---

## 📬 Contact

Correspondence: Prof. Dr. Thomas Röckmann (t.roeckmann@uu.nl) 

Code author: Judith Tettenborn, based on work of Daan Stroeken
