"""
constants.py

This file contains all the constant values used throughout the project.
"""

# General Configurations
APP_NAME = "IAMEE"
VERSION = "0.0b"

# # File Paths
DATA_DIR = "data/"
OUTPUT_DIR = DATA_DIR + "outputs/"
INPUT_DIR = DATA_DIR + "inputs/"
PROCESSED_DIR = DATA_DIR + "processed/"
DATABASE_DIR = "database/"
# LOG_FILE = "logs/app.log"


# File Names/Paths
DATABASE_FILE = DATABASE_DIR + "AR6_Scenarios_Database_World_v1.1.csv"
REGIONAL_DATABASE_FILE = DATABASE_DIR + "AR6_Scenarios_Database_R10_regions_v1.1.csv"
META_FILE = DATABASE_DIR + "meta_data.csv"
BASELINE_SCENARIOS_FILEPATH = INPUT_DIR + "baselines"
BASELINE_DATA_FILEPATH = INPUT_DIR + "baseline_data"
ISO3C_REGIONS = INPUT_DIR + 'iso3c_regions.csv' #https://github.com/setupelz/regioniso3c/blob/main/iso3c_region_mapping_20240319.csv    

# IAM Data constants
CATEGORIES_ALL = ['C1', 'C2', 'C3', 'C4', 'C5','C6', 'C7', 'C8']
CATEGORIES_DEFAULT = CATEGORIES_ALL[:2]

# probably remove?
C1AR10_SCENARIOS = ['EN_NPi2020_300f', 'SSP1-DACCS-1p9-3pctHR', 'SSP1-noDACCS-1p9-3pctHR', 'SSP4-noDACCS-1p9-3pctHR', 
                'SSP1_SPA1_19I_D_LB', 'SSP1_SPA1_19I_LIRE_LB', 'SSP1_SPA1_19I_RE_LB', 'SSP2_SPA2_19I_LI', 
                'CEMICS_GDPgrowth_1p5', 'CEMICS_HotellingConst_1p5', 'CEMICS_Linear_1p5', 
                'LeastTotalCost_LTC_brkLR15_SSP1_P50', 'R2p1_SSP1-PkBudg900', 'R2p1_SSP5-PkBudg900', 
                'CD-LINKS_NPi2020_400', 'PEP_1p5C_full_eff', 'PEP_1p5C_red_eff', 'CEMICS_SSP1-1p5C-fullCDR',
                    'EN_NPi2020_200f', 'EN_NPi2020_400f', 'SusDev_SDP-PkBudg1000', 'SusDev_SSP1-PkBudg900', 
                    'DeepElec_SSP2_def_Budg900', 'DISCRATE_cb400_cdrno_dr5p', 'EN_NPi2020_450f', 'EN_NPi2020_500f']

PRESENT_WARMING = 1.25
REMAINING_2030_BUDGET = 250000 # in MtCO2

# Framework Constants
FRAMEWORK_MANDATORY_VARIABLES = ['Emissions|CO2',
                                 'Investment|Energy Supply',
                                 'Capacity|Electricity|Wind', 
                                 'Capacity|Electricity|Solar|PV', 
                                 'Final Energy', 
                                 'Final Energy|Electricity', 
                                 'Population',
                                 'Primary Energy|Coal',
                                 'Primary Energy|Oil', 
                                 'Primary Energy|Gas', 
                                 'Primary Energy|Nuclear', 
                                 'Primary Energy|Biomass',
                                 'Primary Energy|Non-Biomass Renewables',
                                 'Carbon Sequestration|CCS|Biomass',
                                 'GDP|MER', 
                                 'Land Cover', 
                                 'Carbon Sequestration|Land Use',
                                 'Price|Secondary Energy|Electricity',
                                 'Food Demand|Crops', 
                                 'Food Demand|Livestock', 
                                 'Primary Energy|Fossil|w/ CCS',
                                 'Primary Energy|Fossil|w/o CCS', 
                                 'Land Cover|Forest|Natural Forest']

FRAMEWORK_VARIABLES = FRAMEWORK_MANDATORY_VARIABLES + ['Carbon Sequestration|Direct Air Capture']


CDR_VARIABLES = ['Carbon Sequestration|Direct Air Capture', 'Carbon Sequestration|Land Use','Carbon Sequestration|CCS|Biomass']

ENERGY_VARIABLES = ['Primary Energy|Coal','Primary Energy|Oil',
                    'Primary Energy|Gas', 'Primary Energy|Nuclear',
                    'Primary Energy|Biomass', 'Primary Energy|Non-Biomass Renewables']

LOW_CARBON_ENERGY_VARIABLES = ['Primary Energy|Nuclear', 'Primary Energy|Biomass', 
                                   'Primary Energy|Non-Biomass Renewables', 'Primary Energy|Fossil|w/ CCS']

TRANSITION_SPEED_VARIABLES = ['Final Energy', 'Final Energy|Electricity', 
                              'Food Demand|Crops', 'Food Demand|Livestock',
                              'Population']

BASELINE_DATA_VARIABLES = ['Price|Secondary Energy|Electricity', 'GDP|MER']

# Regions
R10 = ['Countries of Latin America and the Caribbean','Countries of South Asia; primarily India',
    'Countries of Sub-Saharan Africa', 'Countries of centrally-planned Asia; primarily China',
    'Countries of the Middle East; Iran, Iraq, Israel, Saudi Arabia, Qatar, etc.',
    'Eastern and Western Europe (i.e., the EU28)',
    'Other countries of Asia',
    'Pacific OECD', 'Reforming Economies of Eastern Europe and the Former Soviet Union; primarily Russia',
    'North America; primarily the United States of America and Canada']

R10_DICT = {'Countries of Latin America and the Caribbean': 'R10LATIN_AM',
            'Countries of South Asia; primarily India': 'R10INDIA+',
            'Countries of Sub-Saharan Africa': 'R10AFRICA',
            'Countries of centrally-planned Asia; primarily China': 'R10CHINA+',
            'Countries of the Middle East; Iran, Iraq, Israel, Saudi Arabia, Qatar, etc.': 'R10MIDDLE_EAST',
            'Eastern and Western Europe (i.e., the EU28)': 'R10EUROPE',
            'Other countries of Asia': 'R10REST_ASIA',
            'Pacific OECD': 'R10PAC_OECD',
            'Reforming Economies of Eastern Europe and the Former Soviet Union; primarily Russia': 'R10REF_ECON',
            'North America; primarily the United States of America and Canada': 'R10NORTH_AM'}

R10_CODES = ['R10LATIN_AM', 'R10INDIA+', 'R10AFRICA', 'R10CHINA+', 'R10MIDDLE_EAST',
                'R10EUROPE', 'R10REST_ASIA', 'R10PAC_OECD', 'R10REF_ECON','R10NORTH_AM'] #r10_iamc

R10_DEVELOPMENT = {'R10LATIN_AM':'South',
                    'R10INDIA+':'South',
                    'R10AFRICA':'South',
                    'R10CHINA+':'South',
                    'R10MIDDLE_EAST':'South',
                    'R10EUROPE':'North',
                    'R10REST_ASIA':'South',
                    'R10PAC_OECD':'North',
                    'R10REF_ECON':'North',
                    'R10NORTH_AM':'North'}

R5 = ['Asian countries except Japan',
    'Countries from the Reforming Economies of the Former Soviet Union',
    'Countries of the Middle East and Africa',
    'Latin American countries',
    'OECD90 and EU (and EU candidate) countries']

R5_DICT = {'Asian countries except Japan': 'R5ASIA', 
        'Countries from the Reforming Economies of the Former Soviet Union': 'R5REF',
        'Countries of the Middle East and Africa': 'R5MAF',
        'Latin American countries': 'R5LAM',
        'OECD90 and EU (and EU candidate) countries': 'R5OECD90+EU'}

R5_CODES = ['R5ASIA', 'R5REF', 'R5MAF', 'R5LAM', 'R5OECD90+EU'] #r5_iamc

R10_R5 = R10_CODES + R5_CODES



# Environmental Sustainability Constants

BECCS_THRESHOLD = 2800 # in mtCO2 / year medium threshold, high conversion efficiency value from deprez et al 2024
BIOENERGY_THRESHOLD = 100 # in EJ / year medium threshold, high conversion efficiency value from Creutzig et al 2015


# Economic Feasibility Constants
GLOBAL_BASE = 0.023
REGIONAL_BASE_PATH = INPUT_DIR + 'energy_investment_regions.csv'




# Error Messages
ERROR_FILE_NOT_FOUND = "The requested file could not be found."
ERROR_INVALID_INPUT = "The input provided is invalid."


