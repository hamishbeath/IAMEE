import numpy as np
import pyam
import pandas as pd
# from utils import Utils
from utils import Data
from pygini import gini

class Fairness:

    energy_variables = variable=['Primary Energy|Coal','Primary Energy|Oil', 
                        'Primary Energy|Gas', 'Primary Energy|Nuclear',
                        'Primary Energy|Biomass', 'Primary Energy|Non-Biomass Renewables']

    gini_between_countries = pd.read_csv('inputs/gini_btw_6.csv')
    ssp_gini_data = pd.read_csv('inputs/ssp_population_gdp_projections.csv')
    regional_gini = pd.read_csv('outputs/within_region_gini.csv')