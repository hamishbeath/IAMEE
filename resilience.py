import numpy as np
import pyam
import pandas as pd
from utils import Utils
from utils import Data



class Resilience:

    test = 0



def main() -> None:

    test()


def shannon_index_energy_mix(pyam_df, scenario_model_list, end_year):


    # filter for the variables needed
    df = pyam_df.filter(variable=['Capacity|Electricity|Wind','Capacity|Electricity|Solar|PV'],region='World',
                        year=range(2020, end_year+1),
                        scenario=scenario_model_list['scenario'], 
                        model=scenario_model_list['model'])
    
    