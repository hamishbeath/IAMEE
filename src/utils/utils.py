import numpy as np
import pyam
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import country_converter as coco
import pickle as pkl
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from src.constants import *
# import logging

# # Set up logging
# logging.basicConfig(level=logging.DEBUG)

cc = coco.CountryConverter()
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']


"""
This is a set of utils that can be used to do analysis, create plots, or do other things with the data, 
regardless of the variables, database or dimensions being explored.

"""

class Data:




    region_country_df = pd.read_csv('iso3c_regions.csv')
    #https://github.com/setupelz/regioniso3c/blob/main/iso3c_region_mapping_20240319.csv    
    

    mandatory_variables = ['Emissions|CO2', 'Investment|Energy Supply','Capacity|Electricity|Wind',
                           'Capacity|Electricity|Solar|PV', 'Final Energy', 'Population',
                           'Primary Energy|Coal', 'Primary Energy|Oil', 'Primary Energy|Gas', 'Primary Energy|Nuclear',
                           'Primary Energy|Biomass', 'Primary Energy|Non-Biomass Renewables','Carbon Sequestration|CCS|Biomass',
                            'GDP|MER', 'Land Cover|Forest','Land Cover', 'Carbon Sequestration|Land Use', 'Price|Secondary Energy|Electricity']

    mandatory_econ_variables = ['GDP|MER', 'Investment|Energy Supply',
                                'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|5.0th Percentile',
                                'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile',
                                'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|95.0th Percentile',
                                'Policy Cost|GDP Loss']
    
    mandatory_econ_variables_regional = ['GDP|MER', 'Investment|Energy Supply',
                            'Policy Cost|GDP Loss']


    # mandatory_test_variables = ['Investment|Energy Supply', 'Policy Cost|GDP Loss']
    # mandatory_CDR_variables = ['Carbon Sequestration|Enhanced Weathering']
    mandatory_CDR_variables = ['Carbon Sequestration|Direct Air Capture', 'Carbon Sequestration|Land Use','Carbon Sequestration|CCS|Biomass']
    paola_variables = ['Carbon Sequestration|Direct Air Capture', 'Carbon Sequestration|Land Use','Carbon Sequestration|CCS|Biomass', 'Carbon Sequestration|Enhanced Weathering']
    # paola_variables = ['Carbon Sequestration|Land Use','Carbon Sequestration|CCS|Biomass']
    narrative_variables = ['Final Energy|Transportation', 'Final Energy|Transportation|Liquids', 'Final Energy|Transportation|Liquids|Oil', 
                           'Primary Energy|Fossil', 'Land Cover|Pasture', 'Land Cover|Cropland', 'Land Cover|Forest', 'Land Cover',  'Carbon Sequestration|CCS|Biomass', 
                           'Carbon Sequestration|CCS|Fossil', 'Primary Energy|Non-Biomass Renewables', 'Primary Energy|Fossil|w/ CCS', 'Carbon Sequestration|Direct Air Capture',
                           'Primary Energy|Fossil|w/o CCS','Final Energy', 'Agricultural Demand', 'Emissions|CO2|AFOLU', 'Capacity|Electricity|Wind',
                           'Capacity|Electricity|Solar|PV','Energy Service|Transportation|Passenger', 'Energy Service|Transportation|Freight', 'Carbon Sequestration|Land Use' ]

    


    

    c1a_scenarios_selected = ['PEP_1p5C_red_eff', 'SSP1_SPA1_19I_RE_LB', 'EN_NPi2020_300f', 'EN_NPi2020_400f']
    c1a_models_selected = ['REMIND-MAgPIE 1.7-3.0', 'IMAGE 3.2', 'AIM/CGE 2.2', 'WITCH 5.0']
    # ar6_world = pyam.IamDataFrame(data='database/AR6_Scenarios_Database_World_v1.1.csv', meta='database/meta_data.csv')
    # ar6_world = pyam.read_datapackage('database/', data='AR6_Scenarios_Database_World_v1.1.csv', meta='meta_data.csv')

    
    # ar6_R10 = pyam.IamDataFrame(data='database/AR6_Scenarios_Database_Regions_v1.1.csv', meta='database/meta_data.csv')
    # ar6_meta = pd.DataFrame('database/metadata.csv')
    

    # categories = ''
    cdr_categories = ['C1', 'C2', 'C3']
    categories = ['C1', 'C2']

    model_scenarios = pd.read_csv('Countries of Sub-Saharan Africa_mandatory_variables_scenarios' + str(categories) + '.csv')
    dimensions_pyamdf = cat_df = pyam.IamDataFrame(data='cat_df' + str(categories) + '.csv')
    meta_df = pd.read_csv('cat_meta' + str(categories) + '.csv') 

    energy_variables = ['Primary Energy|Coal','Primary Energy|Oil',
                        'Primary Energy|Gas', 'Primary Energy|Nuclear',
                        'Primary Energy|Biomass', 'Primary Energy|Non-Biomass Renewables']
    econ_regions = ['Countries of Sub-Saharan Africa']
    try:
        narrative_data = pyam.IamDataFrame(data='outputs/narrative_data' + str(categories) + '.csv')
    except FileNotFoundError:
        print('No narrative data found')

    try:
        regional_dimensions_pyamdf = pyam.IamDataFrame(data='pyamdf_dimensions_data_R10' + str(categories) + '.csv')
    except FileNotFoundError:
        print('No regional dimensions data found')
    
    try:
        scenario_archetypes = pd.read_csv('outputs/scenario_archetypes '+ str(categories) + '.csv')
    except:
        print('No scenario archetypes file found for the category of', categories)
    try:
        land_use_seq_data = pyam.IamDataFrame(data='land_sequestration_imputed.csv')
    except:
        print('No land use imputed data found')
    try:
        scenario_baselines = pd.read_csv('baselines' + str(categories) + '.csv')
    except FileNotFoundError:
        print('No baselines file found for the category of', categories)

class Utils:



    # function that takes as an input a list of mandatory variables and regional coverage and 
    # provides a list of scenarios that report on all of the mandatory variables for the given region
    def manadory_variables_scenarios(self, categories, regions, variables, subset=False, special_file_name=None
                                     ,call_sub=None, save_data=False, local=False):

        """
        Function that takes as an input a list of mandatory variables and regional coverage and 
        provides a list of scenarios that report on all of the mandatory variables for the given region

        Inputs:
        - database (AR6 or SR15)
        - temperature scenarios
        - regions
        - variables

        Outputs:
        - list of scenarios that report on all of the mandatory variables for the given region

        """
        if local == False:
        
            connAr6 = pyam.iiasa.Connection(name='ar6-public', 
                                creds=None, 
                                auth_url='https://api.manager.ece.iiasa.ac.at')    

            print('Querying data')
            df = connAr6.query(model='*', scenario='*',
                    variable=variables, region=regions)

        else:
            if regions == 'World':
                df = Data.ar6_world
            else:
                # check region in R10 list
                if regions in Data.R10:
                    df = Data.ar6_R10.filter(region=regions)
                else:
                    raise ValueError('Region not in R10 list')
                    
        print(df)
        # ensure filtering by temperature category (subset or not)
        if subset == True:
            cat_df = df.filter(Category_subset=categories)        
        elif subset == False:
            cat_df = df.filter(Category=categories)
        else:
            raise ValueError('Subset must be a boolean')
        
        # cat_df = df.filter(Category_subset=categories)
        if call_sub == None:
            
            if save_data == True:
        # except:
        # cat_df = df.filter(Category=categories)
                if special_file_name != None:
                    cat_df.to_csv(special_file_name + 'cat_df' + str(categories) + '.csv')
                    cat_meta = cat_df.as_pandas(meta_cols=True)
                    cat_meta.to_csv(special_file_name + 'cat_meta' + str(categories) +  '.csv')
                else:    
                    cat_df.to_csv('cat_df' + str(categories) + '.csv')
                    cat_meta = cat_df.as_pandas(meta_cols=True)
                    cat_meta.to_csv('cat_meta' + str(categories) + '.csv')

        # cat_df = pyam.IamDataFrame(data='cat_df.csv')

        # Get the list of model scenario pairs reporting on all of the mandatory variables
        for region in regions:
            output_df = pd.DataFrame()
            model_list = []
            scenario_list = []
            region_df = cat_df.filter(region=region)
            for model in region_df['model'].unique().tolist():
                model_df = region_df.filter(model=model)
                # make list of available scenarios
                model_scenarios = model_df['scenario'].unique().tolist()
                for scenario in model_scenarios:
                    scenario_df = model_df.filter(scenario=scenario)
                    if scenario_df['variable'].nunique() == len(variables):
                        model_list.append(model)
                        scenario_list.append(scenario)

            output_df['model'] = model_list
            output_df['scenario'] = scenario_list

            if call_sub == None:
                if special_file_name != None:
                    output_df.to_csv(special_file_name + '.csv')
                else:
                    output_df.to_csv(region + '_mandatory_variables_scenarios' + str(categories) + '.csv')
            elif call_sub != None:
                return output_df



    # function that takes as an input a list of variables and regions coverage and
    # provides a list of scenarios that report on each variable variables for the given region
    def create_variable_scenario_count(self, df, variables, regions, categories):

        # make outout dataframe
        output_df = pd.DataFrame()

        df = df.filter(year=2040, Category=categories)

        # add the variables to the output dataframe
        output_df['variable'] = variables

        # loop through each region
        for region in regions:

            region_df = df.filter(region=region)

            # count the number of scenarios for each variable
            scenario_count_list = region_df.data.groupby('variable')['scenario'].nunique().reindex(variables, fill_value=0).tolist()

            output_df[region] = scenario_count_list

        print(output_df)
        return output_df

    
     
    def data_download(variables, models, scenarios, region, categories,
                      end_year, file_name=str):
   
        connAr6 = pyam.iiasa.Connection(name='ar6-public', 
                    creds=None, 
                    auth_url='https://api.manager.ece.iiasa.ac.at')    

        df = connAr6.query(model=models, scenario=scenarios,
            variable=variables, region=region, category=categories,
            year=range(2020, end_year+1))
        df = df.filter(Category=categories)

        df.to_csv(file_name + '.csv')


    def data_download_sub(variables, models, scenarios, categories, region, end_year):

        connAr6 = pyam.iiasa.Connection(name='ar6-public', 
                    creds=None, 
                    auth_url='https://api.manager.ece.iiasa.ac.at')    

        df = connAr6.query(model=models, scenario=scenarios, category=categories,
            variable=variables, region=region, year=range(2020, end_year+1)
            )

        df = df.filter(Category=categories)


        return df


    def map_countries_to_regions(country_groups, country_data):

        """
        Function to map countries to regions based on country groupings. 
        The function takes as an input a dataframe of country groupings and a dataframe of country data.
        The function returns a dataframe with the countries mapped to their respective regions.

        Inputs: country_groups - dataframe of country groupings
                country_data - dataframe of country data

        Outputs: output_df - dataframe with countries mapped to regions
        
        """
        output_dict = {}
        # get the list groups of countries
        country_groups_list = country_groups['group'].unique().tolist()
        country_data_countries = country_data['country'].unique().tolist()
        
        output_df = pd.DataFrame()

        # Convert country names to ISO3 codes in both dataframes
        country_groups['countries'] = country_groups['countries'].str.split(', | and ').apply(lambda x: coco.convert(names=x, to='ISO3'))
        country_data['country'] = coco.convert(names=country_data['country'], to='ISO3')

        # Explode the lists of countries into separate rows
        country_groups = country_groups.explode('countries')

        # Merge the dataframes on the ISO3 country codes
        output_df = country_data.merge(country_groups, left_on='country', right_on='countries', how='left')

        # Drop the redundant 'countries' column
        output_df = output_df.drop(columns=['countries'])
        
        return output_df

