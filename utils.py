import numpy as np
import pyam
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']


"""
This is a set of utils that can be used to do analysis, create plots, or do other things with the data, 
regardless of the variables, database or dimensions being explored.

"""

class Utils:

    
    R10 = ['Countries of Latin America and the Caribbean','Countries of South Asia; primarily India',
           'Countries of Sub-Saharan Africa', 'Countries of centrally-planned Asia; primarily China',
           'Countries of the Middle East; Iran, Iraq, Israel, Saudi Arabia, Qatar, etc.',
           'Eastern and Western Europe (i.e., the EU28)',
           'Other countries of Asia',
           'Pacific OECD', 'Reforming Economies of Eastern Europe and the Former Soviet Union; primarily Russia',
           'World']

    test_variables = ['Water Consumption', 'Land Cover|Pasture', 'Land Cover|Forest',
                            'Land Cover', 'Land Cover|Cropland',
                            'Land Cover|Cropland|Energy Crops']
    
    categories = ['C1', 'C2', 'C3', 'C4', 'C5']
    
    connAr6 = pyam.iiasa.Connection(name='ar6-public', 
                                creds=None, 
                                auth_url='https://api.manager.ece.iiasa.ac.at')
    
    connSR15 = pyam.iiasa.Connection(name='iamc15', 
                                creds=None, 
                                auth_url='https://api.manager.ece.iiasa.ac.at')
    
    selected_variables = pd.read_csv('variables_filtered.csv')['variables'].tolist()
    
    # Function that provides simple statistics for a given set of inputs for Pyam
    # Inputs:
    # - database (AR6 or SR15)
    # - temperature scenarios OR SSPs
    # - region
    # - variable(s)


    def simple_stats(self, db, region, variables, categories):
        
        if db == 'AR6':

            df = Utils.connAr6.query(model='*', scenario='*',
                variable=variables, region=region
                )
        if db == 'SR15':

            df = Utils.connSR15.query(model='*', scenario='*',
                variable=variables, region=region,
                )
        
        # filter by temperature category
        df = df.filter(Category=categories)

        # filter by SSP
        #df = df.filter(SSP=SSPs)

        for region in region:
            df_region = df.filter(region=region)
            # count number of scenarios, allowing for multiple models per scenario
            for variable in variables:
                df_variable = df_region.filter(variable=variable)
                total_scenario_count = 0
                model_list = df_variable['model'].unique().tolist()
                for model in model_list:
                    
                    model_df = df_variable.filter(model=model)
                    scenario_list = model_df['scenario'].unique().tolist()
                    scenario_count = len(scenario_list)
                    total_scenario_count += scenario_count

                print('Number of scenarios for', categories, 'categories with data for', 
                    variable, 'in the region of', region, 'is: ', total_scenario_count)
    
    
    def export_variable_list(self, db, categories):

        if db == 'AR6':

            df = Utils.connAr6.query(model='*', scenario='*',
                variable=Utils.selected_variables, year=2100, region='World', 
                )
            
        if db == 'SR15':

            df = Utils.connSR15.query(model='*', scenario='*',
                variable='*', year=2100, region='World'
                )
        
        # filter by temperature category
        df = df.filter(Category=categories)

        variable_list = df['variable'].unique().tolist()
        
        # Export variable list to csv
        variable_list = pd.DataFrame(variable_list)
        variable_list.columns = ['variables']
        variable_list.to_csv('variable_list_checked.csv')

        return variable_list
    
    
    def create_variable_sheet(self, db, categories, regions, variables):

        """
        Creates a datasheet with stats for each variable, against the regions and categories provided

            Inputs:
            - database (AR6 or SR15)
            - temperature scenarios
            - regions
            - variables

            Outputs:
            - datasheet with stats for each region, variable and temperature category
        
        """
            
        if db == 'AR6':

            df = Utils.connAr6.query(model='*', scenario='*',
                variable=variables, region=regions, year=2100
                )
        if db == 'SR15':

            df = Utils.connSR15.query(model='*', scenario='*',
                variable=variables, region=regions, year=2100
                    )
        
        # Make dataframe with a row for each variable
        
        datasheet = pd.DataFrame(variables)
        datasheet.columns = ['variable']
        
        # loop through each temperature category
        for category in categories:
            df_category = df.filter(Category=category)


            # Get the number of scenarios reporting emissions for CO2 in 2100 for World as a basis for %
            emissions_scenarios = 0
            df_category_emissions = df_category.filter(variable='Emissions|CO2')
            model_list_emissions = df_category_emissions['model'].unique().tolist()
            for model in model_list_emissions:
                model_emissions_df = df_category_emissions.filter(model=model)
                scenario_list_emissions = model_emissions_df['scenario'].unique().tolist()
                scenario_count_emissions = len(scenario_list_emissions)
                emissions_scenarios += scenario_count_emissions

            # loop through each region
            for region in regions:
                
                df_region = df_category.filter(region=region)

                # count number of scenarios, allowing for multiple models per scenario
                variables_count_list = []
                variables_percentage_list = []
                for variable in variables:
                    current_scenario_count = 0
                    
                    df_variable = df_region.filter(variable=variable)
                    model_list = df_variable['model'].unique().tolist()
                    for model in model_list:
                        model_df = df_variable.filter(model=model)
                        scenario_list = model_df['scenario'].unique().tolist()
                        scenario_count = len(scenario_list)
                        current_scenario_count += scenario_count

                    variables_count_list.append(current_scenario_count)
                    variables_percentage_list.append(current_scenario_count/emissions_scenarios*100)
                
                # add to datasheet
                datasheet[str(region),str(category), 'count'] = pd.Series(variables_count_list)
                datasheet[str(region),str(category), 'percentage'] = pd.Series(variables_percentage_list)

        # Export datasheet to csv
        datasheet.to_csv('stats_datasheet.csv')

    
    # def filter_data_sheet_variable_prevelance(self, db, categories, region, threshold):

    #     """"
    #     Function that takes a datasheet from the create_variable_sheet function and filters it to only 
    #     include variables that are reported by a certain percentage of scenarios for a given region and
    #     temperature category.
    #     Inputs: 
        
        
    #     """
        

