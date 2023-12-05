import numpy as np
import pyam
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import country_converter as coco
import pickle as pkl
from sklearn.cluster import KMeans
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

    c1aR10_scenarios = ['EN_NPi2020_300f', 'SSP1-DACCS-1p9-3pctHR', 'SSP1-noDACCS-1p9-3pctHR', 'SSP4-noDACCS-1p9-3pctHR', 
                    'SSP1_SPA1_19I_D_LB', 'SSP1_SPA1_19I_LIRE_LB', 'SSP1_SPA1_19I_RE_LB', 'SSP2_SPA2_19I_LI', 
                    'CEMICS_GDPgrowth_1p5', 'CEMICS_HotellingConst_1p5', 'CEMICS_Linear_1p5', 
                    'LeastTotalCost_LTC_brkLR15_SSP1_P50', 'R2p1_SSP1-PkBudg900', 'R2p1_SSP5-PkBudg900', 
                    'CD-LINKS_NPi2020_400', 'PEP_1p5C_full_eff', 'PEP_1p5C_red_eff', 'CEMICS_SSP1-1p5C-fullCDR',
                        'EN_NPi2020_200f', 'EN_NPi2020_400f', 'SusDev_SDP-PkBudg1000', 'SusDev_SSP1-PkBudg900', 
                        'DeepElec_SSP2_def_Budg900', 'DISCRATE_cb400_cdrno_dr5p', 'EN_NPi2020_450f', 'EN_NPi2020_500f']

    R10 = ['Countries of Latin America and the Caribbean','Countries of South Asia; primarily India',
        'Countries of Sub-Saharan Africa', 'Countries of centrally-planned Asia; primarily China',
        'Countries of the Middle East; Iran, Iraq, Israel, Saudi Arabia, Qatar, etc.',
        'Eastern and Western Europe (i.e., the EU28)',
        'Other countries of Asia',
        'Pacific OECD', 'Reforming Economies of Eastern Europe and the Former Soviet Union; primarily Russia',
        'World']

class Utils:

    
    test_variables = ['Water Consumption', 'Land Cover|Pasture', 'Land Cover|Forest',
                            'Land Cover', 'Land Cover|Cropland',
                            'Land Cover|Cropland|Energy Crops']
    
    categories = ['C1', 'C2', 'C3', 'C4', 'C5']
    
    connAr6 = pyam.iiasa.Connection(name='ar6-public', 
                                creds=None, 
                                auth_url='https://api.manager.ece.iiasa.ac.at')
    
    # connSR15 = pyam.iiasa.Connection(name='iamc15', 
    #                             creds=None, 
    #                             auth_url='https://api.manager.ece.iiasa.ac.at')
    
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
    
    
    def create_variable_sheet(self, db, categories, regions, variables, variable_sheet):

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
            df_category = df.filter(Category_subset=category)
            

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

        datasheet['category_1'] = variable_sheet['category_1']
        datasheet['category_2'] = variable_sheet['category_2']
        
        # Export datasheet to csv
        datasheet.to_csv('stats_datasheet.csv')

    
    # def filter_data_sheet_variable_prevelance(self, db, categories, region, threshold):

    #     """"
    #     Function that takes a datasheet from the create_variable_sheet function and filters it to only 
    #     include variables that are reported by a certain percentage of scenarios for a given region and
    #     temperature category.
    #     Inputs: 
        
        
    # Function that performs cluster analysis for a set of input variables, a set region and a set of predetermined scenarios. 
    # The cluster analysis is performed using the k-means algorithm but implements time as feature. 
    # The function returns a dataframe with the cluster labels for each scenario and the cluster centroids.



    # Cluster analysis for a set of input variables, a set region and a set of predetermined scenarios. This function performs 
    # cluster analysis using the k-means algorithm for a snapshot in time (2100) and for the entire time series.
    def snapshot_cluster_analysis(self, region, scenarios, variables, category, n_clusters, snapshot_year):
        
        # # query data
        # df = Utils.connAr6.query(model='*', scenario=scenarios,
        #     variable=variables, region=region, year=snapshot_year
        #     )

        # read in data


        # # filter by temperature category
        # df_category = df.filter(Category_subset=category)

        # # save df as a csv
        # df_category.to_csv('snapshot_data.csv')

        df_category = pyam.IamDataFrame(data='snapshot_data.csv')
        
        k_means_data = pd.DataFrame()
        for variable in variables:
            df_variable = df_category.filter(variable=variable)
            df_variable = df_variable.filter(year=snapshot_year)
            df_variable = df_variable.as_pandas()
            print(df_variable)
            # set index to scenario and model
            df_variable = df_variable.set_index(['model', 'scenario'])
            k_means_data[variable] = df_variable['value']
            
        
        print(k_means_data)


        # # arrange data in a way that is suitable for clustering so that each scenario is a row and each variable is a column
        # df_category = df_category.pivot_table(index=['model', 'scenario'], columns=variables, values='value').reset_index()
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(k_means_data[variables])

        # Get cluster assignments
        labels = kmeans.labels_

        # Get cluster centroids
        centroids = kmeans.cluster_centers_

        # visualise
        plt.scatter(k_means_data[variables[0]], k_means_data[variables[1]], c=labels, s=50, cmap='viridis')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5)
        plt.xlabel(variables[0])
        plt.ylabel(variables[1])
        plt.show()


    # function that performs cluster analysis for a set of n variables, 
    # but for each year in the time series including time as a feature
    def time_series_cluster_analysis(self, region, scenarios, variables, category, n_clusters):
        
        # # query data
        # df = Utils.connAr6.query(model='*', scenario=scenarios,
        #     variable=variables, region=region
        #     )

        df_category = pyam.IamDataFrame(data='snapshot_data.csv')

        # filter by temperature category
        df_category = df_category.filter(year=range(2020, 2101))

        k_means_data = pd.DataFrame()
        for variable in variables:
            df_variable = df_category.filter(variable=variable)
            df_variable = df_variable.as_pandas()
            print(df_variable)
            # set index to scenario and model
            df_variable = df_variable.set_index(['model', 'scenario', 'year'])
            k_means_data[variable] = df_variable['value']
        
        
        
        
        # # arrange data in a way that is suitable for clustering so that each scenario is a row and each variable is a column
        # df_category = df_category.pivot_table(index=['model', 'scenario', 'year'], columns=variables, values='value').reset_index()
        
        print(k_means_data)
            # Elbow Method
        sse = []
        list_k = list(range(1, 10))

        for k in list_k:
            km = KMeans(n_clusters=k)
            km.fit(k_means_data[variables])
            sse.append(km.inertia_)

        # Plot sse against k
        plt.figure(figsize=(6, 6))
        plt.plot(list_k, sse, '-o')
        plt.xlabel(r'Number of clusters *k*')
        plt.ylabel('Sum of squared distance')
        plt.show()

        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(k_means_data[variables])

        # # Get cluster assignments
        labels = kmeans.labels_

        # # Get cluster centroids
        centroids = kmeans.cluster_centers_

        # visualise
        plt.scatter(k_means_data[variables[0]], k_means_data[variables[1]], c=labels, s=50, cmap='viridis')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5)
        plt.xlabel(variables[0])
        plt.ylabel(variables[1])
        plt.show()

        # # add cluster labels to dataframe
        k_means_data['cluster'] = labels

        # save dataframe as csv
        k_means_data.to_csv('cluster_data_timeseries.csv')

        return df_category