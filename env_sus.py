import numpy as np
import pyam
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
from utils import Utils
from utils import Data
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# Defines simple weighting system for indicators of sustainability, adjusting the weights will change the spread of how scenarios 
# within each temperature category will score. The idea is to tease out tradeoffs and synergies. 
class EnvSus:
    
    sustainability_weighting = {'water': 0.2, 'land': 0.2, 'resources': 0.2, 'biodiversity': 0.2, 'GHGs': 0.2}

    def __init__(self, water, land, resources, biodiversity, GHGs):
        self.water = water
        self.land = land
        self.resources = resources
        self.biodiversity = biodiversity
        self.GHGs = GHGs


# import the scenario data for C1, C2, C3 and C4

    connAr6 = pyam.iiasa.Connection(name='ar6-public', 
                                    creds=None, 
                                    auth_url='https://api.manager.ece.iiasa.ac.at')

    categories = ['C1', 'C2', 'C3', 'C4', 'C5']
    category_subset_paris = ['C1a_NZGHGs']
    categories = ['C1']
    # variable=['Water Consumption', 'Land Cover|Pasture', 'Land Cover|Forest', 
    #           'Land Cover', 'Land Cover|Cropland', 'Land Cover|Cropland', 
    #           'Ecotoxicity|Marine|Electricity', 'Material recycling|Plastics']
    variables=['Water Consumption', 'Land Cover|Pasture', 'Land Cover|Forest', 
               'Land Cover|Cropland', 'Land Cover|Cropland|Energy Crops']
    plotting_variables = ['Water Consumption', 'Land Cover|Pasture', 'Land Cover|Forest',
                          'Land Cover|Cropland', 'Land Cover|Cropland|Energy Crops']            
    emissions = ['Emissions|CO2']
    plotting_categories = ['C1', 'C3', 'C5']
    # plotting_categories = ['C3']
    plotting_category_colours = {'C1':'#f57200', 'C3':'#6302d9', 'C5':'#1b9e77'}
    violin_colours = ['#8CCFF4','#7FCACC','#006B7F']
    run_mode = 'cat'
    regions = ['World','Countries of Sub-Saharan Africa', 'Asian countries except Japan']
    checked_variables = pd.read_csv('variable_categories.csv')

def main() -> None:

    # data_download()
    # plot_outputs()
    # plot_using_pyam()
    # violin_plots()
    # Utils.simple_stats(Utils, 'AR6', EnvSus.regions, EnvSus.emissions, EnvSus.categories)
    # joel_data_download()
    # Utils.export_variable_list(Utils, 'AR6', ['C1', 'C2'])
    # Utils.create_variable_sheet(Utils,
    #                              'AR6',
    #                              EnvSus.category_subset_paris,
    #                                regions=EnvSus.regions, 
    #                                variables=EnvSus.checked_variables['variable'].tolist(), 
    #                                variable_sheet=EnvSus.checked_variables)
    # Utils.test_coco()
    # Utils.snapshot_cluster_analysis(Utils, 'World', Data.c1aR10_scenarios,['Land Cover|Forest', 'Land Cover|Cropland'],'C1a_NZGHGs' , 3, 2100)
    # Utils.time_series_cluster_analysis(Utils, 'World', Data.c1aR10_scenarios,['Land Cover|Forest', 'Land Cover|Cropland'],'C1a_NZGHGs' , 4)
    # joel_data_download()
    make_scenario_project_list()

def joel_data_download():
   
    variable_data = pd.read_csv('variable_categories.csv')
    
    # filter out all variables with low R10
    variable_data = variable_data[variable_data["sufficientR10"] > 0]

    # make a list of the variables
    variables = variable_data['variable'].unique().tolist()


    df = EnvSus.connAr6.query(model='*', scenario=Data.c1aR10_scenarios,
                variable=variables, region=Data.R10)

    print(df)
    df.to_csv('joel_test.csv')

def make_scenario_project_list():
   


    # get data
    df = EnvSus.connAr6.query(model='*', scenario='*',
                variable='Emissions|CO2', region='World', year=2010)

    df_pd = df.as_pandas(meta_cols=True)

    df_pd = df_pd[['scenario', 'model', 'Project_study']]    
    
    # # new dataframe
    # list_df = pd.DataFrame()
    # list_df['scenario'], list_df['model'], list_df['Project_study'] = df['scenario'], df['model'], df['Project_study']
    
    # remove identical rows
    df_pd = df_pd.drop_duplicates()
    
    print(df_pd)

    # export to csv
    df_pd.to_csv('scenario_project_list.csv')
    
    


def data_download():
    # Main query of AR6 data, returns a IamDataFrame object
    df = EnvSus.connAr6.query(model='*', scenario='*',
                variable=['Water Consumption', 'Land Cover|Pasture', 'Land Cover|Forest',
                          'Land Cover', 'Land Cover|Cropland',
                          'Land Cover|Cropland|Energy Crops'],
                )
    # Put variables in a list
    
    # print(df.variables().unique().tolist())

    # Export all query data to a pandas dataframe
    pddf = df.as_pandas(meta_cols=True)
    
        
    # Filter out by range of values from column header
    pddf = pddf[pddf['year'].between(2020, 2100)]
    
    # Filter out by desired categories C1, C2, C3, C4 & C5
    pddf = pddf[pddf['Category'].isin(EnvSus.categories)]

    # Ensure only global data is included
    pddf = pddf[pddf['region'] == 'World']

    scenario_list = pddf['scenario'].unique().tolist()

    for variables in EnvSus.variables:
        variable_df = pddf[pddf['variable'] == variables]
        print('The number of rows for ', variables, 'is: ', len(variable_df))

    complete_scenarios = []
    
    # Make a list of the scenarios that have data on all the variables
    for scenario in scenario_list:
        scenario_df = pddf[pddf['scenario'] == scenario]
        print(len(scenario_df))
        
        include = 1
        for variables in EnvSus.variables:
            variable_df = scenario_df[scenario_df['variable'] == variables]
            print('The number of rows for ', variables, 'is: ', len(variable_df))
            if len(variable_df) == 0:
                print(scenario, 'is missing ', variables)
                include = 0
        
        if include == 1:
            complete_scenarios.append(scenario)
    
    # Filter out scenarios that don't have data on all the variables
    
    print(complete_scenarios)
    pddf = pddf[pddf['scenario'].isin(complete_scenarios)]


    print(pddf)
    pddf.to_csv('ar6_test_scenarios.csv')


def plot_using_pyam():

    # df = pyam.IamDataFrame(data='ar6_test_scenarios.csv')
    # # Plot each of the variables in the list on subplots, with different colours for

    # Main query of AR6 data, returns a IamDataFrame object
    df = EnvSus.connAr6.query(model='*', scenario='*',
                variable=['Water Consumption', 'Land Cover|Pasture', 'Land Cover|Forest',
                          'Land Cover|Cropland','Land Cover|Cropland|Energy Crops'],
                )
    
    # Filter out scenarios that do not have all the variables
    df = df.filter(variable=EnvSus.variables)

    df = df.filter(region='World')

    # Filter out by desired categories as above:
    df = df.filter(Category=EnvSus.plotting_categories)

    # Filter out values that are not in the range of years 2025 to 2100
    df = df.filter(year=range(2025, 2101))

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=False, sharex=False)
    axs = axs.flatten()
    
    # for each variable, create a subplot
    variable_int = 0
    for i in EnvSus.variables:
        print(i)    
        df.filter(variable=i).plot(color='model', legend=False, ax=axs[variable_int],
                                                        title=i,
                                                        y='value', x='year',
            )
        # set the y axis label as the unit
        axs[variable_int].set_ylabel(df.filter(variable=i).unit[0])
        
        # Set the x-axis limits
        axs[variable_int].set_xlim(2025, 2100)

        # reduce the line width
        for line in axs[variable_int].lines:
            line.set_linewidth(0.3)

        variable_int += 1
    
    fig.delaxes(axs[5])
    plt.show()


# Plot the data showing warming categories and different variables
def plot_outputs():

    
    df = pd.read_csv('ar6_test_scenarios.csv')

    # Make a list of unique scenarios
    scenario_list = df['scenario'].unique().tolist()
    categories = EnvSus.plotting_categories
    category_colours = EnvSus.plotting_category_colours

    # filter out scenarios that are not in the categories list
    df = df[df['Category'].isin(categories)]
    
    # Plot the scenarios as lines on subplots for each variable. 
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=False, sharex=False)
    axs = axs.flatten()
    for i, variables in enumerate(EnvSus.plotting_variables):
        for scenario in scenario_list:
            
            scenario_df = df[df['scenario'] == scenario]
            scenario_df = scenario_df[scenario_df['variable'] == variables]

            # Make a unique dataframe for each model run of scenario
            scenario_models = scenario_df['model'].unique().tolist()
            for model in scenario_models:
                
                scenario_model_df = scenario_df[scenario_df['model'] == model]
                
                # Make a variable of the category for the scenario
                category = scenario_df['Category'].unique().tolist()
                category = category[0]

                # Get units for the variable
                units = scenario_df['unit'].unique().tolist()
                units = units[0]

                # Colour the lines based on the category
                if EnvSus.run_mode == 'cat':
                    # Plot the data
                    axs[i].plot(scenario_model_df['year'], scenario_model_df['value'], 
                                label=scenario, color=category_colours[category], 
                                linewidth=0.3, alpha=0.5) 
                # # Colour the lines based on the model
                # elif EnvSus.run_mode == 'model':
                #     # Plot the data so the color of the lines is based on the model
                #     axs[i].plot(scenario_model_df['year'], scenario_model_df['value'], 
                #                 label=scenario, linewidth=0.3, alpha=0.5, 
                #                 color=

        axs[i].set_xlim(2020, 2100)
        axs[i].set_title(variables)
        
        # set the y axis label as the unit
        axs[i].set_ylabel(units)

        # axs[i].legend()
    

    # make legend based on category colours
    handles = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in category_colours.values()]
    labels = category_colours.keys()
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.85, 0.25))

    fig.delaxes(axs[5])
    # plt.tight_layout()
    plt.show()

    # print(df)


# Plot seaborn violin plots with snapshots of the data at 2050 and 2100, across each variable and each category
def violin_plots():

    df = pd.read_csv('ar6_test_scenarios.csv')
    
    snapshot_year = [2050]
    
    # filter out scenarios that are not in the categories list
    df = df[df['Category'].isin(EnvSus.plotting_categories)]
    
    # Make a list of unique scenarios
    scenario_list = df['scenario'].unique().tolist()
    
    # Make a subplot for each variable that has three violins of the scenario results at snapshot year, by category
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=False, sharex=False)
    axs = axs.flatten()

    # for each variable, create a subplot, first create the dataset
    variable_int = 0
    for variable in EnvSus.plotting_variables:

        # Make a dataframe for each variable
        variable_df = df[df['variable'] == variable]

        # Make a dataframe for the snapshot year
        snapshot_df = variable_df[variable_df['year'].isin(snapshot_year)]


        
        # Make a dataframe that has columns for each category with the corresponding
        # variable value

        # plot violins on the figure subplot for each category
        data= pd.DataFrame()
        category_int = 0
        for category in EnvSus.plotting_categories:
    
                category_df = snapshot_df[snapshot_df['Category'] == category]
                # reset index
                category_df = category_df.reset_index(drop=True)
                data[category_int] = category_df['value']

                # plot the violin
                sns.boxplot(data=data, ax=axs[variable_int], 
                palette=EnvSus.violin_colours)
                sns.stripplot(data=data, ax=axs[variable_int], 
                               color=".3")
                category_int += 1
        
        # put x axis labels on the figure as per the categories
        axs[variable_int].set_xticklabels(EnvSus.plotting_categories)
        # add a title to the subplot
        axs[variable_int].set_title(variable + str(snapshot_year[0]))
        
        # set y axis labels
        units = variable_df['unit'].unique().tolist()
        axs[variable_int].set_ylabel(units[0])
    
        # Add ticks on the right hand y axis as well as left
        axs[variable_int].yaxis.set_ticks_position('both')

    
    # remove 6th subplot
    
    # add legend to the figure with category colours
    

                
        variable_int += 1
    fig.delaxes(axs[5])
    plt.show()
    
        
  


        
    
    plt.show()


if __name__ == "__main__":
    main()