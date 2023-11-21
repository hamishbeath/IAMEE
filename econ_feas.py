import numpy as np
import pyam
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# Defines simple weighting system for indicators of sustainability, adjusting the weights will change the spread of how scenarios 
# within each temperature category will score. The idea is to tease out tradeoffs and synergies. 
class EconFeas:
    
# import the scenario data for C1, C2, C3 and C4

    connAr6 = pyam.iiasa.Connection(name='ar6-public', 
                                    creds=None, 
                                    auth_url='https://api.manager.ece.iiasa.ac.at')

    categories = ['C1', 'C2', 'C3', 'C4', 'C5']
    # variable=['Water Consumption', 'Land Cover|Pasture', 'Land Cover|Forest', 
    #           'Land Cover', 'Land Cover|Cropland', 'Land Cover|Cropland', 
    #           'Ecotoxicity|Marine|Electricity', 'Material recycling|Plastics']
    variables=['GDP|PPP', 'Investment', 'Investment|Energy Supply']

    # plotting_variables = ['Water Consumption', 'Land Cover|Pasture', 'Land Cover|Forest',
    #                       'Land Cover|Cropland', 'Land Cover|Cropland|Energy Crops']
                            
    plotting_categories = ['C1', 'C3', 'C5']
    # plotting_categories = ['C3']
    plotting_category_colours = {'C1':'#f57200', 'C3':'#6302d9', 'C5':'#1b9e77'}
    # violin_colours = ['#f57200','#6302d9','#1b9e77']
    
    run_mode = 'cat'

def main() -> None:

    # data_download()
    # plot_outputs()
    # plot_using_pyam()
    violin_plots()
    # assess_variable_data()


def assess_variable_data():

    # Main query of AR6 data, returns a IamDataFrame object
    df = EconFeas.connAr6.query(model='*', scenario='*',
                variable=['GDP|PPP', 'Investment', 'Investment|Energy Supply'])

    df_filtered = df.filter(region='World', Category=EconFeas.categories)

        
    # Filter out by range of values from column header
    df_filtered_2050 = df_filtered.filter()

    
    # Export all query data to a pandas dataframe
    pddf = df_filtered_2050.as_pandas(meta_cols=True)
    
    
    # export to csv for analysis
    pddf.to_csv('ar6_econfeas.csv')
    
    # print(df_filtered_2050, len(df_filtered_2050))

    # # Find the number of scenarios reporting each variable in the year 2100

    # print('The number of C1-C5 scenarios reporting Energy Supply investment in 2050 is', len(df_filtered_2050.filter(variable='Investment|Energy Supply')))
    # print('The number of C1-C5 scenarios reporting GDP in 2050 is', len(df_filtered_2050.filter(variable='GDP|PPP')))
    # print('The number of C1-C5 scenarios reporting total investment in 2050 is', len(df_filtered_2050.filter(variable='Investment')))

    # # print the unique model names reporting investment in 2050
    # # print(df_filtered_2050.filter(variable='Investment')['model'].unique().tolist())
    
    # # import model classification data and model types
    # model_classification = pd.read_csv('model_classification.csv')
    # model_type = model_classification['Model Type'].unique().tolist()


       
def data_download():
    # Main query of AR6 data, returns a IamDataFrame object
    df = EconFeas.connAr6.query(model='*', scenario='*',
                variable=['GDP|PPP', 'Investment', 'Investment|Energy Supply']
                )
    # Put variables in a list
    
    # print(df.variables().unique().tolist())

    # Export all query data to a pandas dataframe
    pddf = df.as_pandas(meta_cols=True)
    
        
    # Filter out by range of values from column header
    pddf = pddf[pddf['year'].between(2020, 2100)]
    
    # Filter out by desired categories C1, C2, C3, C4 & C5
    pddf = pddf[pddf['Category'].isin(EconFeas.categories)]

    # Ensure only global data is included
    pddf = pddf[pddf['region'] == 'World']

    scenario_list = pddf['scenario'].unique().tolist()

    for variables in EconFeas.variables:
        variable_df = pddf[pddf['variable'] == variables]
        print('The number of rows for ', variables, 'is: ', len(variable_df))

    complete_scenarios = []
    
    # Make a list of the scenarios that have data on all the variables
    for scenario in scenario_list:
        scenario_df = pddf[pddf['scenario'] == scenario]
        print(len(scenario_df))
        
        include = 1
        for variables in EconFeas.variables:
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
    pddf.to_csv('ar6_econfeas.csv')



def plot_using_pyam():

    # df = pyam.IamDataFrame(data='ar6_test_scenarios.csv')
    # # Plot each of the variables in the list on subplots, with different colours for

    # Main query of AR6 data, returns a IamDataFrame object
    df = EconFeas.connAr6.query(model='*', scenario='*',
                variable=['Water Consumption', 'Land Cover|Pasture', 'Land Cover|Forest',
                          'Land Cover|Cropland','Land Cover|Cropland|Energy Crops'],
                )
    
    # Filter out scenarios that do not have all the variables
    df = df.filter(variable=EconFeas.variables)

    df = df.filter(region='World')

    # Filter out by desired categories as above:
    df = df.filter(Category=EconFeas.plotting_categories)

    # Filter out values that are not in the range of years 2025 to 2100
    df = df.filter(year=range(2025, 2101))

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=False, sharex=False)
    axs = axs.flatten()
    
    # for each variable, create a subplot
    variable_int = 0
    for i in EconFeas.variables:
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
    categories = EconFeas.plotting_categories
    category_colours = EconFeas.plotting_category_colours

    # filter out scenarios that are not in the categories list
    df = df[df['Category'].isin(categories)]
    
    # Plot the scenarios as lines on subplots for each variable. 
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=False, sharex=False)
    axs = axs.flatten()
    for i, variables in enumerate(EconFeas.plotting_variables):
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
                if EconFeas.run_mode == 'cat':
                    # Plot the data
                    axs[i].plot(scenario_model_df['year'], scenario_model_df['value'], 
                                label=scenario, color=category_colours[category], 
                                linewidth=0.3, alpha=0.5) 
                # # Colour the lines based on the model
                # elif EconFeas.run_mode == 'model':
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

    df = pd.read_csv('ar6_econfeas.csv')
    
    snapshot_year = [2050]
    
    # # filter out scenarios that are not in the categories list
    # df = df[df['Category'].isin(EconFeas.plotting_categories)]
    
    # # Make a list of unique scenarios
    # scenario_list = df['scenario'].unique().tolist()
    
    # Make a subplot for each variable that has three violins of the scenario results at snapshot year, by category
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=False, sharex=False)
    axs = axs.flatten()

    # for each variable, create a subplot, first create the dataset
    variable_int = 0
    for variable in EconFeas.variables:

        # Make a dataframe for each variable
        variable_df = df[df['variable'] == variable]

        # Make a dataframe for the snapshot year
        snapshot_df = variable_df[variable_df['year'].isin(snapshot_year)]


        # plot violins on the figure subplot for each category
        data= pd.DataFrame()
        category_int = 0
        for category in EconFeas.categories:
    
                category_df = snapshot_df[snapshot_df['Category'] == category]
                # reset index
                category_df = category_df.reset_index(drop=True)
                data[category_int] = category_df['value']

                # plot the violin
                sns.violinplot(data=data, ax=axs[variable_int], 
                               palette='tab10')
                category_int += 1
        
        # put x axis labels on the figure as per the categories
        axs[variable_int].set_xticklabels(EconFeas.categories)
        axs[variable_int].set_title(variable + str(snapshot_year[0]))
        
        # Set the ymin to zero
        axs[variable_int].set_ylim(bottom=0)
       
        # axs[variable_int].set_ylabel(snapshot_year[0])
        # axs[variable_int].set_xlabel('')
        # axs[variable_int].set_xticks([])
        # axs[variable_int].set_xticklabels(['c1', 'c3', 'c5'])
    
    # add legend to the figure with category colours
    

                
        variable_int += 1
    plt.show()
    
        
  


        
    
    plt.show()


if __name__ == "__main__":
    main()