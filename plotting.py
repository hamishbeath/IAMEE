import numpy as np
import pyam
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import mplcyberpunk
from matplotlib import rcParams
from utils import Data
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']


"""
Title: Plotting for IAM scenario anaylsis
Author: Hamish Beath
Date: November 2023
Description:
This page contains plotting functions that can be called either stand alone or 
from the main dimension specific scripts. 

"""

def main() -> None:
    
    # Plotting.polar_bar_plot_variables(Plotting, 'stats_datasheet.csv', Plotting.dimensions, 'C1a_NZGHGs')
    Plotting.box_plot_variables(Plotting, 'variable_categories.csv', 'robust', 'C1a_NZGHGs', [2050, 2100])
    # Plotting.single_variable_box_line(Plotting, Data.c1aR10_scenarios, 'Land Cover|Forest', 'C1a_NZGHGs', 'World',years=range(2020, 2101))

class Plotting:

    dimensions = ['economic', 'environment', 'resilience', 'resource', 'robust']
    dimension_colours = {'economic': 'red', 'environment': 'green', 'resilience': 'blue', 'resource': 'orange', 'robust': 'purple'}
    dimension_titles = {'economic': 'Economic Feasibility', 'environment': 'Non-climate Environmental Sustainability', 'resilience': 'Societal Resilience', 'resource': 'Resource Availability', 'robust': 'Scenario Robustness'}
    dimention_cmaps = {'economic': 'Reds', 'environment': 'Greens', 'resilience': 'Blues', 'resource': 'Oranges', 'robust': 'Purples'}

    # Create a detailed polar bar plot that categorises 
    def polar_bar_plot_variables(self, file_name, dimensions, category):
        
        variable_data = pd.read_csv(file_name)
        variable_data = variable_data[variable_data['category_1'].isin(dimensions)]

        # print(variable_data)

        for dimension in dimensions:
            
            print(dimension)
            
            # Filter the data to only include the dimension of interest, whether it is found in category_1 or category_2
            dimension_data = variable_data[(variable_data['category_1'] == dimension) | (variable_data['category_2'] == dimension)]
            
            # Rank the variables largest to smallest on the column of world value
            dimension_data = dimension_data.sort_values(by=["('World', '" + category +"', 'percentage')"], ascending=False)
            
            with plt.style.context('fivethirtyeight'):
                fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10,10))

                angles = np.linspace(0, 2*np.pi, len(dimension_data), endpoint=False)
                indexes = list(range(0, len(dimension_data)))
                width = 2*np.pi / len(dimension_data)
                angles = [element * width for element in indexes]

                label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(dimension_data))
                bars_bg = ax.bar(x = angles, height=100, width=width, color='grey',
                        edgecolor='black', zorder=1, alpha=0.1)

                rankings = dimension_data["('World', '" + category +"', 'percentage')"].tolist()    
                rankings_r5 = dimension_data["('Asian countries except Japan', '" + category +"', 'percentage')"].tolist()                
                rankings_r10 = dimension_data["('Countries of Sub-Saharan Africa', '" + category +"', 'percentage')"].tolist()

                variables = dimension_data['variable']
                bars = ax.bar(x = angles, height=rankings, width=width, color=self.dimension_colours[dimension],
                        edgecolor='white', zorder=2, alpha=0.5)
                bars_r5 = ax.bar(x = angles, height=rankings_r5, width=width, color='black',
                        edgecolor='white', zorder=3, alpha=0.3)
                bars_r10 = ax.bar(x = angles, height=rankings_r10, width=width, color='gray',
                        edgecolor='white', zorder=4, alpha=0.3)

                for angle, height, variables in zip(angles, rankings, variables):
                    rotation_angle = np.degrees(angle)
                    if angle < np.pi:
                        rotation_angle -= 90
                    elif angle == np.pi:
                        rotation_angle -= 90
                    else:
                        rotation_angle += 90
                    ax.text(angle, 105, variables, 
                            ha='center', va='bottom', 
                            rotation=rotation_angle, rotation_mode='anchor', fontsize=9)

                ax.set_xticks([])
                ax.grid(alpha=0.1, color='white', lw=3)
                plt.ylim(0, 100)
                ax.yaxis.set_zorder(10)
                ax.yaxis.set_tick_params(zorder=10)
                
                title = self.dimension_titles[dimension] + '\n% ' + category + ' Scenarios with Variables Provided'
                # add title at the top of the plot in bold
                ax.set_title(title, fontsize=13, y=1.07, fontweight='bold')
                
                # add a legend for the different regions R5 and World
                ax.legend([bars, bars_r5, bars_r10], ['World', 'R5', 'R10'], loc='upper right', bbox_to_anchor=(1.1, 1.1))

                plt.show()

                # Save the figure
                fig.savefig('figures/' + dimension + '_polar_bar_r10.png', dpi=300, bbox_inches='tight')


    # Create a box plot showing the levels of each variable for each dimension in 
    # 2020 and 2050. The box plot contains boxes for all the variables for the 
    # dimension of interest, with 2050 and 2100 values plotted in adjacent boxes. 

    def box_plot_variables(self, file_name, dimension, category, years):
        
        # import the data
        variable_data = pd.read_csv(file_name)
        
        # filter out all variables with low R10
        variable_data = variable_data[variable_data["sufficientR10"] > 0]
        
        # Filter the data to only include the dimension of interest, whether it is found in category_1 or category_2
        dimension_data = variable_data[(variable_data['category_1'] == dimension) | (variable_data['category_2'] == dimension)]

        variables = dimension_data['variable']
        variables = variables.unique()
        variables = variables.tolist()

        # import util for getting the data 
        from utils import Utils

        df = Utils.connAr6.query(model='*', scenario='*',
            variable=variables, year=[2050,2100], region=['Countries of Sub-Saharan Africa','World']
            )
        
        df_category = df.filter(Category_subset=category)
        
        
        # Find R10 scenarios
        df_R10_scenarios = df_category.filter(region='Countries of Sub-Saharan Africa')
        df_R10_scenarios = df_R10_scenarios.filter(year=2050)
        R10_scenarios = df_R10_scenarios['scenario'].unique().tolist()
        print(R10_scenarios)


        # Filter for just R10 scenarios
        df_category = df_category.filter(scenario=R10_scenarios)

        # Filter for the world
        df_category = df_category.filter(region='World')

        # set up subplots for the number of variables
        fig, axs = plt.subplots(1, len(variables), figsize=(30,10))
        axs = axs.flatten()

        # add space between subplots
        fig.subplots_adjust(wspace=0.7)

        # loop through each variable
        for i, variable in enumerate(variables):
            

            # filter the data to only include the variable of interest
            df_variable = df_category.filter(variable=variable)
            if df_variable.empty:
                continue
            
            # make a subplot for the variable
            ax = axs[i]


            units = df_variable['unit'].unique().tolist()
            units = units[0]


            data = pd.DataFrame()
            for year in range(2050, 2150, 50):
                
                print(year)
                # get the values for the variable in the year of interest
                df_values = df_variable.filter(year=year)
                
                data[year] = df_values['value']

                # set the box plot in seaborn
            sns.boxplot(data=data, ax=ax, palette=Plotting.dimention_cmaps[dimension],showfliers=False)
            sns.stripplot(data=data, ax=ax, 
            color=".3") # get the values for the variable in the year of interest

            # set the title of the plot
            ax.set_title(variable, fontsize=10,fontweight='bold')
            
            # set the units of the y axis
            ax.set_ylabel(units)

            # Add ticks on the right hand y axis as well as left
            ax.yaxis.set_ticks_position('both')
            
        # set the title of the plot
        title = 'Box plots of ' + dimension + ' variable values for '+ category + ' scenarios in 2050 and 2100'
        fig.suptitle(title, fontsize=20)
        
        
        plt.show()


    def single_variable_box_line(self, scenarios, variable, category,region, years):
        
        # import util for getting the data 
        from utils import Utils

        df = Utils.connAr6.query(model='*', scenario=scenarios,
            variable=variable, year=years, region=region
            )
        
        df_category = df.filter(Category_subset=category)
        
        # set up subplots for the box plot and line plot.
        fig = plt.figsize=(10,15)

        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3]) 

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])


        # set the box plot in seaborn

        units = df_category['unit'].unique().tolist()
        units = units[0]
        data = pd.DataFrame()
        for year in range(2050, 2150, 50):
            
            # get the values for the variable in the year of interest
            df_values = df_category.filter(year=year)
            
            data[year] = df_values['value']

        sns.boxplot(data=data, ax=ax1, palette='cool',showfliers=False)
        sns.stripplot(data=data, ax=ax1,color=".3")     

        # set the units of the y axis
        ax1.set_ylabel(units)
        
        # Add ticks on the right hand y axis as well as left
        ax1.yaxis.set_ticks_position('both')
        ax1.xaxis.set_ticks_position('both')
        ax1.set_xlabel('Year')
        # set up the line plot showing all scenarios for the variable
        # sns.lineplot(data=df_category,x='year', y='value', hue='scenario', ax=axs[1]) #'PuBuGn'
        pyam.plotting.line(df_category, x='year', y='value', legend=None, color='scenario', ax=ax2, 
                           cmap='winter', title=False)
        
        ax2.set_xlim(2020, 2100)

        ax2.yaxis.set_ticks_position('both')
        ax2.xaxis.set_ticks_position('both')

        # set the title of the plot

        # fig.title = 'Distribution of values for ' + variable + '2020-2100'

        plt.show()

if __name__ == "__main__":
    main()
