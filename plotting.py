import numpy as np
import pyam
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import mplcyberpunk
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']


"""

Title: Plotting for IAM scenario anaylsis
Author: Hamish Beath
Date: November 2023
Description:
This page contains plotting functions that can be called either stand alone or from the main dimension specific scripts

"""

def main() -> None:
    
    Plotting.polar_bar_plot_variables(Plotting, 'stats_datasheet.csv', Plotting.dimensions, 'C1')


class Plotting:

    dimensions = ['economic', 'environment', 'resilience', 'resource']
    dimension_colours = {'economic': 'red', 'environment': 'green', 'resilience': 'blue', 'resource': 'orange'}
    dimension_titles = {'economic': 'Economic Feasibility', 'environment': 'Non-climate Environmental Sustainability', 'resilience': 'Societal Resilience', 'resource': 'Resource Availability'}
    
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
            
            # print(dimension_data)


            with plt.style.context('cyberpunk'):
                fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10,10))

                angles = np.linspace(0, 2*np.pi, len(dimension_data), endpoint=False)
                indexes = list(range(0, len(dimension_data)))
                width = 2*np.pi / len(dimension_data)
                angles = [element * width for element in indexes]

                label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(dimension_data))
                bars_bg = ax.bar(x = angles, height=100, width=width, color='lightgrey',
                        edgecolor='white', zorder=1, alpha=0.05)

                rankings = dimension_data["('World', '" + category +"', 'percentage')"].tolist()    
                rankings_r5 = dimension_data["('Asian countries except Japan', '" + category +"', 'percentage')"].tolist()                

                variables = dimension_data['variable']
                bars = ax.bar(x = angles, height=rankings, width=width, color=self.dimension_colours[dimension],
                        edgecolor='white', zorder=2, alpha=0.5)
                bars_r5 = ax.bar(x = angles, height=rankings_r5, width=width, color='black',
                        edgecolor='white', zorder=3, alpha=0.3)

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
                ax.legend([bars, bars_r5], ['World', 'R5'], loc='upper right', bbox_to_anchor=(1.1, 1.1))

                

                plt.show()

                # Save the figure
                fig.savefig('figures/' + dimension + '_polar_bar.png', dpi=300, bbox_inches='tight')


    # Create


if __name__ == "__main__":
    main()
