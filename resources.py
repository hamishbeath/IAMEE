import numpy as np
import pyam
import pandas as pd
from utils import Utils
from utils import Data

class NaturalResouces:

    # annual_retention_improvement = 0.01
    material_intensities = pd.read_csv('inputs/material_intensities.csv')
    material_recycling = pd.read_csv('inputs/material_recycling.csv')
    material_reserves = pd.read_csv('inputs/material_reserves.csv')
    historical_production = pd.read_csv('inputs/mine_production_hist.csv') 
    minerals = ['Nd', 'Dy', 'Cd', 'Te', 'Se', 'In']
    material_recycling_scenario = 're_con' # 're_opt' or 're_con'
    growth_rate_ceiling = 2 # times the maximum growth rate of the historical data
    maximum_circularity_rate = 0.75

"""
Will need function that calculates the total global availability (at decadal intervals)
of each of the chosen minerals. 

This will need as inputs:
1) current production and recycling rates
2) assumptions about growth in mining 
3) assumptions about improvements in recycling levels of each

Outputs:
CSV file with decedal levels of total global availability of each material

"""



def main() -> None:

    calculate_global_availability(NaturalResouces.material_recycling, 
                                  NaturalResouces.material_reserves, 
                                  NaturalResouces.historical_production)

def calculate_global_availability(recycling, reserves, production):

    # firstly take the mine production in the first year and the recycling rate in the first year available. 
    # Then calculate the total availability of each material in each year by calculating the total production and recycling in that year.
    # Then calculate the total amount in that year.
    print(recycling)
    print(production)
    recycling = recycling.set_index('scenario')
    production = production.set_index('mineral')

    historical_availability = pd.DataFrame()
    
    for mineral in NaturalResouces.minerals:

        # extract the relevant data for the mineral
        relevant_data = recycling[mineral]
        recycling_current_rate = relevant_data.loc['re_cur']
        future_recycling_rate = relevant_data.loc[NaturalResouces.material_recycling_scenario]
        mineral_production = production.loc[mineral]

        # establish the first year in the production data (column names are years)
        first_year = production.columns[0]
        total_recycling = {}
        total_availability = {}
        for year in production.columns:
            int_year = int(year)    
            previous_year = int_year - 1
            if year == first_year:
                # calculate the total availability of the mineral in the first year
                production_current = mineral_production[year]
                recycling_current = production_current * recycling_current_rate
                
                # add the recycling and production to the dictionary
                total_recycling[year] = recycling_current
                total_availability[year] = production_current 
            else:
                
                # add the recycling from the previous year to the current year's production
                production_current = mineral_production[year]
                production_current = production_current + total_recycling[str(previous_year)]

                # calculate the recycling in the current year
                recycling_current = production_current * recycling_current_rate

                # add the recycling and production to the dictionary
                total_recycling[year] = recycling_current
                total_availability[year] = production_current

        # add the dictionary to the dataframe
        historical_availability[mineral] = total_availability

        # find the maximum growth rate and the average growth rate of the historical data
        max_growth_rate = 0
        growth_rates = []
        for year in range(0, len(mineral_production)):
            if year == 0:
                pass     
            else:
                
                current_production = mineral_production[year]
                previous_production = mineral_production[year-1]
                growth_rate = (current_production - previous_production) / previous_production
                if growth_rate > max_growth_rate:
                    max_growth_rate = growth_rate   
                if growth_rate > 0:
                    growth_rates.append(growth_rate)
        
        average_growth_rate = sum(growth_rates) / len(growth_rates)
        # print(mineral, max_growth_rate, average_growth_rate)

        # calculate the annual improvement in the recycling rate based on the 2050 recycling rate and the current recycling rate
        
        first_future_year = 2023
        rate_2050 = future_recycling_rate
        rate_current = recycling_current_rate
        annual_improvement = (rate_2050 - rate_current) / (2050 - int(first_future_year))
        print('annual improvement', annual_improvement)

        # calculate the future availability of the mineral
        future_availability = {}
        
        for future_year in range(2023, 2100):
            print('the future year is', future_year)
            # calculate the future recycling rate
            future_recycling_rate = rate_current + (annual_improvement * (future_year - first_future_year))
            if future_recycling_rate > NaturalResouces.maximum_circularity_rate:
                future_recycling_rate = NaturalResouces.maximum_circularity_rate
            print(future_recycling_rate)
            # # calculate the future production
            # future_production = mineral_production[year] * (1 + average_growth_rate)

            # # calculate the future recycling
            # future_recycling = future_production * future_recycling_rate

            # # add the future recycling and production to the dictionary
            # future_availability[future_year] = future_production + future_recycling

        break
        



        
    # print(historical_availability)

if __name__ == "__main__":
    main()