import numpy as np
import pyam
import pandas as pd
from utils import Utils
from utils import Data

class NaturalResouces:

    # annual_retention_improvement = 0.01
    material_intensities = pd.read_csv('inputs/material_intensities.csv')
    material_intensities = material_intensities.set_index('tech')
    material_recycling = pd.read_csv('inputs/material_recycling.csv')
    material_reserves = pd.read_csv('inputs/material_reserves.csv')
    historical_production = pd.read_csv('inputs/mine_production_hist.csv') 
    minerals = ['Nd', 'Dy', 'Cd', 'Te', 'Se', 'In']
    material_recycling_scenario = 're_opt' # 're_opt' or 're_con'
    growth_rate_ceiling = 2 # times the maximum growth rate of the historical data
    maximum_circularity_rate = 0.9
    product_life = 20 # years
    reserve_growth_rate = 0.001 # 1% growth rate in reserves per year
    solar_base_capacity_added = 171 # GW (2022 values from IRENA)
    wind_base_capacity_added = 75 # GW (2022 values from IRENA) https://www.irena.org/News/pressreleases/2023/Mar/Record-9-point-6-Percentage-Growth-in-Renewables-Achieved-Despite-Energy-Crisis
    offshore_share = 0.5
    thin_film_share = 0.1
    material_thresholds = pd.read_csv('inputs/mineral_renewables_amounts.csv')

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

    # calculate_global_availability(NaturalResouces.material_recycling, 
    #                               NaturalResouces.material_reserves, 
    #                               NaturalResouces.historical_production)
    
    scenario_assessment_minerals(Data.dimensions_pyamdf, 
                                 NaturalResouces.minerals, 
                                 Data.model_scenarios, 
                                 NaturalResouces.material_thresholds, 
                                 2050,
                                 Data.categories)
    # calculate_base_shares_minerals()
# function with adjustable parameters that calculates the total global availability of each material
def calculate_global_availability(recycling, reserves, production):

    # firstly take the mine production in the first year and the recycling rate in the first year available. 
    # Then calculate the total availability of each material in each year by calculating the total production and recycling in that year.
    # Then calculate the total amount in that year.
    print(recycling)
    print(production)
    print(reserves)
    recycling = recycling.set_index('scenario')
    production = production.set_index('mineral')
    reserves = reserves.set_index('mineral')
    historical_availability = pd.DataFrame()
    future_availability_df = pd.DataFrame()
    all_availability = pd.DataFrame()

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
        
        stock_outflows = production.loc[mineral, first_year]
        
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
                total_availability[year] = total_availability[str(previous_year)] + production_current
                total_availability[year] -= stock_outflows
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
        # calculate the future availability of the mineral
        future_availability = {}
        future_year_production_dict = {}
        reserves_total = reserves.loc[mineral, 'reserves (tonnes)']
        reserves_remaining = reserves_total
        
        # this loop is for the future mineral availability
        
        for future_year in range(2024, 2101):
            
            # update the reserve total
            # new_reserves_found = reserves_total * NaturalResouces.reserve_growth_rate
            # print(new_reserves_found)
            # reserves_remaining += new_reserves_found

            # calculate the future recycling rate
            future_recycling_rate = rate_current + (annual_improvement * (future_year - first_future_year))
            if future_recycling_rate > NaturalResouces.maximum_circularity_rate:
                future_recycling_rate = NaturalResouces.maximum_circularity_rate
            
            # get stock outflow and number as basis for recycling from previous year
            if (future_year - 2024) > NaturalResouces.product_life:
                try:
                    stock_outflows = future_availability[future_year - NaturalResouces.product_life]
                except:
                    stock_outflows = total_availability[str(future_year - NaturalResouces.product_life)]
            else:
                stock_outflows = total_availability[str(2016)]

            # calculate the future production value depending on which data is available
            if future_year == 2024:
                future_year_production = mineral_production[-1] * (1 + average_growth_rate)
                previous_availability = total_availability[str(2023)]
            else:
                future_year_production = future_year_production_dict[2024] * (1 + (average_growth_rate * (future_year - 2024)))
                previous_availability = future_availability[future_year-1]
            # add the future production to the dictionary
            future_year_production_dict[future_year] = future_year_production
            
            # calculate the recycling value for the future year
            future_year_recycling = stock_outflows * future_recycling_rate # update this to be the figures from the availability dictionary
        
            if reserves_remaining <= 0:

                future_year_production = 0
                future_year_recycling = stock_outflows * future_recycling_rate
                future_availability[future_year] = previous_availability + future_year_production + future_year_recycling
                future_year_production_dict[future_year] = 0
                reserves_remaining = 0
            else:    
                if reserves_remaining > future_year_production:
                    future_year_production_dict[future_year] = future_year_production 
                    future_year_recycling = stock_outflows * future_recycling_rate
                    future_availability[future_year] = previous_availability + future_year_production + future_year_recycling
                    reserves_remaining -= future_year_production
                elif reserves_remaining < future_year_production:
                    future_year_production_dict[future_year] = reserves_remaining
                    future_year_recycling = stock_outflows * future_recycling_rate
                    future_availability[future_year] = previous_availability + future_year_production + future_year_recycling
                    reserves_remaining = 0
            future_availability[future_year] -= stock_outflows
            if future_availability[future_year] < 0:
                future_availability[future_year] = 0

       
        # save to the dataframe
        future_availability_df[mineral] = future_availability
        
    # concat the historical and future availability dataframes
    all_availability = pd.concat([historical_availability, future_availability_df], axis=0)
    all_availability.to_csv('outputs/mineral_availability.csv')
    print(all_availability)
            



        
        
        



        
    # print(historical_availability)


# function that takes input of the scenarios and assesses 
def scenario_assessment_minerals(pyam_df, minerals, scenario_model_list, base_thresholds, end_year, categories):

        # filter for the variables needed
    df = pyam_df.filter(variable=['Capacity|Electricity|Wind','Capacity|Electricity|Solar|PV'],region='World',
                        year=range(2020, end_year+1),
                        scenario=scenario_model_list['scenario'], 
                        model=scenario_model_list['model'])

    year_list = list(range(2030, end_year+1, 10))
    
    material_intensities = NaturalResouces.material_intensities
    wind_material_intensity = material_intensities.loc['wind_neu']
    solar_material_intensity = material_intensities.loc['solar_neu']
    mineral_availability = pd.read_csv('outputs/mineral_availability.csv', index_col=0)

    material_use_ratios = pd.DataFrame(columns=minerals)
    
    # loop through the scenarios and models
    for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']):
        
        # Filter out the data for the required scenario
        scenario_df = df.filter(scenario=scenario)
        scenario__model_df = scenario_df.filter(model=model)

        # empty dictionary to store the mineral shares with zeros 
        ratios = {keys: [] for keys in minerals}
        counter = 0
        for year in range(2030, end_year+1, 10):

            # calculate the total capacity added for solar and wind
            solar_capacity_current = scenario__model_df.filter(variable='Capacity|Electricity|Solar|PV', year=year).data['value'].values
            wind_capacity_current = scenario__model_df.filter(variable='Capacity|Electricity|Wind', year=year).data['value'].values
            solar_capacity_previous = scenario__model_df.filter(variable='Capacity|Electricity|Solar|PV', year=year-10).data['value'].values
            wind_capacity_previous = scenario__model_df.filter(variable='Capacity|Electricity|Wind', year=year-10).data['value'].values
            solar_capacity_added = solar_capacity_current - solar_capacity_previous
            wind_capacity_added = wind_capacity_current - wind_capacity_previous
            solar_capacity_added = solar_capacity_added[0]
            wind_capacity_added = wind_capacity_added[0]
            
            # calculate the material intensity for solar and wind
            scenario_solar_material_quantities = solar_material_intensity * (solar_capacity_added * NaturalResouces.thin_film_share)
            scenario_wind_material_quantities = wind_material_intensity * wind_capacity_added
            total_scenario_material_quantities = scenario_solar_material_quantities + scenario_wind_material_quantities

            # calculate the mineral quantities for solar and wind from relevant decade
            decade_mineral_quantities = mineral_availability.loc[year-10:year]
            for mineral in minerals:
                total_availability = decade_mineral_quantities[mineral].sum()
                scenario_mineral_use = total_scenario_material_quantities[mineral + ' (g/kW)']
                mineral_scenario_share = scenario_mineral_use / total_availability
                base_threshold = base_thresholds[mineral].values[0]
                # calculate the ratio of share to the threshold
                ratio = mineral_scenario_share / base_threshold
                if counter == 0:
                    ratios[mineral] = ratio
                else:
                    ratios[mineral] += ratio
            counter += 1
        # calculate the average ratio for each mineral
        average_ratios = {key: value / counter for key, value in ratios.items()}
        
        #add the average ratios to the dataframe
        average_ratios = pd.DataFrame(average_ratios, index=[0])
        material_use_ratios = pd.concat([material_use_ratios, average_ratios], axis=0)
        material_use_ratios = material_use_ratios.reset_index(drop=True)
        material_use_ratios['scenario'] = scenario_model_list['scenario']
        material_use_ratios['model'] = scenario_model_list['model']
        material_use_ratios = material_use_ratios.set_index(['scenario', 'model'])
    
    # save the dataframe to a csv
    material_use_ratios.to_csv('outputs/material_use_ratios' + str(categories) + '.csv')


def calculate_base_shares_minerals():

    # read in the mineral availability data
    mineral_availaiblility = pd.read_csv('outputs/mineral_availability.csv')
    mineral_availaiblility = mineral_availaiblility.set_index('Unnamed: 0')
    material_intensities = NaturalResouces.material_intensities
    material_intensities_wind = material_intensities.loc['wind_neu']
    material_intensities_solar = NaturalResouces.material_intensities.loc['solar_neu']

    # calculate the mineral quantities for 2022
    mineral_quantities_2022_wind = material_intensities_wind * NaturalResouces.wind_base_capacity_added
    mineral_quantities_2022_solar = material_intensities_solar * (NaturalResouces.solar_base_capacity_added * NaturalResouces.thin_film_share)
    mineral_quantities = mineral_quantities_2022_wind + mineral_quantities_2022_solar

    # empty dictionary to store the mineral shares
    mineral_renewables_amounts = {}

    # loop through and calculate 2022 shares
    for mineral in NaturalResouces.minerals:
        
        mineral_amount = mineral_availaiblility.loc[2022, mineral]
        renewables_amount = mineral_quantities[mineral + ' (g/kW)']
        print('The mineral is', mineral, 'The availability is ', mineral_amount, 'renewables consumption is', renewables_amount)
        mineral_renewables_amounts[mineral] = renewables_amount / mineral_amount    

    # save the dictionary to a dataframe
    mineral_renewables_amounts = pd.DataFrame(mineral_renewables_amounts, index=[0])
    mineral_renewables_amounts.to_csv('inputs/mineral_renewables_amounts.csv')


if __name__ == "__main__":
    main()