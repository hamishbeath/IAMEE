import env_sus
import econ_feas
import robust
import resources
import fairness
import resilience
import transition_speed
import analysis
from utils.file_parser import *
from constants import *
from utils import mandatory_variables_scenarios
import os


categories = CATEGORIES_DEFAULT
regional = True

def main():
    """
    Main function that runs the framework analysis
    """
    # setup(categories, regional)
    # env_sus.main(categories=categories)
    # print('Environmental sustainability done')
    # econ_feas.main()
    # print('Economic feasibility done')
    # resources.main(categories=categories)
    # print('Resource efficiency done')
    # robust.main(categories=categories)
    # print('Robustness done')
    # fairness.main(categories=categories)
    # print('Fairness done')
    # resilience.main(categories=categories)
    # print('Resilience done')
    transition_speed.main(categories=categories)
    print('Transition speed done')
    analysis.main(categories=categories, run_regional=True)
    
    # fairness.main(categories=categories)
    print('Framework run complete')

# set up function that ensures all the necessary data/files are ready
def setup(categories=list, regional=True):
    """
    Function that sets up the necessary data for the analysis. This includes the following:
    - Ensuring the necessary database files are present
    - Getting a list of scenarios that have all the mandatory variables
    - Getting the baseline scenarios for the scenario list # to do item
    - Getting the pyam dataframe database snippet ready for the analysis    
    
    Inputs:
    - categories: The category(ies) that will be compared in the analysis
    - regional: Whether the analysis is regional or global

    Outputs:
    - The pyam dataframe database snippet ready for the analysis
    - The scenario list for scenarios that have all the mandatory variables
    - The baseline scenarios for the scenario list # to do item

    """

    if not regional:
        database_filename = DATABASE_FILE
    else:
        database_filename = REGIONAL_DATABASE_FILE

    if not os.path.exists(database_filename):
        print(database_filename)
        raise FileNotFoundError('The database file is not present')
    if not os.path.exists(META_FILE):
        raise FileNotFoundError('The meta data file is not present')
    
    # check categories is a list
    assert isinstance(categories, list), 'Categories must be a list of strings'
    
    meta_data = read_meta_data(META_FILE)

    # import the database with meta data
    database = read_pyam_add_metadata(database_filename, meta_data)

    
    # check whether scenario list exists
    if not os.path.isfile(os.path.join(PROCESSED_DIR, 'Framework_scenarios' + str(categories) + '.csv')):

        print('Getting scenario list')
        # get the list of scenarios that have all the mandatory variables and save it
        scenario_list = mandatory_variables_scenarios(categories, regional, FRAMEWORK_MANDATORY_VARIABLES, database)
        save_dataframe_csv(scenario_list, os.path.join(PROCESSED_DIR, 'Framework_scenarios' + str(categories)))
        scenarios = scenario_list['scenario'].tolist()
        models = scenario_list['model'].tolist()

    else:
        print('Scenario list already in directory')
        # extract models and scenarios from csv
        scenario_list = read_csv(os.path.join(PROCESSED_DIR, 'Framework_scenarios' + str(categories) + '.csv'))
        scenarios = scenario_list['scenario'].tolist()
        models = scenario_list['model'].tolist()


    if not os.path.exists(os.path.join(PROCESSED_DIR, 'Framework_pyam' + str(categories))):

        print('Saving required framework data')
        # get the pyam dataframe database snippet ready for the analysis
        if not regional:
            pyam_df = database.filter(scenario=scenarios, model=models, variable=FRAMEWORK_VARIABLES,
                                    year=range(2020, 2101), region='World', Category=categories)
            
        else:
            pyam_df = database.filter(scenario=scenarios, model=models, 
                                    variable=FRAMEWORK_VARIABLES,
                                    year=range(2020, 2101), region=R10_CODES, 
                                    Category=categories)
            world_pyam = read_pyam_add_metadata(DATABASE_FILE, meta_data)
            pyam_df = pyam_df.append(world_pyam.filter(scenario=scenarios, model=models, 
                                                    variable=FRAMEWORK_VARIABLES,
                                    year=range(2020, 2101), region='World', 
                                    Category=categories))
        
        # save the pyam dataframe database snippet ready for the analysis
        save_pyam_dataframe_csv(pyam_df, os.path.join(PROCESSED_DIR, 'Framework_pyam' + str(categories)))

    else:
        print('Framework data already in directory')


if __name__ == '__main__':
    main()




