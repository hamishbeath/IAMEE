import env_sus
import econ_feas
import robust
import resources
import resilience
from src.utils.file_parser import *
from src.constants import *
from src.utils import mandatory_variables_scenarios
import os


# set up function that ensures all the necessary data/files are ready
def setup(categories=list, regional=True, custom_database_filename=None,
          custom_meta_filename=None):

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

    # check if the necessary files are present
    if custom_database_filename is not None:
        database_filename = custom_database_filename
    else:
        if regional:
            database_filename = 'database/AR6_Scenarios_Database_R10_regions_v1.1.csv'
        else:
            database_filename = 'database/AR6_Scenarios_Database_World_v1.1.csv'
    if custom_meta_filename is not None:
        meta_filename = custom_meta_filename
    else:
        meta_filename = 'database/meta_data.csv'
    if not os.path.exists(database_filename):
        raise FileNotFoundError('The database file is not present')
    if not os.path.exists(meta_filename):
        raise FileNotFoundError('The meta data file is not present')
    
    # check categories is a list
    assert isinstance(categories, list), 'Categories must be a list of strings'
    
    # import the database with meta data
    database = read_pyam_add_metadata(database_filename, meta_filename)

    # get the list of scenarios that have all the mandatory variables
    scenario_list = mandatory_variables_scenarios(categories, regional, FRAMEWORK_VARIABLES, database)

    #


print('Running the framework')
env_sus.main()
print('Environmental sustainability done')
econ_feas.main()
print('Economic feasibility done')
robust.main()
print('Robustness done')
resources.main()
print('Resource efficiency done')
resilience.main()
print('Resilience done')

print('Framework run complete')