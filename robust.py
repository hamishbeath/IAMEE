import numpy as np
import pyam
import pandas as pd
from utils import Utils
from utils import Data

class Resilience:

    historic_emissions = pd.read_csv('inputs/historical_emissions_co2.csv')



def main() -> None:

    harmonize_emissions(Data.dimensions_pyamdf, 'Emissions|CO2', Resilience.historic_emissions, 2005, False,
              2050)



"""
Code below based on code Robin Lamboll (2024) for harmonising data
https://www.nature.com/articles/s41558-023-01848-5

"""
# Harmonize a variable in a dataframe to match a reference dataframe
def harmonize_emissions(df, var, harm_df, startyear, offset=False, unity_year=int):
    # Harmonises the variable var in the dataframe df to be equal to the values in the dataframe harmdf
    # for years before startyear up until unity_year. If offset is true, uses a linear offset tailing to 0 in unity_year.
    # If offset is false, uses a ratio correction that tends to 1 in unity_year
    harm_years = np.array([y for y in df.year if y>2005 and y<unity_year])
    print(harm_years)
    ret = df.filter(variable=var)
    print(ret)
    # harm_unit = harm_df.filter(variable=var).unit
    # to_harm_unit = ret.unit
    # assert (harm_unit == to_harm_unit) or (harm_unit == ["Mt CO2-equiv/yr"]) or (
    #     (harm_unit == ["Mt NOx/yr"]) and (to_harm_unit == ["Mt NO2/yr"])
    # ), "Invalid units {} (desired) and {} (current) for variable {}".format(
    #     harm_unit, to_harm_unit, var
    # )
    # if (harm_unit != to_harm_unit):
    #     print(
    #         "unit mismatch for  {} (desired) and {} (current) for variable {}".format(
    #             harm_unit, to_harm_unit, var)
    #     )
    #     if (harm_unit == ["Mt CO2-equiv/yr"]):
    #         df = pyam.convert_unit(df, current=to_harm_unit[0], to=harm_unit[0], context="AR6GWP100")
    #     print("Converted unit of {} to {}".format(var, ret.unit))
    assert unity_year >= max(harm_years)
    canon2015
    canon2015 = harm_df.filter(year=startyear, variable=var).data["value"]
    print(canon2015)
    if len(canon2015) != 1:
        print(canon2015)
        raise ValueError
    ret = ret.timeseries()
    canon2015 = canon2015[0]
    origval = ret[startyear].copy()
    for y in [y for y in ret.columns if y<=startyear]:
        try:
            canony = harm_df.filter(year=y, variable=var).data["value"][0]
            ret[y] = canony
        except IndexError as e:
            print(f"We have only years {harm_df.filter(variable=var).year}, need {y}")
    assert df.variable==[var]
    if not offset:
        fractional_correction_all = canon2015 / origval
        for year in [y for y in harm_years if y > startyear]:
            ret[year] *= (fractional_correction_all - 1) * (1 - (year - startyear) / (unity_year-startyear)) + 1
    else:
        offset_val = canon2015 - origval
        for year in [y for y in harm_years if y > startyear]:
            ret[year] += offset_val * (1 - (year - startyear) / (unity_year-startyear))
    return pyam.IamDataFrame(ret)





if __name__ == "__main__":
    main()