"""
Download vetted SCI 2025 scenarios for IAMEE inspection at R10 resolution.

This script intentionally avoids the AR6-oriented pyam download path because the
SCI ixmp4 platform uses different metadata keys and one different mandatory
variable name.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The active local environment has an old compiled numexpr against NumPy 2.
# Pandas can run without it, and blocking the optional import keeps ixmp4 usable.
sys.modules["numexpr"] = None

import ixmp4
import pandas as pd

from constants import FRAMEWORK_MANDATORY_VARIABLES, FRAMEWORK_VARIABLES


PLATFORM = "scenariocompass"
CATEGORY_KEY = "Climate Category|SCI 2025 [Tier I]"
VETTING_KEY = "Vetting|SCI 2025"
CATEGORIES = ["GW2", "GW3", "GW4"]
VETTING_VALUE = "ok"
EXCLUDED_IAMEE_VARIABLES = [
    "Food Demand|Crops",
    "Food Demand|Livestock",
    "Price|Secondary Energy|Electricity",
]

SCI_CCS_BIOMASS = "Carbon Capture|Geological Storage|Biomass"
AR6_CCS_BIOMASS = "Carbon Sequestration|CCS|Biomass"
SCI_LAND_USE_REMOVAL = "Carbon Removal|Land Use"
AR6_LAND_USE_SEQUESTRATION = "Carbon Sequestration|Land Use"
SCI_FOOD_AVAILABILITY_CROPS = "Food Availability|Crops [per capita]"
AR6_FOOD_DEMAND_CROPS = "Food Demand|Crops"
SCI_FOOD_AVAILABILITY_LIVESTOCK = "Food Availability|Livestock [per capita]"
AR6_FOOD_DEMAND_LIVESTOCK = "Food Demand|Livestock"
SCI_PRIMARY_FOREST = "Land Cover|Forest"
AR6_NATURAL_FOREST = "Land Cover|Forest|Natural Forest"
SCI_DAC_REMOVAL = "Carbon Removal|Geological Storage|Direct Air Capture"
AR6_DAC_SEQUESTRATION = "Carbon Sequestration|Direct Air Capture"

R10_REGIONS = [
    "Africa (R10)",
    "China+ (R10)",
    "Europe (R10)",
    "India+ (R10)",
    "Latin America (R10)",
    "Middle East (R10)",
    "North America (R10)",
    "Pacific OECD (R10)",
    "Reforming Economies (R10)",
    "Rest of Asia (R10)",
]

R10_REGION_TO_IAMEE_CODE = {
    "Africa (R10)": "R10AFRICA",
    "China+ (R10)": "R10CHINA+",
    "Europe (R10)": "R10EUROPE",
    "India+ (R10)": "R10INDIA+",
    "Latin America (R10)": "R10LATIN_AM",
    "Middle East (R10)": "R10MIDDLE_EAST",
    "North America (R10)": "R10NORTH_AM",
    "Pacific OECD (R10)": "R10PAC_OECD",
    "Reforming Economies (R10)": "R10REF_ECON",
    "Rest of Asia (R10)": "R10REST_ASIA",
}

OUTPUT_STEM = "SCI_GW2_GW3_GW4_vetted_R10_no_food_price"
OUTPUT_DIR = Path("database")
PROCESSED_DIR = Path("data/processed")


def sci_variable(variable: str) -> str:
    """Map AR6/IAMEE mandatory variable names to the SCI equivalent."""
    if variable == AR6_CCS_BIOMASS:
        return SCI_CCS_BIOMASS
    if variable == AR6_LAND_USE_SEQUESTRATION:
        return SCI_LAND_USE_REMOVAL
    if variable == AR6_FOOD_DEMAND_CROPS:
        return SCI_FOOD_AVAILABILITY_CROPS
    if variable == AR6_FOOD_DEMAND_LIVESTOCK:
        return SCI_FOOD_AVAILABILITY_LIVESTOCK
    if variable == AR6_NATURAL_FOREST:
        return SCI_PRIMARY_FOREST
    if variable == AR6_DAC_SEQUESTRATION:
        return SCI_DAC_REMOVAL
    return variable


def selected_iamee_variables(variables: list[str]) -> list[str]:
    """Drop SCI application indicators that will not be used in scoring."""
    return [v for v in variables if v not in EXCLUDED_IAMEE_VARIABLES]


def get_platform() -> ixmp4.Platform:
    os.environ.setdefault("IXMP4_STORAGE_DIRECTORY", "/private/tmp/ixmp4-data")
    return ixmp4.Platform(PLATFORM)


def get_candidate_scenarios(platform: ixmp4.Platform) -> pd.DataFrame:
    keys = [
        CATEGORY_KEY,
        "Climate Category|SCI 2025 [Tier II]",
        "Climate Category|SCI 2025 [Tier III]",
        VETTING_KEY,
    ]
    meta = platform.meta.tabulate(key__in=keys)
    wide = (
        meta.pivot_table(
            index=["model", "scenario", "version"],
            columns="key",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    mask = wide[CATEGORY_KEY].isin(CATEGORIES) & (
        wide[VETTING_KEY].astype(str).str.lower() == VETTING_VALUE
    )
    return wide.loc[mask].copy()


def download_iamc(
    platform: ixmp4.Platform,
    scenarios: pd.DataFrame,
    variables: list[str],
    regions: list[str],
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for model, model_scenarios in scenarios.groupby("model", sort=True):
        df = platform.iamc.tabulate(
            model={"name": model},
            scenario={"name__in": sorted(model_scenarios["scenario"].unique())},
            variable={"name__in": variables},
            region={"name__in": regions},
            step_year__gte=2020,
            step_year__lte=2100,
        )
        if not df.empty:
            chunks.append(df)

    if not chunks:
        return pd.DataFrame(
            columns=["model", "scenario", "version", "region", "variable", "unit", "year", "value"]
        )

    data = pd.concat(chunks, ignore_index=True)
    return data.merge(
        scenarios[["model", "scenario", "version"]].drop_duplicates(),
        on=["model", "scenario", "version"],
        how="inner",
    )


def scenarios_with_complete_r10_coverage(
    data: pd.DataFrame, mandatory_variables: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = len(mandatory_variables) * len(R10_REGIONS)

    coverage = (
        data.drop_duplicates(["model", "scenario", "version", "region", "variable"])
        .groupby(["model", "scenario", "version"])
        .size()
        .rename("reported_r10_variable_pairs")
        .reset_index()
    )
    coverage["required_r10_variable_pairs"] = required
    coverage["has_all_mandatory_variables_at_all_r10_regions"] = (
        coverage["reported_r10_variable_pairs"] == required
    )

    complete = coverage.loc[
        coverage["has_all_mandatory_variables_at_all_r10_regions"],
        ["model", "scenario", "version"],
    ]
    return complete, coverage


def add_metadata(data: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    return data.merge(metadata, on=["model", "scenario", "version"], how="left")


def write_outputs(
    selected: pd.DataFrame,
    metadata: pd.DataFrame,
    regional: pd.DataFrame,
    world: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    selected.to_csv(PROCESSED_DIR / f"{OUTPUT_STEM}_selected_scenarios.csv", index=False)
    coverage.to_csv(PROCESSED_DIR / f"{OUTPUT_STEM}_coverage_summary.csv", index=False)
    metadata.to_csv(OUTPUT_DIR / f"{OUTPUT_STEM}_metadata.csv", index=False)
    regional.to_csv(OUTPUT_DIR / f"{OUTPUT_STEM}_raw_regions.csv", index=False)
    world.to_csv(OUTPUT_DIR / f"{OUTPUT_STEM}_world.csv", index=False)

    mapped = regional.copy()
    mapped["region_raw"] = mapped["region"]
    mapped["region"] = mapped["region"].map(R10_REGION_TO_IAMEE_CODE)
    mapped.to_csv(OUTPUT_DIR / f"{OUTPUT_STEM}_iamee_region_codes.csv", index=False)

    excluded = pd.DataFrame(
        {
            "excluded_iamee_variable": EXCLUDED_IAMEE_VARIABLES,
            "excluded_sci_variable": [sci_variable(v) for v in EXCLUDED_IAMEE_VARIABLES],
        }
    )
    excluded.to_csv(PROCESSED_DIR / f"{OUTPUT_STEM}_excluded_variables.csv", index=False)


def main() -> None:
    platform = get_platform()

    mandatory_iamee_variables = selected_iamee_variables(FRAMEWORK_MANDATORY_VARIABLES)
    download_iamee_variables = selected_iamee_variables(FRAMEWORK_VARIABLES)
    mandatory_variables = [sci_variable(v) for v in mandatory_iamee_variables]
    download_variables = [sci_variable(v) for v in download_iamee_variables]

    candidates = get_candidate_scenarios(platform)
    print(f"Candidate scenarios after {CATEGORY_KEY}={CATEGORIES} and {VETTING_KEY}=ok: {len(candidates)}")
    print("Excluded SCI-application variables:")
    for variable in EXCLUDED_IAMEE_VARIABLES:
        print(f"- {variable} -> {sci_variable(variable)}")

    coverage_data = download_iamc(platform, candidates, mandatory_variables, R10_REGIONS)
    complete_keys, coverage = scenarios_with_complete_r10_coverage(
        coverage_data, mandatory_variables
    )

    selected = candidates.merge(complete_keys, on=["model", "scenario", "version"], how="inner")
    print(f"Scenarios with all retained mandatory variables in all ten R10 regions: {len(selected)}")

    regional = download_iamc(platform, selected, download_variables, R10_REGIONS)
    world = download_iamc(platform, selected, download_variables, ["World"])
    metadata = selected.copy()

    regional = add_metadata(regional, metadata)
    world = add_metadata(world, metadata)

    write_outputs(selected, metadata, regional, world, coverage)

    print("Wrote:")
    print(f"- {PROCESSED_DIR / (OUTPUT_STEM + '_selected_scenarios.csv')}")
    print(f"- {PROCESSED_DIR / (OUTPUT_STEM + '_coverage_summary.csv')}")
    print(f"- {PROCESSED_DIR / (OUTPUT_STEM + '_excluded_variables.csv')}")
    print(f"- {OUTPUT_DIR / (OUTPUT_STEM + '_metadata.csv')}")
    print(f"- {OUTPUT_DIR / (OUTPUT_STEM + '_raw_regions.csv')}")
    print(f"- {OUTPUT_DIR / (OUTPUT_STEM + '_iamee_region_codes.csv')}")
    print(f"- {OUTPUT_DIR / (OUTPUT_STEM + '_world.csv')}")


if __name__ == "__main__":
    main()
