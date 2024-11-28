columns_in_sols = [
    "solution_group",
    "solution_priority_action_area",
    "solution_name",
    "sector",
    "sub_sector",
    "sub_sector_category",
    "state",
    "food_type",
    "annual_tons_diversion_potential",
    "annual_100_year_mtco2e_reduction_potential",
    "annual_100_year_mtch4_reduction_potential",
    "annual_gallons_water_savings_potential",
    "annual_meal_equivalents_diverted",
    "jobs_created",
    "annual_us_dollars_cost",
    "annual_us_dollars_gross_financial_benefit",
    "annual_us_dollars_net_financial_benefit"
]

print("Columns in solutions:")
for index, column in enumerate(columns_in_sols, start=1):
    print(f"{index:2}. {column}")

columns_in_surplus = [
    "year", "sector", "sub_sector", "sub_sector_category", "food_type",
    "food_category", "tons_surplus", "tons_supply", "us_dollars_surplus",
    "tons_waste", "tons_uneaten", "tons_inedible_parts",
    "tons_not_fit_for_human_consumption", "tons_donations", "tons_industrial_uses",
    "tons_animal_feed", "tons_anaerobic_digestion", "tons_composting",
    "tons_not_harvested", "tons_incineration", "tons_land_application",
    "tons_landfill", "tons_sewer", "tons_dumping",
    "surplus_upstream_100_year_mtco2e_footprint",
    "surplus_downstream_100_year_mtco2e_footprint",
    "surplus_total_100_year_mtco2e_footprint",
    "surplus_upstream_100_year_mtch4_footprint",
    "surplus_downstream_100_year_mtch4_footprint",
    "surplus_total_100_year_mtch4_footprint",
    "gallons_water_footprint", "meals_wasted"
]

print("Columns in surplus:")
for index, column in enumerate(columns_in_surplus, start=1):
    print(f"{index:2}. {column}")