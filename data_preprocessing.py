import pandas as pd

solutions_df = pd.read_csv('ReFED_US_Food_Waste_Solutions_Detail.csv', skiprows=1, low_memory=False)
solutions_df = solutions_df[solutions_df['sector'] == 'Retail']
solutions_df = solutions_df.drop(columns=['sub_sector', 'sub_sector_category'])

aggregation_functions = {
    'annual_tons_diversion_potential': 'sum',
    'annual_100_year_mtco2e_reduction_potential': 'sum',
    'annual_100_year_mtch4_reduction_potential': 'sum',
    'annual_gallons_water_savings_potential': 'sum',
    'annual_meal_equivalents_diverted': 'sum',
    'jobs_created': 'sum',
    'annual_us_dollars_cost': 'sum',
    'annual_us_dollars_gross_financial_benefit': 'sum',
    'annual_us_dollars_net_financial_benefit': 'sum'
}

solutions_df = solutions_df.groupby(['solution_group', 
                         'solution_priority_action_area', 
                         'solution_name', 
                         'sector', 
                         #'state', 
                         'food_type']).agg(aggregation_functions)


solutions_df.reset_index(inplace=True)

solutions_df.to_csv('food_waste_solutions_processed.csv', index=False)

surplus_df = pd.read_csv('ReFED_US_Food_Surplus_Detail.csv', skiprows=1, low_memory=False)
surplus_df = surplus_df[surplus_df['sector'] == 'Retail']
surplus_df = surplus_df.drop(columns=['sub_sector', 'sub_sector_category', 'food_category'])

aggregation_functions = {
    'tons_surplus': 'sum',
    'tons_supply': 'sum',
    'us_dollars_surplus': 'sum',
    'tons_waste': 'sum',
    'tons_uneaten': 'sum',
    'tons_inedible_parts': 'sum',
    'tons_not_fit_for_human_consumption': 'sum',
    'tons_donations': 'sum',
    'tons_industrial_uses': 'sum',
    'tons_animal_feed': 'sum',
    'tons_anaerobic_digestion': 'sum',
    'tons_composting': 'sum',
    'tons_not_harvested': 'sum',
    'tons_incineration': 'sum',
    'tons_land_application': 'sum',
    'tons_landfill': 'sum',
    'tons_sewer': 'sum',
    'tons_dumping': 'sum',
    'surplus_upstream_100_year_mtco2e_footprint': 'sum',
    'surplus_downstream_100_year_mtco2e_footprint': 'sum',
    'surplus_total_100_year_mtco2e_footprint': 'sum',
    'surplus_upstream_100_year_mtch4_footprint': 'sum',
    'surplus_downstream_100_year_mtch4_footprint': 'sum',
    'surplus_total_100_year_mtch4_footprint': 'sum',
    'gallons_water_footprint': 'sum',
    'meals_wasted': 'sum'
}

surplus_df = surplus_df.groupby(['year', 
                         'sector', 
                         'food_type']).agg(aggregation_functions)

# Reset index to convert grouped keys into columns
surplus_df = surplus_df.reset_index()
surplus_df.to_csv('food_waste_surplus_processed.csv', index=False)
