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
                         'state', 
                         'food_type']).agg(aggregation_functions)


solutions_df.reset_index(inplace=True)

solutions_df.to_csv('food_waste_solutions_processed.csv', index=False)