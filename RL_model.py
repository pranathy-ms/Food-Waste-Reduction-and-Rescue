# RL_model.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Custom Environment Definition
class FoodWasteEnv(gym.Env):
    def __init__(self, solutions_df, surplus_df):
        super(FoodWasteEnv, self).__init__()
        
        # Store the dataframes
        self.solutions_df = solutions_df
        self.surplus_df = surplus_df
        
        # Define action space: each action corresponds to applying a specific solution
        self.action_space = spaces.Discrete(len(self.solutions_df))
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(5,), dtype=np.float32
        )
        
        # Initialize state
        self.state = self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        self.state = np.array([
            self.surplus_df['tons_surplus'].sum(),
            self.surplus_df['tons_waste'].sum(),
            self.surplus_df['surplus_total_100_year_mtco2e_footprint'].sum(),
            self.surplus_df['gallons_water_footprint'].sum(),
            self.surplus_df['us_dollars_surplus'].sum()
        ])
        return self.state, {}

    def step(self, action):
        solution = self.solutions_df.iloc[action]
        
        tons_diversion = solution['annual_tons_diversion_potential']
        co2_reduction = solution['annual_100_year_mtco2e_reduction_potential']
        water_savings = solution['annual_gallons_water_savings_potential']
        financial_benefit = solution['annual_us_dollars_net_financial_benefit']
        
        self.state = np.array([
            max(self.state[0] - tons_diversion, 0),
            max(self.state[1] - tons_diversion, 0),
            max(self.state[2] - co2_reduction, 0),
            self.state[3] + water_savings,
            self.state[4] - financial_benefit
        ])

        reward = (
            tons_diversion * 0.5 + 
            co2_reduction * 0.3 + 
            water_savings * 0.1 - 
            solution['annual_us_dollars_cost'] * 0.1
        )

        terminated = bool(self.state[0] <= 0 or self.state[1] <= 0)
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def render(self, mode='human'):
        print(f"Current State: {self.state}")

# Function to load datasets
def load_data():
    solutions_df = pd.read_csv('ReFED_US_Food_Waste_Solutions_Detail.csv', skiprows=1, low_memory=False)
    surplus_df = pd.read_csv('ReFED_US_Food_Surplus_Detail.csv', skiprows=1, low_memory=False)

    # Filter for the "Retail" sector
    solutions_df = solutions_df[solutions_df['sector'] == 'Retail']
    surplus_df = surplus_df[surplus_df['sector'] == 'Retail']

    return solutions_df, surplus_df

# Function to create and train the model
def train_model():
    solutions_df, surplus_df = load_data()
    env = FoodWasteEnv(solutions_df, surplus_df)

    # Initialize and train the RL model
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    
    # Save the trained model
    model.save("food_waste_solution_model")
    return model

# Main script for training the model
if __name__ == "__main__":
    train_model()