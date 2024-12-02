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
        
        # Create a mapping from food_type to integer
        self.food_type_mapping = {food: idx for idx, food in enumerate(surplus_df['food_type'].unique())}
        
        # Define observation space including food_type as an integer index
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(6,), dtype=np.float32
        )
        
        # Initialize state and valid actions
        self.state = None
        self.valid_actions = None
        
        # Initialize environment state
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        # Initialize state with the first food type and its corresponding data
        first_food_type = self.surplus_df['food_type'].iloc[0]
        self.state = np.array([
            self.surplus_df['tons_surplus'].sum(),
            self.surplus_df['tons_waste'].sum(),
            self.surplus_df['surplus_total_100_year_mtco2e_footprint'].sum(),
            self.surplus_df['gallons_water_footprint'].sum(),
            self.surplus_df['us_dollars_surplus'].sum(),
            self.food_type_mapping[first_food_type]  # Map food type to integer
        ])
        
        # Filter actions for the initial food type
        self.valid_actions = self.get_valid_actions(first_food_type)
        
        # Define action space based on valid actions for current food type
        self.action_space = spaces.Discrete(len(self.valid_actions))
        
        return self.state, {}

    def get_valid_actions(self, food_type):
        """Return a list of valid action indices for the given food type."""
        return list(self.solutions_df[self.solutions_df['food_type'] == food_type].index)

    def step(self, action_index):
        current_food_type_index = int(self.state[5])
        
        # Get the current food type string from the mapping
        current_food_type = list(self.food_type_mapping.keys())[current_food_type_index]
        
        # Ensure action is within valid actions for current food type
        if action_index >= len(self.valid_actions):
            raise ValueError("Invalid action index for current food type.")
        
        # Map action index to actual solution index in DataFrame
        solution_index = self.valid_actions[action_index]
        solution = self.solutions_df.loc[solution_index]
        
        tons_diversion = solution['annual_tons_diversion_potential']
        co2_reduction = solution['annual_100_year_mtco2e_reduction_potential']
        water_savings = solution['annual_gallons_water_savings_potential']
        financial_benefit = solution['annual_us_dollars_net_financial_benefit']
        
        # Update state based on selected solution's impact
        self.state = np.array([
            max(self.state[0] - tons_diversion, 0),
            max(self.state[1] - tons_diversion, 0),
            max(self.state[2] - co2_reduction, 0),
            self.state[3] - water_savings,
            self.state[4] - financial_benefit,
            current_food_type_index  # Keep current food type unchanged for simplicity
        ])

        reward = (
            tons_diversion * 0.3 + 
            co2_reduction * 0.2 + 
            water_savings * 0.2 +
            financial_benefit * 0.2 - 
            solution['annual_us_dollars_cost'] * 0.1
        )

        terminated = bool(self.state[0] <= 10000000 or self.state[1] <= 10000000)
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def render(self, mode='human'):
        print(f"Current State: {self.state}")

# Function to load datasets and create mappings
def load_data():
    solutions_df = pd.read_csv('food_waste_solutions_processed.csv', low_memory=False)
    surplus_df = pd.read_csv('food_waste_surplus_processed.csv', low_memory=False)

    return solutions_df, surplus_df

# Function to create and train the model
def train_model():
    solutions_df, surplus_df = load_data()
    env = FoodWasteEnv(solutions_df, surplus_df)

    hyperparameters = {
        'learning_rate': 0.003,
        'n_steps': 2048,
        'batch_size': 16,
        'n_epochs': 10,
        'gamma': 0.99,
        'ent_coef': 0.01,
        'clip_range': 0.2,
        'tensorboard_log': './ppo_tensorboard'
    }
    
    # Initialize and train the RL model
    model = PPO("MlpPolicy", env, verbose=1, **hyperparameters)
    model.learn(total_timesteps=10000)
    
    # Save the trained model
    model.save("food_waste_solution_model")
    return model

# Main script for training the model
if __name__ == "__main__":
    train_model()