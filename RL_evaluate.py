import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from RL_model import FoodWasteEnv, load_data  # Import the necessary components

# Custom function to evaluate the trained model
def evaluate_model(env, model, num_episodes=100):
    rewards = []
    surplus_metrics = []
    emissions_metrics = []
    episode_lengths = []

    for episode in range(num_episodes):
        state, _ = env.reset()  # Reset now returns two values: observation and info
        done = False
        total_reward = 0
        steps = 0

        # Run an episode
        while not done:
            action, _states = model.predict(state, deterministic=True)
            state, reward, done, truncated, _ = env.step(action)
            done = done or truncated  # Combine done and truncated
            total_reward += reward
            steps += 1

        # Append metrics after each episode
        rewards.append(total_reward)
        surplus_metrics.append(state[0])  # Tons Surplus remaining
        emissions_metrics.append(state[2])  # CO2 emissions remaining
        episode_lengths.append(steps)

    # Calculate average metrics
    avg_reward = np.mean(rewards)
    avg_surplus = np.mean(surplus_metrics)
    avg_emissions = np.mean(emissions_metrics)
    avg_episode_length = np.mean(episode_lengths)

    # Print results
    print(f"Average Reward: {avg_reward}")
    print(f"Average Tons Surplus Remaining: {avg_surplus}")
    print(f"Average CO2 Emissions Remaining: {avg_emissions}")
    print(f"Average Episode Length: {avg_episode_length}")

    # Plot rewards over episodes
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Over Episodes")
    plt.legend()
    plt.show()
    
    return rewards, surplus_metrics, emissions_metrics, episode_lengths

# Example usage with the trained model
solutions_df, surplus_df = load_data()  # Load data using the function from RL_model.py
env = FoodWasteEnv(solutions_df, surplus_df)  # Create the environment
model = PPO.load("food_waste_solution_model")  # Load the saved model
rewards, surplus_metrics, emissions_metrics, episode_lengths = evaluate_model(env, model)