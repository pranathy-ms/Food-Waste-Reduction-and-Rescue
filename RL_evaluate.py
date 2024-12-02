import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from RL_model import FoodWasteEnv, load_data 

# Custom function to evaluate the trained model
def evaluate_model(env, model, num_episodes=100):
    rewards = []
    surplus_metrics = []
    emissions_metrics = []
    episode_lengths = []
    best_actions = []

    for episode in range(num_episodes):
        state, _ = env.reset()  # Reset now returns two values: observation and info
        done = False
        #total_reward = 0
        total_reward = []
        all_actions = []
        steps = 0
        total_sum = 0

        # Run an episode
        while not done:
            action, _states = model.predict(state, deterministic=False)
            #print("action is", action)
            state, reward, done, truncated, _ = env.step(action)
            done = done or truncated  # Combine done and truncated
            #total_reward += reward
            total_reward.append(reward)
            all_actions.append(action)
            steps += 1
        for i in range(len(total_reward)):
            total_sum += (total_reward[i] - np.mean(total_reward)) / (np.std(total_reward)+1e-10)

        index = total_reward.index(max(total_reward))
        print("Rewards, Action: ",total_reward[:12],best_actions[:12])
        best_action = all_actions[index]
        best_actions.append(best_action)
        
        # Append metrics after each episode
        rewards.append(total_sum)

        # Run an episode
        # while not done:
        #     action, _states = model.predict(state, deterministic=False)
        #     print("ACTION IS: ",action)
        #     state, reward, done, truncated, _ = env.step(action)
        #     done = done or truncated  # Combine done and truncated
        #     total_reward += reward
        #     steps += 1
        #     env.render()

        # # Append metrics after each episode
        # rewards.append(total_reward/steps)
        surplus_metrics.append(state[0])  # Tons Surplus remaining
        emissions_metrics.append(state[2])  # CO2 emissions remaining
        episode_lengths.append(steps)

    avg_reward = np.mean(rewards)
    avg_surplus = np.mean(surplus_metrics)
    avg_emissions = np.mean(emissions_metrics)
    avg_episode_length = np.mean(episode_lengths)

    print(f"Average Reward: {avg_reward}")
    print(f"Average Tons Surplus Remaining: {avg_surplus}")
    print(f"Average CO2 Emissions Remaining: {avg_emissions}")
    print(f"Average Episode Length: {avg_episode_length}")

    solution_names = {}
    for i in range(len(best_actions)):
        current_solution = solutions_df["solution_name"][best_actions[i]]
        if current_solution not in solution_names:
            solution_names[current_solution] = 1
        else:
            solution_names[current_solution] += 1
    print(solution_names)
    for key in solution_names.keys():
        print(key,solution_names[key])

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