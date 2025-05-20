import numpy as np
import time
import matplotlib.pyplot as plt
import os
from environment import DeliveryEnv, EnvironmentPayload
from delivery_env_wrapper import DeliveryEnvWrapper
from ddpg_tf2 import Agent
from utils import plot_learning_curve

def evaluate_agent(env, agent, verbose=False):
    """
    Evaluate the agent on a single episode and return the resulting schedule and metrics.
    
    Args:
        env: The environment to evaluate on
        agent: The trained agent
        verbose: Whether to print detailed information during evaluation
    
    Returns:
        schedule: The resulting schedule (list of cities visited)
        total_distance: The total distance traveled
        total_priority: The total priority accumulated
        total_reward: The total reward accumulated
    """
    observation = env.reset()
    done = False
    schedule = [env.current_city]  # Start with the initial city
    total_distance = 0.0
    total_priority = 0.0
    total_reward = 0.0
    total_exptime = 0.0  # Track total exptime
    
    # For tracking time windows in astro data
    time_windows = []
    if hasattr(env.delivery_env, 'time_windows'):
        time_windows = env.delivery_env.time_windows
    
    # Maximum steps to prevent infinite loops
    max_steps = env.delivery_env.n_targets * 2
    step_count = 0
    
    # Track cities visited and missed observations
    cities_visited = set([env.current_city])
    missed_observations = 0
    
    while not done and step_count < max_steps:
        action = agent.choose_action(observation, evaluate=True)
        
        # Ensure action is in the correct format
        if not isinstance(action, (list, np.ndarray)):
            action = np.array([action])
        
        # Get the next city before taking the step
        next_city = env._map_action_to_city(action)
        
        # Calculate distance and priority using the previous city in the schedule
        prev_city = schedule[-1]
        distance = env.delivery_env.distance_matrix[prev_city, next_city]
        priority = env.delivery_env.priorities[next_city]
        
        # Get exptime
        exptime = env.delivery_env.get_exptime(next_city, env.current_time)
        
        # Take the step
        observation_, reward, done, info = env.step(action)
        
        # Update metrics
        total_distance += distance
        total_priority += priority
        total_reward += reward
        total_exptime += exptime
        
        # Update schedule and tracking
        if next_city not in schedule:
            schedule.append(next_city)
            cities_visited.add(next_city)
            if 'in_time_window' in info and not info['in_time_window']:
                missed_observations += 1
        
        # Update observation
        observation = observation_
        step_count += 1
        
        if verbose:
            # Fix the formatting of the action value
            action_value = float(action[0]) if isinstance(action[0], (np.ndarray, np.generic)) else action[0]
            city_info = f"City: {prev_city}, Action: {action_value:.4f}, Next City: {next_city}"
            metrics = f"Distance: {distance:.2f}, Priority: {priority}, Exptime: {exptime:.2f}, Reward: {reward:.4f}"
            
            # Add time window information if available
            time_window_info = ""
            if time_windows and len(time_windows) > next_city:
                time_window = time_windows[next_city]
                time_window_info = f", Time Window: {time_window}"
            
            in_window = info.get('in_time_window', 'Unknown')
            window_status = "✓" if in_window else "✗"
            
            print(f"{city_info}{time_window_info}, {metrics}, In Window: {window_status}")
    
    # Set the environment's schedule for formatting
    env.delivery_env.schedule = schedule
    
    # Calculate coverage
    coverage = len(cities_visited) / env.delivery_env.n_targets * 100
    
    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"Cities Visited: {len(cities_visited)}/{env.delivery_env.n_targets} ({coverage:.1f}%)")
    print(f"Missed Observations: {missed_observations}/{len(schedule) - 1}")
    print(f"Total Distance: {total_distance:.2f}")
    print(f"Total Priority: {total_priority:.2f}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Average Exptime: {total_exptime/len(schedule):.2f}")
    
    return schedule, total_distance, total_priority, total_reward

if __name__ == '__main__':
    print("Starting DDPG training for Delivery Environment...")
    
    # Check if astro_data directory exists
    astro_data_path = "astro_data"
    if not os.path.exists(astro_data_path):
        # Try alternative paths
        alternative_paths = [
            "data/astro_data",
            "priority_approach/data/astro_data",
            "../data/astro_data"
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                astro_data_path = path
                break
    
    print(f"Using astro data from: {astro_data_path}")
    
    # Create the environment payload with astro data
    payload = EnvironmentPayload(
        data_directory=astro_data_path,
        min_gen_prio=1,
        max_gen_prio=10
    )
    
    # Create the delivery environment
    delivery_env = DeliveryEnv(payload)
    
    # Wrap the environment to make it compatible with DDPG
    env = DeliveryEnvWrapper(delivery_env)
    
    # Create the DDPG agent with improved exploration
    agent = Agent(
        input_dims=env.observation_space.shape[0],
        env=env,
        n_actions=env.action_space.shape[0],
        batch_size=128,        
        noise=0.3,            
        alpha=0.0001,         
        gamma=0.99,           
        tau=0.001,            
        max_size=1000000,     
        fc1=400,              
        fc2=300
    )
    
    # Training parameters
    n_episodes = int(input("Enter the number of episodes: "))
    max_steps_per_episode = delivery_env.n_targets # Limit steps per episode
    
    # Create directory for plots
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Create directory for checkpoints
    if not os.path.exists('tmp/ddpg'):
        os.makedirs('tmp/ddpg', exist_ok=True)
    
    figure_file = 'plots/delivery_ddpg.png'
    
    best_score = float('-inf')
    best_coverage = 0.0
    score_history = []
    window_hit_rate_history = []
    coverage_history = []
    critic_loss_history = []
    actor_loss_history = []
    avg_exptime_history = []  # Track average exptime
    load_checkpoint = False
    
    if load_checkpoint:
        print("Loading checkpoint and initializing memory...")
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            n_steps += 1
            if n_steps % 10 == 0:
                print(f"Initializing memory: {n_steps}/{agent.batch_size}")
        print("Learning from initial experiences...")
        agent.learn()
        print("Loading models...")
        agent.load_models()
        evaluate = True
    else:
        evaluate = False
    
    print("Starting training loop...")
    start_time = time.time()
    
    # Evaluate periodically
    eval_interval = 10
    
    for i in range(n_episodes):
        episode_start_time = time.time()
        observation = env.reset()
        done = False
        score = 0
        step_count = 0
        
        # Track metrics for this episode
        cities_visited = set([env.current_city])
        window_hits = 0
        window_misses = 0
        exptimes = []  # Track exptimes for this episode
        
        print(f"Episode {i} started")
        while not done and step_count < max_steps_per_episode:
            action = agent.choose_action(observation, evaluate)
            
            # Ensure action is in the correct format
            if not isinstance(action, (list, np.ndarray)):
                action = np.array([action])
                
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            
            # Track metrics
            cities_visited.add(env.current_city)
            if 'in_time_window' in info:
                if info['in_time_window']:
                    window_hits += 1
                else:
                    window_misses += 1
                    
            # Track exptime
            if 'exptime' in info:
                exptimes.append(info['exptime'])
            
            if not load_checkpoint:
                agent.learn()
            
            observation = observation_
            step_count += 1
            
            # Print progress every 20 steps
            if step_count % 20 == 0:
                actor_loss, critic_loss = agent.get_loss_stats()
                print(f"Episode {i}, Step {step_count}, Score: {score:.1f}, "
                      f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, "
                      f"Epsilon: {agent.epsilon:.4f}")
        
        # Calculate metrics for this episode
        coverage = len(cities_visited) / env.delivery_env.n_targets * 100
        window_hit_rate = window_hits / (window_hits + window_misses) * 100 if (window_hits + window_misses) > 0 else 0
        avg_exptime = np.mean(exptimes) if exptimes else 0
        
        # Store metrics
        score_history.append(score)
        coverage_history.append(coverage)
        window_hit_rate_history.append(window_hit_rate)
        avg_exptime_history.append(avg_exptime)
        
        # Get loss values
        actor_loss, critic_loss = agent.get_loss_stats()
        actor_loss_history.append(actor_loss)
        critic_loss_history.append(critic_loss)
        
        # Calculate average metrics
        avg_score = np.mean(score_history[-100:])
        avg_coverage = np.mean(coverage_history[-100:])
        avg_window_hit_rate = np.mean(window_hit_rate_history[-100:])
        avg_actor_loss = np.mean(actor_loss_history[-100:]) if actor_loss_history else 0
        avg_critic_loss = np.mean(critic_loss_history[-100:]) if critic_loss_history else 0
        avg_exptime_overall = np.mean(avg_exptime_history[-100:]) if avg_exptime_history else 0
        
        episode_time = time.time() - episode_start_time
        
        # Save best model based on both score and coverage
        save_model = False
        if avg_score > best_score:
            best_score = avg_score
            save_model = True
            print("*** New best score! ***")
            
        if avg_coverage > best_coverage:
            best_coverage = avg_coverage
            save_model = True
            print("*** New best coverage! ***")
            
        if save_model and not load_checkpoint:
            agent.save_models()
            print(f"*** Models saved at episode {i} ***")
        
        print(f'Episode {i}, Score: {score:.1f}, Avg score: {avg_score:.1f}, '
              f'Coverage: {coverage:.1f}%, Avg coverage: {avg_coverage:.1f}%, '
              f'Window Hit Rate: {window_hit_rate:.1f}%, Avg Exptime: {avg_exptime:.2f}, '
              f'Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, '
              f'Steps: {step_count}, Time: {episode_time:.2f}s')
        
        # Evaluate the agent periodically
        if i % eval_interval == 0 and i > 0:
            print(f"\nEvaluating agent at episode {i}...")
            evaluate_agent(env, agent, verbose=False)
    
    total_training_time = time.time() - start_time
    print(f"Total training time: {total_training_time:.2f}s")
    
    if not load_checkpoint:
        # Plot learning curves
        plt.figure(figsize=(15, 15))
        
        # Plot score
        plt.subplot(5, 1, 1)
        x = [i+1 for i in range(n_episodes)]
        plt.plot(x, score_history)
        plt.title('Learning Curve: Score')
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        
        # Plot coverage
        plt.subplot(5, 1, 2)
        plt.plot(x, coverage_history)
        plt.title('Learning Curve: Coverage (%)')
        plt.xlabel('Episodes')
        plt.ylabel('Coverage (%)')
        
        # Plot window hit rate
        plt.subplot(5, 1, 3)
        plt.plot(x, window_hit_rate_history)
        plt.title('Learning Curve: Window Hit Rate (%)')
        plt.xlabel('Episodes')
        plt.ylabel('Window Hit Rate (%)')
        
        # Plot average exptime
        plt.subplot(5, 1, 4)
        plt.plot(x, avg_exptime_history)
        plt.title('Learning Curve: Average Exptime')
        plt.xlabel('Episodes')
        plt.ylabel('Average Exptime')
        
        # Plot losses
        plt.subplot(5, 1, 5)
        plt.plot(x, actor_loss_history, label='Actor Loss')
        plt.plot(x, critic_loss_history, label='Critic Loss')
        plt.title('Learning Curve: Losses')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(figure_file)
        plt.close()
    
    # Evaluate the trained agent
    print("\nDDPG TRAINED SCHEDULE\n--------------------")
    schedule, total_distance, total_priority, total_reward = evaluate_agent(env, agent, verbose=True)
    
    print("\nDDPG Algorithm Path:")
    print(schedule)
    
    # Display the schedule in the same format as the environment
    if hasattr(env.delivery_env, 'format_schedule'):
        print("\nFormatted DDPG Schedule:")
        # Set the environment's schedule to our result for formatting
        env.delivery_env.schedule = schedule
        print(env.delivery_env.format_schedule())
    
    # Calculate coverage percentage for each approach
    ddpg_coverage = len(set(schedule)) / delivery_env.n_targets * 100
    
    print(f"Coverage: DDPG={ddpg_coverage:.1f}%")
    
    print("\nTraining completed!") 