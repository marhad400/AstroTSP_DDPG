import gym
import numpy as np
import time
from ddpg_tf2 import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    print("Starting DDPG training...")
    # Set max_episode_steps explicitly to ensure episodes terminate
    env = gym.make('Pendulum-v1', max_episode_steps=200)
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])
    n_games = 250

    figure_file = 'plots/pendulum.png'

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        print("Loading checkpoint and initializing memory...")
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation, _ = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, _, info = env.step(action)
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
    for i in range(n_games):
        start_time = time.time()
        observation, _ = env.reset()
        done = False
        score = 0
        step_count = 0
        max_steps = 200  # Set a maximum number of steps per episode
        
        print(f"Episode {i} started")
        while not done and step_count < max_steps:
            action = agent.choose_action(observation, evaluate)
            observation_, reward, done, truncated, info = env.step(action)
            # Consider episode done if either done or truncated is True
            done = done or truncated
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
            step_count += 1
            
            # Print progress every 20 steps
            if step_count % 20 == 0:
                print(f"Episode {i}, Step {step_count}, Current score: {score:.1f}")

        episode_time = time.time() - start_time
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print(f'Episode {i}, Score: {score:.1f}, Avg score: {avg_score:.1f}, Steps: {step_count}, Time: {episode_time:.2f}s')

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
