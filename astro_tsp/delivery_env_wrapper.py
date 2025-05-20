import numpy as np
import gym
from gym import spaces
from environment import DeliveryEnv, EnvironmentPayload
from collections import deque

class DeliveryEnvWrapper(gym.Wrapper):
    """
    A wrapper for the DeliveryEnv to make it compatible with the DDPG agent.
    This wrapper handles the priority/distance ratio action space.
    """
    
    def __init__(self, env: DeliveryEnv):
        """
        Initialize the wrapper.
        
        Args:
            env: The DeliveryEnv instance to wrap.
        """
        super(DeliveryEnvWrapper, self).__init__(env)
        
        # Store the original environment
        self.delivery_env = env
        
        # Current city
        self.current_city = None
        
        # Current time
        self.current_time = 0.0
        
        # Define action and observation spaces
        # Action space: continuous value between 0 and 1 (will be mapped to city selection)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        
        # Observation space: current city one-hot, priority ratio, time, visited status, and recent history
        # [current_city_one_hot (n_targets) + priority_ratio (1) + time (1) + visited_status (n_targets) + recent_history (5)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 * self.delivery_env.n_targets + 7,), dtype=np.float32
        )
        
        # Store the schedule for evaluation
        self.schedule = []
        
        # Track visited cities
        self.visited_cities = []
        
        # Track missed observations
        self.missed_observations = 0
        
        # Track recent history to prevent loops
        self.recent_cities = deque(maxlen=5)
        
        # Track revisit counts for each city
        self.revisit_counts = {}
    
    def reset(self):
        """
        Reset the environment and return the initial observation.
        
        Returns:
            The initial observation.
        """
        # Reset the delivery environment
        self.current_city = self.delivery_env.reset()
        
        # Initialize time
        if hasattr(self.delivery_env, 'earliest_start') and self.delivery_env.earliest_start is not None:
            # If earliest_start is a tuple, use the first element (time)
            if isinstance(self.delivery_env.earliest_start, tuple):
                self.current_time = self.delivery_env.earliest_start[0]
            else:
                self.current_time = 0.0
        else:
            self.current_time = 0.0
        
        # Reset schedule and visited cities
        self.schedule = [self.current_city]
        self.visited_cities = [self.current_city]
        self.missed_observations = 0
        
        # Reset step count
        self.step_count = 0
        
        # Reset recent cities
        self.recent_cities = deque(maxlen=5)
        self.recent_cities.append(self.current_city)
        
        # Reset revisit counts
        self.revisit_counts = {i: 0 for i in range(self.delivery_env.n_targets)}
        
        # Return observation
        return self._get_observation()
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: A continuous value between 0 and 1 that will be mapped to a city selection
                   based on the priority/distance ratio. Can be scalar or array.
        
        Returns:
            observation: The new observation.
            reward: The reward for the action.
            done: Whether the episode is done.
            info: Additional information.
        """
        # Ensure action is in the correct format
        if not isinstance(action, (list, np.ndarray)):
            action = np.array([action])
        
        # Map the continuous action to a city selection
        next_city = self._map_action_to_city(action)
        
        # Calculate distance and priority
        distance = self.delivery_env.distance_matrix[self.current_city, next_city]
        priority = self.delivery_env.priorities[next_city]
        
        # Get the exptime for the observation at the current time
        exptime = self.delivery_env.get_exptime(next_city, self.current_time)
        
        # Calculate time passed - modified by exptime (longer exptime means more time required)
        base_time = self.delivery_env.time_to(distance)
        time_passed = base_time * exptime
        
        # Update current time
        self.current_time += time_passed
        
        # Check if the observation is within its time window
        in_time_window = False
        time_window_penalty = 0.0
        window_value = 0.0
        
        if hasattr(self.delivery_env, 'time_windows') and self.delivery_env.time_windows:
            time_windows = self.delivery_env.time_windows[next_city]
            
            # Check if current time is within any time window
            for window_start, window_val in time_windows:
                if self.current_time <= window_start:
                    in_time_window = True
                    window_value = window_val
                    break
            
            if not in_time_window:
                # Apply penalty for missing the time window
                time_window_penalty = -10.0
                self.missed_observations += 1
        
        # Take a step in the delivery environment
        new_city, base_reward, done = self.delivery_env.step(next_city, time_passed)
        
        # Check for loops - detect if we're revisiting recent cities
        is_in_recent = next_city in self.recent_cities
        
        # Update revisit count
        if next_city in self.visited_cities:
            self.revisit_counts[next_city] += 1
        
        # Calculate additional reward components
        
        # 1. Priority/distance ratio reward
        ratio_reward = (priority / max(distance, 0.1)) * 0.3
        
        # 2. Exploration reward - encourage visiting new cities (increased)
        exploration_reward = 10.0 if next_city not in self.visited_cities else 0.0
        
        # 3. Time window reward/penalty (significantly increased)
        time_reward = window_value * 2.0 if in_time_window else time_window_penalty * 2.0
        
        # 4. Penalty for revisiting cities (exponentially increasing with revisit count)
        revisit_count = self.revisit_counts.get(next_city, 0)
        revisit_penalty = -2.0 * (2 ** revisit_count) if next_city in self.visited_cities else 0.0
        
        # 5. Coverage reward - encourage visiting more unique cities
        coverage_ratio = len(self.visited_cities) / self.delivery_env.n_targets
        coverage_reward = coverage_ratio * 8.0
        
        # 6. Progress reward - encourage making progress through the environment
        progress_reward = 0.1 * self.step_count
        
        # 7. Loop penalty - strongly discourage getting stuck in loops
        loop_penalty = -20.0 if is_in_recent else 0.0
        
        # 8. Exptime efficiency reward - reward for efficient observation timing
        exptime_efficiency_reward = 8.0 * (1.0 / max(exptime, 0.1))
        
        # 9. Time window awareness reward - encourage planning for future windows
        time_window_awareness = 0.0
        if hasattr(self.delivery_env, 'time_windows') and self.delivery_env.time_windows:
            # Look ahead to next possible time window
            for window_start, window_val in self.delivery_env.time_windows[next_city]:
                if window_start > self.current_time:
                    # Reward for choosing a city that can be reached before its next window
                    travel_time = self.delivery_env.time_to(distance) * exptime
                    if self.current_time + travel_time <= window_start:
                        time_window_awareness = 5.0 * window_val
                        break
        
        # Combine rewards
        reward = base_reward + ratio_reward + exploration_reward + time_reward + revisit_penalty + coverage_reward + progress_reward + loop_penalty + exptime_efficiency_reward + time_window_awareness
        
        # Update current city
        self.current_city = new_city
        
        # Update visited cities and schedule
        if next_city not in self.visited_cities:
            self.visited_cities.append(next_city)
        self.schedule.append(next_city)
        
        # Update recent cities
        self.recent_cities.append(next_city)
        
        # Increment step count
        self.step_count += 1
        
        # Check if all cities have been visited or if we're out of time
        if len(self.visited_cities) >= self.delivery_env.n_targets:
            done = True
            # Bonus reward for visiting all cities
            reward += 100.0  # Increased bonus
        
        # Force termination if we detect a loop (same city visited 3 times in a row)
        if self.step_count > 5 and len(set(list(self.recent_cities)[-3:])) == 1:
            done = True
            reward -= 50.0  # Penalty for getting stuck
        
        # Return observation, reward, done, info
        info = {
            'in_time_window': in_time_window,
            'missed_observations': self.missed_observations,
            'visited_cities': len(self.visited_cities),
            'total_cities': self.delivery_env.n_targets,
            'coverage': len(self.visited_cities) / self.delivery_env.n_targets * 100,
            'is_loop': is_in_recent,
            'exptime': exptime  # Add exptime to info
        }
        
        return self._get_observation(), reward, done, info
    
    def _map_action_to_city(self, action):
        """
        Map the continuous action to a city selection based on the priority/distance ratio.
        
        Args:
            action: A continuous value between 0 and 1, can be scalar or array.
        
        Returns:
            The index of the selected city.
        """
        # Get the normalized action value
        # Handle both scalar and array inputs
        if isinstance(action, (list, np.ndarray)):
            if len(action) > 0:
                normalized_action = action[0]
            else:
                normalized_action = 0.5  # Default if empty array
        else:
            normalized_action = action  # Handle scalar input
        
        # Get all cities except the current one
        # We'll allow revisiting cities, but with a penalty
        available_cities = [i for i in range(self.delivery_env.n_targets) if i != self.current_city]
        
        # Filter out cities that would create a loop (visited in the last 3 steps)
        recent_cities = list(self.recent_cities)[-3:] if len(self.recent_cities) >= 3 else []
        non_loop_cities = [city for city in available_cities if city not in recent_cities]
        
        # If we have non-loop cities available, prefer those
        if non_loop_cities:
            available_cities = non_loop_cities
        
        if not available_cities:
            # If no cities are available, return a random city
            return np.random.randint(0, self.delivery_env.n_targets)
        
        # Calculate priority/distance/time ratios for all available cities
        ratios = []
        for city in available_cities:
            # Calculate base ratio (priority/distance)
            distance = self.delivery_env.distance_matrix[self.current_city, city]
            if distance <= 0:
                ratio = 0  # Invalid distance
            else:
                ratio = self.delivery_env.priorities[city] / distance
            
            # Adjust for exptime if available - lower exptime is better
            exptime = self.delivery_env.get_exptime(city, self.current_time)
            if exptime > 0:
                # Penalize cities with higher exptimes
                ratio = ratio / exptime
            
            # Adjust ratio based on time windows if available
            if hasattr(self.delivery_env, 'time_windows') and self.delivery_env.time_windows:
                time_windows = self.delivery_env.time_windows[city]
                
                # Check if we can reach this city in time for any window
                travel_time = self.delivery_env.time_to(distance) * exptime
                arrival_time = self.current_time + travel_time
                
                # Find the best time window
                best_window_value = 0
                for window_start, window_value in time_windows:
                    if arrival_time <= window_start:
                        best_window_value = window_value
                        break
                
                # Adjust ratio based on time window value
                if best_window_value > 0:
                    ratio *= (1 + best_window_value / 100)  # Boost ratio for good time windows
                else:
                    ratio *= 0.1  # Significantly reduce ratio for missed windows
            
            # Apply penalty for revisiting based on revisit count
            revisit_count = self.revisit_counts.get(city, 0)
            if city in self.visited_cities:
                ratio *= (0.5 ** revisit_count)  # Exponentially reduce appeal with each revisit
                
            ratios.append(ratio)
        
        # Normalize ratios to [0, 1]
        min_ratio = min(ratios) if ratios else 0
        max_ratio = max(ratios) if ratios else 1
        
        if min_ratio == max_ratio:
            # If all ratios are the same, return a random city
            return np.random.choice(available_cities)
        
        normalized_ratios = [(r - min_ratio) / (max_ratio - min_ratio) if max_ratio > min_ratio else 0.5 for r in ratios]
        
        # Find the city with the closest normalized ratio to the normalized action
        closest_city_idx = np.argmin(np.abs(np.array(normalized_ratios) - normalized_action))
        
        return available_cities[closest_city_idx]
    
    def _get_observation(self):
        """
        Get the current observation.
        
        Returns:
            The observation as a numpy array.
        """
        # One-hot encoding of current city
        current_city_one_hot = np.zeros(self.delivery_env.n_targets)
        current_city_one_hot[self.current_city] = 1
        
        # Calculate accumulated priority
        accumulated_priority = sum(self.delivery_env.priorities[city] 
                                 for city in self.visited_cities)
        
        # Calculate total possible priority
        total_priority = sum(self.delivery_env.priorities)
        
        # Calculate priority ratio
        priority_ratio = accumulated_priority / total_priority  # This will be between 0 and 1
        
        # Visited status for all cities
        visited_status = np.zeros(self.delivery_env.n_targets)
        for city in self.visited_cities:
            visited_status[city] = 1
        
        # Normalize current time
        if hasattr(self.delivery_env, 'earliest_start') and self.delivery_env.earliest_start is not None:
            # Normalize based on earliest start and some reasonable time horizon
            time_horizon = 86400  # 24 hours in seconds
            
            # If earliest_start is a tuple, use the first element (time)
            earliest_time = self.delivery_env.earliest_start[0] if isinstance(self.delivery_env.earliest_start, tuple) else 0.0
            normalized_time = (self.current_time - earliest_time) / time_horizon
        else:
            normalized_time = self.current_time / 100.0  # Simple normalization
        
        # Recent history (last 5 cities visited, padded with -1)
        recent_history = np.ones(5) * -1  # Initialize with -1 (no city)
        for i, city in enumerate(list(self.recent_cities)[-5:]):
            recent_history[i] = city
        
        # Normalize recent history
        recent_history = recent_history / self.delivery_env.n_targets
        
        # Combine features
        observation = np.concatenate([
            current_city_one_hot,    # Keep current city information
            [priority_ratio],        # Add priority ratio
            [normalized_time],
            visited_status,
            recent_history
        ])
        
        return observation 