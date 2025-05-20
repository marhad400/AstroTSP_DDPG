import tensorflow as tf
import keras
import numpy as np
from keras.optimizers.legacy import Adam
from keras.layers import BatchNormalization
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, input_dims, alpha=0.0005, beta=0.001, env=None,
                 gamma=0.999, n_actions=1, max_size=1000000, tau=0.001,
                 fc1=400, fc2=300, batch_size=64, noise=0.2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_shape=(input_dims,), n_actions=n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        
        # Store the environment for action mapping
        self.env = env
        
        # For TSP with priority/distance ratio
        self.priority_distance_ratio = True
        
        # Action bounds for the normalized output (0 to 1 for sigmoid)
        self.min_action = 0
        self.max_action = 1
        
        # Parameters for exploration
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay = 0.995  # Decay rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        
        # Use Ornstein-Uhlenbeck noise for better exploration
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.noise) * np.ones(1))
        
        # Create networks with batch normalization
        self.actor = ActorNetwork(n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(name='critic')
        self.target_actor = ActorNetwork(n_actions=n_actions,
                                         name='target_actor')
        self.target_critic = CriticNetwork(name='target_critic')

        # Use lower learning rates and gradient clipping
        self.actor_optimizer = Adam(learning_rate=alpha, clipnorm=1.0)
        self.critic_optimizer = Adam(learning_rate=beta, clipnorm=1.0)
        
        self.actor.compile(optimizer=self.actor_optimizer)
        self.critic.compile(optimizer=self.critic_optimizer)
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)
        
        # Training metrics
        self.actor_loss_history = []
        self.critic_loss_history = []
        
        # Learning rate scheduler
        self.lr_decay_steps = 0
        self.lr_decay_rate = 0.9999  # Slow decay
        
        # Huber loss for critic (more robust than MSE)
        self.critic_loss_fn = keras.losses.Huber()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        # Ensure state and new_state are properly formatted as numpy arrays
        # Handle case where state might be a tuple or other non-array type
        if isinstance(state, tuple):
            state = state[0]  # Extract the observation from the tuple
        if isinstance(new_state, tuple):
            new_state = new_state[0]  # Extract the observation from the tuple
            
        # Convert to numpy arrays and flatten
        state = np.array(state, dtype=np.float32).flatten()
        new_state = np.array(new_state, dtype=np.float32).flatten()
        
        # Ensure action is in the correct format
        if not isinstance(action, (list, np.ndarray)):
            action = np.array([action])
        
        # For TSP with priority/distance ratio, we don't need to scale the reward
        # as it's already appropriately scaled in the environment
        
        # Clip reward to prevent extreme values
        reward = np.clip(reward, -100, 100)
        
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self, observation, evaluate=False):
        if isinstance(observation, tuple):  # Ensure observation is a NumPy array
            observation = observation[0]
        
        # Convert observation to tensor
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        
        # Get raw action from actor network (output is in range [0, 1] due to sigmoid)
        actions = self.actor(state)
        
        if not evaluate:
            # Apply exploration strategies
            
            # 1. Epsilon-greedy exploration
            if np.random.random() < self.epsilon:
                # Random action between 0 and 1
                random_action = np.random.uniform(low=0, high=1, size=(1,))
                actions = tf.convert_to_tensor(random_action, dtype=tf.float32)
            
            # 2. Add Ornstein-Uhlenbeck noise for temporally correlated exploration
            noise = self.ou_noise()
            actions = actions.numpy() + noise
            
            # Ensure actions are within bounds
            actions = np.clip(actions, self.min_action, self.max_action)
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        else:
            actions = actions.numpy()
        
        # For debugging
        # print(f"Raw action from network: {actions[0]}, Epsilon: {self.epsilon:.4f}")
        
        # Return as a numpy array to ensure consistent format
        return np.array([actions[0]])  # Return as array with shape (1,)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        # Decay learning rates
        self.lr_decay_steps += 1
        if self.lr_decay_steps % 100 == 0:  # Apply decay every 100 steps
            new_actor_lr = self.actor_optimizer.learning_rate * self.lr_decay_rate
            new_critic_lr = self.critic_optimizer.learning_rate * self.lr_decay_rate
            self.actor_optimizer.learning_rate.assign(new_actor_lr)
            self.critic_optimizer.learning_rate.assign(new_critic_lr)

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        # Update critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(
                                states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            
            # Use target network for more stable learning
            target = rewards + self.gamma * critic_value_ * (1-done)
            
            # Stop gradient to prevent backpropagation through target network
            target = tf.stop_gradient(target)
            
            # Use Huber loss instead of MSE for robustness
            critic_loss = self.critic_loss_fn(target, critic_value)
            
            # Add L2 regularization to prevent overfitting
            for weight in self.critic.trainable_variables:
                critic_loss += 0.0001 * tf.nn.l2_loss(weight)  # Reduced regularization strength

        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        
        # Clip gradients to prevent exploding gradients
        critic_network_gradient, _ = tf.clip_by_global_norm(critic_network_gradient, 1.0)
        
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        # Update actor (less frequently than critic for stability)
        if self.lr_decay_steps % 2 == 0:  # Update actor every 2 steps
            with tf.GradientTape() as tape:
                new_policy_actions = self.actor(states)
                actor_loss = -self.critic(states, new_policy_actions)
                actor_loss = tf.math.reduce_mean(actor_loss)
                
                # Add entropy regularization to encourage exploration
                entropy = -tf.reduce_mean(new_policy_actions * tf.math.log(new_policy_actions + 1e-8))
                actor_loss -= 0.01 * entropy  # Encourage diverse actions
                
                # Add L2 regularization
                for weight in self.actor.trainable_variables:
                    actor_loss += 0.0001 * tf.nn.l2_loss(weight)  # Reduced regularization strength

            actor_network_gradient = tape.gradient(actor_loss,
                                                self.actor.trainable_variables)
            
            # Clip gradients
            actor_network_gradient, _ = tf.clip_by_global_norm(actor_network_gradient, 1.0)
            
            self.actor.optimizer.apply_gradients(zip(
                actor_network_gradient, self.actor.trainable_variables))
            
            # Store losses for monitoring
            self.actor_loss_history.append(float(actor_loss))
        
        self.critic_loss_history.append(float(critic_loss))

        # Update target networks less frequently for stability
        if self.lr_decay_steps % 5 == 0:  # Update targets every 5 steps
            self.update_network_parameters()
        
    def get_loss_stats(self):
        """Return the average actor and critic losses from recent training."""
        if not self.actor_loss_history:
            return 0.0, 0.0
            
        recent_actor_loss = np.mean(self.actor_loss_history[-100:])
        recent_critic_loss = np.mean(self.critic_loss_history[-100:])
        return recent_actor_loss, recent_critic_loss


class OUActionNoise:
    """
    Ornstein-Uhlenbeck process noise for better exploration in continuous action spaces.
    Provides temporally correlated noise that helps with exploration in physical environments.
    """
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)