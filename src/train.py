from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import random
from evaluate import evaluate_HIV

# Configure the environment
env = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)

class ReplayBuffer:
    def __init__(self, max_size, device):
        self.max_size = int(max_size)
        self.buffer = []
        self.pointer = 0  # Circular buffer pointer
        self.device = device

    def append(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.pointer] = (state, action, reward, next_state, done)
        self.pointer = (self.pointer + 1) % self.max_size

    def sample(self, batch_size):
        sample_batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminals = zip(*sample_batch)
        return (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.LongTensor(np.array(actions)).to(self.device),
            torch.FloatTensor(np.array(rewards)).to(self.device),
            torch.FloatTensor(np.array(next_states)).to(self.device),
            torch.FloatTensor(np.array(terminals)).to(self.device),
        )

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=256, depth=6):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(state_dim, hidden_units)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_units, hidden_units) for _ in range(depth - 1)])
        self.output_layer = nn.Linear(hidden_units, action_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)

class ProjectAgent:
    def __init__(self, state_space=env.observation_space.shape[0], action_space=env.action_space.n):
        self.state_space = state_space
        self.action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = "dqn_agent.pt"

        self.learning_rate = 0.001
        self.discount_factor = 0.90 #few interactions
        self.exploration_rate = 1.0

        self.batch_size = 5000
        self.buffer_size = 5000000
        self.target_sync_frequency = 500
        self.gradient_steps = 1

        # Replay buffer
        self.memory = ReplayBuffer(max_size=self.buffer_size, device=self.device)

        # Models
        self.q_network = DQN(state_dim=self.state_space, action_dim=self.action_space, hidden_units=256, depth=6).to(self.device)
        self.target_network = DQN(state_dim=self.state_space, action_dim=self.action_space, hidden_units=256, depth=6).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer and scheduler
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def act(self, obs, explore=False):
        if explore and random.random() < self.exploration_rate:
            return env.action_space.sample()
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(obs_t)
            return q_values.argmax().item()

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, terminals = self.memory.sample(self.batch_size)
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.discount_factor * next_q_values * (1 - terminals)

        loss = F.smooth_l1_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.memory.__len__() % self.target_sync_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_exploration(self, step):
        if step > 10000:
            self.exploration_rate = 0.01

    def save(self, filepath):
        torch.save(self.q_network.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load(self):
        self.q_network.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        self.target_network = deepcopy(self.q_network).to(self.device)
        self.q_network.eval()
        print("Model loaded successfully")

    def train(self, max_episodes=200):
        episode_rewards = []
        best_reward = 0
        step = 0

        for episode in range(max_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.act(state, explore=True)
                next_state, reward, done, truncated, _ = env.step(action)
                self.memory.append(state, action, reward, next_state, done)
                self.optimize()
                self.decay_exploration(step)
                state = next_state
                total_reward += reward
                step += 1

                if done or truncated:
                    break

            # Evaluate performance
            score = evaluate_HIV(agent=self, nb_episode=1)

            if total_reward > best_reward:
                best_reward = total_reward
                self.save(self.checkpoint_path)

            print(f"Episode {episode:4d}, Epsilon: {self.exploration_rate:.2f}, "
                  f"Total Reward: {total_reward:.3e}, Score: {score:.3e}")
            episode_rewards.append(total_reward)

        return episode_rewards
    
    

if __name__ == "__main__":
    agent = ProjectAgent()
    episode_rewards = agent.train(max_episodes=200)
