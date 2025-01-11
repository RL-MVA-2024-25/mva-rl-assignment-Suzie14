from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from copy import deepcopy
import random
import os
from pathlib import Path
from evaluate import evaluate_HIV

env= TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)

class ProjectAgent:
    def __init__(self, state_space=env.observation_space.shape[0], action_space=env.action_space.n):
        self.action_space = action_space
        self.state_space = state_space
        self.checkpoint_path = "dqn_agent.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Replay buffer for experience storage
        self.memory = ReplayBuffer(max_size=60000, device=self.device)

        # Learning parameters
        self.learning_rate = 1e-3
        self.discount_factor = 0.99
        self.exploration_rate = 1.0
        self.min_exploration = 0.01
        self.exploration_decay = 0.99
        
        self.batch_size = 1000
        

        # Target network synchronization frequency
        self.target_sync_frequency = 1000
        self.sync_counter = 0

        # Model initialization
        self.q_network = DQN(state_dim=self.state_space, action_dim=action_space).to(self.device)
        self.target_network = DQN(state_dim=self.state_space, action_dim=action_space).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=350, gamma=0.5)

    def act(self, obs, explore=False):
        if explore and random.random() < self.exploration_rate:
            return env.action_space.sample()
        else:
            Q_values = self.q_network(torch.Tensor(obs).unsqueeze(0).to(self.device))
            return torch.argmax(Q_values).item()

    def save(self, filepath):
        torch.save(self.q_network.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load(self):
        self.q_network.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        self.target_network = deepcopy(self.q_network).to(self.device)
        self.q_network.eval()
        print("Model loaded successfully")

    def adjust_scheduler(self):
        self.lr_scheduler.step()

    def decay_exploration(self):
        if self.exploration_rate > self.min_exploration:
            self.exploration_rate *= self.exploration_decay
    
    # Greedy action selection for the agent
    def select_action(network, obs):
        device = next(network.parameters()).device
        with torch.no_grad():
            Q_values = network(torch.Tensor(obs).unsqueeze(0).to(device))
            return torch.argmax(Q_values).item()

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, terminals = self.memory.sample(self.batch_size)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            target_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            expected_q_values = rewards + self.discount_factor * target_q_values * (1 - terminals)

        loss = F.mse_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.sync_counter += 1
        if self.sync_counter % self.target_sync_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self, max_episodes=200):
        episode_rewards = []
        best_reward = float('-inf')
        for episode in range(max_episodes):
            state, _ = env.reset()
            total_reward = 0.0
            for _ in range(max_episodes):
                action = self.act(state, explore=True)
                next_state, reward, done, truncated, _info = env.step(action)

                self.memory.append(state, action, reward, next_state, done)
                self.optimize()

                state = next_state
                total_reward += reward
                if done or truncated:
                    break
            self.decay_exploration()
            self.adjust_scheduler()

            score = evaluate_HIV(agent=self, nb_episode=1)
            if total_reward > best_reward:
                best_reward = total_reward
                self.save("dqn_agent.pt")


            print(f"t {episode:4d}, Epsilon: {self.exploration_rate:.2f}, Total Reward: {int(total_reward):11d}, Score: {score:.3e} ")
            episode_rewards.append(total_reward)
            
        return episode_rewards


class ReplayBuffer:
    def __init__(self, max_size, device):
        self.max_size = int(max_size)  # Maximum capacity of the buffer
        self.buffer = []
        self.pointer = 0  # Tracks the next position to overwrite
        self.device = device

    def append(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)  # Expand the buffer until full
        self.buffer[self.pointer] = (state, action, reward, next_state, done)
        self.pointer = (self.pointer + 1) % self.max_size  # Circular buffer

    def sample(self, batch_size):
        sample_batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminals = zip(*sample_batch)
        return (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.LongTensor(np.array(actions)).to(self.device),
            torch.FloatTensor(np.array(rewards, dtype=np.float32)).to(self.device),
            torch.FloatTensor(np.array(next_states)).to(self.device),
            torch.FloatTensor(np.array(terminals, dtype=np.float32)).to(self.device)
        )

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_units=1024):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(state_dim, 256)  # First fully connected layer
        self.hidden_layer1 = nn.Linear(256, hidden_units)
        self.hidden_layer2 = nn.Linear(hidden_units, hidden_units)
        self.hidden_layer3 = nn.Linear(hidden_units, hidden_units)
        self.hidden_layer4 = nn.Linear(hidden_units, 256)  # Compress back to 256 units
        self.output_layer = nn.Linear(256, action_dim)  # Final layer maps to action space

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        x = F.relu(self.hidden_layer3(x))
        x = F.relu(self.hidden_layer4(x))
        x = self.output_layer(x)  # No activation on the output layer
        return x
    
if __name__ == "__main__":

    agent = ProjectAgent()
    epoch = agent.train()

