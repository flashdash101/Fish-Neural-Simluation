import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import deque
import random
import torch.optim as optim
from collections import deque


class ReplayBuffer:
    def __init__(self, buffer_size = 10000, batch_size = 64):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, k=batch_size)
    
    def __len__(self):
        return len(self.buffer)


#Implement a simple Q network to control the fish's movement based on the distance and angle to the shark
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, seed=42):
        super(QNetwork, self).__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
#Create the DQN agent class that will use the Q network and the replay buffer to train the fish to avoid the shark
class DQNAgent:
    def __init__(self, state_size, action_size, device=None):
        # Auto-detect GPU if available
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.buffer_size = int(1e5)
        self.batch_size = 64
        self.lr = 5e-4
        self.gamma = 0.99
        self.tau = 1e-3
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999

        # Networks — moved to GPU if available
        self.qnetwork_local = QNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(self.device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)





#Define the epsilon greedy action selection method for the DQN agent
    def greedy_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
        

    #It's time to learn! Implement the learn method for the DQN agent that will sample a batch of experiences from the replay buffer and use them to update the Q network
    def learn(self):
        if len(self.memory) < self.batch_size:
            return

    

        experiences = self.memory.sample(self.batch_size)
        if experiences is None:
            return
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Guard against NaN/Inf (e.g. from unstable targets)
        if torch.isnan(loss) or torch.isinf(loss):
            self.optimizer.zero_grad()
            return None
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), max_norm=1.0) #Added gradient clipping to prevent exploding gradients
        self.optimizer.step()   

        #Soft update the target network
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

        return loss.item()
    

