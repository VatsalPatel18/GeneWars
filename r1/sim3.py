import numpy as np
import random
from enum import Enum
import pygame
from pygame.locals import *
import torch
import torch.nn as nn
import torch.optim as optim

class ActionType(Enum):
    MOVE = 0
    REPLICATE = 1
    ATTACK = 2
    COLLECT = 3

class Bacterium:
    def __init__(self, dna=None, position=(0,0)):
        self.health = 100
        self.position = position
        self.resources = 0
        self.dna = dna if dna else {
            'replication': np.random.rand(10),
            'attack': np.random.rand(10),
            'defense': np.random.rand(10),
            'metabolism': np.random.rand(10)
        }
        
    def mutate(self, mutation_rate=0.01):
        for key in self.dna:
            if random.random() < mutation_rate:
                idx = random.randint(0,9)
                self.dna[key][idx] = np.random.rand()

class BioEnvironment:
    def __init__(self, size=20):
        self.size = size
        self.grid = np.zeros((size, size))
        self.resources = np.random.randint(0, 10, (size, size))
        self.players = []
        self.max_resources = 10
        self.resource_regrowth = 0.1
        
    def add_player(self, bacterium):
        self.players.append(bacterium)
        
    def step(self):
        # Regrow resources
        self.resources = np.minimum(
            self.resources + self.resource_regrowth,
            self.max_resources
        )
        return self.get_state()
    
    def get_state(self):
        # Return state representation for neural network
        state = []
        for p in self.players:
            state += [
                p.health/100,
                p.resources/self.max_resources,
                *p.dna['replication'],
                *p.dna['attack'],
                *p.dna['defense']
            ]
        return np.array(state)
    

class BioNet(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(128, action_size)
        self.value_head = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.fc(x)
        policy = torch.softmax(self.policy_head(x), dim=-1)
        value = torch.tanh(self.value_head(x))
        return policy, value
    

class BioMCTS:
    def __init__(self, network, env, simulations=100):
        self.network = network
        self.env = env
        self.simulations = simulations
        
    def search(self, current_player):
        # Simplified MCTS for biological simulation
        root = MCTSNode(env=self.env.copy())
        
        for _ in range(self.simulations):
            node = root
            env_copy = self.env.copy()
            
            # Selection
            while not node.is_leaf():
                action = node.select_action()
                env_copy.apply_action(action)
                node = node.children[action]
                
            # Expansion
            if not env_copy.game_over():
                policy, value = self.network(torch.tensor(env_copy.get_state()).float())
                node.expand(policy.detach().numpy())
                
            # Backpropagation
            while node is not None:
                node.update(value.item())
                node = node.parent
                value = -value  # Alternate player perspective
                
        return root.get_action_probs()
    

def train():
    env = BioEnvironment()
    players = [Bacterium() for _ in range(2)]
    network = BioNet(input_size=env.get_state().size, action_size=4)
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    
    for episode in range(1000):
        # Self-play
        mcts = BioMCTS(network, env)
        states = []
        policies = []
        values = []
        
        while not env.game_over():
            state = env.get_state()
            action_probs = mcts.search(env.current_player)
            
            # Store training data
            states.append(state)
            policies.append(action_probs)
            
            # Choose action
            action = np.random.choice(len(action_probs), p=action_probs)
            env.apply_action(action)
            
            # Get value estimate
            _, value = network(torch.tensor(state).float())
            values.append(value.item())
            
        # Train network
        optimizer.zero_grad()
        policy_pred, value_pred = network(torch.tensor(states).float())
        loss = policy_loss(policy_pred, policies) + value_loss(value_pred, values)
        loss.backward()
        optimizer.step()

class BioVisualizer:
    def __init__(self, env):
        pygame.init()
        self.env = env
        self.cell_size = 30
        self.screen = pygame.display.set_mode(
            (env.size*self.cell_size, env.size*self.cell_size))
        
    def draw(self):
        self.screen.fill((255,255,255))
        
        # Draw resources
        for y in range(self.env.size):
            for x in range(self.env.size):
                intensity = int(255 * (self.env.resources[y,x]/self.env.max_resources))
                pygame.draw.rect(self.screen, (0, intensity, 0),
                                (x*self.cell_size, y*self.cell_size,
                                 self.cell_size, self.cell_size))
                                 
        # Draw bacteria
        for p in self.env.players:
            x, y = p.position
            pygame.draw.circle(self.screen, (0,0,255),
                              (x*self.cell_size + self.cell_size//2,
                               y*self.cell_size + self.cell_size//2),
                              self.cell_size//3)
        
        pygame.display.flip()


if __name__ == "__main__":
    env = BioEnvironment()
    vis = BioVisualizer(env)
    
    # Add bacteria
    env.add_player(Bacterium(position=(5,5)))
    env.add_player(Bacterium(position=(15,15)))
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
                
        env.step()
        vis.draw()
        pygame.time.wait(100)
        
    pygame.quit()