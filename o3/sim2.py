#!/usr/bin/env python
"""
A simplified simulation of a two-player (two–bacteria) game where each bacterium
has a DNA sequence (here of length 20) that determines its replication efficiency
and virus strength. On its turn a bacterium may choose one of five actions:
  0. Do nothing
  1. Mutate replication gene (first half of DNA)
  2. Mutate virus gene (second half of DNA)
  3. Attempt replication (if enough energy)
  4. Attack the opponent
The game is turn–based, resources replenish slowly, and the outcome is determined
by health, replication count, and so on.

A very simple AlphaZero–style RL agent is built using a PyTorch MLP network and an
MCTS search that calls this network for evaluations.

A basic Tkinter GUI is provided so you can play (e.g. as player 1) against the AI.
"""

import argparse
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from tkinter import messagebox

# ---------------------------
# Parameters and Configuration
# ---------------------------
DNA_LENGTH = 20  # For demonstration; change to 1000 for a “realistic” simulation.
MAX_TURNS = 50
ACTION_SPACE = 5  # 0: Do nothing, 1: Mutate replication, 2: Mutate virus, 3: Replicate, 4: Attack
ENERGY_GAIN = 5         # Energy gained from global resource per turn
TURN_ENERGY_COST = 1    # Energy cost for just existing per turn
# Energy cost for each action (action: cost)
ACTION_COST = {0: 1, 1: 2, 2: 2, 3: 5, 4: 3}
REPLICATION_ENERGY_THRESHOLD = 10  # Minimum energy needed to attempt replication
ATTACK_DAMAGE_MULTIPLIER = 10        # Damage scales with virus strength
GLOBAL_RESOURCE_MAX = 50.0           # Maximum global resource level

# ---------------------------
# Bacteria and Environment Classes
# ---------------------------

class Bacteria:
    def __init__(self):
        # DNA is a vector of integers in {0,1,2,3}
        self.DNA = np.random.randint(0, 4, size=DNA_LENGTH)
        self.health = 100.0
        self.energy = 20.0
        self.replication_count = 0
        self.update_traits()
    
    def update_traits(self):
        # First half of DNA codes for replication; second half for virus.
        rep_gene = self.DNA[:DNA_LENGTH//2]
        virus_gene = self.DNA[DNA_LENGTH//2:]
        # Normalize so that maximum possible sum is (length * 3)
        self.rep_eff = np.sum(rep_gene) / ((DNA_LENGTH//2) * 3)  # 0 to 1
        self.virus_str = np.sum(virus_gene) / ((DNA_LENGTH//2) * 3)
    
    def mutate_gene(self, gene_type):
        # gene_type 1: mutate replication gene; 2: mutate virus gene.
        if gene_type == 1:
            start, end = 0, DNA_LENGTH//2
        elif gene_type == 2:
            start, end = DNA_LENGTH//2, DNA_LENGTH
        else:
            return
        idx = random.randint(start, end - 1)
        current = self.DNA[idx]
        choices = [x for x in range(4) if x != current]
        self.DNA[idx] = random.choice(choices)
        self.update_traits()

class BacteriaEnv:
    """
    A turn-based environment for two bacteria.
    The state (for the player whose turn it is) is a 12–dimensional vector:
      [player health, energy, replication count, rep_eff, virus_str,
       opponent health, energy, replication count, rep_eff, virus_str,
       global resource, turn/MAX_TURNS]
    """
    def __init__(self):
        self.bacteria = [Bacteria(), Bacteria()]
        self.current_player = 0  # 0 or 1
        self.turn = 0
        self.global_resource = GLOBAL_RESOURCE_MAX
        self.done = False
    
    def reset(self):
        self.bacteria = [Bacteria(), Bacteria()]
        self.current_player = 0
        self.turn = 0
        self.global_resource = GLOBAL_RESOURCE_MAX
        self.done = False
        return self.get_state()
    
    def get_state(self):
        bp = self.bacteria[self.current_player]
        op = self.bacteria[1 - self.current_player]
        state = np.array([
            bp.health / 100.0,
            bp.energy / 50.0,
            bp.replication_count / 10.0,
            bp.rep_eff,
            bp.virus_str,
            op.health / 100.0,
            op.energy / 50.0,
            op.replication_count / 10.0,
            op.rep_eff,
            op.virus_str,
            self.global_resource / GLOBAL_RESOURCE_MAX,
            self.turn / MAX_TURNS
        ], dtype=np.float32)
        return state
    
    def step(self, action):
        """
        Applies the chosen action for the current bacterium.
        Returns (next_state, reward, done, info).
        """
        if self.done:
            raise Exception("Game is over")
        bp = self.bacteria[self.current_player]
        op = self.bacteria[1 - self.current_player]
        reward = 0.0
        
        # Apply energy cost for chosen action.
        cost = ACTION_COST.get(action, 1)
        bp.energy -= cost
        
        # Process the chosen action.
        if action == 0:  # Do nothing
            pass
        elif action == 1:  # Mutate replication gene
            bp.mutate_gene(1)
        elif action == 2:  # Mutate virus gene
            bp.mutate_gene(2)
        elif action == 3:  # Attempt replication
            if bp.energy >= REPLICATION_ENERGY_THRESHOLD:
                bp.replication_count += 1
                bp.energy -= 5  # extra replication cost
                # Small chance to automatically mutate replication gene after replication.
                if random.random() < 0.1:
                    bp.mutate_gene(1)
        elif action == 4:  # Attack the opponent
            damage = bp.virus_str * ATTACK_DAMAGE_MULTIPLIER
            op.health -= damage
        
        # Natural cost per turn.
        bp.energy -= TURN_ENERGY_COST
        
        # Gain energy from the global resource.
        gain = min(ENERGY_GAIN, self.global_resource)
        bp.energy += gain
        self.global_resource -= gain
        
        # Replenish global resource (up to maximum).
        self.global_resource = min(self.global_resource + 2, GLOBAL_RESOURCE_MAX)
        
        # Increase turn count every time player 2 has acted.
        if self.current_player == 1:
            self.turn += 1
        
        # Check for termination.
        if bp.health <= 0 or op.health <= 0 or self.turn >= MAX_TURNS:
            self.done = True
            # Determine winner based on death or a score combining replication count and remaining health.
            if bp.health <= 0 and op.health <= 0:
                reward = 0.0
            elif bp.health <= 0:
                reward = -1.0
            elif op.health <= 0:
                reward = 1.0
            else:
                score_bp = bp.replication_count + bp.health / 100.0
                score_op = op.replication_count + op.health / 100.0
                if score_bp > score_op:
                    reward = 1.0
                elif score_bp < score_op:
                    reward = -1.0
                else:
                    reward = 0.0
        
        # Switch turn.
        self.current_player = 1 - self.current_player
        
        next_state = self.get_state()
        return next_state, reward, self.done, {}
    
    def render(self):
        bp = self.bacteria[0]
        op = self.bacteria[1]
        print(f"Turn: {self.turn}")
        print("Bacteria 1:")
        print(f"  Health: {bp.health:.1f}, Energy: {bp.energy:.1f}, Replications: {bp.replication_count}, "
              f"RepEff: {bp.rep_eff:.2f}, VirusStr: {bp.virus_str:.2f}")
        print("Bacteria 2:")
        print(f"  Health: {op.health:.1f}, Energy: {op.energy:.1f}, Replications: {op.replication_count}, "
              f"RepEff: {op.rep_eff:.2f}, VirusStr: {op.virus_str:.2f}")
        print(f"Global Resource: {self.global_resource:.1f}")
        print(f"Current Player: {self.current_player + 1}")
        print("-" * 50)

# ---------------------------
# Neural Network (PyTorch)
# ---------------------------
class BacteriaNet(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, output_dim=ACTION_SPACE):
        super(BacteriaNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy_logits = self.policy_head(x)
        policy = F.softmax(policy_logits, dim=-1)
        value = torch.tanh(self.value_head(x))
        return policy, value

# ---------------------------
# A Very Simplified MCTS (AlphaZero–style)
# ---------------------------
class MCTSNode:
    def __init__(self, state, parent=None, action_taken=None):
        self.state = state          # The (vector) state at this node.
        self.parent = parent        # Parent node.
        self.action_taken = action_taken  # The action that led here.
        self.children = {}          # Map: action -> child node.
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = 0.0            # Prior probability from the network.
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

def ucb_score(child, parent_visits, c_puct=1.0):
    if child.visit_count == 0:
        Q = 0
    else:
        Q = child.value()
    U = c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
    return Q + U

def clone_env(env):
    """Create a shallow copy of the environment (by copying bacteria states)."""
    new_env = BacteriaEnv()
    for i in [0, 1]:
        new_env.bacteria[i].DNA = np.copy(env.bacteria[i].DNA)
        new_env.bacteria[i].health = env.bacteria[i].health
        new_env.bacteria[i].energy = env.bacteria[i].energy
        new_env.bacteria[i].replication_count = env.bacteria[i].replication_count
        new_env.bacteria[i].update_traits()
    new_env.current_player = env.current_player
    new_env.turn = env.turn
    new_env.global_resource = env.global_resource
    new_env.done = env.done
    return new_env

def simulate_action(env, action):
    """Simulate applying an action on a cloned environment; return next state, reward, done."""
    sim = clone_env(env)
    next_state, reward, done, _ = sim.step(action)
    return next_state, reward, done

def mcts_search(env, net, num_simulations=50):
    root = MCTSNode(env.get_state())
    # Use the network to get prior probabilities for the root.
    policy, value = net(root.state)
    policy = policy.detach().numpy()
    # Initialize children for all actions.
    for a in range(ACTION_SPACE):
        next_state, _, _ = simulate_action(env, a)
        child = MCTSNode(next_state, parent=root, action_taken=a)
        child.prior = policy[a]
        root.children[a] = child

    for _ in range(num_simulations):
        node = root
        sim_env = clone_env(env)
        path = [node]
        # Selection: descend until a leaf node.
        while node.children:
            best_score = -float('inf')
            best_action = None
            for action, child in node.children.items():
                score = ucb_score(child, node.visit_count + 1)
                if score > best_score:
                    best_score = score
                    best_action = action
            node = node.children[best_action]
            path.append(node)
            _, _, done = simulate_action(sim_env, best_action)
            if done:
                break
        # Evaluation: if nonterminal, use network evaluation.
        if not sim_env.done:
            policy_eval, value_eval = net(sim_env.get_state())
            value_eval = value_eval.detach().numpy()[0]
        else:
            # Terminal: use the game outcome as value.
            # (Assume reward is already computed in sim_env; here we use it as value.)
            _, reward, _, _ = sim_env.step(0)  # Dummy step; sim_env.done is True.
            value_eval = reward
        # Backpropagation.
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value_eval
            # Flip the value for the opponent.
            value_eval = -value_eval

    # Compute the search policy from visit counts.
    visits = np.array([child.visit_count for action, child in root.children.items()])
    if np.sum(visits) > 0:
        pi = visits / np.sum(visits)
    else:
        pi = np.ones(ACTION_SPACE) / ACTION_SPACE
    best_action = int(np.argmax(visits))
    return best_action, pi

# ---------------------------
# Self–Play and Training
# ---------------------------
def self_play_episode(net, num_simulations=50):
    env = BacteriaEnv()
    trajectory = []
    state = env.reset()
    while not env.done:
        action, pi = mcts_search(env, net, num_simulations)
        trajectory.append((state, pi, None))  # (state, MCTS policy, value placeholder)
        state, reward, done, _ = env.step(action)
    # Back–fill the outcome (from the perspective of each move).
    outcome = reward
    new_trajectory = []
    for (s, pi, _) in trajectory:
        new_trajectory.append((s, pi, outcome))
        outcome = -outcome  # alternate perspective each move
    return new_trajectory

def train(net, optimizer, episodes=100, num_simulations=50):
    net.train()
    for ep in range(episodes):
        trajectory = self_play_episode(net, num_simulations)
        loss_total = 0.0
        for state, pi_target, value_target in trajectory:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            pi_pred, value_pred = net(state_tensor)
            # Compute losses:
            loss_pi = -torch.sum(torch.tensor(pi_target, dtype=torch.float32) * torch.log(pi_pred + 1e-8))
            loss_v = F.mse_loss(value_pred, torch.tensor([value_target], dtype=torch.float32))
            loss = loss_pi + loss_v
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        avg_loss = loss_total / len(trajectory)
        print(f"Episode {ep+1}/{episodes}  Average Loss: {avg_loss:.4f}")
    net.eval()

# ---------------------------
# Tkinter GUI for Interactive Play
# ---------------------------
class BacteriaGUI(tk.Tk):
    def __init__(self, net):
        super().__init__()
        self.title("Bacteria Simulation Game")
        self.net = net
        self.env = BacteriaEnv()
        self.state = self.env.reset()
        
        # Display area
        self.info_text = tk.StringVar()
        self.info_label = tk.Label(self, textvariable=self.info_text, font=("Helvetica", 12), justify="left")
        self.info_label.pack(pady=10)
        
        # Action buttons
        actions = ["Do Nothing", "Mutate Replication", "Mutate Virus", "Replicate", "Attack"]
        self.buttons = []
        for i, action_name in enumerate(actions):
            btn = tk.Button(self, text=action_name, command=lambda a=i: self.player_move(a), width=20)
            btn.pack(pady=2)
            self.buttons.append(btn)
        
        self.update_display()
    
    def update_display(self):
        bp = self.env.bacteria[0]
        op = self.env.bacteria[1]
        info = f"Turn: {self.env.turn}\n"
        info += "Bacteria 1:\n"
        info += f"  Health: {bp.health:.1f}, Energy: {bp.energy:.1f}, Replications: {bp.replication_count}\n"
        info += f"  RepEff: {bp.rep_eff:.2f}, VirusStr: {bp.virus_str:.2f}\n"
        info += "Bacteria 2:\n"
        info += f"  Health: {op.health:.1f}, Energy: {op.energy:.1f}, Replications: {op.replication_count}\n"
        info += f"  RepEff: {op.rep_eff:.2f}, VirusStr: {op.virus_str:.2f}\n"
        info += f"Global Resource: {self.env.global_resource:.1f}\n"
        info += f"Current Player: {self.env.current_player + 1}\n"
        self.info_text.set(info)
    
    def player_move(self, action):
        if self.env.done:
            messagebox.showinfo("Game Over", "The game is over!")
            return
        # Process the human player's move.
        self.state, reward, done, _ = self.env.step(action)
        self.update_display()
        if done:
            self.end_game(reward)
            return
        # For the opponent’s move, let the AI decide.
        if self.env.current_player == 1:
            ai_action, _ = mcts_search(self.env, self.net, num_simulations=30)
            self.state, reward, done, _ = self.env.step(ai_action)
            self.update_display()
            if done:
                self.end_game(reward)
    
    def end_game(self, reward):
        if reward > 0:
            result = "You win!"
        elif reward < 0:
            result = "You lose!"
        else:
            result = "Draw!"
        messagebox.showinfo("Game Over", f"Game Over! {result}")

# ---------------------------
# Main function: Train or Play
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "play"], default="play",
                        help="Mode: 'train' to train the RL agent, 'play' to launch the GUI game.")
    args = parser.parse_args()
    
    net = BacteriaNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    
    if args.mode == "train":
        print("Training mode: starting self-play training...")
        train(net, optimizer, episodes=50, num_simulations=20)
        # Save the trained model if desired:
        torch.save(net.state_dict(), "bacteria_net.pt")
    else:
        print("Play mode: launching GUI...")
        # If you have a trained model, you can load it:
        # net.load_state_dict(torch.load("bacteria_net.pt"))
        app = BacteriaGUI(net)
        app.mainloop()

if __name__ == "__main__":
    main()
