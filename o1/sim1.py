#!/usr/bin/env python3

"""
A minimal AlphaZero-style pipeline for a 'Two-Bacteria' game environment.
--------------------------------------------------------
Author: ChatGPT (adapted from the reference alpha_zero code structure)
License: MIT (same spirit as the alpha zero reference code)

Run:
  python bacteria_alpha_zero.py --train
  python bacteria_alpha_zero.py --eval
  python bacteria_alpha_zero.py --gui

This single-file script demonstrates:
1) A toy BacteriaEnv environment (OpenAI Gym-like).
2) A small MCTS-based AlphaZero approach in PyTorch.
3) A training function that performs self-play, then trains a network from the gathered data.
4) A tiny Tkinter-based GUI to visualize the environment.

Feel free to expand or reorganize into multiple files:
 - bacteria_env.py
 - mcts.py
 - network.py
 - train_bacteria.py
 - evaluate_bacteria.py
 - gui_bacteria.py
etc.

This is intentionally minimal and not heavily optimized!
"""

import argparse
import math
import random
import copy
import numpy as np
import tkinter as tk

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ------------------------------------------
# 1) The Environment
# ------------------------------------------
class BacteriaEnv:
    """
    A toy environment in which two bacteria 'A' and 'B' fight for survival
    on a discrete grid of resources. Each has a DNA vector of length 1000
    controlling replication, virus strategy, etc.
    """

    # Constants for environment
    GRID_H = 5
    GRID_W = 5
    MAX_STEPS = 50       # Steps after which the game is considered done if no one died
    RESOURCE_REFILL = 10 # After every 10 steps, each cell in the grid gets some resources
    INIT_RESOURCES = 5   # initial resources in each cell
    MUTATION_PROB = 0.10 # Probability of random DNA mutation on replicate
    
    def __init__(self):
        """
        State representation:
        - self.grid: 2D array [H, W], each cell has 'resource_amount'
        - self.health: array of shape [2], health of each bacterium
        - self.positions: array of shape [2, 2], positions (row, col) for each bacterium
        - self.dna: array of shape [2, 1000], the DNA for each player
        - self.to_play: 0 or 1, whose turn it is
        - self.steps: how many steps have elapsed
        - Possibly store 'offspring_count' for tie-breaking or a “score”
        """
        self.num_players = 2
        self.reset()

    def reset(self):
        # Reset grid
        self.grid = np.full((self.GRID_H, self.GRID_W), self.INIT_RESOURCES, dtype=np.int32)
        # Bacteria health
        self.health = np.array([10, 10], dtype=np.float32)
        # Positions (spawn them in corners for fun)
        self.positions = np.array([[0, 0], [self.GRID_H-1, self.GRID_W-1]], dtype=np.int32)
        # DNA
        self.dna = np.random.randint(0, 4, size=(2, 1000), dtype=np.int32)
        # Alternatively, you can store “float DNA” or even one-hot. This is just a placeholder.
        
        self.to_play = 0
        self.steps = 0
        
        # Optional: track how many times each has replicated
        self.offspring_count = np.array([0, 0], dtype=np.int32)
        
        # Provide a standard Gym-like observation
        return self._make_observation()

    def _make_observation(self):
        """
        You can flatten or stack the info. This is a naive approach:
          - The grid resource map
          - Bacteria A health & position
          - Bacteria B health & position
          - Possibly partial or entire DNA (for RL partial information).
        For simplicity, we flatten everything into 1D. In practice, you might keep a 2D or 3D tensor.
        """
        # Flatten grid
        grid_flat = self.grid.flatten()
        # Health, positions
        feats = np.concatenate([
            self.health,
            self.positions.flatten()
        ])
        # For demonstration, we only give the current player's DNA chunk
        # (or entire DNA for both players, up to you):
        # We'll just give the entire DNA for both as a flattened float, but it's large. 
        # Here we’ll do a small snippet for demonstration.
        dna_snippet = self.dna.flatten()[:50].astype(np.float32)  # just 50 out of 2000 for brevity

        obs = np.concatenate([grid_flat, feats, dna_snippet])
        return obs.astype(np.float32)
    
    @property
    def observation_size(self):
        # Rough dimension
        return self.GRID_H*self.GRID_W + 2 + 2*2 + 50  # grid + 2 health + 2 positions + snippet(50)
    
    @property
    def action_size(self):
        """
        We define a small set of discrete actions each turn:
         0. Move Up
         1. Move Down
         2. Move Left
         3. Move Right
         4. Gather Resource
         5. Replicate
         6. Mutate DNA region #1 (virus portion)
         7. Mutate DNA region #2 (replication portion)
         8. Attack Opponent (deploy virus)
        (You can expand this as you wish.)
        """
        return 9

    def step(self, action):
        """
        Execute one action for the current player. Then switch turns if game not ended.
        Return (obs, reward, done, info).
        """
        player = self.to_play
        other = 1 - player
        
        r, c = self.positions[player]

        # Basic moves
        if action == 0 and r > 0:              # Move Up
            self.positions[player][0] -= 1
        elif action == 1 and r < self.GRID_H-1: # Move Down
            self.positions[player][0] += 1
        elif action == 2 and c > 0:            # Move Left
            self.positions[player][1] -= 1
        elif action == 3 and c < self.GRID_W-1: # Move Right
            self.positions[player][1] += 1

        elif action == 4:
            # Gather resource in the cell
            cell_r, cell_c = self.positions[player]
            amount = self.grid[cell_r, cell_c]
            self.health[player] += amount * 0.5  # convert resource to health
            self.grid[cell_r, cell_c] = max(0, self.grid[cell_r, cell_c]-amount) 
            
        elif action == 5:
            # Replicate => cost some health, might fail if not enough health
            # On success, offspring_count increases
            if self.health[player] > 2:
                self.health[player] -= 2
                self.offspring_count[player] += 1
                # Possibly chance to mutate automatically
                if random.random() < self.MUTATION_PROB:
                    idx = random.randint(0, 999)
                    self.dna[player][idx] = np.random.randint(0, 4)
                    
        elif action == 6:
            # mutate virus region #1 => e.g. indices [0..100)
            start, end = 0, 100
            mutation_spot = random.randint(start, end-1)
            self.dna[player][mutation_spot] = np.random.randint(0, 4)
            
        elif action == 7:
            # mutate replication region #2 => e.g. indices [100..200)
            start, end = 100, 200
            mutation_spot = random.randint(start, end-1)
            self.dna[player][mutation_spot] = np.random.randint(0, 4)
            
        elif action == 8:
            # Attack Opponent => reduce opponent's health (some function of virus code)
            # We'll do a small random effect
            dmg = 2.0
            self.health[other] -= dmg
            
        # else: if it's out of range or we don’t do anything, do nothing

        # Check if we need to refill resources
        if (self.steps+1) % self.RESOURCE_REFILL == 0:
            self._refill_resources()

        self.steps += 1

        # Check terminal conditions
        done = False
        reward = 0.0
        
        # If either player's health drops below 1 => game ends
        if self.health[0] <= 0:
            done = True
            reward = 1.0 if player == 1 else -1.0  # the active player is player, so if we just killed ourselves, the other side wins
        elif self.health[1] <= 0:
            done = True
            reward = 1.0 if player == 0 else -1.0
        else:
            # If we reached max steps => game ends
            if self.steps >= self.MAX_STEPS:
                done = True
                # Some tie-break or who has more offspring. Could also compare health.
                # For example, we do 0 if tie, else +1 for the one with more offspring.
                if self.offspring_count[0] > self.offspring_count[1]:
                    reward = 1.0 if player == 0 else -1.0
                elif self.offspring_count[1] > self.offspring_count[0]:
                    reward = 1.0 if player == 1 else -1.0
                else:
                    # if still tied, compare health
                    if self.health[0] > self.health[1]:
                        reward = 1.0 if player == 0 else -1.0
                    elif self.health[1] > self.health[0]:
                        reward = 1.0 if player == 1 else -1.0
                    else:
                        # truly tie
                        reward = 0.0

        # Switch turn if not done
        if not done:
            self.to_play = 1 - self.to_play

        next_obs = self._make_observation()
        return next_obs, reward, done, {}

    def _refill_resources(self):
        # Refill each cell with a small random amount of resources
        bonus = np.random.randint(1, 4, size=self.grid.shape)
        self.grid += bonus

    def clone(self):
        """Make a deep copy of the entire state for MCTS expansions."""
        env2 = BacteriaEnv()
        env2.grid = self.grid.copy()
        env2.health = self.health.copy()
        env2.positions = self.positions.copy()
        env2.dna = self.dna.copy()
        env2.to_play = self.to_play
        env2.steps = self.steps
        env2.offspring_count = self.offspring_count.copy()
        return env2

    def is_done(self):
        # Simple check
        if self.health[0] <= 0 or self.health[1] <= 0 or self.steps >= self.MAX_STEPS:
            return True
        return False


# ------------------------------------------
# 2) Minimal MCTS for Two-Player
# ------------------------------------------
class MCTSNode:
    def __init__(self, prior, to_play):
        self.prior = prior
        self.to_play = to_play

        self.visit_count = 0
        self.value_sum = 0.0

        self.children = {}  # action -> MCTSNode
        self._Q = 0.0

    @property
    def Q(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

def softmax_temperature(x, temp):
    x = np.array(x, dtype=np.float32)
    x_exp = np.power(x, 1.0/temp)
    return x_exp / np.sum(x_exp)

def run_mcts(root_env, root_node, evaluate_fn, num_simulations=50, c_puct=1.0):
    """
    root_env: environment in current state (not to be changed)
    root_node: MCTSNode for the root
    evaluate_fn: function that, given state-observation, returns (policy_logits, value)
    """
    for _ in range(num_simulations):
        env_copy = root_env.clone()
        node = root_node
        search_path = [node]

        # 1) Selection
        while node.children:
            # pick child by UCB
            best_score = -999999.0
            best_action = None
            best_child = None

            # to_play is the perspective from which we evaluate
            for action, child in node.children.items():
                # UCB
                q = child.Q
                u = c_puct * child.prior * math.sqrt(node.visit_count+1)/(child.visit_count+1)
                score = q + u
                if score > best_score:
                    best_score = score
                    best_action = action
                    best_child = child

            # step environment
            _, _, done, _ = env_copy.step(best_action)
            node = best_child
            search_path.append(node)
            if done:
                break

        # 2) Expansion & Evaluation
        if not env_copy.is_done():
            obs = env_to_observation(env_copy)  # shape = ...
            policy_logits, value_est = evaluate_fn(obs)
            # softmax or clamp
            policy = F.softmax(torch.from_numpy(policy_logits), dim=-1).numpy()

            # expand
            for action in range(env_copy.action_size):
                node.children[action] = MCTSNode(prior=policy[action], to_play=env_copy.to_play)
        else:
            # if done => you might set value_est from the environment outcome
            # we do a simplistic approach
            # for the last player who made the move, we can check if we got a reward
            # but for now, let's do 0
            value_est = 0.0

        # 3) Backprop the value
        # alpha zero flips sign each turn for two-player zero-sum
        # but here let's keep a simpler approach:
        #   if root_node.to_play = p, and we get a value_est from p's perspective,
        #   then each next child is from the other perspective => sign flip
        # This is an example; adapt to your game’s perspective logic.
        for i, nd in enumerate(reversed(search_path)):
            # flip sign if needed
            if i % 2 != 0:
                value_est = -value_est
            nd.visit_count += 1
            nd.value_sum += value_est

    # Return action probabilities from root_node
    counts = []
    for a in range(root_env.action_size):
        child = root_node.children.get(a)
        if child:
            counts.append(child.visit_count)
        else:
            counts.append(0)
    counts = np.array(counts, dtype=np.float32)
    # Temperature 1.0 for early moves, near 0 for later?
    if np.sum(counts) == 0:
        # If expansions failed, pick uniform
        counts = np.ones(root_env.action_size, dtype=np.float32)
    probs = counts / np.sum(counts)
    return probs

def env_to_observation(env):
    # The environment’s _make_observation gives a 1D float vector
    obs = env._make_observation()
    return obs

# ------------------------------------------
# 3) Minimal Neural Network
# ------------------------------------------
class BacteriaNet(nn.Module):
    """
    A small feedforward network to map observation -> (policy_logits, value)
    You can replace with ResNet or anything more powerful.
    """
    def __init__(self, obs_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x: shape [batch_size, obs_size]
        returns policy_logits (shape [batch_size, action_size]), value (shape [batch_size, 1])
        """
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        policy_logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h))  # in [-1,1]
        return policy_logits, value

# ------------------------------------------
# 4) Self-Play + Training Loop
# ------------------------------------------
def self_play_episode(env, net, num_sim=50):
    """
    Plays one full game using MCTS + the current net for policy/value.
    Returns a list of (obs, pi, z) for training. 
      obs: observation
      pi: MCTS action distribution
      z: final game outcome from the perspective of the player who had that obs
    """
    trajectory = []
    root_obs = env_to_observation(env)
    root_node = MCTSNode(prior=1.0, to_play=env.to_play)

    # We'll define a small function for net evaluation
    def evaluate_fn(obs_np):
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_np).unsqueeze(0).float()
            policy_logits, value = net(obs_t)
            return policy_logits[0].cpu().numpy(), value[0].item()

    while True:
        # run MCTS from root
        pi = run_mcts(env, root_node, evaluate_fn, num_simulations=num_sim, c_puct=1.0)
        
        # store this for training
        trajectory.append((root_obs, pi, env.to_play))

        # sample an action from pi or pick argmax
        action = np.random.choice(len(pi), p=pi)
        
        # step env
        next_obs, reward, done, _ = env.step(action)
        # if game ended, we will set final outcome
        if done:
            # game outcome for the last step from perspective of the last player who moved
            # e.g. if reward=1 => last mover is winner => that is env.to_play^1, but we want 
            # to assign +1 or -1 from perspective. We do a simpler approach: 
            # if reward>0 => the last mover got the positive result
            # We’ll gather that and backfill.
            # In a typical 2p zero-sum, if the last mover is p, then we want z=reward for p, z=-reward for the other
            returns = []
            last_player = 1 - env.to_play
            for i, (_, _, who) in enumerate(trajectory):
                if who == last_player:
                    returns.append(reward)
                else:
                    returns.append(-reward)
            break
        else:
            # create new node for MCTS root in next step
            root_obs = next_obs
            root_node = MCTSNode(prior=1.0, to_play=env.to_play)
            # Evaluate the new root so it at least has children
            # (In advanced code, we can do MCTS “root reuse.”)
    
    # produce final training data
    data = []
    for i, (obs, pi, who) in enumerate(trajectory):
        z = returns[i]
        data.append((obs, pi, z))
    return data


def train_on_data(data, net, optimizer, batch_size=32):
    random.shuffle(data)
    net.train()

    # simple loop
    total_policy_loss = 0.0
    total_value_loss = 0.0
    n_batches = 0

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        obs_batch = []
        pi_batch = []
        z_batch = []
        for obs, pi, z in batch:
            obs_batch.append(obs)
            pi_batch.append(pi)
            z_batch.append(z)

        obs_batch_t = torch.tensor(obs_batch, dtype=torch.float32)
        pi_batch_t = torch.tensor(pi_batch, dtype=torch.float32)
        z_batch_t = torch.tensor(z_batch, dtype=torch.float32).unsqueeze(-1)

        policy_logits, value = net(obs_batch_t)
        # policy loss => cross-entropy
        policy_log_softmax = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -(pi_batch_t * policy_log_softmax).sum(dim=-1).mean()

        # value loss => MSE between z and predicted
        value_loss = F.mse_loss(value, z_batch_t)

        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        n_batches += 1
    
    return total_policy_loss/n_batches, total_value_loss/n_batches

def run_training_loop(num_episodes=50, num_sim=50, lr=1e-3):
    # create net
    env = BacteriaEnv()
    net = BacteriaNet(env.observation_size, env.action_size, hidden_size=64)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for eps in range(num_episodes):
        # do self play for 1 (or more) games, gather data
        env.reset()
        data = self_play_episode(env, net, num_sim=num_sim)
        
        # train on the game data
        pol_loss, val_loss = train_on_data(data, net, optimizer, batch_size=16)
        print(f"Episode {eps+1}/{num_episodes}, pol_loss={pol_loss:.4f}, val_loss={val_loss:.4f}")

    print("Training done. Returning final net.")
    return net

# ------------------------------------------
# 5) A Simple “Evaluation” or “Play vs. AI”
# ------------------------------------------
def evaluate_game(net, render=False):
    env = BacteriaEnv()
    env.reset()

    def evaluate_fn(obs_np):
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_np).unsqueeze(0).float()
            policy_logits, value = net(obs_t)
            return policy_logits[0].cpu().numpy(), value[0].item()

    root_node = MCTSNode(prior=1.0, to_play=env.to_play)
    total_reward_p0 = 0.0
    total_reward_p1 = 0.0

    while True:
        pi = run_mcts(env, root_node, evaluate_fn, num_simulations=30)
        action = np.argmax(pi)  # pick best for evaluation
        next_obs, reward, done, _ = env.step(action)
        if done:
            if env.to_play == 1:
                # last mover was 0
                total_reward_p0 = reward
                total_reward_p1 = -reward
            else:
                total_reward_p0 = -reward
                total_reward_p1 = reward
            break
        root_node = MCTSNode(prior=1.0, to_play=env.to_play)

    print("Evaluation ended. p0 reward=%.2f, p1 reward=%.2f" % (total_reward_p0, total_reward_p1))
    return total_reward_p0, total_reward_p1


# ------------------------------------------
# 6) Minimal Tkinter GUI
# ------------------------------------------
class BacteriaGUI:
    """
    A tiny GUI that shows the 5x5 grid of resources and each bacterium's position,
    plus some basic stats. This is not a sophisticated real-time animation:
    each button press steps the environment once (like a debugging environment).
    """

    CELL_SIZE = 60
    
    def __init__(self, env, net=None):
        self.env = env
        self.net = net
        self.root = tk.Tk()
        self.root.title("Two-Bacteria RL Environment")

        self.canvas = tk.Canvas(self.root,
                                width=self.env.GRID_W * self.CELL_SIZE,
                                height=self.env.GRID_H * self.CELL_SIZE)
        self.canvas.pack()

        self.info_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.info_label.pack()

        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.pack()

        self.step_button = tk.Button(self.btn_frame, text="Step (MCTS Best Action)", command=self.on_step)
        self.step_button.grid(row=0, column=0)

        self.reset_button = tk.Button(self.btn_frame, text="Reset", command=self.on_reset)
        self.reset_button.grid(row=0, column=1)
        
        self.draw_env()

    def draw_env(self):
        self.canvas.delete("all")

        # draw grid resources
        for r in range(self.env.GRID_H):
            for c in range(self.env.GRID_W):
                x0 = c*self.CELL_SIZE
                y0 = r*self.CELL_SIZE
                x1 = x0 + self.CELL_SIZE
                y1 = y0 + self.CELL_SIZE

                # color scale
                res = self.env.grid[r,c]
                shade = max(0, 200 - 10*res)
                color = f"#{shade:02x}{255:02x}{shade:02x}"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        
        # draw bacteria
        for i in range(self.env.num_players):
            rr, cc = self.env.positions[i]
            x0 = cc*self.CELL_SIZE
            y0 = rr*self.CELL_SIZE
            x1 = x0 + self.CELL_SIZE
            y1 = y0 + self.CELL_SIZE
            col = "blue" if i == 0 else "red"
            self.canvas.create_oval(x0+5, y0+5, x1-5, y1-5, fill=col)

        # update label
        msg = (f"Player to move: {self.env.to_play}\n"
               f"Health: {self.env.health}\n"
               f"Offspring: {self.env.offspring_count}\n"
               f"Steps: {self.env.steps}/{self.env.MAX_STEPS}")
        self.info_label.config(text=msg)

    def on_step(self):
        if self.env.is_done():
            return
        # pick action from MCTS
        if self.net is None:
            # random
            action = random.randrange(self.env.action_size)
        else:
            # MCTS
            root_node = MCTSNode(prior=1.0, to_play=self.env.to_play)

            def evaluate_fn(obs_np):
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs_np).unsqueeze(0).float()
                    policy_logits, value = self.net(obs_t)
                    return policy_logits[0].cpu().numpy(), value[0].item()

            pi = run_mcts(self.env, root_node, evaluate_fn, num_simulations=30)
            action = np.argmax(pi)

        _, reward, done, _ = self.env.step(action)
        self.draw_env()
        if done:
            tk.messagebox = tk.Message(self.root, text=f"Game over! Reward={reward}", width=200)
            tk.messagebox.show()

    def on_reset(self):
        self.env.reset()
        self.draw_env()

    def start(self):
        self.root.mainloop()


def run_gui():
    env = BacteriaEnv()
    env.reset()

    # either load a net or pass net=None
    net = None

    # If you wanted to load a trained net, do so here
    # net = BacteriaNet(env.observation_size, env.action_size)
    # net.load_state_dict(torch.load("my_bacteria_net.pth"))

    gui = BacteriaGUI(env, net=net)
    gui.start()


# ------------------------------------------
# 7) Command-line Interface
# ------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run training loop")
    parser.add_argument("--eval", action="store_true", help="Run evaluation game")
    parser.add_argument("--gui", action="store_true", help="Run a Tkinter GUI")
    parser.add_argument("--episodes", type=int, default=10, help="Number of training episodes")
    parser.add_argument("--sims", type=int, default=30, help="Number of MCTS simulations per move")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()

    if args.train:
        net = run_training_loop(num_episodes=args.episodes, num_sim=args.sims, lr=args.lr)
        # Save the final model
        torch.save(net.state_dict(), "bacteria_net.pth")
        print("Saved final model to bacteria_net.pth")

    elif args.eval:
        net = BacteriaNet(BacteriaEnv().observation_size, BacteriaEnv().action_size, hidden_size=64)
        # net = BacteriaNet(obs_size, action_size, hidden_size=64)
        net.load_state_dict(torch.load("bacteria_net.pth"))
        net.eval()
        evaluate_game(net, render=False)

    elif args.gui:
        run_gui()

    else:
        parser.print_help()
