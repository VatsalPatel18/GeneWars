#!/usr/bin/env python3

"""
AlphaZero-ish pipeline for a "Two-Bacteria" environment with expanded DNA logic:
- 10x10 board
- Each bacterium has 1000 bp of "DNA" (A,C,G,T => 0,1,2,3), subdivided into 5 regions.
- Some actions mutate specific DNA regions.
- On replicate, that bacterium's resource usage doubles (or generally multiples by replicate_count+1).
- We store the starting DNA in a file 'dna_start_log.txt'.
- There's a minimal MCTS and training loop in PyTorch, plus an optional Tkinter GUI.
"""

import argparse
import math
import random
import numpy as np
import tkinter as tk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# -----------------------------
# ENVIRONMENT PARAMETERS
# -----------------------------
GRID_H = 10
GRID_W = 10
MAX_STEPS = 80          # Number of steps after which the game ends if no one died
INIT_RESOURCES = 5      # initial resources in each cell
RESOURCE_REFILL_STEPS = 10  # how often we refill resources
DNA_LENGTH = 1000
NUM_REGIONS = 5         # we break DNA into 5 regions of length 200 each
MUTATION_ACTIONS = [0,1,2,3,4]  # we will map each region to one action
# We'll use a few "global" actions
MOVE_UP    = 5
MOVE_DOWN  = 6
MOVE_LEFT  = 7
MOVE_RIGHT = 8
GATHER     = 9
REPLICATE  = 10
ATTACK     = 11
# So total actions = 5 (for mutate region 0..4) + these 7 = 12
ACTION_SIZE = NUM_REGIONS + 7


class BacteriaEnv:
    """
    Turn-based environment with two bacteria (player 0 and player 1) on a 10x10 grid.
    Each has:
      - health
      - positions
      - replication_count
      - DNA[1000], stored as integers {0:A,1:C,2:G,3:T}
    We define a set of discrete actions:
      - MutateRegion0,1,2,3,4  (which picks a random spot in that region to mutate)
      - MoveUp, MoveDown, MoveLeft, MoveRight
      - Gather (collect resources in cell => convert to health)
      - Replicate (doubles resource usage cost for that bacteria)
      - Attack (deal damage to opponent)
    """

    def __init__(self):
        self.num_players = 2
        self.grid_h = GRID_H
        self.grid_w = GRID_W
        self.max_steps = MAX_STEPS
        self.action_size = ACTION_SIZE

        self.reset()

    def reset(self):
        # Initialize grid resources
        self.grid = np.full((self.grid_h, self.grid_w), INIT_RESOURCES, dtype=np.int32)

        # Bacteria states
        self.health = np.array([20.0, 20.0], dtype=np.float32)
        # positions: place each in a corner (top-left, bottom-right)
        self.positions = np.array([[0, 0], [self.grid_h - 1, self.grid_w - 1]], dtype=np.int32)
        self.replication_count = np.array([0, 0], dtype=np.int32)  # how many times each has replicated

        # Each has DNA[1000]
        self.dna = np.random.randint(0,4,size=(2, DNA_LENGTH), dtype=np.int32)

        self.steps = 0
        self.to_play = 0  # whose turn

        # Possibly log the initial DNA
        with open("dna_start_log.txt", "w") as f:
            f.write("Initial DNA for Bacteria A:\n")
            dna_str_A = "".join(str(bp) for bp in self.dna[0])
            f.write(dna_str_A + "\n\n")
            f.write("Initial DNA for Bacteria B:\n")
            dna_str_B = "".join(str(bp) for bp in self.dna[1])
            f.write(dna_str_B + "\n")

        return self._make_observation()

    def _refill_resources(self):
        # Refill each cell with random bonus
        bonus = np.random.randint(1,5,size=self.grid.shape)
        self.grid += bonus

    def _make_observation(self):
        """
        For simplicity, we'll flatten:
          grid (10x10) => 100
          health => 2
          positions => 4
          replication_count => 2
          step => 1
          plus a small DNA snippet for the current player (e.g. first 50 bp).
        """
        grid_flat = self.grid.flatten()  # shape (100,)
        feats = np.concatenate([
            self.health,
            self.positions.flatten(),
            self.replication_count,
            [self.steps],
        ]).astype(np.float32)
        # snippet from current player's DNA
        snippet = self.dna[self.to_play,:50].astype(np.float32)

        obs = np.concatenate([grid_flat, feats, snippet])
        return obs.astype(np.float32)

    @property
    def observation_size(self):
        # 100 (grid) + 2(health) +4(positions) +2(rep_count) +1(steps) +50(dna snippet) =159
        return 159

    def clone(self):
        """Make a deep copy for MCTS expansions."""
        new_env = BacteriaEnv()
        # we do a manual copy to preserve arrays
        new_env.grid = self.grid.copy()
        new_env.health = self.health.copy()
        new_env.positions = self.positions.copy()
        new_env.replication_count = self.replication_count.copy()
        new_env.dna = self.dna.copy()
        new_env.steps = self.steps
        new_env.to_play = self.to_play
        return new_env

    def is_done(self):
        # If steps >= max_steps or any health <=0 => done
        if self.steps >= self.max_steps:
            return True
        if self.health[0] <= 0 or self.health[1] <= 0:
            return True
        return False

    def step(self, action):
        """
        Applies the action for self.to_play.
        Then either ends or switches turn.
        Return (next_obs, reward, done, {})
        """
        player = self.to_play
        other = 1 - player
        r, c = self.positions[player]

        # resource usage cost => e.g. based on replication_count
        usage_multiplier = (self.replication_count[player] + 1)
        # e.g. small "upkeep" cost
        upkeep_cost = 0.5 * usage_multiplier
        self.health[player] -= upkeep_cost  # or you might drain "energy" if you prefer

        # Handle the chosen action
        # 0..4 => mutate region i
        if 0 <= action < NUM_REGIONS:
            # region i => mutate
            region_len = DNA_LENGTH // NUM_REGIONS  # 200
            region_i = action
            start_idx = region_i * region_len
            end_idx = (region_i+1)*region_len
            mutation_spot = random.randint(start_idx, end_idx-1)
            old_val = self.dna[player][mutation_spot]
            # pick a new base among {0,1,2,3} excluding old
            new_val = random.choice([x for x in [0,1,2,3] if x!=old_val])
            self.dna[player][mutation_spot] = new_val

        elif action == MOVE_UP and r>0:
            self.positions[player][0] -= 1
        elif action == MOVE_DOWN and r< self.grid_h-1:
            self.positions[player][0] += 1
        elif action == MOVE_LEFT and c>0:
            self.positions[player][1] -= 1
        elif action == MOVE_RIGHT and c< self.grid_w-1:
            self.positions[player][1] += 1
        elif action == GATHER:
            amount = self.grid[r,c]
            # convert resources to health
            self.health[player] += amount
            self.grid[r,c] = 0
        elif action == REPLICATE:
            # cost some health
            if self.health[player] > 5:
                self.health[player] -= 5
                self.replication_count[player] += 1
                # maybe auto mutate a region randomly
                region_i = random.randint(0,NUM_REGIONS-1)
                region_len = DNA_LENGTH // NUM_REGIONS
                start_idx = region_i * region_len
                end_idx = (region_i+1)*region_len
                mut_spot = random.randint(start_idx,end_idx-1)
                old_val = self.dna[player][mut_spot]
                new_val = random.choice([x for x in [0,1,2,3] if x!=old_val])
                self.dna[player][mut_spot] = new_val

        elif action == ATTACK:
            dmg = 5.0 # can be scaled by e.g. portion of virus region
            self.health[other] -= dmg

        # Possibly refill resources
        if (self.steps+1) % RESOURCE_REFILL_STEPS == 0:
            self._refill_resources()

        self.steps += 1

        # check terminal
        done = self.is_done()
        reward = 0.0
        if done:
            # if a single player's health <=0 => that means the other side presumably wins
            if self.health[player] <= 0 and self.health[other] <= 0:
                # tie
                # compare replication_count
                if self.replication_count[0] > self.replication_count[1]:
                    reward = 1.0 if player==0 else -1.0
                elif self.replication_count[1] > self.replication_count[0]:
                    reward = 1.0 if player==1 else -1.0
                else:
                    reward = 0.0
            elif self.health[player] <= 0:
                # current mover died => other side might get +1
                reward = -1.0
            elif self.health[other] <= 0:
                reward = 1.0
            else:
                # we ended on steps
                # check replication_count or total health
                sc0 = self.replication_count[0] + self.health[0]*0.1
                sc1 = self.replication_count[1] + self.health[1]*0.1
                if abs(sc0 - sc1) < 0.01:
                    reward = 0.0
                else:
                    if sc0>sc1:
                        reward = 1.0 if player==0 else -1.0
                    else:
                        reward = 1.0 if player==1 else -1.0
        else:
            # switch turn
            self.to_play = other

        next_obs = self._make_observation()
        return next_obs, reward, done, {}

# -----------------------------
# MCTS Node
# -----------------------------
class MCTSNode:
    def __init__(self, prior, to_play):
        self.prior = prior
        self.to_play = to_play
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    @property
    def Q(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

def run_mcts(root_env, root_node, evaluate_fn, num_simulations=50, c_puct=1.0):
    """
    root_env: current environment
    root_node: MCTSNode for the root
    evaluate_fn(obs_np) -> (policy_logits, value)
    """
    for _ in range(num_simulations):
        env_copy = root_env.clone()
        node = root_node
        search_path = [node]

        # selection
        while node.children:
            best_score = -999999
            best_action = None
            best_child = None
            for action, child in node.children.items():
                q = child.Q
                u = c_puct * child.prior * math.sqrt(node.visit_count+1)/(child.visit_count+1)
                score = q+u
                if score>best_score:
                    best_score = score
                    best_action = action
                    best_child = child
            # step environment
            _, _, done, _ = env_copy.step(best_action)
            node = best_child
            search_path.append(node)
            if done: break

        # expansion
        if not env_copy.is_done():
            obs = env_copy._make_observation()
            policy_logits, value_est = evaluate_fn(obs)
            policy_probs = F.softmax(torch.tensor(policy_logits), dim=-1).detach().numpy()

            # expand all possible actions
            for a in range(env_copy.action_size):
                node.children[a] = MCTSNode(prior=policy_probs[a], to_play=env_copy.to_play)
        else:
            # terminal => set value_est based on environment
            # let's do a naive approach: if both died => 0, if we died => -1, else +1
            # we can glean from last step's reward or do 0
            # We'll do 0 for simplicity
            value_est = 0.0

        # backprop
        # alpha-zero style sign-flip each ply
        for i, nd in enumerate(reversed(search_path)):
            if i%2!=0:
                value_est = -value_est
            nd.visit_count +=1
            nd.value_sum += value_est

    # return policy from root
    counts = []
    for a in range(root_env.action_size):
        if a in root_node.children:
            counts.append(root_node.children[a].visit_count)
        else:
            counts.append(0)
    counts = np.array(counts, dtype=np.float32)
    if np.sum(counts)==0:
        # fallback
        probs = np.ones_like(counts)/len(counts)
    else:
        probs = counts/np.sum(counts)
    return probs

# -----------------------------
# Minimal Neural Net
# -----------------------------
class BacteriaNet(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if len(x.shape)==1:
            x = x.unsqueeze(0)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        policy_logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h))  # in [-1,1]
        return policy_logits, value


# -----------------------------
# Self-play + Training
# -----------------------------
def self_play_episode(env, net, num_sim=50):
    """
    Return list of (obs, pi, z).
    z is final outcome from that player's perspective
    """
    trajectory = []
    def evaluate_fn(obs_np):
        with torch.no_grad():
            obs_t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
            plogits, val = net(obs_t)
            return plogits[0].numpy(), val[0].item()

    while True:
        root_obs = env._make_observation()
        root_node = MCTSNode(prior=1.0, to_play=env.to_play)
        # quick policy/value from net to init root children
        policy_logits, _ = evaluate_fn(root_obs)
        policy_probs = F.softmax(torch.tensor(policy_logits), dim=-1).numpy()
        for a in range(env.action_size):
            root_node.children[a] = MCTSNode(prior=policy_probs[a], to_play=env.to_play)

        pi = run_mcts(env, root_node, evaluate_fn, num_simulations=num_sim, c_puct=1.0)
        # store
        trajectory.append((root_obs, pi, env.to_play))
        # sample or argmax
        action = np.random.choice(len(pi), p=pi)
        next_obs, reward, done, _ = env.step(action)
        if done:
            # figure out final outcome from perspective of last mover
            # last mover was 1-env.to_play
            last_mover = 1- env.to_play
            # if reward>0 => last_mover is winner => that is +1 for last_mover, -1 for the other
            # we can do a backfill
            returns = []
            for (_,_,who) in trajectory:
                if who==last_mover:
                    returns.append(reward)
                else:
                    returns.append(-reward)
            break

    data = []
    for i, (o, pi, w) in enumerate(trajectory):
        z = returns[i]
        data.append((o, pi, z))
    return data

def train_on_data(data, net, optimizer, batch_size=32):
    random.shuffle(data)
    net.train()
    total_policy_loss=0.
    total_value_loss=0.
    n_batches=0

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        obs_batch = []
        pi_batch = []
        z_batch = []
        for (o, pi, z) in batch:
            obs_batch.append(o)
            pi_batch.append(pi)
            z_batch.append(z)
        obs_batch_t = torch.tensor(obs_batch,dtype=torch.float32)
        pi_batch_t = torch.tensor(pi_batch,dtype=torch.float32)
        z_batch_t = torch.tensor(z_batch,dtype=torch.float32).unsqueeze(-1)

        policy_logits, value = net(obs_batch_t)
        logp = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -(pi_batch_t*logp).sum(dim=-1).mean()
        value_loss = F.mse_loss(value, z_batch_t)
        loss = policy_loss+ value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_policy_loss+= policy_loss.item()
        total_value_loss+= value_loss.item()
        n_batches+=1

    return total_policy_loss/n_batches, total_value_loss/n_batches

def run_training_loop(num_episodes=50, num_sim=50, lr=1e-3):
    env = BacteriaEnv()
    net = BacteriaNet(env.observation_size, env.action_size, hidden_size=128)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for ep in range(num_episodes):
        env.reset()
        data = self_play_episode(env, net, num_sim=num_sim)
        pol_loss, val_loss = train_on_data(data, net, optimizer, batch_size=16)
        print(f"Ep {ep+1}/{num_episodes} pol_loss={pol_loss:.4f}, val_loss={val_loss:.4f}")
    print("Training done.")
    return net

# -----------------------------
# EVALUATION
# -----------------------------
def evaluate_game(net):
    env = BacteriaEnv()
    env.reset()
    with torch.no_grad():
        while not env.is_done():
            # MCTS or direct policy
            root_obs = env._make_observation()
            policy_logits, value = net(torch.tensor(root_obs))
            policy_probs = F.softmax(policy_logits, dim=-1).squeeze(0).numpy()
            action = np.argmax(policy_probs)
            _, reward, done, _ = env.step(action)
            if done:
                print("Game End. Reward=%.2f from last mover perspective" % reward)
                break

# -----------------------------
# MINIMAL TKINTER GUI
# -----------------------------
class BacteriaGUI:
    CELL_SIZE=50
    def __init__(self, env, net=None):
        self.env = env
        self.net = net
        self.root = tk.Tk()
        self.root.title("Two-Bacteria (Expanded)")

        w = self.env.grid_w*self.CELL_SIZE
        h = self.env.grid_h*self.CELL_SIZE
        self.canvas = tk.Canvas(self.root, width=w, height=h)
        self.canvas.pack()

        self.info_label = tk.Label(self.root, text="", font=("Arial",12))
        self.info_label.pack()

        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.pack()

        self.step_btn = tk.Button(self.btn_frame, text="Step (Argmax Net)", command=self.on_step)
        self.step_btn.grid(row=0, column=0)

        self.reset_btn = tk.Button(self.btn_frame, text="Reset", command=self.on_reset)
        self.reset_btn.grid(row=0, column=1)

        self.draw_env()

    def draw_env(self):
        self.canvas.delete("all")
        # draw resources
        for r in range(self.env.grid_h):
            for c in range(self.env.grid_w):
                x0 = c*self.CELL_SIZE
                y0 = r*self.CELL_SIZE
                x1 = x0+self.CELL_SIZE
                y1 = y0+self.CELL_SIZE
                amt = self.env.grid[r,c]
                shade = max(0, 255-amt*10)
                color = f"#{shade:02x}ff{shade:02x}"
                self.canvas.create_rectangle(x0,y0,x1,y1, fill=color)

        # draw bacteria
        for i in range(self.env.num_players):
            rr,cc = self.env.positions[i]
            x0 = cc*self.CELL_SIZE
            y0 = rr*self.CELL_SIZE
            x1 = x0+self.CELL_SIZE
            y1 = y0+self.CELL_SIZE
            col = "blue" if i==0 else "red"
            self.canvas.create_oval(x0+5,y0+5,x1-5,y1-5, fill=col)

        msg = (f"Player to move: {self.env.to_play}\n"
               f"Health: {self.env.health}\n"
               f"Replication: {self.env.replication_count}\n"
               f"Steps: {self.env.steps}/{self.env.max_steps}")
        self.info_label.config(text=msg)

    def on_step(self):
        if self.env.is_done():
            return
        if self.net is None:
            # random
            action = random.randrange(self.env.action_size)
        else:
            obs = self.env._make_observation()
            obs_t = torch.tensor(obs,dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                plogits, val = self.net(obs_t)
                p = F.softmax(plogits, dim=-1).squeeze(0).numpy()
            action = np.argmax(p)
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
    net = None
    # optionally load a trained net:
    # net = BacteriaNet(env.observation_size, env.action_size)
    # net.load_state_dict(torch.load("bigger_bacteria_net.pth"))
    gui = BacteriaGUI(env, net)
    gui.start()

# -----------------------------
# MAIN
# -----------------------------
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--sims", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.train:
        net = run_training_loop(num_episodes=args.episodes, num_sim=args.sims, lr=args.lr)
        torch.save(net.state_dict(), "bigger_bacteria_net.pth")
        print("Saved model to bigger_bacteria_net.pth")
    elif args.eval:
        env = BacteriaEnv()
        net = BacteriaNet(env.observation_size, env.action_size)
        net.load_state_dict(torch.load("bigger_bacteria_net.pth"))
        net.eval()
        evaluate_game(net)
    elif args.gui:
        run_gui()
    else:
        parser.print_help()
