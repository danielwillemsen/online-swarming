import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
import random
import agents
import pygame

from celluloid import Camera
matplotlib.use('TkAgg')
COLORS = ["r", "g", "b"]

class PatternEnvironment:
    def __init__(self, n_agents=8, draw=False):
        self.n_agents = n_agents
        self.n_neighbors_max = 8
        self.draw = draw
        self.size_grid = (self.n_agents + 1, self.n_agents + 1)
        self.grid = np.zeros(self.size_grid, dtype=int)
        self.grid_flat = self.grid.ravel()
        self.observation_list = []
        self.create_observation_list()
        self.agent_ids = [i for i in range(1, self.n_agents+1)]

        self.fig = None
        self.ax = None
        if self.draw:
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlim(0, self.size_grid[1])
            self.ax.set_ylim(0, self.size_grid[0])
            self.ax.grid()
            plt.show()
        self.reset()

    def create_observation_list(self):
        self.observation_list = [seq for seq in itertools.product([i for i in range(1+1)], repeat=self.n_neighbors_max)
                        if sum(seq) > 0 and sum(seq) <= self.n_agents - 1]

    def get_joint_observation(self):
        observation = {}
        for agent_id in self.agent_ids:
            location = np.where(self.grid == agent_id)
            location = (int(location[0]), int(location[1]))
            grid_around = np.zeros((3,3))
            down, up, left, right = 0, 0, 0, 0
            if location[0] == self.size_grid[0] - 1:
                down = 1
            if location[0] == 0:
                up = 1
            if location[1] == self.size_grid[1] - 1:
                right = 1
            if location[1] == 0:
                left = 1
            grid_around[0+down:3-up,0+right:3-left] = self.grid[max(location[0]-1,0):min(location[0]+2,self.size_grid[0]), max(location[1]-1,0):min(location[1]+2,self.size_grid[1])]
            grid_around = (grid_around.ravel() > 0).astype(int).tolist()
            del grid_around[4]
            observation[agent_id] = tuple(grid_around)
        return observation

    def is_terminal(self):
        opinion = self.opinions[1]
        for opinion2 in self.opinions.values():
            if opinion2 != opinion:
                return False
        if self.draw:
            plt.close()
        return True

    def reset(self):
        self.grid = np.zeros(self.size_grid, dtype=int)
        self.grid_flat = self.grid.ravel()
        self.opinions = dict()
        if self.draw:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlim(0, self.size_grid[1])
            self.ax.set_ylim(0, self.size_grid[0])
            self.ax.grid()
            plt.show()
        for agent in self.agent_ids:
            placed = False
            empty_indices = np.flatnonzero(self.grid_flat == 0)
            while not placed:
                index = np.random.choice(empty_indices)
                self.grid_flat[index] = agent
                location = np.where(self.grid == agent)
                location = (int(location[0]), int(location[1]))
                grid_around = self.grid[max(location[0]-1,0):min(location[0]+2,self.size_grid[0]), max(location[1]-1,0):min(location[1]+2,self.size_grid[1])]
                if np.sum(grid_around) != agent or agent == 1:
                    placed = True
                else:
                    self.grid_flat[index] = 0
        if self.draw:
            self.update_plot()

    def step(self, joint_action):
        """ Updates the environment based on a joint action. It randomly selects an action from one of the agents to
        execute.

        :param joint_action:
        :return:
        """
        # Select random agent and associated action
        active_agent = np.random.choice(self.agent_ids)
        action = joint_action[active_agent]
        location = np.where(self.grid == active_agent)
        location = (int(location[0]), int(location[1]))
        if action == 0: # Right
            if location[1] == self.size_grid[1]-1:
                self.grid = np.roll(self.grid, -1, axis=1)
                location = np.where(self.grid == active_agent)
                location = (int(location[0]), int(location[1]))
            if self.grid[location[0], location[1] + 1] == 0:
                self.grid[location] = 0
                self.grid[location[0], location[1] + 1] = active_agent
                grid_around = self.grid[max(location[0]-1,0):min(location[0]+2,self.size_grid[0]), max(location[1]-1,0):min(location[1]+2,self.size_grid[1])]
                safe = True
                if grid_around.size == 9:
                    if grid_around[0,0] and not (grid_around[0,1] or (grid_around[1,0] and sum(grid_around[:,1]))):
                        safe = False
                    if grid_around[1,0] and not sum(grid_around[:,1]):
                        safe = False
                    if grid_around[2,0] and not (grid_around[2,1] or (grid_around[1,0] and sum(grid_around[:,1]))):
                        safe = False
                if not safe:
                    self.grid[location] = active_agent
                    self.grid[location[0], location[1] + 1] = 0
        if action == 1: # Down
            if location[0] == self.size_grid[0]-1:
                self.grid = np.roll(self.grid, -1, axis=0)
                location = np.where(self.grid == active_agent)
                location = (int(location[0]), int(location[1]))
            if self.grid[location[0] + 1, location[1]] == 0:
                self.grid[location] = 0
                self.grid[location[0]+ 1, location[1] ] = active_agent
                grid_around_temp = self.grid[max(location[0]-1,0):min(location[0]+2,self.size_grid[0]), max(location[1]-1,0):min(location[1]+2,self.size_grid[1])]
                safe = True
                grid_around = np.rot90(grid_around_temp, -1)
                if grid_around.size == 9:
                    if grid_around[0,0] and not (grid_around[0,1] or (grid_around[1,0] and sum(grid_around[:,1]))):
                        safe = False
                    if grid_around[1,0] and not sum(grid_around[:,1]):
                        safe = False
                    if grid_around[2,0] and not (grid_around[2,1] or (grid_around[1,0] and sum(grid_around[:,1]))):
                        safe = False
                if not safe:
                    self.grid[location] = active_agent
                    self.grid[location[0] + 1, location[1]] = 0
        if action == 2: #left
            if location[1] == 0:
                self.grid = np.roll(self.grid, 1, axis=1)
                location = np.where(self.grid == active_agent)
                location = (int(location[0]), int(location[1]))
            if self.grid[location[0], location[1] -1] == 0:
                self.grid[location] = 0
                self.grid[location[0], location[1] - 1] = active_agent
                grid_around_temp = self.grid[max(location[0]-1,0):min(location[0]+2,self.size_grid[0]), max(location[1]-1,0):min(location[1]+2,self.size_grid[1])]
                safe = True
                grid_around = np.rot90(grid_around_temp, -2)
                if grid_around.size == 9:
                    if grid_around[0,0] and not (grid_around[0,1] or (grid_around[1,0] and sum(grid_around[:,1]))):
                        safe = False
                    if grid_around[1,0] and not sum(grid_around[:,1]):
                        safe = False
                    if grid_around[2,0] and not (grid_around[2,1] or (grid_around[1,0] and sum(grid_around[:,1]))):
                        safe = False
                if not safe:
                    self.grid[location] = active_agent
                    self.grid[location[0], location[1] - 1] = 0

        if action == 3: # up
            if location[0] == 0:
                self.grid = np.roll(self.grid, 1, axis=0)
                location = np.where(self.grid == active_agent)
                location = (int(location[0]), int(location[1]))
            if self.grid[location[0] - 1, location[1]] == 0:
                self.grid[location] = 0
                self.grid[location[0] - 1, location[1]] = active_agent
                grid_around_temp = self.grid[max(location[0] - 1, 0):min(location[0] + 2, self.size_grid[0]),
                                   max(location[1] - 1, 0):min(location[1] + 2, self.size_grid[1])]
                safe = True
                grid_around = np.rot90(grid_around_temp, -3)
                if grid_around.size == 9:
                    if grid_around[0, 0] and not (grid_around[0, 1] or (grid_around[1, 0] and sum(grid_around[:, 1]))):
                        safe = False
                    if grid_around[1, 0] and not sum(grid_around[:, 1]):
                        safe = False
                    if grid_around[2, 0] and not (grid_around[2, 1] or (grid_around[1, 0] and sum(grid_around[:, 1]))):
                        safe = False
                if not safe:
                    self.grid[location] = active_agent
                    self.grid[location[0] - 1, location[1]] = 0
        self.grid_flat = self.grid.ravel()
        if self.draw:
            self.update_plot()
        return self.get_joint_observation()

    def update_plot(self):
        self.ax.clear()
        self.ax.grid()
        for agent in self.agent_ids:
            location = np.where(self.grid == agent)
            location = (int(location[0]), int(location[1]))
            color = COLORS[0]
            self.ax.scatter(location[0], location[1], color=color, s=500)
        plt.pause(0.01)


class ConsensusEnvironment:
    def __init__(self, n_agents=5, n_opinions=3, draw=False):
        self.n_agents = n_agents
        self.n_opinions = n_opinions
        self.n_neighbors_max = min(8, n_agents - 1)
        self.draw = draw
        self.size_grid = (self.n_agents, self.n_agents)
        self.grid = np.zeros(self.size_grid, dtype=int)
        self.grid_flat = self.grid.ravel()
        self.opinions = dict()
        self.observation_list = []
        self.create_observation_list()
        self.agent_ids = [i for i in range(1, self.n_agents+1)]

        self.fig = None
        self.ax = None
        if self.draw:
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlim(0, self.size_grid[1])
            self.ax.set_ylim(0, self.size_grid[0])
            self.ax.grid()
            plt.show()
        self.reset()

    def create_observation_list(self):
        self.observation_list = []
        combinations = [seq for seq in itertools.product([i for i in range(self.n_neighbors_max + 1)], repeat=self.n_opinions)
                        if sum(seq) > 0 and sum(seq) <= self.n_neighbors_max]
        for opinion in range(self.n_opinions):
            for comb in combinations:
                self.observation_list.append((opinion,comb))

    def get_joint_observation(self):
        observation = {}
        for agent_id in self.agent_ids:
            location = np.where(self.grid == agent_id)
            location = (int(location[0]), int(location[1]))
            opinions_around = [0 for retval in range(self.n_opinions)]
            grid_around = self.grid[max(location[0]-1,0):min(location[0]+2,self.size_grid[0]), max(location[1]-1,0):min(location[1]+2,self.size_grid[1])]
            for agent_id_2 in grid_around.ravel():
                if agent_id_2 and agent_id_2 != agent_id:
                    opinions_around[self.opinions[agent_id_2]] += 1
            observation[agent_id] = ((self.opinions[agent_id], tuple(opinions_around)))
        return observation

    def is_terminal(self):
        opinion = self.opinions[1]
        for opinion2 in self.opinions.values():
            if opinion2 != opinion:
                return False
        if self.draw:
            plt.close()
        return True

    def reset(self):
        self.grid = np.zeros(self.size_grid, dtype=int)
        self.grid_flat = self.grid.ravel()
        self.opinions = dict()
        if self.draw:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlim(0, self.size_grid[1])
            self.ax.set_ylim(0, self.size_grid[0])
            self.ax.grid()
            plt.show()
        for agent in self.agent_ids:
            placed = False
            empty_indices = np.flatnonzero(self.grid_flat == 0)
            self.opinions[agent] = np.random.randint(self.n_opinions)
            while not placed:
                index = np.random.choice(empty_indices)
                self.grid_flat[index] = agent
                location = np.where(self.grid == agent)
                location = (int(location[0]), int(location[1]))
                opinions_around = [0 for retval in range(self.n_opinions)]
                grid_around = self.grid[max(location[0]-1,0):min(location[0]+2,self.size_grid[0]), max(location[1]-1,0):min(location[1]+2,self.size_grid[1])]
                if np.sum(grid_around) != agent or agent == 1:
                    placed = True
                else:
                    self.grid_flat[index] = 0
        if self.draw:
            self.update_plot()

    def step(self, joint_action):
        """ Updates the environment based on a joint action. It randomly selects an action from one of the agents to
        execute.

        :param joint_action:
        :return:
        """
        # Select random agent and associated action
        active_agent = np.random.choice(self.agent_ids)
        action = joint_action[active_agent]
        self.opinions[active_agent] = action
        if self.draw:
            self.update_plot()
        return self.get_joint_observation()

    def update_plot(self):
        self.ax.clear()
        self.ax.grid()
        for agent in self.agent_ids:
            location = np.where(self.grid == agent)
            location = (int(location[0]), int(location[1]))
            color = COLORS[self.opinions[agent]]
            self.ax.scatter(location[0], location[1], color=color, s=500)
        plt.pause(0.01)

if __name__ == "__main__":
    env = PatternEnvironment(draw=True)
    while True:
        joint_action = {}
        for i in range(1,8+1):
            action = random.randint(0,3)
            joint_action[i] = action
        env.step(joint_action)