import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
import random

from celluloid import Camera
matplotlib.use('TkAgg')
COLORS = ["r", "g", "b"]

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
    n_agents = 10
    n_opinions = 3

    env_temp = ConsensusEnvironment(n_opinions=n_opinions, n_agents=n_agents)
    agent_temp = ConsensusAgent(observation_list=env_temp.observation_list, n_actions=n_opinions)
    policy = agent_temp.init_random_policy()

    best_score = float("inf")
    for iteration in range(250):
        policy = agent_temp.init_random_policy()
        score_list = []
        print(iteration)
        for i in range(250):
            score = run_episode(policy=policy, n_opinions=n_opinions, n_agents=n_agents)
            score_list.append(score)
        if np.mean(score_list) < best_score:
            best_score = np.mean(score_list)
            best_policy = policy
            print(best_score)

    score_list = []
    print("Done")
    for i in range(500):
        score = run_episode(policy=best_policy, n_opinions=n_opinions, n_agents=n_agents)
        score_list.append(score)
    print(np.mean(score_list))

    plt.hist(score_list, bins=20)
    plt.show()