import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
import random
import agents
import pygame
import time
from celluloid import Camera
matplotlib.use('TkAgg')
COLORS = ["r", "g", "b"]
BLACK = (0,0,0)
WHITE = (255,255,255)
GREEN = (0,255,0)

class MultiAgentNavigation:
    def __init__(self, size=4, n_agents=1, draw=False):
        self.n_agents = n_agents
        self.n_actions = 4
        self.size_grid = (size,size)
        self.agent_ids = [i for i in range(1, self.n_agents+1)]
        self.draw = draw
        self.grid_agents = np.zeros(self.size_grid, dtype=int)
        self.grid_rewards = np.zeros(self.size_grid, dtype=int)
        self.observation_list = None
        self.observation_dict2 = {}

        self.create_observation_list()
        self.observation_dict = {observation: idx for idx, observation in enumerate(self.observation_list)}

        self.agent_locations = {}
        self.reward_locations = {}
        self.terminal = False

        if self.draw:
            pygame.init()
            self.gameDisplay = pygame.display.set_mode((800, 600))
            self.gameDisplay.fill(WHITE)
        self.reset()


    def get_desired_observations(self):
        observation_dict = {observation: idx for idx, observation in enumerate(self.observation_list)}
        n_observations = len(self.observation_list)
        s_des = []
        r = np.zeros((n_observations, self.n_actions))
        for s1, s1_idx in observation_dict.items():
            obs = self.observation_dict2[s1]
            if np.all(obs[0] == obs[1]):
                r[s1_idx, :] = 1.0
                s_des.append(s1)
        return s_des

    def step(self, joint_action):
        for id in self.agent_ids:
            if np.all(self.agent_locations[id] == self.reward_locations[id]):
                self.terminal = True
            # Right
            if joint_action[id] == 0:
                self.agent_locations[id][1] = min(self.agent_locations[id][1]+1, self.size_grid[1]-1)
            if joint_action[id] == 1:
                self.agent_locations[id][0] = min(self.agent_locations[id][0]+1, self.size_grid[0]-1)
            if joint_action[id] == 2:
                self.agent_locations[id][1] = max(self.agent_locations[id][1]-1, 0)
            if joint_action[id] == 3:
                self.agent_locations[id][0] = max(self.agent_locations[id][0]-1,0)
        if self.draw:
            self.update_plot()

    def is_terminal(self):
        if self.terminal:
            return True
        else:
            return False

    def create_observation_list(self):
        observations_self = [np.zeros(self.size_grid)]
        observations_self = []
        observations_reward = []
        # Own observations and rewards
        for i in range(self.size_grid[0]*self.size_grid[1]):
            temp = np.zeros(self.size_grid, dtype=int)
            temp2 = np.zeros(self.size_grid, dtype=int)
            temp.ravel()[i] = 1
            temp2.ravel()[i] = 1

            observations_self.append(temp)
            observations_reward.append(temp2)

        observations_others = []
        for tup in itertools.product([i for i in range(self.size_grid[0]*self.size_grid[1])], repeat=self.n_agents-1):
            temp = np.zeros(self.size_grid, dtype=int)
            for ind in tup:
                temp.ravel()[ind] = 1
            observations_others.append(temp)
        somelists = [observations_self, observations_reward, observations_others]
        self.observation_list = []
        self.observation_dict2 = {}
        for seq in itertools.product(*somelists):
            s = bytes()
            for item in seq:
                s = s + item.tostring()
            self.observation_list.append(s)
            self.observation_dict2[s] = seq

        # self.observation_list = [seq for seq in itertools.product(*somelists)]
        return

    def reset(self):
        self.grid_agents = np.zeros(self.size_grid, dtype=int)
        self.grid_flat_agents = self.grid_agents.ravel()
        self.opinions = dict()
        self.terminal = False

        # Place agents
        for agent in self.agent_ids:
            location0 = random.randrange(0, self.size_grid[0])
            location1 = random.randrange(0, self.size_grid[1])
            self.agent_locations[agent] = np.array([location0, location1], dtype=int)
        # Place rewards
        for agent in self.agent_ids:
            location0 = random.randrange(0, self.size_grid[0])
            location1 = random.randrange(0, self.size_grid[1])
            self.reward_locations[agent] = np.array([location0, location1], dtype=int)
        if self.draw:
            self.update_plot()

    def get_joint_observation(self):
        joint_observation = {}
        for agent in self.agent_ids:
            observations_self = np.zeros(self.size_grid, dtype=int)
            observations_self[tuple(self.agent_locations[agent])] = 1
            observations_reward = np.zeros(self.size_grid, dtype=int)
            observations_reward[tuple(self.reward_locations[agent])] = 1
            observations_others = np.zeros(self.size_grid, dtype=int)
            for agent2 in self.agent_ids:
                if agent2 != agent:
                    observations_others[tuple(self.agent_locations[agent2])] = 1
            s = bytes()
            for item in [observations_self, observations_reward, observations_others]:
                s = s + item.tostring()
            joint_observation[agent] = s
                    # for agent in self.agent_ids:
        #     location = np.ravel_multi_index(tuple(self.agent_locations[agent]), self.size_grid)
        #     location_reward = np.ravel_multi_index(tuple(self.reward_locations[agent]), self.size_grid)
        #     joint_observation[agent] = (location, location_reward)
        return joint_observation

    def update_plot(self, active_agent=None):
        sleep = True
        self.gameDisplay.fill(WHITE)
        for agent in self.agent_ids:
            location = tuple(self.agent_locations[agent])
            location = (int(location[1])*50, int(location[0])*50)
            pygame.draw.circle(self.gameDisplay, BLACK, location, 25)

            location = tuple(self.reward_locations[agent])
            location = (int(location[1])*50, int(location[0])*50)

            pygame.draw.circle(self.gameDisplay, GREEN, location, 25)

        pygame.display.update()
        if sleep:
            time.sleep(0.05)


class PatternEnvironment:
    def __init__(self, type="square", size=2, draw=False):
        self.type = type
        self.size = size
        self.pattern = None
        self.n_agents = None
        self.setup_pattern()
        self.n_actions = 5
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
            pygame.init()
            self.gameDisplay = pygame.display.set_mode((800, 600))
            self.gameDisplay.fill(WHITE)
        self.reset()

    def check_desired_pattern(self):
        shape = np.shape(self.pattern)
        occupancy_grid = self.grid > 0
        for y in range(0,self.size_grid[0] - shape[0] + 1):
            for x in range(0, self.size_grid[1] - shape[1] + 1):
                if np.all(occupancy_grid[y:shape[0]+y, x:shape[1]+x] == self.pattern):
                    return True
        return False

    def get_desired_observations(self):
        shape = np.shape(self.pattern)
        observations = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                if self.pattern[i,j]:
                    grid_around = np.zeros((3, 3))
                    down, up, left, right = 0, 0, 0, 0
                    if i == shape[0] - 1:
                        down = 1
                    if i == 0:
                        up = 1
                    if j == shape[1] - 1:
                        right = 1
                    if j == 0:
                        left = 1
                    grid_around[0 + down:3 - up, 0 + right:3 - left] = self.pattern[
                                                                       max(i - 1, 0):min(i + 2,
                                                                                                   shape[0]),
                                                                       max(j - 1, 0):min(j + 2,
                                                                                                   shape[1])]
                    grid_around = (grid_around.ravel() > 0).astype(int).tolist()
                    del grid_around[4]
                    observations.append(tuple(grid_around))
        return observations

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
            grid_around[0+up:3-down,0+left:3-right] = self.grid[max(location[0]-1,0):min(location[0]+2,self.size_grid[0]), max(location[1]-1,0):min(location[1]+2,self.size_grid[1])]
            grid_around = (grid_around.ravel() > 0).astype(int).tolist()
            del grid_around[4]
            observation[agent_id] = tuple(grid_around)
        return observation

    def is_terminal(self):
        return self.check_desired_pattern()


    def reset(self):
        self.grid = np.zeros(self.size_grid, dtype=int)
        self.grid_flat = self.grid.ravel()
        self.opinions = dict()
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

    def step(self, joint_action, active_agent=None):
        """ Updates the environment based on a joint action. It randomly selects an action from one of the agents to
        execute.

        :param joint_action:
        :return:
        """
        # Select random agent and associated action
        if not active_agent:
            active_agent = np.random.choice(self.agent_ids)
        action = joint_action[active_agent]
        location = np.where(self.grid == active_agent)
        location = (int(location[0]), int(location[1]))
        if self.draw:
            self.update_plot(active_agent=active_agent)
        if action == 0: # Right
            if location[1] == self.size_grid[1]-1:
                self.grid = np.roll(self.grid, -1, axis=1)
                location = np.where(self.grid == active_agent)
                location = (int(location[0]), int(location[1]))
            safe = True
            if self.grid[location[0], location[1] + 1] == 0:
                self.grid[location] = 0
                self.grid[location[0], location[1] + 1] = active_agent
                grid_around = self.grid[max(location[0]-1,0):min(location[0]+2,self.size_grid[0]), max(location[1]-1,0):min(location[1]+2,self.size_grid[1])]
                safe = self.check_safe(active_agent, grid_around)

                if not safe:
                    self.grid[location] = active_agent
                    self.grid[location[0], location[1] + 1] = 0
        if action == 1: # Down
            if location[0] == self.size_grid[0]-1:
                self.grid = np.roll(self.grid, -1, axis=0)
                location = np.where(self.grid == active_agent)
                location = (int(location[0]), int(location[1]))
            safe = True
            if self.grid[location[0] + 1, location[1]] == 0:
                self.grid[location] = 0
                self.grid[location[0]+ 1, location[1] ] = active_agent
                grid_around_temp = self.grid[max(location[0]-1,0):min(location[0]+2,self.size_grid[0]), max(location[1]-1,0):min(location[1]+2,self.size_grid[1])]
                grid_around = np.rot90(grid_around_temp, 1)
                safe = self.check_safe(active_agent, grid_around)

                if not safe:
                    self.grid[location] = active_agent
                    self.grid[location[0] + 1, location[1]] = 0
        if action == 2: #left
            if location[1] == 0:
                self.grid = np.roll(self.grid, 1, axis=1)
                location = np.where(self.grid == active_agent)
                location = (int(location[0]), int(location[1]))
            safe = True
            if self.grid[location[0], location[1] -1] == 0:
                self.grid[location] = 0
                self.grid[location[0], location[1] - 1] = active_agent
                grid_around_temp = self.grid[max(location[0]-1,0):min(location[0]+2,self.size_grid[0]), max(location[1]-1,0):min(location[1]+2,self.size_grid[1])]
                safe = True
                grid_around = np.rot90(grid_around_temp, -2)
                safe = self.check_safe(active_agent, grid_around)

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
                grid_around = np.rot90(grid_around_temp, -1)
                safe = self.check_safe(active_agent, grid_around)
                if not safe:
                    self.grid[location] = active_agent
                    self.grid[location[0] - 1, location[1]] = 0
        if action == 4:     # Do nothing!
            pass
        self.grid_flat = self.grid.ravel()
        if self.draw:
            self.update_plot()
        return self.get_joint_observation()

    def check_safe(self, active_agent, grid_around):
        safe = True
        if grid_around.size == 9:
            if np.sum(grid_around[:, 1:3]) == active_agent:
                safe = False
            if grid_around[0, 0] and not (grid_around[0, 1] or (grid_around[1, 0] and sum(grid_around[:, 1]))):
                safe = False
            if grid_around[1, 0] and not sum(grid_around[:, 1]):
                safe = False
            if grid_around[2, 0] and not (grid_around[2, 1] or (grid_around[1, 0] and sum(grid_around[:, 1]))):
                safe = False
        if grid_around.size == 6:
            if sum(grid_around[:, 0]) and not sum(grid_around[:, 1]):
                safe = False
        return safe

    def update_plot(self, active_agent=None):
        self.gameDisplay.fill(WHITE)
        if self.check_desired_pattern():
            color = GREEN
            sleep = True
        else:
            sleep = False
            color = BLACK
        for agent in self.agent_ids:

            location = np.where(self.grid == agent)
            location = (int(location[1])*50, int(location[0])*50)
            if agent == active_agent:
                pygame.draw.circle(self.gameDisplay, GREEN, location, 25)
            else:
                pygame.draw.circle(self.gameDisplay, color, location, 25)

        pygame.display.update()
        if sleep:
            time.sleep(0.05)

    def setup_pattern(self):
        if self.type == "square":
            self.pattern = np.ones((self.size,self.size))
            self.n_agents = self.size*self.size
        elif self.type == "triangle":
            if self.size == 3:
                self.n_agents = 4
                self.pattern = np.array([[0,1,0],[1,1,1]])
            if self.size == 5:
                self.n_agents = 9
                self.pattern = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1]])


class ConsensusEnvironment:
    def __init__(self, n_agents=5, n_opinions=3, draw=False, DQN=False):
        self.n_agents = n_agents
        self.DQN = DQN
        self.n_actions = n_opinions
        self.n_neighbors_max = min(8, n_agents - 1)
        self.draw = draw
        self.size_grid = (self.n_agents, self.n_agents)
        self.grid = np.zeros(self.size_grid, dtype=int)
        self.grid_flat = self.grid.ravel()
        self.opinions = dict()
        self.observation_list = []
        self.create_observation_list()
        self.agent_ids = [i for i in range(1, self.n_agents+1)]
        self.desired_observations = self.get_desired_observations()
        self.fig = None
        self.ax = None
        self.current_observation = None
        self.extended_observation = None

        if self.draw:
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlim(0, self.size_grid[1])
            self.ax.set_ylim(0, self.size_grid[0])
            self.ax.grid()
            plt.show()
        self.reset()

    def get_desired_observations(self):
        observation_dict = {observation: idx for idx, observation in enumerate(self.observation_list)}
        n_observations = len(self.observation_list)
        s_des = []
        r = np.zeros((n_observations, self.n_actions))
        for s1, s1_idx in observation_dict.items():
            arr = np.array(s1[1])
            if sum(arr > 0) == 1:
                if np.where(arr > 0)[0] == s1[0]:
                    s_des.append(s1)
                    r[s1_idx, :] = 1.0
        return s_des

    def create_observation_list(self):
        self.observation_list = []
        combinations = [seq for seq in itertools.product([i for i in range(self.n_neighbors_max + 1)], repeat=self.n_actions)
                        if sum(seq) > 0 and sum(seq) <= self.n_neighbors_max]
        for opinion in range(self.n_actions):
            for comb in combinations:
                self.observation_list.append((opinion,comb))

    def get_joint_observation(self):
        observation = {}
        extended_observation = {}
        for agent_id in self.agent_ids:
            location = np.where(self.grid == agent_id)
            location = (int(location[0]), int(location[1]))
            down,up,right,left = 0,0,0,0
            if location[0] == self.size_grid[0] - 1:
                down = 1
            if location[0] == 0:
                up = 1
            if location[1] == self.size_grid[1] - 1:
                right = 1
            if location[1] == 0:
                left = 1
            opinions_around = [0 for retval in range(self.n_actions)]
            # grid_around = self.grid[max(location[0]-1,0):min(location[0]+2,self.size_grid[0]), max(location[1]-1,0):min(location[1]+2,self.size_grid[1])]
            grid_around = np.zeros((3,3))
            grid_around[0+up:3-down,0+left:3-right] = self.grid[max(location[0]-1,0):min(location[0]+2,self.size_grid[0]), max(location[1]-1,0):min(location[1]+2,self.size_grid[1])]

            extended_obs_agent = np.zeros((3,3))
            for index, agent_id_2 in enumerate(grid_around.ravel()):
                if agent_id_2:
                    extended_obs_agent[np.unravel_index(index,(3,3))] = self.opinions[agent_id_2]+1
                if agent_id_2 and agent_id_2 != agent_id:
                    opinions_around[self.opinions[agent_id_2]] += 1
            extended_observation[agent_id] = extended_obs_agent
            observation[agent_id] = ((self.opinions[agent_id], tuple(opinions_around)))
        self.current_observation = observation
        self.extended_observation = extended_observation
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
        self.current_observation = None
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
            self.opinions[agent] = np.random.randint(self.n_actions)
            while not placed:
                index = np.random.choice(empty_indices)
                self.grid_flat[index] = agent
                location = np.where(self.grid == agent)
                location = (int(location[0]), int(location[1]))
                opinions_around = [0 for retval in range(self.n_actions)]
                grid_around = self.grid[max(location[0]-1,0):min(location[0]+2,self.size_grid[0]), max(location[1]-1,0):min(location[1]+2,self.size_grid[1])]
                if np.sum(grid_around) != agent or agent == 1:
                    placed = True
                else:
                    self.grid_flat[index] = 0
        if self.draw:
            self.update_plot()

    def step(self, joint_action, active_agent=None):
        """ Updates the environment based on a joint action. It randomly selects an action from one of the agents to
        execute.

        :param joint_action:
        :return:
        """
        # Select random agent and associated action
        if not active_agent:
            active_agent = np.random.choice(self.agent_ids)
        action = joint_action[active_agent]
        if not self.current_observation[active_agent] in self.desired_observations:
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
    env = PatternEnvironment(type="triangle",size=3,draw=True)
    env.grid = env.grid*0
    env.grid[0:2,0:3] = np.array([[1,0,0],[2,3,4]])
    env.grid_flat = env.grid.ravel()
    env.update_plot()
    joint_action = {1:0, 2:0, 3:0, 4:0}
    env.step(joint_action)