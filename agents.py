import numpy as np
import random
import pagerank_method
import torch
import torch.nn as nn
import torch.nn.functional as F

class PermanentAgentStorage:
    def __init__(self, env, agent_init_fn, **kwargs):
        self.agent_dict = {}
        self.n_agents = env.n_agents
        self.next_agent = 1
        for i in range(1, env.n_agents + 1):
            self.agent_dict[i] = agent_init_fn(env, **kwargs)

    def new_episode(self):
        for agent in self.agent_dict.values():
            agent.new_episode()

    def get_next_agent(self, *args, **kwargs):
        agent = self.agent_dict[self.next_agent]
        self.next_agent += 1
        if self.next_agent > self.n_agents:
            self.next_agent = 1
        return agent

class DQNAgent:
    def __init__(self, env, experiences=None, simulation_type="EXP", k=10, discount=0.98, lr=0.05, randomness=0.1):
        self.env = env

        self.n_actions = env.n_actions
        self.observation_list = env.observation_list
        self.observation_dict = {observation: idx for idx, observation in enumerate(self.observation_list)}

        self.r = np.zeros((len(self.observation_list), self.n_actions))
        for item in env.get_desired_observations():
            self.r[self.observation_dict[item]] = 1

        if np.any(experiences):
            self.experiences = np.copy(experiences)
        else:
            self.experiences = np.zeros((len(self.observation_list), len(self.observation_list), self.n_actions))

        self.k = k
        self.simulation_type = simulation_type
        self.randomness = randomness

        self.Q = Network(env)
        self.Q.double()

        self.encountered = set()

        self.last_observation = None
        self.last_action = None

        self.replay_buffer = []

        self.discount = discount
        self.alpha = lr

        self.optimizer = torch.optim.RMSprop(self.Q.parameters(),lr=0.001)

    def select_action(self, observation, observation_simple):
        observation_idx = self.observation_dict[observation_simple]

        # Select action
        if np.random.random() < self.randomness:
            action = random.choices([i for i in range(self.n_actions)])
        else:
            Q = self.Q.predict(observation)
            choices = np.argmax(Q)
            action = [int(choices)]

        # Learn based on last real experience
        if self.last_action and np.any(self.last_observation):
            self.replay_buffer.append((self.Q.obs_to_input(self.last_observation), self.Q.obs_to_input(observation), self.last_action, self.r[observation_idx,0]))
            self.update_model(self.last_observation, observation, self.last_action, self.r[observation_idx,0])
        # Save experience
        # if self.last_action and self.last_observation:
        #     self.experiences[self.last_observation, observation_idx, self.last_action] += 1
        # if self.last_observation:
        #     self.encountered.add((self.last_observation, self.last_action[0]))
        self.last_action = action
        self.last_observation = np.copy(observation)
        if len(self.replay_buffer) > 20000:
            del self.replay_buffer[0]
        return action[0]

    def update_model(self, *args):
        if np.random.random() < 0.05:
            if len(self.replay_buffer) > 50:
                samples = random.choices(self.replay_buffer, k=50)
                x = torch.stack([item[0] for item in samples])
                new_x = torch.stack([item[1] for item in samples])
                action = torch.tensor([item[2] for item in samples])
                rewards = torch.tensor([item[3] for item in samples], dtype=torch.float64)
                state_action_values = self.Q(x).gather(1, action)
                next_values = self.Q(new_x).max(1)[0].detach()
                targets = (next_values*self.discount + rewards)
                loss = F.smooth_l1_loss(state_action_values, targets.unsqueeze(1))
                # print(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                state_action_values = self.Q(x).gather(1, action)
                loss = F.smooth_l1_loss(state_action_values, targets.unsqueeze(1))
                #print(loss)
                #print("----")
    # def update_model(self, state, next_state, action, reward):
    #     x = self.Q.obs_to_input(state)
    #     state_action_values = self.Q(x)[0,action]
    #     next_values = self.Q(self.Q.obs_to_input(next_state)).max()
    #     target = (next_values*self.discount + reward).detach()
    #     loss = F.smooth_l1_loss(state_action_values[0], target)
    #     #print(loss)
    #     # print(loss)
    #     # print(state_action_values[0])
    #     # if reward > 0.1:
    #     #     print(loss)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     next_values = self.Q(self.Q.obs_to_input(next_state)).max()
    #     loss = F.smooth_l1_loss(self.Q(x)[0,action][0], target)
    #     #
    #     # if reward > 0.1:
    #     #     print(loss)
    #     #     print("-----")

class Network(nn.Module):
    def __init__(self, env, experiences=None, simulation_type="EXP", k=10, discount=0.98, lr=0.05, randomness=0.1, **kwargs):
        super(Network, self).__init__()
        self.env = env
        self.state_shape = (3,3)
        self.n_actions = env.n_actions
        self.device = kwargs.get('device', torch.device('cpu'))
        self.conv1 = nn.Conv2d(self.n_actions+1, self.n_actions, 1, padding=0)
        self.fc1 = nn.Linear(self.state_shape[0]*self.state_shape[1]*self.n_actions, self.n_actions)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = x.view(-1, self.state_shape[0]*self.state_shape[1]*self.n_actions)
        x = self.fc1(x)
        return x

    def obs_to_input(self, observation):
        observation = observation.astype(int)
        x = np.zeros((self.n_actions + 1, observation.shape[0], observation.shape[1]))
        layer_idx = np.arange(observation.shape[0]).reshape(observation.shape[0], 1)

        # this index selects each component separately
        component_idx = np.tile(np.arange(observation.shape[1]), (observation.shape[0], 1))

        # then we use `a` to select indices according to category label
        x[observation, layer_idx, component_idx] = 1
        x = torch.from_numpy(x)
        return x

    def predict(self, observation):
        x = self.obs_to_input(observation).unsqueeze(0)
        return self.forward(x).detach().numpy()

class DynaAgent:
    """
    A consensus agent that adapts its behaviour based on local experiences.
    """

    def __init__(self, env, experiences=None, simulation_type="EXP", k=10, discount=0.98, lr=0.05, randomness=0.1, Q=None):
        self.env = env.n_actions

        self.n_actions = env.n_actions
        self.observation_list = env.observation_list
        self.observation_dict = {observation: idx for idx, observation in enumerate(self.observation_list)}

        self.r = np.zeros((len(self.observation_list), self.n_actions))
        for item in env.get_desired_observations():
            self.r[self.observation_dict[item]] = 1

        if np.any(experiences):
            self.experiences = np.copy(experiences)
        else:
            self.experiences = np.zeros((len(self.observation_list), len(self.observation_list), self.n_actions))

        self.k = k
        self.simulation_type = simulation_type
        self.randomness = randomness
        if not np.any(Q):
            self.Q = np.zeros((len(self.observation_list), self.n_actions))+0.1
        else:
            self.Q = np.copy(Q)
        self.encountered = set()

        self.last_observation = None
        self.last_action = None

        self.discount = discount
        self.alpha = lr

    def select_action(self, observation):

        observation_idx = self.observation_dict[observation]
        # Select action
        if np.random.random() < self.randomness:
            action = random.choices([i for i in range(self.n_actions)])
        else:
            choices = np.ravel(np.argwhere(self.Q[observation_idx, :] == np.max(self.Q[observation_idx, :])))
            action = [int(np.random.choice(choices))]

        # Learn based on last real experience
        if self.last_action and self.last_observation:
            self.Q[self.last_observation, self.last_action] = \
                self.Q[self.last_observation, self.last_action] + \
                self.alpha * (self.r[observation_idx, 0]
                              + self.discount * np.max(self.Q[observation_idx, :])
                              - self.Q[self.last_observation, self.last_action])

        # Save experience
        if self.last_action and self.last_observation:
            self.experiences[self.last_observation, observation_idx, self.last_action] += 1
        if self.last_observation:
            self.encountered.add((self.last_observation, self.last_action[0]))
        self.last_action = action
        self.last_observation = observation_idx

        # Learn based on simulated experiences
        if self.k > 0 and self.simulation_type == "EXP":
            self.do_simulation_training_EXP()
        elif self.k > 0 and self.simulation_type == "SAMPLE":
            self.do_simulation_training_SAMPLE()
        elif self.k > 0:
            print("Simulation training type unknown.")
        return action

    def do_simulation_training_EXP(self):
        k = self.k
        if len(self.encountered) > 0:
            states_actions = random.choices(tuple(self.encountered), k=k)
            states = []
            actions = []
            for item in states_actions:
                states.append(item[0])
                actions.append(item[1])

            weights = self.experiences[states, :, actions]
            weights = (weights / weights.sum(axis=1, keepdims=True))

            self.Q[states, actions] = \
                self.Q[states, actions] + \
                self.alpha * (weights @ self.r[:, 0]
                              + self.discount * weights @ np.max(self.Q[:, :], 1)
                              - self.Q[states, actions])

    def sample_vectorized(self, prob_matrix, items):
        s = prob_matrix.cumsum(axis=0)
        r = np.random.rand(prob_matrix.shape[1])
        k = (s < r).sum(axis=0)
        return items[k]

    def do_simulation_training_SAMPLE(self):
        k = self.k
        if len(self.encountered) > 0:
            states_actions = random.choices(tuple(self.encountered), k=k)
            states = []
            actions = []
            for item in states_actions:
                states.append(item[0])
                actions.append(item[1])

            weights = self.experiences[states, :, actions]
            weights = (weights / weights.sum(axis=1, keepdims=True)).T
            possible_states = np.arange(len(self.observation_list))
            next_states = self.sample_vectorized(weights, possible_states)

            self.Q[states, actions] = \
                self.Q[states, actions] + \
                self.alpha * (self.r[next_states, 0]
                              + self.discount * np.max(self.Q[next_states, :], 1)
                              - self.Q[states, actions])

    def new_episode(self):
        self.last_observation = None
        self.last_action = None


class QLearningAgent(DynaAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, simulation_type="", k=0, **kwargs)


class LearningConsensusAgent:
    """
    A consensus agent that adapts its behaviour based on local experiences.
    """

    def __init__(self, env, experiences=None, policy=None):
        self.env = env
        self.observation_list = env.observation_list
        if np.any(experiences):
            self.experiences = np.copy(experiences)
        else:
            self.experiences = np.zeros((len(self.observation_list), len(self.observation_list), self.n_actions))
        self.policy = np.copy(policy)
        self.observation_dict = {observation: idx for idx, observation in enumerate(self.observation_list)}
        self.n_actions = env.n_actions
        self.last_observation = None
        self.last_action = None

        if not np.any(self.policy):
            self.update_policy()

    def init_equal_policy(self):
        self.policy = np.zeros((len(self.observation_list), self.n_actions)) + 1. / self.n_actions

    def select_action(self, observation):
        observation_idx = self.observation_dict[observation]
        if self.last_action and self.last_observation:
            self.experiences[self.last_observation, observation_idx, self.last_action] += 1

        action = random.choices([i for i in range(self.n_actions)], weights=self.policy[observation_idx])

        self.last_action = action
        self.last_observation = observation_idx

        return action

    def new_episode(self):
        self.last_observation = None
        self.last_action = None
        # 10% chance of updating policy
        if np.random.rand(1) < 0.01:
            self.update_policy()

    def update_policy(self):
        randomness = 0.1
        P = self.experiences / np.sum(self.experiences, (1))[:, np.newaxis, :]
        pagerank_policy = pagerank_method.optimize_value_iteration(P, self.env)
        policy = (1 - randomness) * pagerank_policy + randomness * (pagerank_policy * 0 + 1. / self.env.n_actions)
        self.policy = policy / np.sum(policy, 1)[:, np.newaxis]


class ConsensusAgent:
    """
    A simple consensus agent class that applies a fixed policy as an action selection.
    """

    def __init__(self, env, policy=None):
        self.policy = policy
        self.n_actions = env.n_actions
        self.observation_list = env.observation_list
        self.observation_dict = {observation: idx for idx, observation in enumerate(self.observation_list)}

        if not np.any(self.policy):
            self.init_equal_policy()

    def init_equal_policy(self):
        self.policy = np.zeros((len(self.observation_list), self.n_actions)) + 1. / self.n_actions

    def init_random_policy(self):
        self.policy = np.random.rand(len(self.observation_list), self.n_actions)
        row_sums = self.policy.sum(axis=1)
        self.policy = self.policy / row_sums[:, np.newaxis]
        return self.policy

    def select_action(self, observation):
        observation_idx = self.observation_dict[observation]
        action = random.choices([i for i in range(self.n_actions)], weights=self.policy[observation_idx])
        return action
