import numpy as np
import random
from environments import ConsensusEnvironment
import pagerank_method

class PermanentAgentStorage:
    def __init__(self, env: ConsensusEnvironment, agent_init_fn, experiences, policy=None):
        self.agent_dict = {}
        self.n_agents = env.n_agents
        self.next_agent = 1
        for i in range(1, env.n_agents+1):
            self.agent_dict[i] = agent_init_fn(observation_list=env.observation_list, experiences=experiences, n_actions=env.n_opinions, env=env, policy=policy)

    def new_episode(self):
        for agent in self.agent_dict.values():
            agent.new_episode()

    def get_next_agent(self, **kwargs):
        agent = self.agent_dict[self.next_agent]
        self.next_agent += 1
        if self.next_agent>self.n_agents:
            self.next_agent = 1
        return agent


class LearningConsensusAgent:
    """
    A consensus agent that adapts its behaviour based on local experiences.
    """
    def __init__(self, observation_list, experiences, policy=None, n_actions=3, env=None):
        self.env = env
        self.experiences = experiences
        self.policy = policy
        self.observation_list = observation_list
        self.observation_dict = {observation: idx for idx, observation in enumerate(observation_list)}
        self.n_actions = n_actions
        self.last_observation = None
        self.last_action = None

        if not np.any(self.policy):
            self.update_policy()

    def init_equal_policy(self):
        self.policy = np.zeros((len(self.observation_list), self.n_actions)) + 1./self.n_actions

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
        if np.random.rand(1)<0.005:
            self.update_policy()

    def update_policy(self):
        randomness = 0.1
        P = self.experiences / np.sum(self.experiences, (1))[:, np.newaxis, :]
        pagerank_policy = pagerank_method.optimize_pagerank(P, self.env)
        policy = (1 - randomness) * pagerank_policy + (randomness) * (pagerank_policy * 0 + 1. / self.env.n_opinions)
        self.policy = policy / np.sum(policy, 1)[:, np.newaxis]
        # self.policy = 0.9*self.policy + 0.1*policy


class ConsensusAgent:
    """
    A simple consensus agent class that applies a fixed policy as an action selection.
    """
    def __init__(self, observation_list, policy=None, n_actions=3):
        self.policy = policy
        self.observation_list = observation_list
        self.observation_dict = {observation: idx for idx, observation in enumerate(observation_list)}
        self.n_actions = n_actions

        if not np.any(self.policy):
            self.init_equal_policy()

    def init_equal_policy(self):
        self.policy = np.zeros((len(self.observation_list), self.n_actions)) + 1./self.n_actions

    def init_random_policy(self):
        self.policy = np.random.rand(len(self.observation_list), self.n_actions)
        row_sums = self.policy.sum(axis=1)
        self.policy = self.policy / row_sums[:, np.newaxis]
        return self.policy

    def select_action(self, observation):
        observation_idx = self.observation_dict[observation]
        action = random.choices([i for i in range(self.n_actions)], weights=self.policy[observation_idx])
        return action
