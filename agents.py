import numpy as np
import random
from environments import ConsensusEnvironment

class PermanentAgentStorage:
    def __init__(self, env: ConsensusEnvironment, agent_init_fn):
        self.agent_dict = {}
        self.n_agents = env.n_agents
        self.next_agent = 1
        for i in range(1, env.n_agents+1):
            self.agent_dict[i] = agent_init_fn()

    def get_next_agent(self):
        agent = self.agent_dict[self.next_agent]
        self.next_agent += 1
        if self.next_agent>self.n_agents:
            self.next_agent = 1
        return agent

class LearningConsensusAgent:
    """
    A consensus agent that adapts its behaviour based on local experiences.e
    """
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
