import numpy as np
import random
# from environments import ConsensusEnvironment
import pagerank_method

class PermanentAgentStorage:
    def __init__(self, env, agent_init_fn, experiences, policy=None):
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

class DynaQLearningConsensusAgent:
    """
    A consensus agent that adapts its behaviour based on local experiences.
    """
    def __init__(self, observation_list, experiences, policy=None, n_actions=3, env=None):
        self.env = env
        self.experiences = experiences
        self.policy = policy
        self.Q = np.zeros((len(observation_list),n_actions))+0

        self.observation_list = observation_list
        self.observation_dict = {observation: idx for idx, observation in enumerate(observation_list)}
        self.r = np.zeros((len(observation_list), n_actions))
        for item in env.get_desired_observations():
            self.r[self.observation_dict[item]] = 1
#        self.Q = np.copy(self.r)
        if np.any(policy):
            for action in range(n_actions):
                self.Q[:,action] = np.copy(policy)
        if not np.any(self.experiences):
           self.experiences = np.zeros((len(observation_list),len(observation_list),n_actions))
        self.encountered = set()
        self.n_actions = n_actions
        self.last_observation = None
        self.last_action = None
        self.discount = 0.95
        self.alpha = 0.05
        if not np.any(self.policy):
            self.init_equal_policy()

    def init_equal_policy(self):
        self.policy = np.zeros((len(self.observation_list), self.n_actions)) + 1./self.n_actions

    def select_action(self, observation):
        if np.max(self.r) >1.0:
            print("Wtf")
        observation_idx = self.observation_dict[observation]
        if self.last_action and self.last_observation:
            self.Q[self.last_observation, self.last_action] = \
                self.Q[self.last_observation, self.last_action] + \
                self.alpha * (self.r[observation_idx,0]
                              + self.discount * np.max(self.Q[observation_idx, :])
                              - self.Q[self.last_observation, self.last_action])
        if np.random.random() < 0.10:
            action = random.choices([i for i in range(self.n_actions)])
        else:
            choices = np.ravel(np.argwhere(self.Q[observation_idx, :] == np.max(self.Q[observation_idx, :])))
            action = [int(np.random.choice(choices))]

        if self.last_action and self.last_observation:
            self.experiences[self.last_observation, observation_idx, self.last_action] += 1
        if self.last_observation:
            self.encountered.add(self.last_observation)
        self.last_action = action
        self.last_observation = observation_idx

        self.do_simulation_training()
        return action

    def do_simulation_training(self):
        # P = self.experiences / np.sum(self.experiences, (1))[:, np.newaxis, :]
        # P = np.nan_to_num(P)
        # if np.sum(P) == 0:
        #     return
        k = 5
        failed = 0
        if len(self.encountered)>0:
            for i in range(k):
                last_state = random.choice(tuple(self.encountered))
                if np.random.random() < 0.10:
                    action = random.choices([i for i in range(self.n_actions)])
                else:
                    choices = np.ravel(np.argwhere(self.Q[last_state, :] == np.max(self.Q[last_state, :])))
                    action = int(np.random.choice(choices))
                if np.sum(self.experiences[last_state,:,action]) > 0:
                    next_state = np.random.choice(len(self.observation_list),p=np.ravel(self.experiences[last_state,:,action]/np.sum(self.experiences[last_state,:,action])))
                    self.Q[last_state, action] = \
                        self.Q[last_state, action] + \
                        self.alpha * (self.r[next_state,0]
                                      + self.discount * np.max(self.Q[next_state, :])
                                      - self.Q[last_state, action])
                else:
                    failed += 1
                    #print(failed)

    # def do_simulation_training(self):
    #     P = self.experiences / np.sum(self.experiences, (1))[:, np.newaxis, :]
    #     P = np.nan_to_num(P)
    #     if np.sum(P) == 0:
    #         return
    #     k = 5
    #     failed = 0
    #     for i in range(k):
    #         last_state = np.random.choice(len(P[:,0,0]),p=np.sum(P,(1,2))/np.sum(P))
    #         if np.random.random() < 0.10:
    #             action = random.choices([i for i in range(self.n_actions)])
    #         else:
    #             action = [np.argmax(self.Q[last_state,:])]
    #         if np.sum(P[last_state,:,action]) > 0:
    #             next_state = np.random.choice(len(self.observation_list),p=np.ravel(P[last_state,:,action]))
    #             self.Q[last_state, action] = \
    #                 self.Q[last_state, action] + \
    #                 self.alpha * (self.r[next_state,0]
    #                               + self.discount * np.max(self.Q[next_state, :])
    #                               - self.Q[last_state, action])
    #         else:
    #             failed += 1
    #             #print(failed)

    def new_episode(self):
        self.last_observation = None
        self.last_action = None
        # 10% chance of updating policy
        #if np.random.rand(1)<0.005:
        #    self.update_policy()

class QLearningConsensusAgent:
    """
    A consensus agent that adapts its behaviour based on local experiences.
    """
    def __init__(self, observation_list, experiences, policy=None, n_actions=3, env=None):
        self.env = env
        self.experiences = np.copy(experiences)
        self.policy = np.copy(policy)
        self.Q = np.zeros((len(observation_list),n_actions))+0
        if np.any(policy):
            for action in range(n_actions):
                self.Q[:,action] = np.copy(policy)
        self.observation_list = observation_list
        self.observation_dict = {observation: idx for idx, observation in enumerate(observation_list)}
        self.r = np.zeros((len(observation_list), n_actions))
        for item in env.get_desired_observations():
            self.r[self.observation_dict[item]] = 1
        # pagerank_method.extract_localized_rewards(env)

        self.n_actions = n_actions
        self.last_observation = None
        self.last_action = None
        self.discount = 0.95
        self.alpha = 0.05
        if not np.any(self.policy):
            self.init_equal_policy()

    def init_equal_policy(self):
        self.policy = np.zeros((len(self.observation_list), self.n_actions)) + 1./self.n_actions

    def select_action(self, observation):
        observation_idx = self.observation_dict[observation]
        if self.last_action and self.last_observation:
            self.Q[self.last_observation, self.last_action] = \
                self.Q[self.last_observation, self.last_action] + \
                self.alpha * (self.r[observation_idx,0]
                              + self.discount * np.max(self.Q[observation_idx, :])
                              - self.Q[self.last_observation, self.last_action])
        if np.random.random() < 0.10:
            action = random.choices([i for i in range(self.n_actions)])
        else:
            choices = np.ravel(np.argwhere(self.Q[observation_idx,:] == np.max(self.Q[observation_idx,:])))
            action = [int(np.random.choice(choices))]
            #action = [np.argmax(self.Q[observation_idx,:])]

        self.last_action = action
        self.last_observation = observation_idx

        return action

    def new_episode(self):
        self.last_observation = None
        self.last_action = None
        # 10% chance of updating policy
        #if np.random.rand(1)<0.005:
        #    self.update_policy()

class LearningConsensusAgent:
    """
    A consensus agent that adapts its behaviour based on local experiences.
    """
    def __init__(self, observation_list, experiences, policy=None, n_actions=3, env=None):
        self.env = env
        self.experiences = np.copy(experiences)
        self.policy = np.copy(policy)
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
        if np.random.rand(1)<0.01:
            self.update_policy()

    def update_policy(self):
        randomness = 0.1
        P = self.experiences / np.sum(self.experiences, (1))[:, np.newaxis, :]
        pagerank_policy = pagerank_method.optimize_value_iteration(P, self.env)
        policy = (1 - randomness) * pagerank_policy + (randomness) * (pagerank_policy * 0 + 1. / self.env.n_opinions)
        self.policy = policy / np.sum(policy, 1)[:, np.newaxis]
        # self.policy = 0.9*self.policy + 0.1*policy


class ConsensusAgent:
    """
    A simple consensus agent class that applies a fixed policy as an action selection.
    """
    def __init__(self, observation_list, policy=None, n_actions=3, experiences=None, env=None):
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
