from environments import ConsensusEnvironment
from utils import run_episode
import pagerank_method
import matplotlib.pyplot as plt
import numpy as np

from agents import LearningConsensusAgent
from agents import QLearningAgent
from agents import DynaAgent
from agents import ConsensusAgent
from agents import DQNAgent

from agents import PermanentAgentStorage

# Environment setup:
n_agents = 10
n_opinions = 2

randomness = 0.2
n_tests = 50

env = ConsensusEnvironment(n_agents=n_agents, n_opinions=n_opinions, draw=False)
P = pagerank_method.pagerank_find_P(env)
agent_storage = PermanentAgentStorage(env, DQNAgent, k=0, randomness=randomness)

scores_DQN_online_decentralized = np.zeros(n_tests)
for i in range(n_tests):
    # Generate Data
    steps = run_episode(env, agent_storage.get_next_agent, use_extended_observation=True, return_details=False)
    scores_DQN_online_decentralized[i] = steps
    print(steps)
print(np.mean(scores_DQN_online_decentralized))