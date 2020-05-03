from utils import moving_average
from environments import MultiAgentNavigation
from utils import run_episode
from utils import moving_average
import pagerank_method
import matplotlib.pyplot as plt
import numpy as np

from agents import LearningConsensusAgent
from agents import QLearningAgent
from agents import DynaAgent
from agents import ConsensusAgent

from agents import PermanentAgentStorage

# Environment setup:
n_agents = 2

randomness = 0.05
n_tests = 5000

env = MultiAgentNavigation(size=5, n_agents=n_agents, draw=False)

scores_Q_online_decentralized = np.zeros(n_tests)

agent_storage = PermanentAgentStorage(env, DynaAgent, k=0, randomness=randomness, lr=0.02)

for i in range(n_tests):
    # Generate Data
    steps, observations, actions = run_episode(env, agent_storage.get_next_agent, return_details=True)
    scores_Q_online_decentralized[i] = steps
    if i%10 == 0:
        print(i)
print("Mean Q_initialized - constant: " + str(np.mean(scores_Q_online_decentralized)) + " --- S.E.: " + str(np.std(scores_Q_online_decentralized)/np.sqrt(n_tests)))
plt.plot(moving_average(scores_Q_online_decentralized))
plt.show()