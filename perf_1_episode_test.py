# Experiment to test how much learning can be done within a single episode.

from utils import moving_average
from environments import ConsensusEnvironment
from utils import run_episode
import pagerank_method
import matplotlib.pyplot as plt
import numpy as np

from agents import LearningConsensusAgent
from agents import QLearningAgent
from agents import DynaAgent
from agents import ConsensusAgent

from agents import PermanentAgentStorage

# Environment setup:
n_agents = 10
n_opinions = 3

randomness = 0.05
n_tests = 100

env = ConsensusEnvironment(n_agents=n_agents, n_opinions=n_opinions, draw=False)
P = pagerank_method.pagerank_find_P(env)

pagerank_policy = pagerank_method.pagerank_optimize_for_env(env)
neutral_policy = pagerank_policy*0 + 1./n_opinions

print("Initializing Q")
Q_init = pagerank_method.init_Q(env,P)
print("Q calculated")
# Random behaviour
# scores_random = np.zeros(n_tests)
# for i in range(n_tests):
#     steps = run_episode(env, ConsensusAgent, policy=neutral_policy)
#     scores_random[i] = steps
#
# print("Mean random: " + str(np.mean(scores_random)) + " --- S.E.: " + str(np.std(scores_random)/np.sqrt(n_tests)))
#
# scores_Q_online_decentralized = np.zeros(n_tests)
# for i in range(n_tests):
#     agent_storage = PermanentAgentStorage(env, DynaAgent, k=0, randomness=randomness)
#     # Generate Data
#     steps, observations, actions = run_episode(env, agent_storage.get_next_agent, return_details=True)
#     scores_Q_online_decentralized[i] = steps
#     print(i)
# print("Mean Q: " + str(np.mean(scores_Q_online_decentralized)) + " --- S.E.: " + str(np.std(scores_Q_online_decentralized)/np.sqrt(n_tests)))
#
#
# # Dyna-Q-learning
# scores_DynaQ_online_decentralized = np.zeros(n_tests)
# for i in range(n_tests):
#     agent_storage = PermanentAgentStorage(env, DynaAgent, k=10, randomness=randomness)
#     # Generate Data
#     steps, observations, actions = run_episode(env, agent_storage.get_next_agent, return_details=True)
#     scores_DynaQ_online_decentralized[i] = steps
#     print(i)
# print("Mean Dyna-Q: " + str(np.mean(scores_DynaQ_online_decentralized)) + " --- S.E.: " + str(np.std(scores_DynaQ_online_decentralized)/np.sqrt(n_tests)))
#
# DP behaviour - Nonlearning
scores_DP = np.zeros(n_tests)
for i in range(n_tests):
    steps = run_episode(env, ConsensusAgent, policy=pagerank_policy*(1-randomness) + randomness*neutral_policy, use_joint_actions=False)
    scores_DP[i] = steps

print("Mean DP: " + str(np.mean(scores_DP)) + " --- S.E.: " + str(np.std(scores_DP)/np.sqrt(n_tests)))

scores_Q_online_decentralized = np.zeros(n_tests)
for i in range(n_tests):
    agent_storage = PermanentAgentStorage(env, DynaAgent, k=0, randomness=randomness, lr=0.0000, Q=Q_init)
    # Generate Data
    steps, observations, actions = run_episode(env, agent_storage.get_next_agent, return_details=True, use_joint_actions=False)
    scores_Q_online_decentralized[i] = steps
    if i%10 == 0:
        print(i)
print("Mean Q_initialized - constant: " + str(np.mean(scores_Q_online_decentralized)) + " --- S.E.: " + str(np.std(scores_Q_online_decentralized)/np.sqrt(n_tests)))

scores_Q_online_decentralized = np.zeros(n_tests)
agent_storage = PermanentAgentStorage(env, DynaAgent, k=0, randomness=randomness, Q=Q_init)
for i in range(n_tests):
    # Generate Data
    steps, observations, actions = run_episode(env, agent_storage.get_next_agent, return_details=True, use_joint_actions=False)
    scores_Q_online_decentralized[i] = steps
    if i%10 == 0:
        print(i)
print("Mean Q_initialized: " + str(np.mean(scores_Q_online_decentralized)) + " --- S.E.: " + str(np.std(scores_Q_online_decentralized)/np.sqrt(n_tests)))

scores_Q_online_decentralized = np.zeros(n_tests)
agent_storage = PermanentAgentStorage(env, DynaAgent, experiences=P * 200, k=25, randomness=randomness, Q=Q_init)
for i in range(n_tests):
    # Generate Data
    steps, observations, actions = run_episode(env, agent_storage.get_next_agent, return_details=True, use_joint_actions=False)
    scores_Q_online_decentralized[i] = steps
    if i%10 == 0:
        print(i)
print("Mean Dyna_initialized: " + str(np.mean(scores_Q_online_decentralized)) + " --- S.E.: " + str(np.std(scores_Q_online_decentralized)/np.sqrt(n_tests)))
