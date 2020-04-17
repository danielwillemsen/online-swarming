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

def moving_average(a, n=50) :
    b = np.zeros(a.size)
    for i in range(len(a)):
        if i>=n:
            b[i] = np.mean(a[i-n:i+1])
        else:
            b[i] = np.mean(a[0:i+1])
    return b

n_agents = 6
n_opinions = 3

randomness = 0.1
n_tests_random = 5
n_tests_pagerank = 100
n_gens = 3

env = ConsensusEnvironment(n_agents=n_agents, n_opinions=n_opinions, draw=False)
P = pagerank_method.pagerank_find_P(env)
pagerank_policy = pagerank_method.optimize_value_iteration(P, env)
neutral_policy = pagerank_policy*0 + 1./n_opinions

# Random
# scores_random = np.zeros(n_tests_random)
# for i in range(n_tests_random):
#     steps = run_episode(env, agent_init_fn, policy=neutral_policy)
#     scores_random[i] = steps
#
# print("Score Random:" + str(np.mean(scores_random)))

# PageRank DP + 10% Random
policy = pagerank_policy * (1 - randomness) + neutral_policy * randomness

scores_pagerank = np.zeros(n_tests_pagerank)
for i in range(n_tests_pagerank):
    steps = run_episode(env, ConsensusAgent, policy=policy)
    scores_pagerank[i] =  steps

print("Score PageRank:" + str(np.mean(scores_pagerank)))

# PageRank online, centralized
P = pagerank_method.pagerank_find_P(env)
pagerank_policy_old = pagerank_method.optimize_value_iteration(P,env)
pagerank_policy = np.copy(pagerank_policy_old)
observation_list = env.observation_list
observation_dict = {observation: idx for idx, observation in enumerate(observation_list)}

experiences = P*1000

scores_pagerank_online_centralized = np.zeros(n_tests_pagerank*n_gens)
for i in range(n_gens):
    policy = ((1-randomness) * pagerank_policy + (randomness) * neutral_policy)
    policy = policy / np.sum(policy, 1)[:, np.newaxis]

    steps_list = []
    # Generate Data
    for j in range(n_tests_pagerank):
        steps, observations, actions = run_episode(env, ConsensusAgent, policy=policy, return_details=True)
        for id, agent_observations in observations.items():
            for step, o_t in enumerate(agent_observations[:-1]):
                o_t1 = agent_observations[step+1]
                a = actions[id][step]
                experiences[observation_dict[o_t], observation_dict[o_t1], a] += 1.0
        scores_pagerank_online_centralized[i*n_tests_pagerank + j] = steps
    P = experiences / np.sum(experiences,(1))[:, np.newaxis, :]
    pagerank_policy = pagerank_method.optimize_value_iteration(P, env)
print("Done pagerank, centralized")
# PageRank online, decentralized
P = pagerank_method.pagerank_find_P(env)
pagerank_policy_old = pagerank_method.optimize_value_iteration(P,env)
pagerank_policy = np.copy(pagerank_policy_old)
experiences = P*1000
policy = pagerank_policy * (1 - randomness) + neutral_policy * randomness

observation_list = env.observation_list
observation_dict = {observation: idx for idx, observation in enumerate(observation_list)}

#experiences = P*1000
agent_storage = PermanentAgentStorage(env, LearningConsensusAgent, experiences=experiences, policy=policy)

scores_pagerank_online_decentralized = np.zeros(n_tests_pagerank*n_gens)
for i in range(n_gens):
    policy = ((1-randomness) * pagerank_policy + (randomness) * neutral_policy)
    policy = policy / np.sum(policy, 1)[:, np.newaxis]

    steps_list = []
    # Generate Data
    for j in range(n_tests_pagerank):
        agent_storage.new_episode()
        steps, observations, actions = run_episode(env, agent_storage.get_next_agent, return_details=True)
        scores_pagerank_online_decentralized[i*n_tests_pagerank + j] =  steps
    P = experiences / np.sum(experiences,(1))[:, np.newaxis, :]
    pagerank_policy = pagerank_method.optimize_value_iteration(P, env)

print("Done PageRank")
# Q online, decentralized
P = pagerank_method.pagerank_find_P(env)
pagerank_policy_old = pagerank_method.optimize_value_iteration(P,env)
pagerank_policy = np.copy(pagerank_policy_old)
experiences = P*1000
policy = pagerank_policy * (1 - randomness) + neutral_policy * randomness
V = pagerank_method.optimize_value_iteration_values(P,env)

observation_list = env.observation_list
observation_dict = {observation: idx for idx, observation in enumerate(observation_list)}

agent_storage = PermanentAgentStorage(env, QLearningAgent)

scores_Q_online_decentralized = np.zeros(n_tests_pagerank*n_gens)
for i in range(n_gens):
    policy = ((1-randomness) * pagerank_policy + (randomness) * neutral_policy)
    policy = policy / np.sum(policy, 1)[:, np.newaxis]

    steps_list = []
    # Generate Data
    for j in range(n_tests_pagerank):
        steps, observations, actions = run_episode(env, agent_storage.get_next_agent, return_details=True)
        scores_Q_online_decentralized[i*n_tests_pagerank + j] =  steps
        if j%10 == 0:
            print(j+i*n_tests_pagerank)
print("Done Q")

# DynaQ online, decentralized
P = pagerank_method.pagerank_find_P(env)
pagerank_policy_old = pagerank_method.optimize_value_iteration(P,env)
V = pagerank_method.optimize_value_iteration_values(P,env)

pagerank_policy = np.copy(pagerank_policy_old)
experiences = P*1000
policy = pagerank_policy * (1 - randomness) + neutral_policy * randomness

observation_list = env.observation_list
observation_dict = {observation: idx for idx, observation in enumerate(observation_list)}

agent_storage = PermanentAgentStorage(env, DynaAgent, experiences=experiences)

scores_DynaQ_online_decentralized = np.zeros(n_tests_pagerank*n_gens)
for i in range(n_gens):
    policy = ((1-randomness) * pagerank_policy + (randomness) * neutral_policy)
    policy = policy / np.sum(policy, 1)[:, np.newaxis]

    steps_list = []
    # Generate Data
    for j in range(n_tests_pagerank):
        steps, observations, actions = run_episode(env, agent_storage.get_next_agent, return_details=True)
        scores_DynaQ_online_decentralized[i*n_tests_pagerank + j] =  steps
        if j%10 == 0:
            print(j+i*n_tests_pagerank)

print("Done Dyna-Q")

# Dyna-Q online, decentralized
observation_list = env.observation_list
observation_dict = {observation: idx for idx, observation in enumerate(observation_list)}

# agent_storage = PermanentAgentStorage(env, DynaQLearningConsensusAgent, None, policy=None)
#
# scores_DynaQ_online_decentralized = np.zeros(n_tests_pagerank*n_gens)
# for i in range(n_gens):
#     policy = ((1-randomness) * pagerank_policy + (randomness) * neutral_policy)
#     policy = policy / np.sum(policy, 1)[:, np.newaxis]
#
#     steps_list = []
#     # Generate Data
#     for j in range(n_tests_pagerank):
#         steps, observations, actions = run_episode(env, agent_storage.get_next_agent, policy=None, return_details=True)
#         scores_DynaQ_online_decentralized[i*n_tests_pagerank + j] = steps
#         print(steps)

# Plotting
plt.figure()
#plt.axhline(y=np.mean(scores_random), color='r', linestyle='-', label="Random Actions")
plt.axhline(y=np.mean(scores_pagerank), color='r', linestyle='-.', label="PageRank DP")
plt.plot(moving_average(scores_pagerank_online_centralized), linestyle='-', label="PageRank DP Online (Centralized)")
plt.plot(moving_average(scores_pagerank_online_decentralized), linestyle='-', label="PageRank DP Online (Decentralized)")
plt.plot(moving_average(scores_Q_online_decentralized), linestyle='-', label="Q-learning Online (Decentralized)")
plt.plot(moving_average(scores_DynaQ_online_decentralized), linestyle='-', label="Dyna-Q-learning Online (Decentralized)")

# plt.plot(moving_average(scores_DynaQ_online_decentralized), linestyle='-', label="Dyna-Q-learning Online (Decentralized)")

plt.legend()
plt.ylabel("Mean steps until consensus")
plt.xlabel("Episode")
plt.show()