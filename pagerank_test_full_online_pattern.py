from environments import ConsensusEnvironment
from agents import LearningConsensusAgent
from agents import QLearningAgent
from agents import DynaQLearningConsensusAgent
from agents import ExpDynaQLearningConsensusAgent

from environments import PatternEnvironment
from agents import PermanentAgentStorage
from utils import run_episode
import pagerank_method
import matplotlib.pyplot as plt
import numpy as np

env = PatternEnvironment(type="triangle", size=3, draw=True)

#P = pagerank_method.pagerank_find_P(env)
experiences = []#P*2000
#P = experiences / np.sum(experiences, (1))[:, np.newaxis, :]

#pagerank_policy = pagerank_method.optimize_value_iteration(P,env)
# neutral_policy = pagerank_policy*0 + 1./n_opinions
#randomness = 0.1
#policy = ((1 - randomness) * pagerank_policy + (randomness) * neutral_policy)
#policy = policy / np.sum(policy, 1)[:, np.newaxis]
# policy = None
### Setup

agent_storage = PermanentAgentStorage(env, ExpDynaQLearningConsensusAgent, experiences=None, policy=None)


observation_list = env.observation_list
observation_dict = {observation: idx for idx, observation in enumerate(observation_list)}

mean_steps_list = []
n_gens = 9999
n_episodes_per_gen = 10
# plot_env = ConsensusEnvironment(n_agents=n_agents, n_opinions=n_opinions, draw=True)
episodes_done = []
for i in range(n_gens):
    episodes_done.append(i*n_episodes_per_gen)

    steps_list = []

    # Generate Data
    for j in range(n_episodes_per_gen):
        agent_storage.new_episode()
        # print("Running Episode: " + str(j))
        steps, observations, actions = run_episode(env, agent_storage.get_next_agent, return_details=True)
        steps_list.append(steps)
    mean_steps_list.append(np.mean(steps_list))

    print(np.mean(steps_list))
    # if np.mean(steps_list)<500 and i%5 == 0:
    #    run_episode(plot_env, agent_storage.get_next_agent, return_details=True)

plt.figure()
plt.plot(episodes_done, mean_steps_list)
plt.xlabel("Total episodes performed")
plt.ylabel("Mean number of steps")
plt.grid()
plt.show()