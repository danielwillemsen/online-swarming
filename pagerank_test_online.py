from environments import ConsensusEnvironment
from agents import ConsensusAgent
from utils import run_episode
import pagerank_method
import matplotlib.pyplot as plt
import numpy as np

n_agents = 20
n_opinions = 3
env = ConsensusEnvironment(n_agents=n_agents, n_opinions=n_opinions, draw=False)
agent_init_fn = ConsensusAgent

### Setup
P = pagerank_method.pagerank_find_P(env)
experiences = P*500
# P = experiences / np.sum(experiences, (1))[:, np.newaxis, :]

pagerank_policy = pagerank_method.optimize_pagerank(P,env)

observation_list = env.observation_list
observation_dict = {observation: idx for idx, observation in enumerate(observation_list)}

neutral_policy = pagerank_policy*0 + 1./n_opinions
# s_des = []
# for s1, s1_idx in observation_dict.items():
#     arr = np.array(s1[1])
#     if sum(arr > 0) == 1:
#         if np.where(arr > 0)[0] == s1[0]:
#             s_des.append(s1_idx)
#
# #neutral_policy[s_des,:] = 0

mean_steps_list = []
n_gens = 26
n_episodes_per_gen = 100
#plot_env = ConsensusEnvironment(n_agents=n_agents, n_opinions=n_opinions, draw=True)
randomness = 0.1
episodes_done = []
for i in range(n_gens):
    episodes_done.append(i*n_episodes_per_gen)
    #randomness = randomness*0.9
    policy = ((1-randomness) * pagerank_policy + (randomness) * neutral_policy)
    policy = policy / np.sum(policy, 1)[:, np.newaxis]

    steps_list = []
    # Generate Data
    for j in range(n_episodes_per_gen):
        steps, observations, actions = run_episode(env, agent_init_fn, policy=policy, return_details=True)
        steps_list.append(steps)
        for id, agent_observations in observations.items():
            for step, o_t in enumerate(agent_observations[:-1]):
                o_t1 = agent_observations[step+1]
                a = actions[id][step]
                experiences[observation_dict[o_t], observation_dict[o_t1], a] += 1.0
    mean_steps_list.append(np.mean(steps_list))
    P = experiences / np.sum(experiences,(1))[:, np.newaxis, :]
    pagerank_policy = pagerank_method.optimize_pagerank(P, env)
    print(np.mean(steps_list))
    #if np.mean(steps_list)<500 and i%5 == 0:
    #    run_episode(plot_env, agent_init_fn, policy=policy, return_details=True)

plt.figure()
plt.plot(episodes_done, mean_steps_list)
plt.xlabel("Total episodes performed")
plt.ylabel("Mean number of steps")
plt.grid()
plt.show()