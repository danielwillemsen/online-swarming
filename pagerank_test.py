from environments import ConsensusEnvironment
from agents import ConsensusAgent
from utils import run_episode
import pagerank_method
import matplotlib.pyplot as plt
import numpy as np

n_agents = 20
n_opinions = 3
env = ConsensusEnvironment(n_agents=n_agents, n_actions=n_opinions, draw=False)
agent_init_fn = ConsensusAgent

pagerank_policy = pagerank_method.pagerank_optimize_for_env(env)
neutral_policy = pagerank_policy*0 + 1./n_opinions
randomness_list = [0.01, 0.02, 0.05, 0.1, 0.2]

mean_steps_list = []
for randomness in randomness_list:
    policy = pagerank_policy*(1-randomness) + neutral_policy*randomness
    steps_list = []
    for i in range(100):
        steps = run_episode(env, agent_init_fn, policy=policy)
        steps_list.append(steps)
        print(steps)
    mean_steps_list.append(np.mean(steps_list))
    print("Test complete")
    print(np.mean(steps_list))

plt.semilogx(randomness_list,mean_steps_list)
plt.ylabel("Mean number of steps (lower is better)")
plt.xlabel("uniform policy / optimal policy ratio")
plt.show()