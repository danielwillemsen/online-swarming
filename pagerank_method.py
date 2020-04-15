import environments
import numpy as np
import scipy.optimize as opt
import mdptoolbox
import time

def pagerank_optimize_for_env(env):
    """ Optimizes pagerank for given environment. Assumes transitions according to paper

    :param env:
    :return:
    """
    P = pagerank_find_P(env)
    pi = optimize_pagerank(P, env)
    return pi

def optimize_value_iteration(P, env):
    r = extract_localized_rewards(env)

    # P is (s_old, s_new, action)
    # for mdp is (a, sold, snew)
    transition = np.swapaxes(P, 0, 2)
    transition = np.swapaxes(transition, 1, 2)
    t1 = time.time()
    mdp = mdptoolbox.mdp.PolicyIteration(transitions=transition, reward=r, discount=0.99, max_iter=50)
    mdp.run()
    if mdp.iter == 50:
        print("Max policy-iterations reached:" + str(mdp.iter))
    policy_ind = np.array(mdp.policy)
    policy = np.zeros((policy_ind.size, env.n_opinions))
    policy[np.arange(policy_ind.size),policy_ind] = 1.0
    return policy


def extract_localized_rewards(env):
    n_opinions = env.n_opinions
    observation_list = env.observation_list
    observation_dict = {observation: idx for idx, observation in enumerate(observation_list)}
    n_observations = len(observation_list)
    s_des = []
    r = np.zeros((n_observations, n_opinions))
    for s1, s1_idx in observation_dict.items():
        arr = np.array(s1[1])
        if sum(arr > 0) == 1:
            if np.where(arr > 0)[0] == s1[0]:
                s_des.append(s1_idx)
                r[s1_idx, :] = 1.0
    return r


def optimize_pagerank(P, env, regular=True):
    """

    :param P: state transition matrix: old state x new state x actions
    :param env: environment
    :return: optimized policy
    """
    n_opinions=env.n_opinions
    observation_list = env.observation_list
    observation_dict = {observation: idx for idx, observation in enumerate(observation_list)}
    n_observations = len(observation_list)

    r = extract_localized_rewards(env)

    r = r.reshape(-1, order="F")
    RH = P
    RH = np.swapaxes(RH, 1, 2)
    RH = RH.reshape((-1, n_observations), order="F").T
    I = np.tile(np.eye(n_observations), n_opinions)
    Aeq = RH - I
    beq = np.zeros(n_observations)
    # Uncomment this for problems where agents cannot transition between neighbour counts.
    if regular:
        for i in range(1, env.n_neighbors_max + 1):
            s_i_neighbors = []
            for s1, s1_idx in observation_dict.items():
                if sum(s1[1]) == i:
                    s_i_neighbors.append(s1_idx)
            Aeq2_row = np.zeros((1, n_observations))
            Aeq2_row[0, s_i_neighbors] = 1.0
            Aeq2_row = np.tile(Aeq2_row, n_opinions)
            beq2_row = np.array([1])
            Aeq = np.append(Aeq, Aeq2_row, 0)
            beq = np.append(beq, beq2_row, 0)
    else:
        Aeq2_row = np.ones((1,n_observations))
        Aeq2_row = np.tile(Aeq2_row, n_opinions)
        beq2_row = np.array([1])
        Aeq = np.append(Aeq, Aeq2_row, 0)
        beq = np.append(beq, beq2_row, 0)
    t1 = time.time()
    res = opt.linprog(-1 * r, A_eq=Aeq, b_eq=beq)
    pi = np.reshape(res.x, (n_observations, 3), order="F")
    pi = pi / np.sum(pi, 1)[:, np.newaxis]
    print(time.time()-t1)
    return pi


def pagerank_find_P(env):
    n_opinions=env.n_opinions
    observation_list = env.observation_list
    observation_dict = {observation: idx for idx, observation in enumerate(observation_list)}
    n_observations = len(observation_list)

    P_active = np.zeros((n_observations, n_observations, n_opinions))
    for s1, s1_idx in observation_dict.items():
        for s2, s2_idx in observation_dict.items():
            if s2[1] == s1[1]:
                P_active[s1_idx, s2_idx, s2[0]] = 1.0 / (sum(s1[1]) + 1)
    P_passive = np.zeros((n_observations, n_observations, n_opinions))
    for s1, s1_idx in observation_dict.items():
        for s2, s2_idx in observation_dict.items():
            if s2[0] == s1[0]:
                s2_arr = np.array(s2[1])
                s1_arr = np.array(s1[1])
                diff = s2_arr - s1_arr
                if sum(abs(diff)) == 2 and sum(diff) == 0:
                    idx = np.where(diff == -1)
                    n_options = s1_arr[idx]
                    P_passive[s1_idx, s2_idx, :] = 1.0 / (sum(s1[1]) + 1) / (n_opinions - 1.0) * n_options
    P = P_active + P_passive
    return P

