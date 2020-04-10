import environments

def run_episode(env, agent_init_fn, policy=None, return_details=False):

    # Initialization
    env.reset()
    agents = {}
    observations = {}
    actions = {}
    for i in range(1, env.n_agents+1):
        agents[i] = agent_init_fn(observation_list=env.observation_list, n_actions=env.n_opinions, policy=policy)
        if return_details:
            observations[i] = []
            actions[i] = []

    #Perform loop
    steps = 0

    while not env.is_terminal():
        steps += 1
        joint_observation = env.get_joint_observation()
        # Every agent selects action:
        joint_action = {}
        for id, agent in agents.items():
            action = agent.select_action(joint_observation[id])[0]
            joint_action[id] = action
            if return_details:
                observations[id].append(joint_observation[id])
                actions[id].append(action)
        # Update environment
        env.step(joint_action)
    if return_details:
        joint_observation = env.get_joint_observation()
        for id, agent in agents.items():
            observations[id].append(joint_observation[id])
        return steps, observations, actions
    return steps