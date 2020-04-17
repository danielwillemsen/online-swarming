import environments


def run_episode(env, agent_init_fn, return_details=False, **kwargs):

    # Initialization
    env.reset()
    agents = {}
    observations = {}
    actions = {}
    for i in range(1, env.n_agents+1):
        agents[i] = agent_init_fn(env, **kwargs)
        if return_details:
            observations[i] = []
            actions[i] = []

    #Perform loop
    steps = 0
    steps_correct = 0
    terminal = False
    while not terminal:
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
        terminal = env.is_terminal()
        if terminal:
            steps_correct += 1
    if return_details:
        joint_observation = env.get_joint_observation()
        for id, agent in agents.items():
            agent.select_action(joint_observation[id])[0]
        for id, agent in agents.items():
            observations[id].append(joint_observation[id])
        return steps, observations, actions
    return steps