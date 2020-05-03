import environments
import numpy as np

def moving_average(a, n=50) :
    b = np.zeros(a.size)
    for i in range(len(a)):
        if i>=n:
            b[i] = np.mean(a[i-n:i+1])
        else:
            b[i] = np.mean(a[0:i+1])
    return b

def run_episode(env, agent_init_fn, return_details=False, use_extended_observation=False, use_joint_actions = True, **kwargs):

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
        if use_joint_actions:
            for id, agent in agents.items():
                if not use_extended_observation:
                    action = agent.select_action(joint_observation[id])[0]
                else:
                    action = agent.select_action(env.extended_observation[id], joint_observation[id])
                joint_action[id] = action
                if return_details:
                    observations[id].append(joint_observation[id])
                    actions[id].append(action)
        else:
            active_agent = np.random.choice(list(agents.keys()))
            if not use_extended_observation:
                action = agents[active_agent].select_action(joint_observation[active_agent])[0]
            else:
                action = agents[active_agent].select_action(env.extended_observation[active_agent], joint_observation[active_agent])
            joint_action[active_agent] = action
        # Update environment
        if use_joint_actions:
            env.step(joint_action)
        else:
            env.step(joint_action, active_agent=active_agent)
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