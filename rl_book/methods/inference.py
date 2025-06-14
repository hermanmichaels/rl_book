NUM_STEPS = 1000

def test_single_player(env, method):    
    # Test policy and visualize found solution
    observation, _ = env.reset()

    for _ in range(NUM_STEPS):
        action = method.act(observation, 1000)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()

def test_again_user(env, method):
    env.env.reset()

    for agent in env.env.agent_iter():
        observation, reward, termination, truncation, info = env.env.last()

        if termination or truncation:
            action = None
        else:
            mask = observation["action_mask"]

            state = env.obs_to_state(observation["observation"], 0)
            if agent == "player_1":
                action = method.method.act(state, 1000, mask)
            else:
                # action = methods[0].method.act(state, mask)
                action = int(input("Enter your action (column index): "))

        env.env.step(action)

    env.env.close()