import copy

from env import BaitEnv

from controllers.random import RandomAgent
from controllers.depthfirst import DFSAgent
from controllers.limitdepthfirst import LimitedDFSAgent
from controllers.Astar import AstarAgent
from controllers.MCTS import MCTSAgent

if __name__ == "__main__":
    
    print("Game start!")
    level = 1
    env = BaitEnv(level=level, render=False)
    
    # actions: 0 noop, 1 left, 2 right, 3 down, 4 up
    
    mode = "Astar" # "play", "random", "depthfirst", "limitdepthfirst", "Astar", "MCTS"
    action_lst = None
    if mode == "play":
        # input your own actions here
        tick_max = 30
        action_lst = [3, 2, 3, 1, 3, 4, 4, 4, 1, 0]
    elif mode == "random":
        tick_max = 30
        agent = RandomAgent(env, tick_max)
    elif mode == "depthfirst":
        tick_max = 1000
        agent = DFSAgent(env, tick_max)
        action_lst = agent.solve()
    elif mode == "limitdepthfirst":
        tick_max = 100
        agent = LimitedDFSAgent(env, tick_max)
    elif mode == "Astar":
        tick_max = 100
        agent = AstarAgent(env, tick_max)
        action_lst = agent.solve()
    elif mode == "MCTS":
        tick_max = 1000
        agent = MCTSAgent(env, tick_max)

    print("Action list:", action_lst)
    action_lst_len = len(action_lst) if action_lst else 1e8

    env = BaitEnv(level=level, render=True)
    env.reset()

    action_list = []  # Test

    for step in range(min(1000, action_lst_len)):
        if action_lst:
            action_id = action_lst[step]
        else:
            env_copy = copy.deepcopy(env)
            env_copy.render = False
            action_id = agent.act(env_copy)
        state, reward, isOver, info = env.step(action_id)

        # Test
        action_list.append(action_id)

        print(f"Step: {step}, Action taken: {action_id}, Reward: {reward}, Done: {isOver}, Info: {info}")
        if isOver:
            break

    env.make_gif()