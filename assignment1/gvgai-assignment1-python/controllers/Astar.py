import copy
import heapq

class AstarAgent:
    def __init__(self, env, tick_max):
        self.env = env
        self.tick_max = tick_max
        self.tick = 0
        self.actions = [4, 3, 2, 1]  # 1 left, 2 right, 3 down, 4 up

    def astar(self):
        # 使用优先队列存储待扩展节点
        open_set = []
        # 记录已探索的状态
        observation = []
        state = self.env._get_observation()
        env = copy.deepcopy(self.env)
        forbidden_action = 0

        heapq.heappush(open_set, (0, state, [], env))  # (f_score, state, action_sequence, env)
        reward, isOver, info = 0, False, {}


        while open_set:

            f_score, current_state, action_sequence, cur_env = heapq.heappop(open_set)
            open_set = []
            
            # 检查是否达到目标状态
            if isOver:
                return action_sequence

            # 将当前状态加入已探索的集合
            observation.append(current_state) 

            # 生成当前状态的所有有效动作
            valid_moves = []

            for action in self.actions:
                if action == forbidden_action:
                    forbidden_action = 0
                    continue
                cur_env_check = copy.deepcopy(cur_env)
                new_state, reward, isOver, info = cur_env_check.step(action)
                if new_state not in observation and info not in [{'message': 'Fell into hole. Game over.'}, {'message': 'Hit wall'}, {'message': 'Cannot push box. Obstacle ahead.'}]:
                    valid_moves.append(action)
                    print("Append action:", action)
                    

            if len(valid_moves) == 0:
                if len(action_sequence) != 0:
                    forbidden_action = action_sequence[-1]
                    action_sequence.pop()
                    cur_env.reset()
                    for x in action_sequence:
                        _ = cur_env.step(x)

            else:
                for action in valid_moves:
                    cur_env_action = copy.deepcopy(cur_env)
                    new_state, reward, isOver, info = cur_env_action.step(action)
                    g_score = len(action_sequence)
                    h_score = self.heuristic(cur_env_action, reward)
                    f_score = g_score + h_score

                    print("Score is:", f_score, end='; ')
                    print("g_core is:", g_score, end='; ')
                    print("h_score is:", h_score)
                    
                    # 将新状态加入到优先队列中
                    heapq.heappush(open_set, (f_score, new_state, action_sequence + [action], cur_env_action))
                    
                    print("The action_sequence is:", action_sequence)
                    
                    self.tick += 1

                    if self.tick > self.tick_max:
                        assert 0

    def heuristic(self, env, reward):  # 启发函数
        def manhattan(x, y):
            return abs(x[0]-y[0])+abs(x[1]-y[1])
        
        h, w = env.height, env.width
        state = env._get_observation()

        hole_count = 0

        isover = True
        eat_mushroom = True
        has_hole = False
        box_pos = []
        hole_pos = []

        for i in range(h):
            for j in range(w):
                grid = state[i][j]
                if 'key' in grid:
                    key_cord = i, j
                elif 'box' in grid:
                    box_pos.append((i, j))
                elif 'hole' in grid:
                    has_hole = True
                    hole_count += 1
                    hole_pos.append((i, j))
                elif 'avatar_nokey' in grid:
                    haskey=False
                    nk_cord = i, j
                elif 'avatar_withkey' in grid:
                    haskey = True
                    wk_cord = i, j
                elif 'mushroom' in grid:
                    eat_mushroom = False
                    ms_cord = i, j
                elif 'goal' in grid:
                    isover = False
                    goal_cord = i, j

        if has_hole:
            hole_cost = 0
            if not haskey:
                hole_cost = hole_count
                min_dis = 0
                for pos in box_pos:
                    if manhattan(pos, nk_cord) > 1:
                        min_dis += 2 * manhattan(pos, nk_cord)
                    elif manhattan(pos, hole_pos[0]) > 1:
                        min_dis += 2 * manhattan(pos, hole_pos[0])
                    elif manhattan(pos, nk_cord) == 1:
                        pass 
                hole_cost += min_dis

        if isover:
            return - 10 * reward
        if haskey or eat_mushroom:
            if not eat_mushroom:
                cost = manhattan(wk_cord, ms_cord) + manhattan(ms_cord, goal_cord) - 10 * reward
                if has_hole:
                    return cost + hole_cost
                else:
                    return cost
            elif not haskey:
                cost = manhattan(nk_cord, key_cord) + manhattan(key_cord, goal_cord) - 10 * reward
                if has_hole:
                    return cost + hole_cost
                else:
                    return cost
            else:
                cost = manhattan(wk_cord, goal_cord) - 10 * reward
                if has_hole:
                    return cost + hole_cost
                else:
                    return cost
        else:
            cost = manhattan(nk_cord, key_cord) + manhattan(key_cord, ms_cord) + manhattan(ms_cord, goal_cord) - 10 * reward
            if has_hole:
                return cost + hole_cost
            else:
                return cost 

    def solve(self):
        state = self.env.reset()  # 重置环境
        action_sequence = self.astar()
        return action_sequence

    def act(self, env):
        action_sequence = self.astar()
        return action_sequence[0]
