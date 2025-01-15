import copy

class LimitedDFSAgent:
    def __init__(self, env, tick_max):
        self.env = env
        self.tick_max = tick_max
        self.tick = 0
        self.max_depth = 30
        self.score_list = []
        self.actions = [1, 2, 3, 4]
        # Your can add new attributes if needed


    # 深搜到一定深度后，判断局面好坏
    def score(self, env, steps):
        def manhattan(x, y):
            return abs(x[0]-y[0])+abs(x[1]-y[1]) 
        
        h, w = env.height, env.width
        state = env._get_observation()
        isover = True

        for i in range(h):
            for j in range(w):
                grid = state[i][j]
                if 'key' in grid:
                    key_cord = i, j
                elif 'avatar_nokey' in grid:
                    haskey=False
                    nk_cord = i, j
                elif 'avatar_withkey' in grid:
                    haskey = True
                    wk_cord = i, j
                elif 'goal' in grid:
                    isover = False
                    goal_cord = i, j
        if isover:
            return len(steps) / 100
        elif haskey:
            return manhattan(wk_cord, goal_cord)
        else:
            return manhattan(nk_cord, key_cord) + manhattan(key_cord, goal_cord) 

    # 搜索函数
    def search(self, env, depth, action_sequence, observation):
        valid_moves = []

        for action in self.actions:
            env_copy = copy.deepcopy(env)
            new_state, reward, isOver, info = env_copy.step(action)
            if new_state not in observation and info != {'message': 'Fell into hole. Game over.'} and info != {'message': 'Hit wall'} and info != {'message': 'Cannot push box. Obstacle ahead.'}: 
                valid_moves.append(action)

        moves, state_list = [], []

        for action in valid_moves:
            env_copy = copy.deepcopy(env)
            new_state, reward, isOver, info = env_copy.step(action)

            moves = copy.copy(action_sequence)
            moves.append(action)
                    
            state_list = copy.deepcopy(observation)
            state_list.append(new_state)

            # 搜索结束
            if depth == 1:
                self.score_list.append((moves, self.score(env_copy, moves)))

                if isOver:
                    return moves
            else:
                if isOver:
                    self.score_list.append((moves, self.score(env_copy, moves)))
                    return moves
                self.search(env_copy, depth - 1, moves, state_list)


    def limiteddfs(self, actions):
        state, reward, isOver, info = self.env.reset(), 0, False, {}
        observation = [state]
        action_sequence = self.search(self.env, self.max_depth, [], observation)
        self.tick += len(action_sequence)

        if self.tick > self.tick_max:
            assert 0
        
        return action_sequence
        # print(f"Used steps: {self.n} / {self.tick_max}")

    def solve(self):
        state = self.env.reset()  # Reset environment to start a new episode
        actions = self.actions
        action_sequence = self.limiteddfs(actions)
        return action_sequence

    def act(self, env):
        self.score_list = []
        moves = self.search(env, self.max_depth, [], [env._get_observation()])
        if moves is not None:
            return moves[0]
        else:
            self.score_list = sorted(self.score_list, key = lambda x: x[1])
            return self.score_list[0][0][0]