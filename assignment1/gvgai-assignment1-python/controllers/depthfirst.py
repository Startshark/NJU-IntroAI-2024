actions = [4, 3, 2, 1]

class DFSAgent:
    def __init__(self, env, tick_max):
        self.env = env
        self.tick_max = tick_max
        self.tick = 0
        # Your can add new attributes if needed

    def dfs(self):
        
        state = self.env.reset()
        reward, isOver, info = 0, False, {}
        observation = [state]
        action_sequence = []

        while not isOver:
            valid_move = []
            # state_copy, reward_copy, isOver_copy, info_copy = state, reward, isOver, info
            for action_id in actions:
                state, reward, isOver, info = self.env.step(action_id)  # next_situation
                if state not in observation and info != {'message': 'Fell into hole. Game over.'} and info != {'message': 'Hit wall'} and info != {'message': 'Cannot push box. Obstacle ahead.'}:
                    valid_move.append(action_id)
                    
                    # Debug
                    # print(info, action_id)

                # Back to the previous step
                self.env.reset()
                for x in action_sequence:
                    state, reward, isOver, info = self.env.step(x)
                    
            if len(valid_move) == 0:
                if len(action_sequence) != 0:
                    action_sequence.pop()
                    self.env.reset()
                    for x in action_sequence:
                        state, reward, isOver, info = self.env.step(x)
            else:
                action = valid_move[0]
                state, reward, isOver, info = self.env.step(action)
                action_sequence.append(action)
                observation.append(state)
                self.tick += 1            
   
        return action_sequence

    def solve(self):
        state = self.env.reset()  # Reset environment to start a new episode
        action_sequence = self.dfs()
        return action_sequence