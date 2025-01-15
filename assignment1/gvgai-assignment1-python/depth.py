import copy
class LimitedDFSAgent:
    def __init__(self, env, tick_max):
        self.env = env
        self.tick_max = tick_max
        self.tick = 0
        self.actions=env.action_space
        # Your can add new attributes if needed

    def act(self,env):
        self.lst_score=[]
        DEPTH=12
        steps=self.search(env,DEPTH,[],[env._get_observation()])
        if steps!=None:            
            return steps[0]
        else:
            self.lst_score=sorted(self.lst_score,key=lambda x: x[1])
            return self.lst_score[0][0][0]
        if self.tick > self.tick_max:
            assert 0



        # next_state, reward, isOver, info = env_copy.step(action_id)
        # self.tick += 1
        # print(f"Used steps: {self.n} / {self.tick_max}")

    def solve(self):
        state = self.env.reset()  # Reset environment to start a new episode
        actions = self.env.action_space
        

    
        action_sequence = self.limiteddfs()
        return action_sequence

    def search(self,env,depth,steps,history):
        action_reasonable=[]
        for action in self.actions:
            temp_env=copy.deepcopy(env)
            newstate=temp_env.step(action)
            if newstate[0] not in history and (newstate[3]!={'message': 'Fell into hole. Game over.'}):
                action_reasonable.append(action)
        if depth==1:
            for action in action_reasonable:
                temp_env=copy.deepcopy(env)
                state=temp_env.step(action)
                temp_steps=copy.copy(steps)
                temp_steps.append(action)
                temp_history=copy.deepcopy(history)
                temp_history.append(state[0])                
                self.lst_score.append((temp_steps,self.score(temp_env,temp_steps)))
                if state[2]:
                    return temp_steps
        else:
            for action in action_reasonable:
                temp_env=copy.deepcopy(env)
                state=temp_env.step(action)
                temp_steps=copy.copy(steps)
                temp_steps.append(action)
                temp_history=copy.deepcopy(history)
                temp_history.append(state[0]) 
                if state[2]:
                    self.lst_score.append((temp_steps,self.score(temp_env,temp_steps)))
                    return temp_steps
                self.search(temp_env,depth-1,temp_steps,temp_history)

    def score(self, env,steps):
        def distant(cord1,cord2):
            return abs(cord1[0]-cord2[0])+abs(cord1[1]-cord2[1])
        state=env._get_observation()

        height=env.height
        width=env.width
        win=True
        for i in range(height):
            for j in range(width):
                cell=state[i][j]
                if 'key' in cell:
                    cord_key=i,j
                if 'avatar_nokey' in cell:
                    cord_Anokey=i,j
                    gotkey=False
                if 'avatar_withkey' in cell:
                    cord_Awithkey=i,j
                    gotkey=True
                if 'goal' in cell:
                    cord_goal=i,j
                    win=False
        if win:
            return len(steps)/10
        if gotkey:
            score=distant(cord_Awithkey,cord_goal)
        else:
            score=(distant(cord_Anokey,cord_key)+distant(cord_key,cord_goal))
        return score