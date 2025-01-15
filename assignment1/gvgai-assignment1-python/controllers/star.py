import copy
class AstarAgent:
    def __init__(self, env, tick_max):
        self.env = env
        self.tick_max = tick_max
        self.tick = 0
        self.actions=env.action_space
        # Your can add new attributes if needed

    def astar(self,goal,env):
        lenpreway=self.prewayfunc(env,goal)
        if lenpreway==0:
            return []
        state=env._get_observation()
        boundaries=[(state,[],lenpreway,[state])]
        isOver = False
        while True:
            boundaries=sorted(boundaries,key=lambda x:x[2]+len(x[1]))
            # print(len(boundaries[0][1])+boundaries[0][2],boundaries[0][1],end='|')
            # print('')
            search_state,search_steps,_,search_history=boundaries[0]
            temp_env=copy.deepcopy(env)
            for x in search_steps:
                temp_env.step(x)
            action_reasonable=[]
            for action in self.actions:
                temp_temp_env=copy.deepcopy(temp_env)
                newstate=temp_temp_env.step(action)
                if newstate[0] not in search_history and (newstate[3]!={'message': 'Fell into hole. Game over.'}):
                    action_reasonable.append(action)
            for action in action_reasonable:
                temp_steps=copy.copy(search_steps)
                temp_steps.append(action)
                temp_temp_env=copy.deepcopy(temp_env)
                new_state=temp_temp_env.step(action)
                isOver = new_state[2]
                temp_history=copy.deepcopy(search_history)
                temp_history.append(new_state[0])
                lenpreway=self.prewayfunc(temp_temp_env,goal)
                if lenpreway==0:
                    return temp_steps
                Node=(new_state[0],temp_steps,lenpreway,temp_history)
                boundaries.append(Node)
            del boundaries[0]
        if self.tick > self.tick_max:
            assert 0

    def solve(self):
        state = self.env.reset()  # Reset environment to start a new episode
        actions = self.env.action_space
        steps1=self.astar('mushroom',self.env)
        # print(steps1)
        for x in steps1:
            self.env.step(x)
        steps2=self.astar('key',self.env)
        # print(steps2)
        for x in steps2:
            self.env.step(x)
        steps3=self.astar('goal',self.env)
        # print(steps3)
        return steps1+steps2+steps3
        raise NotImplementedError
    
        action_sequence = self.astar()
        return action_sequence

    def act(self, env):
        
        return self.astar('goal', env=env)
    
    def prewayfunc(self,env,current_goal):
        state=env._get_observation()
        #print(len(state),len(state[0]))
        height=env.height
        width=env.width
        goal_exist=False
        box_lst=[]
        wall=0
        hole=0
        for i in range(height):
            for j in range(width):
                cell=state[i][j]
                if 'avatar_nokey' in cell or 'avatar_withkey' in cell:
                    cord_A=i,j
                if current_goal in cell:
                    cord_goal=i,j
                    goal_exist=True
                    #print("cord_goal:",(i,j))
                if 'box' in cell:
                    box_lst.append((i,j))
        if goal_exist:
            dir_lst=['up','dn','lf','rg']
            ways=[(x,y) for x in dir_lst for y in dir_lst]
            ways.remove(('up','dn'))
            ways.remove(('dn','up'))
            ways.remove(('lf','rg'))
            ways.remove(('rg','lf'))
            def trans(way):
                x,y=cord_goal
                cordlst=[]
                for i in range(2):
                    if way[i]=='up':
                        y=y-1
                    elif way[i]=='dn':
                        y=y+1
                    elif way[i]=='lf':
                        x=x-1
                    else:
                        x=x+1
                    cordlst.append((x,y))
                return cordlst
            def distant(cord1,cord2):
                return abs(cord1[0]-cord2[0])+abs(cord1[1]-cord2[1])
            ways_codes=list(map(trans,ways))
            #print("wayscodes:",ways_codes)
            def near_box(cord):
                x,y=cord
                mindis=height+width
                for box in box_lst:
                    dis=distant((x,y),box)
                    if dis<mindis:
                        mindis=dis
                        cord_box=box
                return mindis,cord_box
            pay_lst=[]
            for way_code in ways_codes:
                cord1=way_code[1]
                cord2=way_code[0]
                if cord1[0]<0 or cord1[0]>=len(state) or cord1[1]<0 or cord1[1]>=len(state[0]):
                    ways_codes.remove(way_code)
            for way_code in ways_codes:
                pay=0
                to=True
                cord1=way_code[1]
                cord2=way_code[0]
                cell1=state[cord1[0]][cord1[1]]
                cell2=state[cord2[0]][cord2[1]]
                if 'hole' in cell1:
                    pay+=near_box(cord1)[0]+distant(cord_A,near_box(cord1)[1])-1
                    if 'hole' in cell2:
                        pay+=distant(near_box(cord2)[1],cord1)-1+near_box(cord2)[0]+2
                    elif 'floor' in cell2:
                        pay+=3
                    elif 'wall' in cell2:
                        to=False
                elif 'floor' in cell1:
                    if 'hole' in cell2:
                        pay+=distant(near_box(cord2)[1],cord_A)+near_box(cord2)[0]+1
                    elif 'floor' in cell2:
                        if 'avatar_nokey' in cell2 or 'avatar_withkey' in cell2:
                            pay+=1
                        else:
                            pay+=distant(cord_A,cord1)+2
                    elif 'wall' in cell2:
                        to=False
                elif 'wall' in cell1:
                    to=False
                if to:
                    pay_lst.append(pay)
            pay_lst=sorted(pay_lst)
            #print("paylist:",pay_lst)
            return pay_lst[0]
        else:
            return 0