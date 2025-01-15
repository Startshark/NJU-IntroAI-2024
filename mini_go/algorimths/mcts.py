import numpy as np
import collections
import copy
import random
import math
import os
from agent.agent import StepOutput

class MCTSNode:
    def __init__(self, time_step, player_id, env, parent=None, action=None):
        self.time_step = time_step
        self.player_id = player_id
        self.env = env
        self.parent = parent
        self.action = action
        self.legal_action = time_step.observations['legal_actions'][self.player_id]
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.legal_action)

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children if child.visits > 0  # 不然会报Division_By_Zero的错误!
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        legal_actions = self.legal_action
        for action in legal_actions:
            new_env = copy.deepcopy(self.env)
            new_time_step = new_env.step(action)
            child = MCTSNode(time_step=new_time_step, player_id=self.player_id, env=new_env, parent=self, action=action)
            self.children.append(child)

    def rollout(self, rollout_policy_net=None):
        current_time_step = copy.deepcopy(self.time_step)
        current_env = copy.deepcopy(self.env)
        while not current_time_step.last():
            possible_moves = current_time_step.observations['legal_actions'][self.player_id]
            if rollout_policy_net:
                action = self.policy_rollout(rollout_policy_net, current_time_step)
                print("Rollout Action is: ", action)
            else:
                action = self.rollout_policy(possible_moves)
                # print("MCTS Action is: ", action)
            current_time_step = current_env.step(action)
        return current_time_step.rewards[self.player_id]

    def rollout_policy(self, possible_moves):
        return random.choice(possible_moves)

    def policy_rollout(self, rollout_policy_net, time_step):
        # 使用 rollout_policy_net 选择动作
        agent_output = rollout_policy_net.step(time_step)
        return agent_output.action

    def ucb(self, c_param=1.4):
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + c_param * math.sqrt(math.log(self.parent.visits) / self.visits)

    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

class MCTS:
    def __init__(self, policy_net, rollout_policy_net, player_id, env, max_depth=15, n_simulations=1000, c_param=1.4):
        self.policy_net = policy_net
        self.player_id = player_id
        self.rollout_policy_net = rollout_policy_net
        self.n_simulations = n_simulations
        self.c_param = c_param
        self.env = env
        self.max_depth = max_depth
        self.version = 0
        self.loss = 0  # 添加 loss 属性

    def search(self, time_step):
        root = MCTSNode(time_step, self.player_id, env=self.env)
        
        # 使用 policy_net 选择初始动作
        if self.policy_net:
            agent_output = self.policy_net.step(time_step)
            initial_action = agent_output.action
            initial_env = copy.deepcopy(self.env)
            initial_time_step = initial_env.step(initial_action)
            root = MCTSNode(initial_time_step, self.player_id, env=initial_env, parent=None, action=initial_action)
        
        for _ in range(self.n_simulations):
            node = root
            while not node.time_step.last() and node.is_fully_expanded():
                node = node.best_child(self.c_param)
            if not node.is_fully_expanded():
                node.expand()
                node = random.choice(node.children)
            reward = node.rollout(self.rollout_policy_net)
            node.backpropagate(reward)
            self.loss += self.calculate_loss(node)  # 计算并累加 loss
        return StepOutput(action=root.best_child().action, probs=1)

    def calculate_loss(self, node):
        if node.visits == 0:
            return 0
        return (node.value / node.visits) ** 2

    def step(self, time_step):
        root = MCTSNode(time_step=time_step, player_id=self.player_id, env=self.env)
        for _ in range(self.max_depth):
            self.explore(root)
        max_ucb = -float('inf')
        for child in root.children:
            if child.ucb() > max_ucb:
                ret = child
                max_ucb = child.ucb()
        return StepOutput(action = ret.action, probs = 1)
    
    def explore(self, node):
        if not node.children:
            if node.visits == 0:
                node.rollout(self.rollout_policy_net)
                self.backtrace(node)
            else:
                node.expand()
        else:
            next_node = max(node.children, key=lambda child: child.ucb(self.c_param))
            self.explore(next_node)

    def backtrace(self, node):
        while node:
            node.visits += 1
            node.value += node.value
            node = node.parent

    def set_version(self, version):
        self.version = version

    def get_version(self):
        return self.version
    
    def save(self, checkpoint_root, checkpoint_name):
        save_prefix = os.path.join(checkpoint_root, checkpoint_name)
        if self.policy_net:
            self.policy_net.save(checkpoint_root, checkpoint_name + '_policy_net')
        if self.rollout_policy_net:
            self.rollout_policy_net.save(checkpoint_root, checkpoint_name + '_rollout_policy_net')
        with open(save_prefix + '_version.txt', 'w') as f:
            f.write(str(self.version))