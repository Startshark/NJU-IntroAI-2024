import os
import random
import tensorflow as tf
from algorimths.dqn import DQN

class OpponentPool:
    def __init__(self, session, env, pool_size=10):
        self.session = session
        self.env = env
        self.pool_size = pool_size
        self.pool = []  # 存储对手模型的列表
        self.current_version = 0

    def add_opponent(self, model_path, version):
        """添加新的对手到池中"""
        if len(self.pool) >= self.pool_size:
            # 随机移除一个旧对手
            self.pool.pop(random.randint(0, len(self.pool) - 1))
        
        # 创建新的DQN实例
        opponent = DQN(
            session=self.session,
            player_id=1,  # 作为对手时为玩家1
            state_representation_size=self.env.state_size,
            num_actions=self.env.action_size,
            hidden_layers_sizes=[128, 128]
        )
        # 加载模型权重
        opponent.restore(model_path)
        self.pool.append((opponent, version))
        self.current_version = max(self.current_version, version)

    def sample_opponent(self):
        """随机采样一个对手"""
        if not self.pool:
            return None, 0
        opponent, version = random.choice(self.pool)
        return opponent, version

    def save_pool(self, save_dir):
        """保存整个对手池"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i, (opponent, version) in enumerate(self.pool):
            opponent.save(
                checkpoint_root=save_dir,
                checkpoint_name=f'opponent_{version}'
            )

    def load_pool(self, load_dir):
        """加载对手池"""
        if not os.path.exists(load_dir):
            return
        for filename in os.listdir(load_dir):
            if filename.startswith('opponent_'):
                version = int(filename.split('_')[1])
                path = os.path.join(load_dir, filename)
                self.add_opponent(path, version)