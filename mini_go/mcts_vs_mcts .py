from absl import logging, flags, app
from environment.GoEnv import Go
from algorimths.mcts import MCTS
from agent.agent import RandomAgent
import time, os
import numpy as np
import tensorflow as tf
import agent.agent as agent
from algorimths.dqn import DQN
import pygame
import random
from algorimths.opponent import OpponentPool

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_train_episodes", 10000,
                     "Number of training episodes for each base policy.")
flags.DEFINE_integer("num_eval", 10,
                     "Number of evaluation episodes")
flags.DEFINE_integer("eval_every", 2000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer("save_every", 2000,
                     "Episode frequency at which the agents save the policies.")
flags.DEFINE_integer("n_simulations", 1,
                     "Number of simulations for MCTS.")
flags.DEFINE_float("c_param", 1.4,
                   "Exploration parameter for MCTS.")
flags.DEFINE_boolean("visualize", True,
                     "Whether to visualize the game.")

flags.DEFINE_integer("pool_size", 10, "Size of the opponent pool")
flags.DEFINE_float("pool_sampling_rate", 0.7, "Rate of sampling from pool vs random agent")
flags.DEFINE_string("pool_directory", "opponent_pool", "Directory to save/load opponent pool")

ret = [0]

def initialize_pygame():
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption('Go Game')
    font = pygame.font.Font(None, 36)
    return screen, font

def draw_board(screen, board_size):
    # Constants for board layout
    GRID_SIZE = 100  # Size of each grid square
    MARGIN = 50     # Margin from screen edge
    BOARD_COLOR = (0, 0, 255)  # Blue
    LINE_COLOR = (0, 0, 0)     # Black
    
    screen.fill(BOARD_COLOR)
    
    # Draw vertical and horizontal lines
    for i in range(board_size):
        # Vertical lines
        pygame.draw.line(screen, LINE_COLOR, 
                        (MARGIN + i * GRID_SIZE, MARGIN),
                        (MARGIN + i * GRID_SIZE, MARGIN + (board_size - 1) * GRID_SIZE))
        # Horizontal lines
        pygame.draw.line(screen, LINE_COLOR,
                        (MARGIN, MARGIN + i * GRID_SIZE),
                        (MARGIN + (board_size - 1) * GRID_SIZE, MARGIN + i * GRID_SIZE))

def draw_stones(screen, board):
    # Constants for stone drawing
    GRID_SIZE = 200 # Size of each grid square
    MARGIN = 50     # Margin from screen edge
    STONE_RADIUS = 20  # Radius of each stone
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    
    for i in range(len(board)):
        for j in range(len(board[i])):
            stone_pos = (MARGIN + j * GRID_SIZE, MARGIN + i * GRID_SIZE)
            if board[i][j] == 1:
                pygame.draw.circle(screen, BLACK, stone_pos, STONE_RADIUS)
            elif board[i][j] == -1:
                pygame.draw.circle(screen, WHITE, stone_pos, STONE_RADIUS)

def main(unused_argv):
    begin = time.time()
    env = Go()

    rollout_hidden_layers_sizes = [128, 128]
    rollout_hidden_layers_sizes_kwargs = {
        "replay_buffer_capacity": int(5e4),
        "epsilon_decay_duration": int(0.6 * 10000),
        "epsilon_start": 0.8,
        "epsilon_end": 0.001,
        "learning_rate": 1e-3,
        "learn_every": 128,
        "batch_size": 128,
        "max_global_gradient_norm": 10,
    }

    value_hidden_layers_sizes = [128, 128]
    value_hidden_layers_sizes_kwargs = {
        "replay_buffer_capacity": int(5e4),
        "epsilon_decay_duration": int(0.6 * 10000),
        "epsilon_start": 0.8,
        "epsilon_end": 0.001,
        "learning_rate": 1e-3,
        "learn_every": 128,
        "batch_size": 128,
        "max_global_gradient_norm": 10,
    }

    # Create a separate graph for each network
    rollout_graph = tf.Graph()
    value_graph = tf.Graph()

    # Load fast-rollout network
    with tf.Session(graph=rollout_graph) as sess:
        with rollout_graph.as_default():
            # 初始化对手池
            opponent_pool = OpponentPool(sess, env, FLAGS.pool_size)
            # 加载已有的对手池（如果存在）
            if os.path.exists(FLAGS.pool_directory):
                opponent_pool.load_pool(FLAGS.pool_directory)

            rollout_agent = DQN(sess, 0, env.state_size, env.action_size, rollout_hidden_layers_sizes, **rollout_hidden_layers_sizes_kwargs)
            sess.run(tf.global_variables_initializer())
            rollout_agent.restore("saved_model/10000_fast")

    # Load value network
    with tf.Session(graph=value_graph) as sess:
        with value_graph.as_default():
            # 初始化对手池
            opponent_pool = OpponentPool(sess, env, FLAGS.pool_size)
            # 加载已有的对手池（如果存在）
            if os.path.exists(FLAGS.pool_directory):
                opponent_pool.load_pool(FLAGS.pool_directory)
            
            value_agent = DQN(sess, 0, env.state_size, env.action_size, value_hidden_layers_sizes, **value_hidden_layers_sizes_kwargs)
            sess.run(tf.global_variables_initializer())
            value_agent.restore("saved_model/10000")

        mcts_random_agent = MCTS(policy_net=None, rollout_policy_net=None, player_id=0, env=env, max_depth=10, n_simulations=FLAGS.n_simulations, c_param=FLAGS.c_param)
        mcts_agent = MCTS(policy_net=value_agent, rollout_policy_net=rollout_agent, player_id=0, env=env, max_depth=10, n_simulations=FLAGS.n_simulations, c_param=FLAGS.c_param)
        agents = [mcts_random_agent, mcts_agent]
        ret = [0]
        max_len = 2000

        if FLAGS.visualize:
            screen, font = initialize_pygame()

        # train the agent
        logging.info("Training Start~")
        for ep in range(FLAGS.num_train_episodes):
            
            
            if (ep + 1) % FLAGS.eval_every == 0:
                losses = agents[0].loss
                logging.info("Episodes: {}: Rewards: {}".format(ep + 1, np.mean(ret)))
                with open('log_{}_{}'.format(os.environ.get('BOARD_SIZE'), begin), 'a+') as log_file:
                    log_file.writelines("{}, {}\n".format(ep+1, np.mean(ret)))
            if (ep + 1) % FLAGS.save_every == 0:
                if not os.path.exists("saved_model"):
                    os.mkdir('saved_model')
                agents[0].save(checkpoint_root='saved_model', checkpoint_name='{}_mcts'.format(ep+1))

                
                # 保存当前模型到对手池
                save_path = os.path.join(FLAGS.pool_directory, 
                                        f'opponent_{ep+1}')
                rollout_agent.save(
                    checkpoint_root=FLAGS.pool_directory,
                    checkpoint_name=f'opponent_{ep+1}'
                )
                opponent_pool.add_opponent(save_path, ep+1)
                
                # 保存整个对手池
                opponent_pool.save_pool(FLAGS.pool_directory)

                # 保存模型

            # 训练回合
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if player_id == 0:
                    action = mcts_agent.search(time_step).action
                else:
                    action = mcts_random_agent.search(time_step).action
                time_step = env.step(action)
                # print(time_step.observations["info_state"][0])
                # print(env.get_current_board())

                if FLAGS.visualize:
                    # 绘制棋盘和棋子
                    draw_board(screen, env.N)
                    draw_stones(screen, env.get_current_board().board)
                    pygame.display.flip()

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return

            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)
            if len(ret) < max_len:
                ret.append(time_step.rewards[0])
            else:
                ret[ep % max_len] = time_step.rewards[0]

        # evaluated the trained agent
        logging.info("Evaluation Start~")
        ret = []
        for ep in range(FLAGS.num_eval):
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if player_id == 0:
                    action = mcts_agent.search(time_step).action
                else:
                    action = mcts_random_agent.search(time_step).action
                time_step = env.step(action)
                # print(env.get_current_board())


                if FLAGS.visualize:
                    # 绘制棋盘和棋子
                    draw_board(screen, env.N)
                    draw_stones(screen, env.get_current_board().board)
                    pygame.display.flip()

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return

            # Episode is over, step all agents with final info state.
            mcts_agent.step(time_step)
            mcts_random_agent.step(time_step)
            ret.append(time_step.rewards[0])
        print(np.mean(ret))

    print('Time elapsed:', time.time()-begin)

if __name__ == '__main__':
    app.run(main)
    # 输出对局结果
    result_agent_0 = sum(1 for r in ret if r > 0)
    result_agent_1 = sum(1 for r in ret if r < 0)
    draws = len(ret) - result_agent_0 - result_agent_1
    print(f'Result: {result_agent_0} - {result_agent_1}')
    print(f'Draws: {draws}')