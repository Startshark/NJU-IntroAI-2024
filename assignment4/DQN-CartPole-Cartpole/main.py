import gym
from collections import deque
import random
import argparse
import torch

from agent import DQNAgent, DDQNAgent

##############################################
import numpy as np
import matplotlib.pyplot as plt
import datetime
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
##############################################

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_name", type=str, default="ddqn")
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--max_steps_per_episode", type=int, default=500)
    parser.add_argument("--epsilon_start", type=float, default=0.9)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay_rate", type=float, default=0.995)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--buffer_size", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--update_frequency", type=int, default=10)
    #####################################################################
    parser.add_argument("--device", type=str, default="cpu") # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #####################################################################

    args = parser.parse_args()
    return args


def eval_policy(agent):
    state = env.reset()
    done = False
    return_ = 0
    while not done:
        action = agent.act(state, eps=0.)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        return_ += reward
    # print(f"Return {return_}") 
    return return_

def train(args, agent, buffer):
    # Training loop

    ####################################################################
    episodes, steps, los, returns = [], [], [], []
    start_time = datetime.datetime.now()
    current_time = start_time.strftime("%Y-%m-%d %H-%M-%S")
    ####################################################################

    for episode in range(args.num_episodes):
        # Reset the environment
        state = env.reset()
        epsilon = max(args.epsilon_end, args.epsilon_start * (args.epsilon_decay_rate ** episode))

        # Run one episode
        losses = []
        return_ = 0
        for step in range(args.max_steps_per_episode):
            # Choose and perform an action
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)


            # env.render()
            
            
            buffer.append((state, action, reward, next_state, done))
            
            if len(buffer) >= args.batch_size:
                batch = random.sample(buffer, args.batch_size)
                # Update the agent's knowledge
                loss = agent.learn(batch, args.gamma)
                losses.append(loss)
            return_ += reward
            
            state = next_state
            
            # Check if the episode has ended
            if done:
                break
        loss = torch.mean(torch.tensor(losses))
        eval_return = eval_policy(agent)

        print(f"Episode {episode + 1} Step {step + 1}: Training Loss {loss}, Return {eval_return}")
        

        ############################
        episodes.append(episode + 1)
        steps.append(step + 1)
        
        if(loss.isnan()):
            los.append(float(1.5))
        else:
            los.append(loss.item())
        
        returns.append(eval_return)
        ############################


    #############################################################################
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    
    _, axes = plt.subplots(3, 1)
    axes[0].plot(episodes, steps, label="Steps", linestyle="-.")
    axes[1].plot(episodes, returns, label="Returns", linestyle="-.", color="red")
    axes[2].plot(episodes, los, label="Losses", linestyle="-.", color="green")
    
    axes[0].set_title("Steps训练曲线")
    axes[1].set_title("Returns训练曲线")
    axes[2].set_title("Losses训练曲线")
    # axes[2].set_ylim(0, 1.5)
    # axes[2].set_yticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5])
    axes[2].annotate("此处应为NAN值", xy=(4, 1.5), xytext=(2, 0.50), arrowprops=dict(arrowstyle='->'))

    for ax in axes:
        ax.grid(alpha=0.5, linewidth=0.5)
        ax.legend()
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Values") 
    
    plt.figtext(0.5, 0, f"训练总时长: {total_time}", ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"./train log/training process {current_time}.png")
    plt.show()
    print(f"训练过程图像已保存至./train log/training process {current_time}.png")
    #############################################################################
    

if __name__ == "__main__":
    args = parser()
    # Set up the environment
    env = gym.make("CartPole-v1")

    buffer = deque(maxlen=args.buffer_size)

    # Initialize the DQNAgent
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    if args.agent_name == "dqn":
        agent = DQNAgent(input_dim, output_dim, buffer_size=args.buffer_size, seed=1234, lr = args.lr)#, device=args.device)
    elif args.agent_name == "ddqn":
        agent = DDQNAgent(input_dim, output_dim, buffer_size=args.buffer_size, seed=1234, lr = args.lr)#, device=args.device)
    else:
        assert False, "Not Implement agent!"
    train(args, agent, buffer)