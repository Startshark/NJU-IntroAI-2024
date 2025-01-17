import os
import sys
import pygame
import pickle
from sklearn.ensemble import RandomForestClassifier

##################################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
##################################################


from play import AliensEnvPygame
from learn import extract_features

def main():

    level = eval(input("请输入要测试的关卡等级(0-4): "))

    ##############################################
    # clf = RandomForestClassifier(n_estimators=100)
    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = DecisionTreeClassifier()
    clf = MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=1000)
    ##############################################  

    pygame.init()

    env = AliensEnvPygame(level=level, render=False)
    
    model = input("请输入要测试的模型名称: ")

    ####################################################################
    env.log_folder =  f'logs/game_ai_play_lvl{env.level}_{env.timing}'
    if not os.path.exists(env.log_folder):
        os.makedirs(env.log_folder)
        os.removedirs(f'logs/game_records_lvl{env.level}_{env.timing}')
    ####################################################################

    # 加载模型
    
    ###############################################################################################
    model_path = f'logs\{model}\gameplay_model.pkl' # 替换为你的模型的路径
    ###############################################################################################
    
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)

    print("模型加载完成")

    observation = env.reset()

    grid_image = env.do_render()

    mode = grid_image.mode
    size = grid_image.size
    data_image = grid_image.tobytes()
    pygame_image = pygame.image.fromstring(data_image, size, mode)

    screen = pygame.display.set_mode(size)
    pygame.display.set_caption('Aliens Game - AI Playing')

    screen.blit(pygame_image, (0, 0))
    pygame.display.flip()

    done = False
    total_score = 0
    step = 0
    while not done:
        features = extract_features(observation)
        features = features.reshape(1, -1)

        action = clf.predict(features)[0]

        observation, reward, game_over, info = env.step(action)
        total_score += reward
        print(f"Step: {step}, Action taken: {action}, Reward: {reward}, Done: {game_over}, Info: {info}")
        step += 1

        grid_image = env.do_render()
        mode = grid_image.mode
        size = grid_image.size
        data_image = grid_image.tobytes()
        pygame_image = pygame.image.fromstring(data_image, size, mode)

        screen.blit(pygame_image, (0, 0))
        pygame.display.flip()

        if game_over or step > 5000:
            print("游戏结束!")
            print(f"信息: {info}，分数：{total_score}")
            done = True

        pygame.time.delay(20)

    env.save_gif(filename=f'replay_ai.gif')

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
