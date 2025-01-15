import os
import re
import pickle
import numpy as np

##################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
##################################################

from play import AliensEnvPygame

def find_files_with_pattern(directory, pattern):
    # 编译正则表达式
    regex = re.compile(pattern)
    matched_files = []

    # 遍历文件夹
    for root, dirs, files in os.walk(directory):
        # print(f"Current directory: {root}")  # Debug: print current directory
        # print(f"Files found: {files}")        # Debug: print files found
        if regex.match(root[7:]):
            matched_files.append(root[7:])

    return matched_files

def extract_features(observation):

    grid = observation
    features = []
    object_mapping = {
            'floor': 0,
            'wall': 1,
            'avatar': 2,
            'alien': 3,
            'bomb': 4,
            'portalSlow': 5,
            'portalFast': 6,
            'sam': 7,
            'base': 8
        }

    def cell_to_feature(cell):
        feature_vector = [0] * len(object_mapping)
        for obj in cell:
            index = object_mapping.get(obj, -1)
            if index >= 0:
                feature_vector[index] = 1
        return feature_vector

    # 提取每个单元格的特征
    for row in grid:
        for cell in row:
            cell_feature = cell_to_feature(cell)
            features.extend(cell_feature)

    ################################################
    avatar_position = None
    col = len(grid[0])

    for row in grid:
        for j, cell in enumerate(row):
            if 'avatar' in cell:
                avatar_position = j
                break
        if avatar_position is not None:
            break
    
    def second_min(distances):
        # 如果元素不足两个，返回默认值
        if len(distances) < 2:
            return 1
        # 对列表进行排序并返回第二小的值
        unique_distances = sorted(set(distances))  # 去重后排序
        return unique_distances[1] if len(unique_distances) > 1 else 0

    def get_relative_position():
        alien_distances = []
        bomb_distances = []
        alien_feature_vector = [0] * len(object_mapping)
        bomb_feature_vector = [0] * len(object_mapping)
        if avatar_position is not None:
            for row in grid:
                for j, cell in enumerate(row):
                    for obj in cell:
                        if obj == 'alien':
                            alien_distances.append(abs(avatar_position - j) / col)  # 使用曼哈顿距离的横向分量计算相对距离, 并且进行归一化
                        if obj == 'bomb':
                            bomb_distances.append(abs(avatar_position - j) / col)  # 使用曼哈顿距离的横向分量计算相对距离, 并且进行归一化
        alien_distance = second_min(alien_distances)
        alien_feature_vector[3] = alien_distance
        bomb_distance = second_min(bomb_distances)
        bomb_feature_vector[4] = bomb_distance
        return alien_feature_vector, bomb_feature_vector


    alien_feature_vector, bomb_feature_vector = get_relative_position()
    for _ in range(col):
        features.extend(alien_feature_vector)
    for _ in range(col):
        features.extend(bomb_feature_vector)

    # 提取区域特征, 但是添加以后极慢无比！
    num_regions = 4
    region_size = len(grid) // num_regions
    for region_i in range(num_regions):
        for region_j in range(num_regions):
            region_counts = [0] * len(object_mapping)
            for i in range(region_i * region_size, (region_i + 1) * region_size):
                for j in range(region_j * region_size, (region_j + 1) * region_size):
                    for obj in grid[i][j]:
                        index = object_mapping.get(obj, -1)
                        if index >= 0:
                            region_counts[index] += 1
            features.extend(region_counts)
    
    return np.array(features)

def main():
    level = eval(input("请输入要训练的关卡等级(0-4): "))
    pattern = rf"^game_records_lvl{level}_.*$"
    # pattern = rf"^game_records_lvl[0-4]_.*$"
    data_list = find_files_with_pattern('./logs', pattern)[1:]
    print(data_list)

    data = []
    for data_load in data_list:
        with open(os.path.join('logs', data_load, 'data.pkl'), 'rb') as f:
            data += pickle.load(f)

    X = []
    y = []
    for observation, action in data:
        features = extract_features(observation)
        X.append(features)
        y.append(action)

    X = np.array(X)
    y = np.array(y)




    ##############################################
    # clf = RandomForestClassifier(n_estimators=100)
    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = DecisionTreeClassifier()
    clf = MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=1000)
    ##############################################

    clf.fit(X, y)

    env = AliensEnvPygame(level=level, render=False)

    ###################################################################
    env.log_folder = f'logs/game_train_lvl{env.level}_{env.timing}'

    if not os.path.exists(env.log_folder):
        os.makedirs(env.log_folder)
        os.removedirs(f'logs/game_records_lvl{env.level}_{env.timing}')
    ###################################################################

    with open(f'{env.log_folder}/gameplay_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print("模型训练完成")
    print("模型已保存到", f'{env.log_folder}')

if __name__ == '__main__':
    main()
