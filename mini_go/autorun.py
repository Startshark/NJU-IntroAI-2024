import subprocess
import re

def run_mcts_vs_mcts():
    # 运行 mcts_vs_mcts.py 脚本
    result = subprocess.run(['python', 'mcts_vs_mcts.py'], capture_output=True, text=True)
    return result.stdout

def parse_results(output):
    # 解析输出结果，统计每个 agent 的胜率
    wins_agent_0 = 0
    wins_agent_1 = 0
    draws = 0

    # 使用正则表达式匹配输出中的胜负信息
    win_pattern = re.compile(r'Result: (\d) - (\d)')
    for line in output.split('\n'):
        match = win_pattern.search(line)
        if match:
            score_0 = int(match.group(1))
            score_1 = int(match.group(2))
            if score_0 > score_1:
                wins_agent_0 += 1
            elif score_1 > score_0:
                wins_agent_1 += 1
            else:
                draws += 1

    return wins_agent_0, wins_agent_1, draws

def main():
    num_games = 1000
    total_wins_agent_0 = 0
    total_wins_agent_1 = 0
    total_draws = 0

    for _ in range(num_games):
        output = run_mcts_vs_mcts()
        wins_agent_0, wins_agent_1, draws = parse_results(output)
        total_wins_agent_0 += wins_agent_0
        total_wins_agent_1 += wins_agent_1
        total_draws += draws

    print(f'Agent 0 wins: {total_wins_agent_0} ({total_wins_agent_0 / num_games * 100:.2f}%)')
    print(f'Agent 1 wins: {total_wins_agent_1} ({total_wins_agent_1 / num_games * 100:.2f}%)')
    print(f'Draws: {total_draws} ({total_draws / num_games * 100:.2f}%)')

if __name__ == '__main__':
    main()