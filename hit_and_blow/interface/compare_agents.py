import argparse
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Tuple

from hit_and_blow.core.environment import Environment
from hit_and_blow.agents import AgentRegistry

def run_simulation(agent_name: str, num_games: int, num_digits: int, digit_range: int, max_turns: int) -> Dict[str, Any]:
    """指定したエージェントで複数回ゲームを実行し、統計を収集する"""
    # エージェントのインスタンス化
    agent_class = AgentRegistry.get(agent_name)
    
    # 統計情報の初期化
    stats = {
        "agent_name": agent_name,
        "win_count": 0,
        "total_turns": 0,
        "turn_history": [],
        "execution_time": 0,
    }
    
    start_time = time.time()
    
    # 指定回数ゲームを実行
    for game_idx in range(num_games):
        # 環境のセットアップ
        env = Environment(digits=num_digits, number_range=digit_range, max_turns=max_turns)
        obs = env.reset()
        
        # エージェントのセットアップ
        agent = agent_class(num_digits=num_digits, digit_range=digit_range)
        agent.reset()
        
        done = False
        while not done:
            # エージェントの予測を取得
            guess = agent.predict(obs["history"])
            
            # 環境を更新
            obs, reward, done, info = env.step(guess)
        
        # ゲーム結果の集計
        if info["hits"] == env.digits:  # 勝利条件
            stats["win_count"] += 1
        
        stats["turn_history"].append(env.turns)
        stats["total_turns"] += env.turns
    
    # 集計・計算
    stats["execution_time"] = time.time() - start_time
    stats["win_rate"] = stats["win_count"] / num_games
    stats["avg_turns"] = stats["total_turns"] / num_games
    stats["turn_counts"] = defaultdict(int)
    
    for turns in stats["turn_history"]:
        stats["turn_counts"][turns] += 1
    
    return stats

def compare_agents(agents: List[str], num_games: int, num_digits: int, digit_range: int, max_turns: int) -> Dict[str, Dict[str, Any]]:
    """複数のエージェントを比較する"""
    results = {}
    
    for agent_name in agents:
        print(f"エージェント '{agent_name}' をテスト中 ({num_games}ゲーム)...")
        stats = run_simulation(agent_name, num_games, num_digits, digit_range, max_turns)
        results[agent_name] = stats
        
        # 中間結果の表示
        print(f"  勝率: {stats['win_rate']:.1%}")
        print(f"  平均ターン数: {stats['avg_turns']:.2f}")
        print(f"  実行時間: {stats['execution_time']:.2f}秒")
        print()
    
    return results

def plot_results(results: Dict[str, Dict[str, Any]], num_digits: int, digit_range: int, max_turns: int) -> None:
    """結果をグラフで可視化する"""
    # 日本語フォントの設定
    try:
        import japanize_matplotlib
    except ImportError:
        print("警告: japanize-matplotlibがインストールされていないため、日本語が正しく表示されない可能性があります")
        print("日本語表示するには: pip install japanize-matplotlib")
    
    # プロット数の設定
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'エージェント比較 ({num_digits}桁, 範囲:0-{digit_range-1}, 最大{max_turns}ターン)', fontsize=16)
    
    agent_names = list(results.keys())
    colors = ['blue', 'red', 'green', 'purple', 'orange']  # 最大5エージェント用の色
    
    # 1. 勝率の比較
    win_rates = [results[agent]["win_rate"] * 100 for agent in agent_names]
    axs[0, 0].bar(agent_names, win_rates, color=colors[:len(agent_names)])
    axs[0, 0].set_title('勝率比較')
    axs[0, 0].set_ylabel('勝率 (%)')
    axs[0, 0].set_ylim(0, 100)
    
    # 2. 平均ターン数の比較
    avg_turns = [results[agent]["avg_turns"] for agent in agent_names]
    axs[0, 1].bar(agent_names, avg_turns, color=colors[:len(agent_names)])
    axs[0, 1].set_title('平均ターン数')
    axs[0, 1].set_ylabel('ターン数')
    
    # 3. ターン数の分布
    max_observed_turns = max([max(results[agent]["turn_history"]) for agent in agent_names])
    bins = list(range(1, max_observed_turns + 2))
    
    for i, agent in enumerate(agent_names):
        axs[1, 0].hist(results[agent]["turn_history"], bins=bins, alpha=0.5,
                   label=agent, color=colors[i % len(colors)])
    
    axs[1, 0].set_title('ターン数の分布')
    axs[1, 0].set_xlabel('ターン数')
    axs[1, 0].set_ylabel('ゲーム数')
    axs[1, 0].legend()
    
    # 4. 実行時間の比較
    exec_times = [results[agent]["execution_time"] for agent in agent_names]
    axs[1, 1].bar(agent_names, exec_times, color=colors[:len(agent_names)])
    axs[1, 1].set_title('実行時間')
    axs[1, 1].set_ylabel('時間 (秒)')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # グラフを保存
    plt.savefig('agent_comparison.png')
    print(f"比較グラフを 'agent_comparison.png' に保存しました")
    
    # グラフを表示
    plt.show()

def print_detailed_results(results: Dict[str, Dict[str, Any]]) -> None:
    """詳細な結果を表示する"""
    print("\n===== 詳細結果 =====")
    
    for agent_name, stats in results.items():
        print(f"\nエージェント: {agent_name}")
        print(f"勝率: {stats['win_rate']:.1%} ({stats['win_count']}勝 / {len(stats['turn_history'])}ゲーム)")
        print(f"平均ターン数: {stats['avg_turns']:.2f}")
        print(f"実行時間: {stats['execution_time']:.2f}秒")
        
        print("\nターン数分布:")
        for turns, count in sorted(stats['turn_counts'].items()):
            percentage = count / len(stats['turn_history']) * 100
            print(f"  {turns}ターン: {count}ゲーム ({percentage:.1f}%)")

def main():
    """エージェント比較のメインエントリポイント"""
    parser = argparse.ArgumentParser(description="Hit and Blow エージェント比較ツール")
    parser.add_argument("--games", type=int, default=100, help="各エージェントの実行ゲーム数 (デフォルト: 100)")
    parser.add_argument("--digits", type=int, default=3, help="桁数 (デフォルト: 3)")
    parser.add_argument("--range", type=int, default=6, help="数字の範囲 (0〜range-1) (デフォルト: 6)")
    parser.add_argument("--turns", type=int, default=10, help="最大ターン数 (デフォルト: 10)")
    
    # 利用可能なエージェント一覧を取得
    available_agents = AgentRegistry.get_all_names()
    
    parser.add_argument(
        "--agents", 
        type=str, 
        nargs='+',
        default=available_agents,
        choices=available_agents,
        help=f"比較するエージェントのリスト (デフォルト: 全エージェント, 選択肢: {', '.join(available_agents)})"
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="グラフ表示を無効にする"
    )
    
    args = parser.parse_args()

    print(f"Hit and Blow エージェント比較")
    print(f"設定: {args.digits}桁, 範囲:0-{args.range-1}, 最大{args.turns}ターン, 各{args.games}ゲーム")
    print(f"比較エージェント: {', '.join(args.agents)}")
    print("===========================================")
    
    # エージェント比較の実行
    results = compare_agents(args.agents, args.games, args.digits, args.range, args.turns)
    
    # 詳細結果の表示
    print_detailed_results(results)
    
    # 結果のグラフ表示
    if not args.no_plot:
        try:
            plot_results(results, args.digits, args.range, args.turns)
        except ImportError:
            print("警告: matplotlibがインストールされていないため、グラフは表示されません")
            print("グラフ表示するには: pip install matplotlib")

if __name__ == "__main__":
    main() 