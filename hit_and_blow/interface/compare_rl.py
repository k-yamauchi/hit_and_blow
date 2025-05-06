#!/usr/bin/env python3
"""
異なる強化学習手法を訓練・比較するスクリプト
"""
import os
import argparse
import time
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
from hit_and_blow.core.environment import Environment
from hit_and_blow.agents import AgentRegistry
from hit_and_blow.interface.train_rl import train_agent

def compare_rl_agents(num_episodes=1000, num_digits=3, digit_range=6, max_turns=10,
                     learning_rate=0.1, discount_factor=0.95, exploration_rate=0.3, 
                     test_episodes=100, verbose=True):
    """
    異なる強化学習手法を訓練し比較する
    """
    # 各エージェントタイプ
    agent_types = ["hybrid_q_learning", "q_learning", "sarsa", "hybrid_sarsa", "bandit", "hybrid_bandit"]
    
    # 結果を保存する辞書
    results = {}
    model_paths = {}
    
    # 訓練とテストの結果格納場所
    training_stats = {}
    testing_stats = {}
    
    start_time = time.time()
    
    # 各エージェントを訓練
    for agent_type in agent_types:
        print(f"\n{'='*50}")
        print(f"{agent_type}エージェントの訓練開始")
        print(f"{'='*50}")
        
        # エージェントを訓練
        model_path, stats = train_agent(
            agent_name=agent_type,
            num_episodes=num_episodes,
            num_digits=num_digits,
            digit_range=digit_range,
            max_turns=max_turns,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate,
            model_path=None,  # 自動生成
            verbose=verbose
        )
        
        training_stats[agent_type] = stats
        model_paths[agent_type] = model_path
    
    print("\n訓練完了。テストフェーズ開始...")
    
    # 各エージェントをテスト
    for agent_type in agent_types:
        print(f"\n{agent_type}エージェントのテスト")
        
        # エージェントのインスタンス化
        agent_class = AgentRegistry.get(agent_type)
        agent = agent_class(
            num_digits=num_digits, 
            digit_range=digit_range,
            learning_rate=0,  # テスト時は学習しない
            discount_factor=discount_factor,
            exploration_rate=0.05,  # テスト時は探索率を低く
            model_path=model_paths[agent_type]
        )
        
        # テスト結果の初期化
        wins = 0
        total_turns = 0
        turn_history = []
        
        # テストの実行
        for episode in range(test_episodes):
            env = Environment(digits=num_digits, number_range=digit_range, max_turns=max_turns)
            obs = env.reset()
            agent.reset()
            
            done = False
            while not done:
                guess = agent.predict(obs["history"])
                obs, reward, done, info = env.step(guess)
            
            # 結果の収集
            success = info["hits"] == env.digits
            if success:
                wins += 1
            total_turns += env.turns
            turn_history.append(env.turns)
            
            if verbose and (episode + 1) % 10 == 0:
                print(f"テストエピソード {episode+1}/{test_episodes}完了")
        
        # 統計情報の計算
        win_rate = wins / test_episodes
        avg_turns = total_turns / test_episodes
        
        # 結果の保存
        testing_stats[agent_type] = {
            "win_rate": win_rate,
            "avg_turns": avg_turns,
            "turn_history": turn_history
        }
        
        print(f"テスト結果: 勝率={win_rate:.1%}, 平均ターン数={avg_turns:.2f}")
    
    # 全体の実行時間
    elapsed_time = time.time() - start_time
    print(f"\n総実行時間: {elapsed_time:.1f}秒")
    
    # 結果のグラフ表示
    plot_results(training_stats, testing_stats, num_episodes, test_episodes)
    
    return training_stats, testing_stats, model_paths

def plot_results(training_stats, testing_stats, num_episodes, test_episodes):
    """訓練とテスト結果をグラフ表示"""
    agent_types = list(training_stats.keys())
    colors = {
        'hybrid_q_learning': 'blue', 
        'q_learning': 'green', 
        'sarsa': 'red',
        'hybrid_sarsa': 'purple',
        'bandit': 'brown',
        'hybrid_bandit': 'orange'
    }
    
    plt.figure(figsize=(15, 10))
    
    # 勝率の推移（訓練）
    plt.subplot(2, 2, 1)
    for agent_type in agent_types:
        success_rate_history = training_stats[agent_type]["success_rate_history"]
        x = np.linspace(100, num_episodes, len(success_rate_history))
        plt.plot(x, success_rate_history, color=colors[agent_type], label=agent_type)
    plt.title('訓練時の勝率推移')
    plt.xlabel('エピソード数')
    plt.ylabel('勝率')
    plt.grid(True)
    plt.legend()
    
    # 平均ターン数の推移（訓練）
    plt.subplot(2, 2, 2)
    for agent_type in agent_types:
        avg_turns_history = training_stats[agent_type]["avg_turns_history"]
        x = np.linspace(100, num_episodes, len(avg_turns_history))
        plt.plot(x, avg_turns_history, color=colors[agent_type], label=agent_type)
    plt.title('訓練時の平均ターン数推移')
    plt.xlabel('エピソード数')
    plt.ylabel('平均ターン数')
    plt.grid(True)
    plt.legend()
    
    # テスト結果（勝率）
    plt.subplot(2, 2, 3)
    win_rates = [testing_stats[agent_type]["win_rate"] for agent_type in agent_types]
    plt.bar(agent_types, win_rates, color=[colors[agent_type] for agent_type in agent_types])
    plt.title('テスト時の勝率比較')
    plt.xlabel('エージェントタイプ')
    plt.ylabel('勝率')
    plt.ylim(0, 1.1)
    for i, v in enumerate(win_rates):
        plt.text(i, v + 0.05, f'{v:.1%}', ha='center')
    
    # テスト結果（平均ターン数）
    plt.subplot(2, 2, 4)
    avg_turns = [testing_stats[agent_type]["avg_turns"] for agent_type in agent_types]
    plt.bar(agent_types, avg_turns, color=[colors[agent_type] for agent_type in agent_types])
    plt.title('テスト時の平均ターン数比較')
    plt.xlabel('エージェントタイプ')
    plt.ylabel('平均ターン数')
    for i, v in enumerate(avg_turns):
        plt.text(i, v + 0.2, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    
    # 保存先の設定
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"rl_comparison_{num_episodes}ep_{test_episodes}test.png")
    
    plt.savefig(save_path)
    print(f"比較結果のグラフを保存しました: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="異なる強化学習手法の訓練と比較")
    parser.add_argument("--episodes", type=int, default=1000, help="訓練エピソード数 (デフォルト: 1000)")
    parser.add_argument("--digits", type=int, default=3, help="桁数 (デフォルト: 3)")
    parser.add_argument("--range", type=int, default=6, help="数字の範囲 (0〜range-1) (デフォルト: 6)")
    parser.add_argument("--turns", type=int, default=10, help="最大ターン数 (デフォルト: 10)")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="学習率 (デフォルト: 0.1)")
    parser.add_argument("--discount", type=float, default=0.95, help="割引率 (デフォルト: 0.95)")
    parser.add_argument("--exploration", type=float, default=0.3, help="探索率 (デフォルト: 0.3)")
    parser.add_argument("--test", type=int, default=100, help="テストエピソード数 (デフォルト: 100)")
    parser.add_argument("--quiet", action="store_true", help="進捗表示を簡略化")
    
    args = parser.parse_args()
    
    print(f"強化学習手法の比較開始")
    print(f"設定: {args.digits}桁, 範囲:0-{args.range-1}, 訓練:{args.episodes}エピソード, テスト:{args.test}エピソード")
    print("=" * 50)
    
    compare_rl_agents(
        num_episodes=args.episodes,
        num_digits=args.digits,
        digit_range=args.range,
        max_turns=args.turns,
        learning_rate=args.learning_rate,
        discount_factor=args.discount,
        exploration_rate=args.exploration,
        test_episodes=args.test,
        verbose=not args.quiet
    )
    
    print("\n比較が完了しました。")
    print("結果のグラフが 'hit_and_blow/results/' ディレクトリに保存されています。")

if __name__ == "__main__":
    main() 