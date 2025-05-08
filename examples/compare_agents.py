#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hit_and_blow.agents.registry import AgentRegistry
from hit_and_blow.core.environment import Environment

def convert_history_format(history, agent_name):
    """
    環境の履歴形式をエージェントが期待する形式に変換
    
    環境: [(action, hits, blows), ...]
    
    エージェントタイプに応じて異なる形式に変換:
    - hybrid_q_learning, q_learning等: [(action, hits, blows), ...] (オリジナルのまま)
    - その他: [(guess, (hits, blows)), ...]
    """
    # hybrid_q_learning等は元の形式をそのまま期待する
    if agent_name in ["hybrid_q_learning", "q_learning", "sarsa", "hybrid_sarsa", "bandit", "hybrid_bandit"]:
        return history
    
    # その他のエージェント用に変換
    return [(guess, (hits, blows)) for guess, hits, blows in history]

def compare_agents(agents=None, games=100, digits=3, number_range=6, max_turns=10, verbose=False):
    """
    複数のエージェントを比較する関数
    
    Args:
        agents (list): 比較するエージェント名のリスト
        games (int): 各エージェントのゲーム回数
        digits (int): 桁数
        number_range (int): 数字の範囲
        max_turns (int): 最大ターン数
        verbose (bool): 詳細な情報を表示するか
        
    Returns:
        dict: エージェント名をキー、結果を値とする辞書
    """
    if agents is None:
        # デフォルトでは全エージェントを比較
        agents = AgentRegistry.get_all_names()
    
    # 結果を保存する辞書
    results = {}
    
    for agent_name in agents:
        if verbose:
            print(f"\n===== エージェント: {agent_name} =====")
        
        # エージェントを初期化
        try:
            agent_class = AgentRegistry.get(agent_name)
            
            # 強化学習エージェント用のモデルパスを設定
            model_path = None
            if agent_name in ["hybrid_q_learning", "q_learning", "sarsa", "hybrid_sarsa", "bandit", "hybrid_bandit"]:
                # デフォルトのモデルパス
                model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
                # train_rlと同じ命名規則を使用（エージェント名、桁数、範囲を含む）
                model_path = os.path.join(model_dir, f"{agent_name}_model_{digits}d_{number_range}r.pkl")
                if not os.path.exists(model_path):
                    print(f"警告: モデルファイル {model_path} が見つかりません。新しいモデルで開始します。")
                
                agent = agent_class(
                    num_digits=digits, 
                    digit_range=number_range,
                    model_path=model_path,
                    exploration_rate=0.05  # テスト時は探索率を下げる
                )
            else:
                agent = agent_class(num_digits=digits, digit_range=number_range)
        except Exception as e:
            print(f"エージェント {agent_name} の初期化エラー: {e}")
            continue
        
        # 統計情報を収集
        wins = 0
        turns_history = []
        total_time = 0
        
        # 指定した回数ゲームを実行
        for game in range(games):
            if verbose and (game + 1) % 10 == 0:
                print(f"ゲーム {game+1}/{games}...")
            
            # 環境を初期化
            env = Environment(digits=digits, number_range=number_range, max_turns=max_turns)
            state = env.reset()
            agent.reset()
            
            done = False
            turn = 0
            start_time = time.time()
            
            # ゲームループ
            while not done and turn < max_turns:
                # 環境の履歴をエージェントが期待する形式に変換
                agent_history = convert_history_format(env.history, agent_name)
                
                # エージェントに予測させる
                prediction = agent.predict(agent_history)
                
                # 環境で予測を評価
                state, reward, done, info = env.step(prediction)
                
                turn += 1
                
                # ゲーム終了条件をチェック
                if done and info['hits'] == digits:
                    wins += 1
                    turns_history.append(turn)
            
            # 勝てなかった場合も最大ターン数を記録
            if not done or info['hits'] != digits:
                turns_history.append(max_turns)
            
            total_time += time.time() - start_time
        
        # 統計情報を計算
        win_rate = (wins / games) * 100
        avg_turns = sum(turns_history) / len(turns_history) if turns_history else 0
        avg_time = total_time / games
        
        # 結果を保存
        results[agent_name] = {
            'wins': wins,
            'games': games,
            'win_rate': win_rate,
            'avg_turns': avg_turns,
            'avg_time': avg_time,
            'turns_history': turns_history
        }
        
        if verbose:
            print(f"勝率: {win_rate:.1f}% ({wins}/{games})")
            print(f"平均ターン数: {avg_turns:.2f}")
            print(f"平均実行時間: {avg_time:.4f}秒")
    
    return results

def visualize_results(results):
    """
    結果を可視化する関数
    
    Args:
        results (dict): エージェント名をキー、結果を値とする辞書
    """
    # 結果が空の場合は何もしない
    if not results:
        print("結果が空のためグラフは表示されません。")
        return
        
    # 日本語フォントの設定（必要に応じて）
    try:
        import japanize_matplotlib
    except ImportError:
        pass
    
    # 結果をDataFrameに変換
    data = []
    for agent_name, stats in results.items():
        data.append({
            'Agent': agent_name,  # 英語に変更
            'Win Rate (%)': stats['win_rate'],
            'Avg Turns': stats['avg_turns'],
            'Exec Time (s)': stats['avg_time']
        })
    
    df = pd.DataFrame(data)
    
    # グラフの作成
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 勝率のグラフ
    df.plot(x='Agent', y='Win Rate (%)', kind='bar', ax=axes[0], color='green')
    axes[0].set_title('Win Rate Comparison')
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis='y')
    
    # 平均ターン数のグラフ
    df.plot(x='Agent', y='Avg Turns', kind='bar', ax=axes[1], color='blue')
    axes[1].set_title('Average Turns Comparison')
    axes[1].grid(axis='y')
    
    # 実行時間のグラフ
    df.plot(x='Agent', y='Exec Time (s)', kind='bar', ax=axes[2], color='red')
    axes[2].set_title('Average Execution Time Comparison')
    axes[2].grid(axis='y')
    
    plt.tight_layout()
    plt.savefig('agent_comparison.png')
    print(f"比較グラフを 'agent_comparison.png' に保存しました")
    
    # ターン数分布のグラフ (改善版)
    plt.figure(figsize=(14, 8))
    
    # エージェントごとに異なる色を使用
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']
    
    # 最大ターン数を取得
    max_turns = max([max(stats['turns_history']) for stats in results.values()])
    
    # エージェントの数
    num_agents = len(results)
    
    # 各エージェントの位置を計算
    for i, (agent_name, stats) in enumerate(results.items()):
        # ターン数の頻度をカウント (1から最大ターン数まで)
        turns_count = {}
        for turn in range(1, max_turns + 1):
            turns_count[turn] = 0
        
        for turn in stats['turns_history']:
            if turn in turns_count:
                turns_count[turn] += 1
            
        # パーセンテージに変換
        total_games = stats['games']
        turns_percentage = {turn: (count / total_games) * 100 for turn, count in turns_count.items()}
        
        # X軸の位置をずらす（バーが重ならないように）
        bar_width = 0.8 / num_agents
        offsets = [(j - (num_agents-1)/2) * bar_width for j in range(num_agents)]
        x_positions = [turn + offsets[i] for turn in range(1, max_turns + 1)]
        
        # プロット
        plt.bar(
            x_positions,
            [turns_percentage.get(turn, 0) for turn in range(1, max_turns + 1)],
            width=bar_width,
            label=agent_name,
            color=colors[i % len(colors)],
            edgecolor='black',
            linewidth=1,
            alpha=0.7
        )
    
    plt.title('Turn Distribution')
    plt.xlabel('Turns')
    plt.ylabel('Percentage of Games (%)')
    plt.xticks(range(1, max_turns + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    # グラフの保存
    plt.tight_layout()
    plt.savefig('turns_distribution.png')
    print(f"ターン数分布グラフを 'turns_distribution.png' に保存しました")
    
    # グラフ表示
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Hit and Blow Agent Comparison')
    parser.add_argument('--agents', nargs='+', help='比較するエージェント名（複数指定可）')
    parser.add_argument('--games', type=int, default=100, help='各エージェントのゲーム回数')
    parser.add_argument('--digits', type=int, default=3, help='桁数')
    parser.add_argument('--range', type=int, default=6, help='数字の範囲（0〜range-1）')
    parser.add_argument('--turns', type=int, default=10, help='最大ターン数')
    parser.add_argument('--verbose', action='store_true', help='詳細情報を表示')
    parser.add_argument('--no-plot', action='store_true', help='グラフを表示しない')
    args = parser.parse_args()
    
    # 利用可能なエージェントを表示
    print("利用可能なエージェント:")
    for agent in AgentRegistry.get_all_names():
        print(f"- {agent}")
    
    # エージェントを比較
    print(f"\n{args.games}回のゲームでエージェントを比較します...")
    results = compare_agents(
        agents=args.agents,
        games=args.games,
        digits=args.digits,
        number_range=args.range,
        max_turns=args.turns,
        verbose=args.verbose
    )
    
    # 結果の表示
    print("\n===== 比較結果 =====")
    print(f"設定: {args.digits}桁, 範囲0-{args.range-1}, 最大{args.turns}ターン, {args.games}ゲーム")
    
    if not results:
        print("有効な結果がありません。エージェントの初期化に失敗した可能性があります。")
        return
    
    # 結果をソート（勝率、平均ターン数の順）
    sorted_results = sorted(
        results.items(),
        key=lambda x: (-x[1]['win_rate'], x[1]['avg_turns'])
    )
    
    for agent_name, stats in sorted_results:
        print(f"エージェント: {agent_name}")
        print(f"  勝率: {stats['win_rate']:.1f}% ({stats['wins']}/{stats['games']})")
        print(f"  平均ターン数: {stats['avg_turns']:.2f}")
        print(f"  平均実行時間: {stats['avg_time']:.4f}秒")
    
    # グラフの表示（指定された場合）
    if not args.no_plot:
        visualize_results(results)

if __name__ == "__main__":
    main() 