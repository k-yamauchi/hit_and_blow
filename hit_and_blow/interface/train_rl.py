#!/usr/bin/env python3
"""
強化学習エージェントを訓練するためのスクリプト
"""
import os
import argparse
import time
from hit_and_blow.core.environment import Environment
from hit_and_blow.agents import AgentRegistry

def train_agent(agent_name="hybrid_q_learning", num_episodes=1000, num_digits=3, digit_range=10, max_turns=10, 
               learning_rate=0.1, discount_factor=0.95, exploration_rate=0.3, model_path=None, verbose=True):
    """強化学習エージェントを訓練する"""
    
    # モデルパスの設定
    if model_path is None:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{agent_name}_model_{num_digits}d_{digit_range}r.pkl")
    
    print(f"エージェント: {agent_name}")
    print(f"モデルパス: {model_path}")
    
    # 統計情報の初期化
    stats = {
        "total_success": 0,
        "total_turns": 0,
        "success_rate_history": [],
        "avg_turns_history": []
    }
    
    # 環境とエージェントの初期化
    agent_class = AgentRegistry.get(agent_name)
    agent = agent_class(
        num_digits=num_digits, 
        digit_range=digit_range,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        exploration_rate=exploration_rate,
        model_path=model_path
    )
    
    start_time = time.time()
    
    # エピソード数分だけ訓練
    for episode in range(num_episodes):
        env = Environment(digits=num_digits, number_range=digit_range, max_turns=max_turns)
        obs = env.reset()
        agent.reset()
        
        done = False
        while not done:
            guess = agent.predict(obs["history"])
            obs, reward, done, info = env.step(guess)
        
        # ゲーム終了時の処理
        success = info["hits"] == num_digits
        
        # 統計情報の更新
        stats["total_turns"] += env.turns
        if success:
            stats["total_success"] += 1
        
        # 報酬の計算
        if success:
            # 成功の場合、速く解くほど報酬が高い
            final_reward = 1.0 + (max_turns - env.turns) / max_turns
        else:
            # 失敗の場合は負の報酬
            final_reward = -0.5
        
        # Q値の更新 (エージェントによって処理が異なる)
        if agent_name in ["sarsa", "hybrid_sarsa", "bandit"]:
            agent.update_final_reward(final_reward)
        else:
            agent.update_q_values(final_reward)
        
        # 100エピソードごとに統計情報を更新
        if (episode + 1) % 100 == 0:
            success_rate = stats["total_success"] / (episode + 1)
            avg_turns = stats["total_turns"] / (episode + 1)
            stats["success_rate_history"].append(success_rate)
            stats["avg_turns_history"].append(avg_turns)
            
            if verbose:
                print(f"エピソード {episode+1}/{num_episodes}: 勝率={success_rate:.1%}, 平均ターン数={avg_turns:.2f}")
        
        # 定期的にモデルを保存
        if (episode + 1) % 100 == 0 or episode == num_episodes - 1:
            agent.save_model(model_path)
            if verbose:
                print(f"モデルを保存しました: {model_path}")
    
    # 訓練結果の表示
    elapsed_time = time.time() - start_time
    final_success_rate = stats["total_success"] / num_episodes
    final_avg_turns = stats["total_turns"] / num_episodes
    
    print(f"\n===== 訓練結果 =====")
    print(f"エージェント: {agent_name}")
    print(f"エピソード数: {num_episodes}")
    print(f"勝率: {final_success_rate:.1%}")
    print(f"平均ターン数: {final_avg_turns:.2f}")
    print(f"訓練時間: {elapsed_time:.1f}秒")
    print(f"モデルパス: {model_path}")
    
    return model_path, stats

def main():
    parser = argparse.ArgumentParser(description="強化学習エージェントの訓練")
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=["hybrid_q_learning", "q_learning", "sarsa", "hybrid_sarsa", "hybrid_bandit"],
        help="エージェントタイプ",
    )
    parser.add_argument("--episodes", type=int, default=1000, help="訓練エピソード数 (デフォルト: 1000)")
    parser.add_argument("--digits", type=int, default=3, help="桁数 (デフォルト: 3)")
    parser.add_argument("--range", type=int, default=10, help="数字の範囲 (0〜range-1) (デフォルト: 10)")
    parser.add_argument("--turns", type=int, default=10, help="最大ターン数 (デフォルト: 10)")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="学習率 (デフォルト: 0.1)")
    parser.add_argument("--discount", type=float, default=0.95, help="割引率 (デフォルト: 0.95)")
    parser.add_argument("--exploration", type=float, default=0.3, help="探索率 (デフォルト: 0.3)")
    parser.add_argument("--model", type=str, default=None, help="モデルファイルのパス")
    parser.add_argument("--quiet", action="store_true", help="進捗表示を簡略化")
    
    args = parser.parse_args()
    
    print(f"強化学習エージェントの訓練開始")
    print(f"設定: {args.digits}桁, 範囲:0-{args.range-1}, {args.episodes}エピソード")
    print("===========================================")
    
    model_path, _ = train_agent(
        agent_name=args.agent,
        num_episodes=args.episodes,
        num_digits=args.digits,
        digit_range=args.range,
        max_turns=args.turns,
        learning_rate=args.learning_rate,
        discount_factor=args.discount,
        exploration_rate=args.exploration,
        model_path=args.model,
        verbose=not args.quiet
    )
    
    print("\n訓練したモデルを使ってみるには:")
    print(f"poetry run play-agent --agent {args.agent} --model {model_path}")

if __name__ == "__main__":
    main() 