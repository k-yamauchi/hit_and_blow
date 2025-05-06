import argparse
import os
from hit_and_blow.core.environment import Environment
from hit_and_blow.agents import AgentRegistry

def main():
    """エージェントのデモ実行"""
    parser = argparse.ArgumentParser(description="Hit and Blow エージェントデモ")
    parser.add_argument("--digits", type=int, default=3, help="桁数 (デフォルト: 3)")
    parser.add_argument("--range", type=int, default=6, help="数字の範囲 (0〜range-1) (デフォルト: 6)")
    parser.add_argument("--turns", type=int, default=10, help="最大ターン数 (デフォルト: 10)")
    
    # 利用可能なエージェント一覧を取得
    available_agents = AgentRegistry.get_all_names()
    
    parser.add_argument(
        "--agent", 
        type=str, 
        default="rule_based", 
        choices=available_agents,
        help=f"使用するエージェント (デフォルト: rule_based, 選択肢: {', '.join(available_agents)})"
    )
    
    # 強化学習用のオプション
    parser.add_argument(
        "--learn", 
        action="store_true",
        help="強化学習エージェントに学習させる（reinforcementエージェントのみ有効）"
    )
    
    parser.add_argument(
        "--model", 
        type=str,
        default=None, 
        help="強化学習エージェントのモデルパス（学習結果の保存先）"
    )
    
    args = parser.parse_args()

    # 注意: 桁数が多いと候補が爆発的に増えるので、デフォルトは3桁・6種類に設定
    env = Environment(digits=args.digits, number_range=args.range, max_turns=args.turns)
    obs = env.reset()
    
    print("======= Hit and Blow エージェントデモ =======")
    print(f"桁数: {args.digits}, 範囲: 0〜{args.range-1}, 最大ターン数: {args.turns}")
    print(f"エージェント: {args.agent}")
    print(f"秘密の数字 (デバッグ用): {env.secret}")
    print("================================")

    # モデルパスの設定（強化学習の場合）
    model_path = None
    if args.agent in ["hybrid_q_learning", "q_learning", "sarsa", "hybrid_sarsa", "bandit", "hybrid_bandit"]:
        if args.model is None:
            # デフォルトのモデルパス
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"q_learning_model.pkl")
        else:
            model_path = args.model
            
        print(f"モデルパス: {model_path}")
        if args.learn:
            print("学習モード: 有効")

    # 選択されたエージェントのクラスを取得し、インスタンス化
    agent_class = AgentRegistry.get(args.agent)
    
    # 強化学習エージェントの場合、モデルパスを渡す
    if args.agent in ["hybrid_q_learning", "q_learning", "sarsa", "hybrid_sarsa", "bandit", "hybrid_bandit"]:
        agent = agent_class(num_digits=args.digits, digit_range=args.range, model_path=model_path)
    else:
        agent = agent_class(num_digits=args.digits, digit_range=args.range)
        
    agent.reset()

    done = False
    while not done:
        guess = agent.predict(obs["history"])
        print(f"\nエージェントの予測: {guess}")
        
        obs, reward, done, info = env.step(guess)
        print(f"結果: {info['hits']} Hit, {info['blows']} Blow")
        
        # エージェント固有の情報表示（RuleBasedAgentの場合は候補数を表示）
        if hasattr(agent, "candidates"):
            print(f"残りの候補数: {len(agent.candidates)}")
        
        if done:
            # ゲーム終了時の処理
            if info['hits'] == env.digits:
                print("\n正解しました！")
                print(f"ターン {env.turns}/{env.max_turns} で正解")
                success = True
            else:
                print("\nゲームオーバー...")
                print(f"正解は: {env.secret}")
                success = False
                
            # 強化学習エージェントの場合、ゲーム終了時にQ値を更新して保存
            if args.agent in ["hybrid_q_learning", "q_learning", "sarsa", "hybrid_sarsa", "bandit", "hybrid_bandit"] and args.learn:
                # 報酬の計算: 成功した場合は高い報酬、ターン数の逆数でボーナス
                if success:
                    # 素早く解くほど報酬が高い
                    final_reward = 1.0 + (env.max_turns - env.turns) / env.max_turns
                else:
                    # 失敗した場合は低い報酬
                    final_reward = -0.5
                
                # Q値の更新
                agent.update_q_values(final_reward)
                
                # モデルの保存
                agent.save_model(model_path)
                print(f"\nモデルを保存しました: {model_path}")

    # 履歴表示
    print("\n----- プレイ履歴 -----")
    for i, entry in enumerate(env.history):
        guess, hits, blows = entry
        print(f"ターン {i+1}: {guess} -> {hits}H {blows}B")

if __name__ == "__main__":
    main() 