# Hit and Blow 高度な使用方法

このドキュメントでは、Hit and Blowパッケージの高度な使用方法と拡張機能について説明します。

## カスタムエージェントの作成

独自のエージェントを実装することで、新しい戦略やアルゴリズムを試すことができます。

### 基本的な実装手順

1. `AgentBase` クラスを継承してエージェントを実装
2. `reset()` と `predict()` メソッドをオーバーライド
3. エージェントを登録して使用可能にする

```python
from hit_and_blow.agents.base import AgentBase
from hit_and_blow.agents.registry import AgentRegistry

class MyCustomAgent(AgentBase):
    def __init__(self, digits=3, number_range=10, **kwargs):
        super().__init__(digits, number_range, **kwargs)
        # エージェント固有の初期化

    def reset(self):
        # ゲームの開始時に呼び出される
        # 内部状態の初期化を行う
        pass

    def predict(self, history):
        # 過去の履歴に基づいて次の予測を行う
        # history: [(予測, hits, blows), ...]
        
        # 何らかのロジックで次の予測を決定
        prediction = [0, 1, 2]  # 例: 常に [0, 1, 2] を予測
        
        return prediction

# エージェントの登録
AgentRegistry.register("my_custom", MyCustomAgent)
```

### 登録したエージェントの使用

コマンドラインから登録したエージェントを使用できます：

```bash
poetry run play-agent --agent my_custom
```

## アドバイザーモードの実装

アドバイザーモードは、ユーザーがヒット&ブローゲームで遊ぶ際にアドバイスを提供する機能です。以下は実装例です。

```python
# examples/advisor_mode.py
import argparse
from hit_and_blow.core.advisor import HitAndBlowAdvisor

def main():
    parser = argparse.ArgumentParser(description='Hit and Blow Advisor')
    parser.add_argument('--digits', type=int, default=3, help='Number of digits')
    parser.add_argument('--range', type=int, default=10, help='Range of numbers (0 to range-1)')
    args = parser.parse_args()
    
    advisor = HitAndBlowAdvisor(digits=args.digits, number_range=args.range)
    
    print(f"=== Hit and Blow Advisor ===")
    print(f"Digits: {args.digits}, Range: 0-{args.range-1}")
    print("Input format: Enter your guess followed by the result (hits blows)")
    print("Example: 123 1 1 (means you guessed 123 and got 1 hit, 1 blow)")
    print("Type 'hint' for a suggestion, 'reset' to start over, or 'exit' to quit")
    print("===============================")
    
    while True:
        user_input = input("\nEnter your guess and result (or command): ").strip().lower()
        
        if user_input == 'exit':
            break
        elif user_input == 'reset':
            advisor.reset()
            print("Game reset. All candidates reinitialized.")
            continue
        elif user_input == 'hint':
            suggestion = advisor.suggest_next_guess()
            print(f"Suggestion: {suggestion}")
            continue
        
        # Parse user input
        try:
            parts = user_input.split()
            if len(parts) < 3:
                print("Invalid input. Please provide guess, hits, and blows.")
                continue
                
            guess_str = parts[0]
            hits = int(parts[1])
            blows = int(parts[2])
            
            # Convert guess string to list of integers
            guess = [int(digit) for digit in guess_str]
            
            if len(guess) != args.digits:
                print(f"Guess must have exactly {args.digits} digits.")
                continue
                
            # Update advisor with result
            advisor.update_with_result(guess, hits, blows)
            
            # Show status
            candidates_count = len(advisor.candidates)
            print(f"Remaining candidates: {candidates_count}")
            
            if candidates_count <= 5:
                print("Possible answers:")
                for candidate in advisor.candidates:
                    print(f"  {candidate}")
            
            if candidates_count == 1:
                print(f"The answer is: {advisor.candidates[0]}")
            elif candidates_count == 0:
                print("No valid candidates found. There might be an error in the provided results.")
                print("Resetting the game...")
                advisor.reset()
            else:
                suggestion = advisor.suggest_next_guess()
                print(f"Suggested next guess: {suggestion}")
                
        except ValueError:
            print("Invalid input format. Please enter numbers only.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
```

## パフォーマンス最適化

大きな桁数や範囲でエージェントを実行する場合、計算効率が重要になります。以下に最適化のヒントを示します。

### 候補管理の最適化

```python
# 効率的な候補フィルタリング
def efficient_filter_candidates(candidates, guess, hits, blows):
    return [c for c in candidates if calculate_hits_blows(c, guess) == (hits, blows)]

# さらに最適化: 集合演算を使用
def super_efficient_filter(candidates, guess, hits, blows):
    result_set = set()
    for c in candidates:
        h, b = calculate_hits_blows(c, guess)
        if h == hits and b == blows:
            result_set.add(tuple(c))
    return [list(c) for c in result_set]
```

### 情報利得計算の最適化

```python
# 情報利得計算の最適化
def optimized_information_gain(guess, candidates):
    # 各結果の発生回数をカウント
    result_counts = {}
    for candidate in candidates:
        result = calculate_hits_blows(candidate, guess)
        result_counts[result] = result_counts.get(result, 0) + 1
    
    # エントロピー計算
    total_candidates = len(candidates)
    information_gain = 0
    for count in result_counts.values():
        probability = count / total_candidates
        information_gain -= probability * math.log2(probability)
    
    return information_gain
```

## データ収集と分析

エージェントのパフォーマンスデータを収集し、分析するフレームワークを構築できます。

```python
# examples/performance_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
from hit_and_blow.interface.compare_agents import compare_agents

# 様々な設定でエージェントを比較
results = []
for digits in [3, 4]:
    for num_range in [6, 10]:
        for games in [100, 500]:
            result = compare_agents(
                agents=['rule_based', 'probabilistic', 'hybrid_q_learning', 'hybrid_bandit'],
                games=games,
                digits=digits,
                number_range=num_range,
                return_results=True
            )
            
            for agent_name, stats in result.items():
                results.append({
                    'agent': agent_name,
                    'digits': digits,
                    'range': num_range,
                    'games': games,
                    'win_rate': stats['win_rate'],
                    'avg_turns': stats['avg_turns']
                })

# DataFrameに変換
df = pd.DataFrame(results)

# 平均ターン数の分析
plt.figure(figsize=(10, 6))
for agent in df['agent'].unique():
    agent_df = df[df['agent'] == agent]
    plt.plot(agent_df['digits'].astype(str) + '_' + agent_df['range'].astype(str),
             agent_df['avg_turns'], marker='o', label=agent)

plt.xlabel('Configuration (digits_range)')
plt.ylabel('Average Turns')
plt.title('Agent Performance Across Configurations')
plt.legend()
plt.grid(True)
plt.savefig('performance_analysis.png')
plt.show()
```

## 学習済みモデルの管理

様々な設定で学習したモデルを管理するためのユーティリティを実装することもできます。

```python
# examples/model_manager.py
import os
import glob
import pickle
import datetime

class ModelManager:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def save_model(self, model, agent_type, digits, number_range, metadata=None):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{agent_type}_{digits}d_{number_range}r_{timestamp}.pkl"
        filepath = os.path.join(self.model_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': model,
                'agent_type': agent_type,
                'digits': digits,
                'number_range': number_range,
                'timestamp': timestamp,
                'metadata': metadata or {}
            }, f)
        
        return filepath
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def list_models(self, agent_type=None, digits=None, number_range=None):
        pattern = f"{agent_type or '*'}_{digits or '*'}d_{number_range or '*'}r_*.pkl"
        return glob.glob(os.path.join(self.model_dir, pattern))
    
    def get_best_model(self, agent_type, digits, number_range, metric='avg_turns'):
        models = self.list_models(agent_type, digits, number_range)
        if not models:
            return None
        
        best_model = None
        best_score = float('inf') if metric == 'avg_turns' else float('-inf')
        
        for model_path in models:
            model_data = self.load_model(model_path)
            if 'metadata' in model_data and metric in model_data['metadata']:
                score = model_data['metadata'][metric]
                if (metric == 'avg_turns' and score < best_score) or \
                   (metric != 'avg_turns' and score > best_score):
                    best_score = score
                    best_model = model_path
        
        return best_model

# 使用例
if __name__ == "__main__":
    manager = ModelManager()
    
    # モデルのリスト表示
    models = manager.list_models(agent_type='hybrid_q_learning')
    for model in models:
        print(model)
    
    # 最良のモデルを取得
    best_model_path = manager.get_best_model('hybrid_q_learning', 3, 6)
    if best_model_path:
        model_data = manager.load_model(best_model_path)
        print(f"Best model: {best_model_path}")
        print(f"Average turns: {model_data['metadata'].get('avg_turns')}")
        print(f"Win rate: {model_data['metadata'].get('win_rate')}")
```

これらの高度な使用方法を応用することで、Hit and Blowパッケージの機能を拡張し、より深く理解することができます。 