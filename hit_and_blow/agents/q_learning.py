from typing import List, Dict, Any, Tuple, Optional
import random
import numpy as np
import pickle
import os
from collections import defaultdict
from hit_and_blow.agents.base import AgentBase

class QLearningAgent(AgentBase):
    """
    シンプルなQ学習エージェント
    状態空間と行動空間を適切に抽象化し、Q値ベースの行動選択を行う
    """
    
    def __init__(self, num_digits: int = 4, digit_range: int = 10,
                 learning_rate: float = 0.1, discount_factor: float = 0.95,
                 exploration_rate: float = 0.2, model_path: Optional[str] = None):
        """
        初期化
        
        Args:
            num_digits: 桁数
            digit_range: 数字の範囲（0からdigit_range-1まで）
            learning_rate: 学習率（Q値の更新速度）
            discount_factor: 割引率（将来の報酬の現在価値への割引率）
            exploration_rate: 探索率（ランダムな行動を選ぶ確率）
            model_path: 保存済みのモデルパス
        """
        super().__init__(num_digits, digit_range)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Q値テーブル: {状態: {行動: Q値}}
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # 状態と行動の履歴
        self.state_history = []
        self.action_history = []
        
        # モデルの読み込み
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
    def reset(self):
        """エージェントの状態をリセット"""
        self.state_history = []
        self.action_history = []
        
    def get_state_representation(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> str:
        """
        履歴から状態表現を生成する
        状態 = これまでの予測結果のシーケンス（最新の3つまで）
        """
        if not history:
            return "initial"
            
        # 最新の3つの結果のみを使用（状態空間を小さく保つ）
        recent_history = history[-3:] if len(history) > 3 else history
        
        state_parts = []
        for entry in recent_history:
            guess, result = entry
            hits, blows = result
            # 予測とその結果を状態に含める
            guess_str = "".join(map(str, guess))
            state_parts.append(f"{guess_str}:{hits}H{blows}B")
            
        return "|".join(state_parts)
        
    def get_valid_actions(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> List[Tuple[int, ...]]:
        """
        有効な行動（予測）のリストを生成する
        過去に予測していない組み合わせをランダムにサンプリング
        """
        # 過去の予測を除外
        past_guesses = set(tuple(entry[0]) for entry in history)
        
        # 可能なすべての予測の中から過去に行っていないものを選ぶ
        all_possible = set()
        
        # サンプリングベースで生成（全組み合わせを生成すると大きすぎる場合がある）
        attempts = 0
        max_attempts = 100  # 最大試行回数
        
        while len(all_possible) < 10 and attempts < max_attempts:
            # ランダムな予測を生成
            digits = list(range(self.digit_range))
            action = tuple(random.sample(digits, self.num_digits))
            
            if action not in past_guesses:
                all_possible.add(action)
                
            attempts += 1
            
        # サンプリングが失敗した場合（非常に稀）
        if not all_possible:
            # 完全ランダムな行動を返す
            action = tuple(random.sample(range(self.digit_range), self.num_digits))
            while action in past_guesses and len(past_guesses) < self.digit_range ** self.num_digits:
                action = tuple(random.sample(range(self.digit_range), self.num_digits))
            return [action]
            
        return list(all_possible)
        
    def predict(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> List[int]:
        """
        次の予測を行う
        Q学習: 状態から最も価値の高い行動を選ぶ
        """
        # 現在の状態
        current_state = self.get_state_representation(history)
        self.state_history.append(current_state)
        
        # 有効な行動を取得
        valid_actions = self.get_valid_actions(history)
        
        # 確率的に探索と活用を切り替え
        if random.random() < self.exploration_rate:
            # 探索: ランダムな行動を選ぶ
            action = random.choice(valid_actions)
        else:
            # 活用: 最も価値の高い行動を選ぶ
            q_values = [self.q_table[current_state][a] for a in valid_actions]
            
            # 最大Q値を持つ行動（複数ある場合はランダム）
            max_q = max(q_values)
            max_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
            action = random.choice(max_actions)
        
        self.action_history.append(action)
        return list(action)
        
    def _calculate_hits_blows(self, guess: List[int], answer: List[int]) -> Tuple[int, int]:
        """
        ヒットとブローを計算
        """
        hits = sum(1 for i in range(self.num_digits) if guess[i] == answer[i])
        
        # ブローの計算（位置が違うが数字は含まれている）
        blows = 0
        for i in range(self.num_digits):
            if guess[i] != answer[i] and guess[i] in answer:
                # 同じ数字の出現回数を考慮する必要がある
                guess_count = guess.count(guess[i])
                answer_count = answer.count(guess[i])
                if guess_count <= answer_count:
                    blows += 1
                    
        return hits, blows
        
    def update_q_values(self, final_reward: float) -> None:
        """
        エピソード終了時にQ値を更新する
        
        Args:
            final_reward: 最終報酬（タスク達成までのターン数に基づく）
        """
        # 状態・行動履歴が空の場合は更新しない
        if not self.state_history or not self.action_history:
            return
            
        # 逆順に状態・行動履歴をたどりながらQ値を更新
        cum_reward = final_reward
        
        for t in range(len(self.state_history) - 1, -1, -1):
            state = self.state_history[t]
            action = self.action_history[t]
            
            # 現在のQ値
            current_q = self.q_table[state][action]
            
            # Q値の更新（TD学習）
            new_q = current_q + self.learning_rate * (cum_reward - current_q)
            self.q_table[state][action] = new_q
            
            # 次の更新のための報酬の割引
            cum_reward *= self.discount_factor
            
    def save_model(self, path: str) -> None:
        """
        モデル（Q値テーブル）を保存する
        """
        # defaultdictは直接pickleできないため、通常の辞書に変換
        q_table_dict = {k: dict(v) for k, v in self.q_table.items()}
        
        with open(path, 'wb') as f:
            pickle.dump(q_table_dict, f)
            
    def load_model(self, path: str) -> None:
        """
        保存されたモデルを読み込む
        """
        try:
            with open(path, 'rb') as f:
                q_table_dict = pickle.load(f)
                
            # 辞書をdefaultdictに変換
            self.q_table = defaultdict(lambda: defaultdict(float))
            for state, actions in q_table_dict.items():
                for action, q_value in actions.items():
                    self.q_table[state][action] = q_value
                    
            print(f"モデルを読み込みました: {path}")
        except Exception as e:
            print(f"モデル読み込み失敗: {e}")
            
    def calculate_reward(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> float:
        """
        報酬を計算する
        
        Args:
            history: 予測履歴
            
        Returns:
            報酬値（正解するほど高い）
        """
        if not history:
            return 0.0
            
        last_entry = history[-1]
        guess, result = last_entry
        hits, blows = result
        
        # ヒットとブローに基づく報酬
        hit_reward = hits / self.num_digits
        blow_reward = blows / (2 * self.num_digits) * 0.5
        
        # 正解した場合は大きな報酬
        if hits == self.num_digits:
            solved_reward = 1.0
        else:
            solved_reward = 0.0
            
        # ターン数に基づくペナルティ（長引くほど報酬減少）
        turn_penalty = -0.1 * len(history)
        
        return hit_reward + blow_reward + solved_reward + turn_penalty 