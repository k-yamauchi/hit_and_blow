from typing import List, Dict, Any, Tuple, Optional
import random
from hit_and_blow.agents.base import AgentBase

class HeuristicAgent(AgentBase):
    """
    ヒューリスティックエージェント
    固定のパターンと単純なルールに基づいて予測を行う
    """
    
    def __init__(self, num_digits: int = 4, digit_range: int = 10):
        super().__init__(num_digits, digit_range)
        self.fixed_positions = {}  # ヒットした位置を記録
        self.known_digits = set()  # 正解に含まれる数字（Hitまたはブロー）
        self.not_used_digits = set()  # 正解に含まれない数字
        self.attempts = 0
        
    def reset(self):
        """エージェントの状態をリセット"""
        self.fixed_positions = {}
        self.known_digits = set()
        self.not_used_digits = set()
        self.attempts = 0
        
    def predict(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> List[int]:
        """
        次の予測を行う
        ヒューリスティック：
        1. 初回は連続した数字（0,1,2,3など）
        2. ヒットした数字は固定
        3. ブローの数字は位置を変えて試す
        4. 残りは未使用の数字からランダム選択
        """
        self.attempts += 1
        
        # 履歴から学習
        self._learn_from_history(history)
        
        # 初回は連続した数字
        if self.attempts == 1:
            start = random.randint(0, self.digit_range - self.num_digits)
            return [i % self.digit_range for i in range(start, start + self.num_digits)]
        
        # 予測を構築
        prediction = [None] * self.num_digits
        
        # 1. まずヒットした位置を固定
        for pos, digit in self.fixed_positions.items():
            prediction[pos] = digit
        
        # 2. ブローの数字を配置（まだ固定されていない位置に）
        blow_digits = [d for d in self.known_digits if d not in self.fixed_positions.values()]
        free_positions = [i for i in range(self.num_digits) if prediction[i] is None]
        
        for digit in blow_digits:
            if free_positions:
                pos = random.choice(free_positions)
                prediction[pos] = digit
                free_positions.remove(pos)
        
        # 3. 残りの位置はまだ試していない数字を使用
        available_digits = [d for d in range(self.digit_range) 
                           if d not in self.known_digits 
                           and d not in self.not_used_digits]
        
        # 利用可能な数字がなければ、全数字から選択
        if not available_digits:
            available_digits = list(range(self.digit_range))
        
        for i in range(self.num_digits):
            if prediction[i] is None:
                if available_digits:
                    digit = random.choice(available_digits)
                    prediction[i] = digit
                    available_digits.remove(digit)
                else:
                    # もう選択肢がなければランダム
                    prediction[i] = random.randint(0, self.digit_range - 1)
        
        return prediction
    
    def _learn_from_history(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> None:
        """履歴から情報を学習する"""
        if not history:
            return
            
        for entry in history:
            guess, result = entry  # (guess, (hits, blows))
            hits, blows = result
            
            # ヒットもブローもなければ、それらの数字は使われていない
            if hits == 0 and blows == 0:
                for digit in guess:
                    self.not_used_digits.add(digit)
            else:
                # ヒットがあれば位置を固定
                for i, digit in enumerate(guess):
                    # ヒットした箇所を特定（最後の予測のみ）
                    if entry == history[-1]:
                        last_fixed = len(self.fixed_positions)
                        # 前回と比較して新しいヒットがあれば記録
                        if len(history) >= 2:
                            prev_guess, prev_result = history[-2]
                            prev_hits, prev_blows = prev_result
                            if hits > prev_hits and i not in self.fixed_positions:
                                # ヒットの増加を検出したら、新しい位置を固定
                                self.fixed_positions[i] = digit
                        # 初回のヒットなら記録
                        elif hits > 0 and len(self.fixed_positions) < hits:
                            self.fixed_positions[i] = digit
                    
                    # 既知の数字として記録
                    if digit not in self.not_used_digits:
                        self.known_digits.add(digit) 