from typing import List, Dict, Any, Tuple, Set
import random
import time
from hit_and_blow.agents.base import AgentBase

class MonteCarloAgent(AgentBase):
    """
    モンテカルロ法によるエージェント
    ランダムサンプリングと評価を繰り返して最適な予測を見つける
    """
    
    def __init__(self, num_digits: int = 4, digit_range: int = 10):
        super().__init__(num_digits, digit_range)
        self.possible_answers = set()  # 可能性のある答えの集合
        self.generated = False
        self.max_samples = 100  # モンテカルロサンプリングの最大数
        
    def reset(self):
        """エージェントの状態をリセット"""
        self.possible_answers = set()
        self.generated = False
        
    def predict(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> List[int]:
        """
        モンテカルロ法を使用して次の予測を行う
        """
        # 最初に可能な答えの集合を更新
        self._update_possible_answers(history)
        
        # 可能な答えがひとつしかなければ、それを返す
        if len(self.possible_answers) == 1:
            return list(next(iter(self.possible_answers)))
            
        # 可能な答えが少ない場合、残りの候補から直接選択
        if len(self.possible_answers) <= 10:
            return list(random.choice(list(self.possible_answers)))
        
        # モンテカルロサンプリングで次の予測を評価
        best_guess = None
        best_score = float('-inf')
        
        # 候補を評価するためのサンプル数を決定
        sample_size = min(self.max_samples, len(self.possible_answers))
        samples = random.sample(list(self.possible_answers), sample_size)
        
        # 各候補について評価
        for candidate in samples:
            score = self._evaluate_guess(list(candidate), history)
            if score > best_score:
                best_score = score
                best_guess = candidate
                
        return list(best_guess) if best_guess else self._random_guess()
        
    def _evaluate_guess(self, guess: List[int], history: List[Tuple[List[int], Tuple[int, int]]]) -> float:
        """
        候補の予測を評価するスコアを計算
        高いスコアほど良い予測
        """
        # 情報利得の近似値として、この予測が平均的に何個の候補を排除できるかを計算
        remaining_after_guess = 0
        
        # サンプリングした可能な答えに対して、この予測がどれだけ候補を絞れるかをシミュレート
        sample_size = min(50, len(self.possible_answers))
        secret_samples = random.sample(list(self.possible_answers), sample_size)
        
        for secret in secret_samples:
            # この予測と仮の答えの間のヒット・ブローを計算
            hits, blows = self._calculate_hits_blows(guess, list(secret))
            
            # この結果に基づいて残る候補数を計算
            remaining = sum(1 for ans in self.possible_answers 
                          if self._calculate_hits_blows(list(ans), guess) == (hits, blows))
            
            remaining_after_guess += remaining
            
        # 平均の残り候補数（少ないほど良い）
        avg_remaining = remaining_after_guess / sample_size
        
        # スコアは残り候補数の逆数（大きいほど良い）
        return -avg_remaining
        
    def _calculate_hits_blows(self, guess: List[int], secret: List[int]) -> Tuple[int, int]:
        """2つの数列間のヒットとブローを計算"""
        hits = sum(1 for g, s in zip(guess, secret) if g == s)
        
        # ブローは、共通の数字の数からヒットを引いたもの
        guess_counts = {}
        secret_counts = {}
        
        for g in guess:
            guess_counts[g] = guess_counts.get(g, 0) + 1
            
        for s in secret:
            secret_counts[s] = secret_counts.get(s, 0) + 1
            
        common = sum(min(guess_counts.get(d, 0), secret_counts.get(d, 0)) for d in range(self.digit_range))
        blows = common - hits
        
        return hits, blows
        
    def _update_possible_answers(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> None:
        """履歴に基づいて可能な答えの集合を更新"""
        # 初回の場合、可能な全ての数列を生成
        if not self.generated:
            self._generate_all_possibilities()
            self.generated = True
            
        # 各履歴エントリに基づいて候補を絞り込む
        for entry in history:
            guess, result = entry
            hits, blows = result
            
            # この予測と同じヒット・ブローを生成する候補だけを残す
            self.possible_answers = {ans for ans in self.possible_answers 
                                   if self._calculate_hits_blows(list(ans), guess) == (hits, blows)}
    
    def _generate_all_possibilities(self) -> None:
        """可能な全ての数列の組み合わせを生成"""
        # 重複を許さない場合の全組み合わせを生成
        def backtrack(current: List[int]) -> None:
            if len(current) == self.num_digits:
                self.possible_answers.add(tuple(current))
                return
                
            for digit in range(self.digit_range):
                if digit not in current:  # 重複を許さない
                    current.append(digit)
                    backtrack(current)
                    current.pop()
        
        backtrack([])
        
    def _random_guess(self) -> List[int]:
        """ランダムな予測を生成"""
        if self.possible_answers:
            return list(random.choice(list(self.possible_answers)))
        else:
            # バックアップとして完全ランダムな予測
            return random.sample(range(self.digit_range), self.num_digits) 