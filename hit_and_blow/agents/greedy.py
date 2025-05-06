from typing import List, Dict, Any, Tuple, Set
import random
from collections import defaultdict
from hit_and_blow.agents.base import AgentBase

class GreedyAgent(AgentBase):
    """
    貪欲法によるエージェント
    各ステップで最も候補を絞り込める予測を選択する
    """
    
    def __init__(self, num_digits: int = 4, digit_range: int = 10):
        super().__init__(num_digits, digit_range)
        self.candidates = set()  # 可能性のある答えの集合
        self.generated = False
        
    def reset(self):
        """エージェントの状態をリセット"""
        self.candidates = set()
        self.generated = False
        
    def predict(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> List[int]:
        """
        貪欲法で次の予測を行う
        情報利得が最大になる予測を選択する
        """
        # 候補の集合を更新
        self._update_candidates(history)
        
        # 候補が1つしかなければそれを返す
        if len(self.candidates) == 1:
            return list(next(iter(self.candidates)))
            
        # 候補からランダムに抽出（候補が多すぎる場合）
        if len(self.candidates) > 300:
            return list(random.choice(list(self.candidates)))
        
        # 全ての可能な予測とその情報利得を評価
        best_guess = None
        min_expected_remaining = float('inf')
        
        # 評価する予測候補
        guesses_to_evaluate = list(self.candidates)
        
        # 評価対象が多すぎる場合はサンプリング
        if len(guesses_to_evaluate) > 50:
            guesses_to_evaluate = random.sample(guesses_to_evaluate, 50)
        
        for guess in guesses_to_evaluate:
            # この予測が各ヒット・ブローの組み合わせを返す確率を計算
            partition = self._partition_by_feedback(list(guess))
            
            # 期待される残り候補数を計算
            expected_remaining = sum(len(group) * len(group) for group in partition.values()) / len(self.candidates)
            
            # 最小の期待残り候補数を持つ予測を選択
            if expected_remaining < min_expected_remaining:
                min_expected_remaining = expected_remaining
                best_guess = guess
        
        return list(best_guess) if best_guess else self._random_guess()
        
    def _update_candidates(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> None:
        """
        履歴に基づいて候補を更新
        """
        # 初回は全ての可能な組み合わせを生成
        if not self.generated:
            self._generate_all_possibilities()
            self.generated = True
        
        # 履歴に基づいて候補を絞り込む
        for entry in history:
            guess, result = entry
            hits, blows = result
            
            # この予測と同じヒット・ブローを生成する候補だけを残す
            self.candidates = {
                candidate for candidate in self.candidates
                if self._calculate_hits_blows(list(candidate), guess) == (hits, blows)
            }
    
    def _generate_all_possibilities(self) -> None:
        """可能な全ての数列の組み合わせを生成"""
        def generate(current: List[int]) -> None:
            if len(current) == self.num_digits:
                self.candidates.add(tuple(current))
                return
                
            for digit in range(self.digit_range):
                if digit not in current:  # 重複を許さない
                    current.append(digit)
                    generate(current)
                    current.pop()
        
        generate([])
        
    def _partition_by_feedback(self, guess: List[int]) -> Dict[Tuple[int, int], Set[Tuple[int, ...]]]:
        """
        この予測に対するすべての可能な答えを、ヒット・ブローの組み合わせごとにグループ化
        """
        partition = defaultdict(set)
        
        for candidate in self.candidates:
            feedback = self._calculate_hits_blows(list(candidate), guess)
            partition[feedback].add(candidate)
            
        return partition
        
    def _calculate_hits_blows(self, candidate: List[int], guess: List[int]) -> Tuple[int, int]:
        """2つの数列間のヒットとブローを計算"""
        hits = sum(1 for a, b in zip(candidate, guess) if a == b)
        
        # ブローは、共通の数字の数からヒットを引いたもの
        candidate_counts = {}
        guess_counts = {}
        
        for a in candidate:
            candidate_counts[a] = candidate_counts.get(a, 0) + 1
            
        for b in guess:
            guess_counts[b] = guess_counts.get(b, 0) + 1
            
        common = sum(min(candidate_counts.get(d, 0), guess_counts.get(d, 0)) for d in range(self.digit_range))
        blows = common - hits
        
        return hits, blows
        
    def _random_guess(self) -> List[int]:
        """ランダムな予測を生成"""
        if self.candidates:
            return list(random.choice(list(self.candidates)))
        else:
            # バックアップとして完全ランダムな予測
            return random.sample(range(self.digit_range), self.num_digits) 