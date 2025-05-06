import itertools
import random
import math
from collections import defaultdict
from typing import List, Tuple, Dict
from hit_and_blow.agents.base import AgentBase

class ProbabilisticAgent(AgentBase):
    """確率的アプローチを用いたエージェント
    
    候補集合から情報量が最大となる予測を選択する
    """
    def __init__(self, num_digits: int, digit_range: int):
        super().__init__(num_digits, digit_range)
        self.candidates = []
        self.max_candidates_for_entropy = 1000  # 情報エントロピー計算の対象とする最大候補数

    def reset(self):
        """初期候補集合生成"""
        digits = list(range(self.digit_range))
        self.candidates = list(itertools.permutations(digits, self.num_digits))

    def predict(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> List[int]:
        """履歴から整合性のある候補集合を絞り、情報量が最大となる予測を返す"""
        if not history:
            # 初手は固定戦略（最も情報量が多い予測を実験的に選定）
            # 例：4桁10種類の場合、[0, 1, 2, 3]が良い初手
            return list(range(min(self.num_digits, self.digit_range)))

        # 履歴ごとに整合性を保つ候補集合に絞り込む
        for past_guess, (past_hits, past_blows) in history:
            self.candidates = [
                c for c in self.candidates
                if self._evaluate_guess(list(c), past_guess) == (past_hits, past_blows)
            ]

        # 候補が見つからない場合（通常は起こりえないが安全策として）
        if not self.candidates:
            digits = list(range(self.digit_range))
            return random.sample(digits, self.num_digits)
            
        # 候補数が1つの場合は確定解
        if len(self.candidates) == 1:
            return list(self.candidates[0])
            
        # 候補が多すぎる場合は、一部をサンプリングして情報量計算
        if len(self.candidates) > self.max_candidates_for_entropy:
            sample_size = min(self.max_candidates_for_entropy, len(self.candidates))
            candidates_sample = random.sample(self.candidates, sample_size)
        else:
            candidates_sample = self.candidates

        # 情報量が最大となる予測を選択
        best_guess = self._select_best_guess(candidates_sample)
        return best_guess

    def _select_best_guess(self, candidates: List[Tuple[int, ...]]) -> List[int]:
        """情報量（エントロピー減少量）が最大となる予測を選択する
        
        Args:
            candidates: 可能な候補のリスト
            
        Returns:
            情報量が最大となる予測
        """
        # 候補数が少ない場合は単にランダム選択
        if len(candidates) <= 2:
            return list(random.choice(candidates))
            
        # 各予測の期待情報量を計算
        max_info_gain = -1
        best_guesses = []

        # 全候補ではなく現実的にあり得る予測のみ評価（最適解と近似解のトレードオフ）
        # 評価対象はcandidatesから最大100個をサンプリング
        guess_pool = candidates[:100] if len(candidates) > 100 else candidates
        
        for guess_tuple in guess_pool:
            guess = list(guess_tuple)
            info_gain = self._calculate_expected_info_gain(guess, candidates)
            
            # より良い情報量を持つ予測が見つかった場合
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_guesses = [guess]
            elif info_gain == max_info_gain:
                best_guesses.append(guess)
        
        # 最も情報量の高い予測からランダムに1つ選択
        return random.choice(best_guesses)
        
    def _calculate_expected_info_gain(self, guess: List[int], candidates: List[Tuple[int, ...]]) -> float:
        """与えられた予測の期待情報利得を計算する
        
        Args:
            guess: 評価する予測
            candidates: 現在の候補集合
            
        Returns:
            期待情報利得
        """
        # 予測gに対する各(hits, blows)の確率分布を計算
        response_counts = defaultdict(int)
        for candidate in candidates:
            response = self._evaluate_guess(list(candidate), guess)
            response_counts[response] += 1
            
        # 現在のエントロピー
        current_entropy = math.log2(len(candidates))
        
        # 予測後の期待エントロピー
        expected_entropy = 0
        total_candidates = len(candidates)
        
        for count in response_counts.values():
            # この結果が出る確率
            prob = count / total_candidates
            # この結果が出た場合の残りの候補数
            remaining = count
            # この結果が出た場合のエントロピー
            if remaining > 0:
                entropy = math.log2(remaining)
                expected_entropy += prob * entropy
                
        # 情報利得 = 現在のエントロピー - 予測後の期待エントロピー
        return current_entropy - expected_entropy

    def _evaluate_guess(self, candidate: List[int], guess: List[int]) -> Tuple[int, int]:
        """ヒット・ブロー判定"""
        hits = sum([c == g for c, g in zip(candidate, guess)])
        blows = sum([min(candidate.count(d), guess.count(d)) for d in set(guess)]) - hits
        return hits, blows 