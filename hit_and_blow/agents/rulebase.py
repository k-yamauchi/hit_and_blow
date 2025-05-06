from typing import List, Dict, Any, Set, Tuple
import itertools
import random
from hit_and_blow.agents.base import AgentBase

class RuleBaseAgent(AgentBase):
    """
    ルールベースエージェント
    ヒットアンドブロー情報を基にした単純な論理的推論を用いる
    """
    
    def __init__(self, num_digits: int = 4, digit_range: int = 10):
        super().__init__(num_digits, digit_range)
        self.candidates = []  # 可能性のある答えの候補
        
    def reset(self):
        """エージェントの状態をリセット"""
        self.candidates = []
        
    def predict(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> List[int]:
        """
        次の予測を行う
        ルールベース: 履歴から候補を絞り込み、残った候補からランダムに選択
        """
        # 初めての予測なら全候補を生成
        if not self.candidates and not history:
            self._initialize_candidates()
            
        # 履歴に基づいて候補を絞り込む
        if history:
            if not self.candidates:  # 途中から始まった場合
                self._initialize_candidates()
                
            for entry in history:
                guess, result = entry  # (guess, (hits, blows))
                hits, blows = result
                
                # この予測と同じヒット・ブロー数を返す候補だけを残す
                self.candidates = [
                    candidate for candidate in self.candidates
                    if self._check_hit_blow(guess, candidate) == (hits, blows)
                ]
                
        # 候補からランダムに選択
        if not self.candidates:
            # 候補がない場合（論理的に不可能な状況）はランダムに生成
            return [random.randint(0, self.digit_range - 1) for _ in range(self.num_digits)]
            
        return random.choice(self.candidates)
        
    def _initialize_candidates(self):
        """可能性のある全ての答えを生成する"""
        digits = list(range(self.digit_range))
        self.candidates = list(itertools.permutations(digits, self.num_digits))
        self.candidates = [list(c) for c in self.candidates]
        
    def _check_hit_blow(self, guess: List[int], answer: List[int]) -> Tuple[int, int]:
        """予測と答えのヒット・ブロー数を計算"""
        hits = sum(1 for i in range(self.num_digits) if guess[i] == answer[i])
        
        # ブロー数の計算
        guess_set = set(guess)
        answer_set = set(answer)
        common = guess_set.intersection(answer_set)
        
        # 共通の数字の数からヒット数を引くとブロー数（簡易計算で重複考慮なし）
        total_matches = sum(1 for g in guess for a in answer if g == a)
        blows = total_matches - hits
        
        return hits, blows 