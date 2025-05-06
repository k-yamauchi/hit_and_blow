from typing import List, Dict, Any, Tuple, Set
import random
import re
from hit_and_blow.agents.base import AgentBase

class PatternMatchingAgent(AgentBase):
    """
    パターンマッチングエージェント
    過去の結果からパターンを学習して予測する
    """
    
    def __init__(self, num_digits: int = 4, digit_range: int = 10):
        super().__init__(num_digits, digit_range)
        self.candidates = []  # 可能性のある答えのリスト
        self.digit_patterns = {}  # 各桁のパターン辞書
        self.digit_pairs = {}  # 桁のペアの関係辞書
        
    def reset(self):
        """エージェントの状態をリセット"""
        self.candidates = []
        self.digit_patterns = {pos: {} for pos in range(self.num_digits)}
        self.digit_pairs = {}
        
    def predict(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> List[int]:
        """
        次の予測を行う
        パターンマッチング: 履歴から学習したパターンを基に予測
        """
        # 初回または少ない履歴のときはランダム
        if len(history) < 2:
            return self._random_prediction()
            
        # 履歴から学習
        self._learn_from_history(history)
        
        # 候補の生成または更新
        if not self.candidates:
            self._generate_candidates(history)
            
        # 候補がなければ新しく生成
        if not self.candidates:
            return self._random_prediction()
            
        # 候補から確率的に選択
        return self._select_candidate()
        
    def _learn_from_history(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> None:
        """
        履歴からパターンを学習する
        """
        # 各桁のパターンを学習
        for entry in history:
            guess, result = entry
            hits, blows = result
            
            self._update_digit_patterns(guess, hits, blows)
            self._update_digit_pairs(guess, hits, blows)
            
        # 候補を絞り込む
        if self.candidates:
            self._filter_candidates(history)
            
    def _update_digit_patterns(self, guess: List[int], hits: int, blows: int) -> None:
        """各桁のパターンを更新"""
        # ヒットが多いほど、その位置での数字の確率が高い
        for pos, digit in enumerate(guess):
            # この位置でのこの数字のスコアを更新
            if digit not in self.digit_patterns[pos]:
                self.digit_patterns[pos][digit] = 0
                
            # ヒットが多いほど重み付け
            if hits > 0:
                weight = hits / self.num_digits  # ヒット数で正規化
                self.digit_patterns[pos][digit] += weight
                
            # ブローがある場合、他の位置の可能性
            if blows > 0 and hits < self.num_digits:
                # この数字が他の位置に出現する可能性
                for other_pos in range(self.num_digits):
                    if other_pos != pos:
                        if digit not in self.digit_patterns[other_pos]:
                            self.digit_patterns[other_pos][digit] = 0
                        
                        blow_weight = blows / (2 * self.num_digits)  # ブロー数で正規化して弱く重み付け
                        self.digit_patterns[other_pos][digit] += blow_weight
                        
    def _update_digit_pairs(self, guess: List[int], hits: int, blows: int) -> None:
        """
        桁間の関係を更新
        例: 0,1の位置に2,3がある場合、それらの組み合わせの確率を更新
        """
        # 2桁のペアの関係
        for i in range(self.num_digits):
            for j in range(i+1, self.num_digits):
                pair = (i, j, guess[i], guess[j])
                if pair not in self.digit_pairs:
                    self.digit_pairs[pair] = 0
                    
                # ヒットが多いほど重み付け
                if hits > 0:
                    weight = hits / self.num_digits
                    self.digit_pairs[pair] += weight
                
    def _generate_candidates(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> None:
        """
        学習したパターンから候補を生成
        """
        # 各桁で最も確率の高い数字を取得
        best_digits = []
        for pos in range(self.num_digits):
            if not self.digit_patterns[pos]:
                # パターンがなければランダム
                best_digits.append(random.sample(range(self.digit_range), 3))  # トップ3
            else:
                # スコアの高い順にソート
                sorted_digits = sorted(
                    self.digit_patterns[pos].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                # トップ3を選択
                best_digits.append([d for d, _ in sorted_digits[:3]])
                
        # 最良の組み合わせを生成
        self._generate_combinations(best_digits, [], 0)
        
        # 候補が少なすぎる場合はランダムに追加
        if len(self.candidates) < 5:
            while len(self.candidates) < 5:
                self.candidates.append(self._random_prediction())
                
    def _generate_combinations(self, best_digits: List[List[int]], current: List[int], pos: int) -> None:
        """
        再帰的に組み合わせを生成
        """
        if pos == self.num_digits:
            if len(set(current)) == self.num_digits:  # 重複チェック
                self.candidates.append(current.copy())
            return
            
        for digit in best_digits[pos]:
            if digit not in current:  # 各桁で異なる数字
                current.append(digit)
                self._generate_combinations(best_digits, current, pos + 1)
                current.pop()
    
    def _filter_candidates(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> None:
        """
        履歴に基づいて候補を絞り込む
        """
        filtered_candidates = []
        
        for candidate in self.candidates:
            valid = True
            
            for entry in history:
                guess, result = entry
                hits, blows = result
                
                # 候補が全ての履歴エントリと一致するか確認
                candidate_hits, candidate_blows = self._calculate_hits_blows(candidate, guess)
                
                if (candidate_hits, candidate_blows) != (hits, blows):
                    valid = False
                    break
            
            if valid:
                filtered_candidates.append(candidate)
                
        self.candidates = filtered_candidates
        
    def _select_candidate(self) -> List[int]:
        """
        候補から選択する
        確率的に選択するが、優先度の高い候補を選びやすくする
        """
        if not self.candidates:
            return self._random_prediction()
            
        # 単純にランダム選択
        return random.choice(self.candidates)
        
    def _random_prediction(self) -> List[int]:
        """
        ランダムな予測を生成
        ただし学習したパターンに基づいて確率的に選ぶ
        """
        prediction = [-1] * self.num_digits
        used_digits = set()
        
        # 各桁で確率的に数字を選ぶ
        for pos in range(self.num_digits):
            if self.digit_patterns[pos]:
                # 正規化された確率を計算
                total = sum(self.digit_patterns[pos].values())
                probs = {d: s/total for d, s in self.digit_patterns[pos].items()}
                
                # 未使用の数字のみフィルタリング
                available_probs = {d: p for d, p in probs.items() if d not in used_digits}
                
                if available_probs:
                    # 確率的に選ぶ
                    digits, probs = zip(*available_probs.items())
                    digit = random.choices(digits, weights=probs)[0]
                    prediction[pos] = digit
                    used_digits.add(digit)
            
        # まだ未割り当ての桁があれば、ランダムに割り当て
        remaining_positions = [i for i, d in enumerate(prediction) if d == -1]
        remaining_digits = [d for d in range(self.digit_range) if d not in used_digits]
        
        if remaining_positions and remaining_digits:
            # 残りの数字をランダムに割り当て
            for pos in remaining_positions:
                if remaining_digits:
                    digit = random.choice(remaining_digits)
                    prediction[pos] = digit
                    remaining_digits.remove(digit)
                else:
                    # もう使える数字がなければランダム
                    while True:
                        digit = random.randint(0, self.digit_range - 1)
                        if digit not in used_digits:
                            prediction[pos] = digit
                            used_digits.add(digit)
                            break
                            
        # 全ての桁が割り当てられていなければ、完全にランダム
        if -1 in prediction:
            return random.sample(range(self.digit_range), self.num_digits)
            
        return prediction
        
    def _calculate_hits_blows(self, prediction: List[int], guess: List[int]) -> Tuple[int, int]:
        """2つの数列間のヒットとブローを計算"""
        hits = sum(1 for i in range(len(prediction)) if prediction[i] == guess[i])
        
        # ブローは共通の数字の数からヒットを引いたもの
        pred_counter = {}
        guess_counter = {}
        
        for p in prediction:
            pred_counter[p] = pred_counter.get(p, 0) + 1
            
        for g in guess:
            guess_counter[g] = guess_counter.get(g, 0) + 1
            
        common = sum(min(pred_counter.get(d, 0), guess_counter.get(d, 0)) for d in range(self.digit_range))
        blows = common - hits
        
        return hits, blows 