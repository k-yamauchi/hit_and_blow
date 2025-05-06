from typing import List, Dict, Any, Tuple, Optional
import random
import numpy as np
import pickle
import os
import math
from collections import defaultdict, Counter
from hit_and_blow.agents.base import AgentBase
import itertools

class HybridSARSAAgent(AgentBase):
    """
    ハイブリッドSARSAエージェント
    SARSAをベースに、ルールベース、情報理論、ヒューリスティックを組み合わせた手法
    """
    
    def __init__(self, num_digits: int = 4, digit_range: int = 10,
                 learning_rate: float = 0.2, discount_factor: float = 0.9,
                 exploration_rate: float = 0.1, model_path: Optional[str] = None):
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
        
        # SARSA用の履歴
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.next_state_history = []
        self.next_action_history = []
        
        # 候補リスト（ルールベース的な要素）
        self.candidates = []
        
        # 最適な初期予測パターン
        self._optimal_first_guesses = {}
        
        # モデルの読み込み
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
    def reset(self):
        """エージェントの状態をリセット"""
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.next_state_history = []
        self.next_action_history = []
        self.candidates = list(itertools.permutations(range(self.digit_range), self.num_digits))
        
    def get_state_representation(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> str:
        """
        履歴から状態表現を生成する
        状態 = これまでの予測と結果のシーケンス + 残りの候補数
        """
        if not history:
            return f"initial|{len(self.candidates)}"
            
        # ルールベースの要素を取り入れて候補を絞り込む
        if history:
            self._update_candidates(history[-1])
            
        # 最新の3つの結果のみを使用（状態空間を小さく保つ）
        recent_history = history[-3:] if len(history) > 3 else history
        
        state_parts = []
        for entry in recent_history:
            guess, result = entry
            hits, blows = result
            state_parts.append(f"{hits}H{blows}B")
            
        # 残りの候補数も状態に加える（重要な情報）
        remaining = len(self.candidates)
        state_parts.append(f"R{remaining}")
        
        # ターン数も状態に含める（SARSAでは重要）
        turn = len(history)
        state_parts.append(f"T{turn}")
        
        return "|".join(state_parts)
        
    def _update_candidates(self, last_entry: Tuple[List[int], Tuple[int, int]]):
        """
        ルールベースエージェントのロジックを用いて候補を絞り込む
        """
        guess, result = last_entry
        hits, blows = result
        
        # 前の候補をフィルタリング
        self.candidates = [
            cand for cand in self.candidates
            if self._check_consistent(list(cand), guess, hits, blows)
        ]
        
    def _check_consistent(self, candidate: List[int], guess: List[int], hits: int, blows: int) -> bool:
        """
        候補が過去の予測結果と整合性があるかチェック
        """
        h, b = 0, 0
        for i in range(self.num_digits):
            if candidate[i] == guess[i]:
                h += 1
            elif candidate[i] in guess:
                b += 1
                
        return h == hits and b == blows
    
    def _get_optimal_first_guess(self) -> Tuple[int, ...]:
        """
        最適な最初の予測を返す（情報量が最大になるよう事前計算）
        """
        # キャッシュキーを作成
        key = (self.num_digits, self.digit_range)
        
        # 既に計算済みならキャッシュから返す
        if key in self._optimal_first_guesses:
            return self._optimal_first_guesses[key]
            
        # 可能な組み合わせすべてが正解の場合を考慮
        all_candidates = list(itertools.permutations(range(self.digit_range), self.num_digits))
        
        # 各可能性について評価
        best_guess = None
        best_score = float('-inf')
        
        # サンプリングして処理負荷を減らす
        sample_size = min(100, len(all_candidates))
        sampled_candidates = random.sample(all_candidates, sample_size)
        
        for guess in sampled_candidates:
            # 各結果の出現確率を計算
            results_count = defaultdict(int)
            
            for answer in all_candidates:
                hits, blows = self._calculate_hits_blows(list(guess), list(answer))
                results_count[(hits, blows)] += 1
                
            # 情報エントロピーを計算
            total = len(all_candidates)
            entropy = 0
            for count in results_count.values():
                p = count / total
                entropy -= p * math.log2(p)
                
            if entropy > best_score:
                best_score = entropy
                best_guess = guess
                
        # キャッシュに保存
        self._optimal_first_guesses[key] = best_guess
        return best_guess
        
    def get_valid_actions(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> List[Tuple[int, ...]]:
        """
        有効な行動（予測）のリストを生成する
        候補リストから取得するが、候補が多すぎる場合は一部をサンプリング
        """
        # 過去の予測
        past_guesses = [tuple(entry[0]) for entry in history]
        
        # 候補からサンプリング（上限10個）
        if not self.candidates:
            # 候補がない場合（整合性のあるものがない）
            return [tuple(random.sample(range(self.digit_range), self.num_digits))]
        
        # 候補が多すぎる場合はサンプリング
        if len(self.candidates) > 20:
            valid_actions = random.sample(self.candidates, min(10, len(self.candidates)))
        else:
            valid_actions = self.candidates[:10]  # 最大10個に制限
            
        # 過去に予測していない候補を優先
        valid_actions = [a for a in valid_actions if a not in past_guesses]
        
        # 有効な行動がない場合は乱数で生成
        if not valid_actions:
            action = tuple(random.sample(range(self.digit_range), self.num_digits))
            while action in past_guesses:
                action = tuple(random.sample(range(self.digit_range), self.num_digits))
            valid_actions = [action]
            
        return valid_actions
        
    def _select_action(self, state: str, valid_actions: List[Tuple[int, ...]]) -> Tuple[int, ...]:
        """
        SARSAポリシーに従って行動を選択する
        """
        # 確率的に探索と活用を切り替え
        if random.random() < self.exploration_rate:
            # 探索: ランダムな行動を選ぶ
            return random.choice(valid_actions)
        else:
            # 活用: 最も価値の高い行動を選ぶ
            q_values = [self.q_table[state][a] for a in valid_actions]
            
            # 最大Q値を持つ行動が複数ある場合は候補数の少ない状態に導く行動を優先
            if all(q == 0 for q in q_values):  # Q値がまだ学習されていない場合
                # 情報利得が最大となる行動を選択（ヒューリスティック）
                return self._select_informative_action(valid_actions)
            else:
                max_q = max(q_values)
                max_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
                return random.choice(max_actions)
                
    def predict(self, history: List[Tuple[List[int], Tuple[int, int]]]) -> List[int]:
        """
        次の予測を行う
        SARSA: 状態から最も価値の高い行動を選び、次の状態と行動を記録
        """
        # 初回リセット（候補リストの初期化）
        if not history and not self.candidates:
            self.reset()
            
        # 最初の予測は最適な初期パターンを使用
        if not history:
            action = self._get_optimal_first_guess()
            self.action_history.append(action)
            # 状態も記録（SARSAでは初期状態も必要）
            initial_state = self.get_state_representation(history)
            self.state_history.append(initial_state)
            return list(action)
            
        # 現在の状態
        current_state = self.get_state_representation(history)
        
        # 前回の行動から学習データを記録（SARSAのオンライン学習）
        if len(self.state_history) > len(self.next_state_history):
            self.next_state_history.append(current_state)
            
            # 前回の行動の報酬を計算
            if history:
                reward = self.calculate_immediate_reward(history[-1])
                self.reward_history.append(reward)
        
        # 候補が1つしかない場合は確定解
        if len(self.candidates) == 1:
            action = self.candidates[0]
            # 前回からのSARSA更新
            if len(self.state_history) > len(self.next_action_history):
                self.next_action_history.append(action)
                self.update_q_values_online()
            
            # 今回の状態と行動を記録
            self.state_history.append(current_state)
            self.action_history.append(action)
            return list(action)
        
        # 候補が少ない場合（5以下）は候補自体を直接試す
        if 1 < len(self.candidates) <= 5:
            # できるだけHitが多くなるものを選ぶ
            best_action = self.candidates[0]
            best_score = -1
            
            for action in self.candidates:
                # 他の候補とのHit数平均を計算
                total_hits = 0
                for candidate in self.candidates:
                    if candidate != action:  # 自分自身とは比較しない
                        hits, _ = self._calculate_hits_blows(list(action), list(candidate))
                        total_hits += hits
                
                avg_hits = total_hits / max(1, len(self.candidates) - 1)
                
                # より多くのHitを持つ行動を優先
                if avg_hits > best_score:
                    best_score = avg_hits
                    best_action = action
            
            # 前回からのSARSA更新
            if len(self.state_history) > len(self.next_action_history):
                self.next_action_history.append(best_action)
                self.update_q_values_online()
                
            # 今回の状態と行動を記録
            self.state_history.append(current_state)
            self.action_history.append(best_action)
            return list(best_action)
        
        # 有効な行動を取得
        valid_actions = self.get_valid_actions(history)
        
        # SARSA方式で行動を選択
        action = self._select_action(current_state, valid_actions)
        
        # 前回からのSARSA更新
        if len(self.state_history) > len(self.next_action_history):
            self.next_action_history.append(action)
            self.update_q_values_online()
        
        # 今回の状態と行動を記録
        self.state_history.append(current_state)
        self.action_history.append(action)
        
        return list(action)
    
    def _select_informative_action(self, actions: List[Tuple[int, ...]]) -> Tuple[int, ...]:
        """
        情報利得が最大となる行動を選択（ヒューリスティック）
        """
        # 単純な場合：候補から直接選ぶ
        if len(actions) <= 3:
            return random.choice(actions)
        
        # 残りの候補数が多い場合（>100）、計算コストを抑えるためサンプリング
        candidate_sample = self.candidates
        if len(self.candidates) > 100:
            candidate_sample = random.sample(self.candidates, 100)
            
        # 残りの候補を最も効率よく絞り込める行動を選ぶ
        best_action = actions[0]
        min_entropy = float('inf')
        
        # サンプリングして評価（すべての可能性を評価すると時間がかかりすぎる）
        sample_actions = random.sample(actions, min(5, len(actions)))
        
        for action in sample_actions:
            # 可能なヒット・ブローの組み合わせごとに残る候補数を計算
            outcome_candidates = defaultdict(int)
            
            for candidate in candidate_sample:
                # この候補が正解だった場合のヒット・ブロー
                hits, blows = self._calculate_hits_blows(list(action), list(candidate))
                outcome = (hits, blows)
                outcome_candidates[outcome] += 1
                
            # エントロピーを計算（情報理論に基づく選択）
            total = len(candidate_sample)
            entropy = 0
            
            for count in outcome_candidates.values():
                p = count / total
                entropy -= p * math.log2(p)
                
            # ヒット数も評価に含める（ヒットが多い行動を優先）
            hit_bonus = 0
            for (h, _), count in outcome_candidates.items():
                hit_bonus += h * count / total
                
            # エントロピーが高いほど情報量が多い（低いほど良い）が、
            # ヒット数ボーナスでバランスを取る
            adjusted_entropy = entropy - 0.5 * hit_bonus
                
            if adjusted_entropy < min_entropy:
                min_entropy = adjusted_entropy
                best_action = action
                
        return best_action
        
    def _calculate_hits_blows(self, guess: List[int], answer: List[int]) -> Tuple[int, int]:
        """
        ヒットとブローを計算
        """
        hits = sum(1 for i in range(self.num_digits) if guess[i] == answer[i])
        
        # ブローの計算（位置が違うが数字は含まれている）
        guess_count = Counter(guess)
        answer_count = Counter(answer)
        common = sum((guess_count & answer_count).values())
        blows = common - hits
        
        return hits, blows
        
    def update_q_values_online(self) -> None:
        """
        オンラインでQ値を更新する（SARSAアルゴリズム）
        直近の（状態、行動、報酬、次の状態、次の行動）の組を使用
        """
        if len(self.state_history) <= 0 or len(self.action_history) <= 0 or \
           len(self.reward_history) <= 0 or len(self.next_state_history) <= 0 or \
           len(self.next_action_history) <= 0:
            return
        
        # インデックスの設定（最新の更新されていないトランジション）
        idx = len(self.reward_history) - 1
        
        state = self.state_history[idx]
        action = self.action_history[idx]
        reward = self.reward_history[idx]
        next_state = self.next_state_history[idx]
        next_action = self.next_action_history[idx]
        
        # 現在のQ値
        current_q = self.q_table[state][action]
        
        # 次の状態・行動のQ値
        next_q = self.q_table[next_state][next_action]
        
        # SARSA更新式: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        td_error = reward + self.discount_factor * next_q - current_q
        new_q = current_q + self.learning_rate * td_error
        
        self.q_table[state][action] = new_q
            
    def calculate_immediate_reward(self, last_entry: Tuple[List[int], Tuple[int, int]]) -> float:
        """
        直前の行動に対する即時報酬を計算
        """
        guess, result = last_entry
        hits, blows = result
        
        # ヒットとブローに基づく報酬（ヒットの方が価値が高い）
        hit_reward = hits / self.num_digits * 0.5
        blow_reward = blows / (2 * self.num_digits) * 0.2
        
        # 正解した場合は大きな報酬
        if hits == self.num_digits:
            solved_reward = 1.0
        else:
            solved_reward = 0.0
            
        # 候補を絞り込んだ場合の報酬（より効率的な絞り込みを促進）
        candidates_reward = 0.0
        if len(self.candidates) > 0:
            candidates_reward = min(0.2, 3.0 / len(self.candidates))
        
        return hit_reward + blow_reward + solved_reward + candidates_reward
        
    def update_final_reward(self, final_reward: float) -> None:
        """
        エピソード終了時に最終報酬を更新する
        
        Args:
            final_reward: 最終報酬（タスク達成までのターン数に基づく）
        """
        # 最後の状態・行動のQ値を更新（次の状態はないのでSARSAの特殊ケース）
        if len(self.state_history) > 0 and len(self.action_history) > 0:
            state = self.state_history[-1]
            action = self.action_history[-1]
            
            # 現在のQ値
            current_q = self.q_table[state][action]
            
            # 最終報酬のみを使って更新
            new_q = current_q + self.learning_rate * (final_reward - current_q)
            self.q_table[state][action] = new_q
            
    def save_model(self, path: str) -> None:
        """
        モデル（Q値テーブル）を保存する
        """
        # defaultdictは直接pickleできないため、通常の辞書に変換
        q_table_dict = {k: dict(v) for k, v in self.q_table.items()}
        
        # 最適な初期予測も保存
        save_data = {
            'q_table': q_table_dict,
            'optimal_first_guesses': dict(self._optimal_first_guesses)
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
            
    def load_model(self, path: str) -> None:
        """
        保存されたモデルを読み込む
        """
        try:
            with open(path, 'rb') as f:
                save_data = pickle.load(f)
                
            # 新形式のモデルファイルの場合
            if isinstance(save_data, dict) and 'q_table' in save_data:
                q_table_dict = save_data['q_table']
                self._optimal_first_guesses = dict(save_data.get('optimal_first_guesses', {}))
            else:
                # 旧形式のモデルファイル（q_tableのみ）の場合
                q_table_dict = save_data
                
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
        エピソード全体の報酬を計算する
        
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
        
        # 正解した場合は大きな報酬
        if hits == self.num_digits:
            base_reward = 2.0
        else:
            # 不正解の場合は小さい報酬
            base_reward = 0.1
            
        # ターン数に基づくボーナスまたはペナルティ
        turn_factor = max(0, 1.0 - 0.1 * len(history))
        
        return base_reward * turn_factor 