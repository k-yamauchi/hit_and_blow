#!/usr/bin/env python3
"""
Hit and Blowゲームを解くための真のハイブリッドエージェント
多腕バンディットと強化学習（Q学習）を組み合わせた二段階アプローチ

フェーズ1: 多腕バンディットで初期の情報収集を最大化（初期の2-3ターン）
フェーズ2: 候補が絞られた後はQ学習で最適解に向かう（後半のターン）
"""

import pickle
import random
import math
import numpy as np
from collections import defaultdict, Counter
import itertools
from hit_and_blow.agents.base import AgentBase


class HybridBanditQLearningAgent(AgentBase):
    """
    多腕バンディットとQ学習を組み合わせた二段階ハイブリッドアプローチ
    
    - 初期段階：情報理論に基づく多腕バンディットで情報収集を最大化
    - 後半段階：Q学習で最適解に向かう
    """
    
    def __init__(self, num_digits=3, digit_range=10, learning_rate=0.1, discount_factor=0.9, 
                 exploration_rate=0.2, model_path=None, **kwargs):
        """
        初期化
        
        Parameters
        ----------
        num_digits : int
            数字の桁数
        digit_range : int
            数字の範囲 (0からdigit_range-1)
        learning_rate : float
            Q学習の学習率
        discount_factor : float
            Q学習の割引率
        exploration_rate : float
            探索率（ε-greedy法用）
        model_path : str
            保存されたモデルのパス
        """
        super().__init__(num_digits, digit_range)
        
        # Q学習のパラメータ
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Q値テーブル
        self.q_values = defaultdict(lambda: defaultdict(float))
        
        # 最適な初期予測
        self.optimal_first_guesses = []
        
        # 最後のエピソードの状態と行動の履歴
        self.last_state = None
        self.last_action = None
        self.history = []
        
        # 候補リスト（有効な数字の組み合わせ）
        self.candidates = []
        
        # モデルの読み込み
        if model_path:
            self.load_model(model_path)
        else:
            self._initialize_candidates()
            
        # 最適な初期予測が未設定の場合、あらかじめ計算された値を使用
        if not self.optimal_first_guesses:
            # 情報理論に基づく最適な初期予測
            if self.num_digits == 3 and self.digit_range == 6:
                self.optimal_first_guesses = [[0, 1, 2], [0, 2, 4], [1, 3, 5]]
            elif self.num_digits == 3 and self.digit_range == 10:
                self.optimal_first_guesses = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
            elif self.num_digits == 4 and self.digit_range == 10:
                self.optimal_first_guesses = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 2, 5, 8]]
            else:
                # その他のケースでは、ターン0に最大の情報利得を持つ予測を生成
                self._generate_optimal_first_guesses()
    
    def _generate_optimal_first_guesses(self):
        """最適な初期予測を計算して生成"""
        if self.optimal_first_guesses:
            return
            
        # 全候補から最も情報利得の高い初期予測を見つける
        best_guesses = []
        all_candidates = self.candidates.copy()
        
        # 少なくとも3つの初期予測を生成
        for _ in range(3):
            if not all_candidates:
                break
                
            # 情報利得が最大の候補を見つける
            best_guess = None
            best_info_gain = -float('inf')
            
            # サンプリングしてパフォーマンスを向上
            sample_size = min(100, len(all_candidates))
            sample = random.sample(all_candidates, sample_size)
            
            for candidate in sample:
                info_gain = self._calculate_candidate_reduction(candidate, all_candidates)
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_guess = candidate
            
            if best_guess:
                best_guesses.append(best_guess)
                # 同じ数字を含まない予測を優先するため、類似の予測を除外
                all_candidates = [c for c in all_candidates 
                                 if sum(1 for i in range(self.num_digits) if c[i] in best_guess) < 2]
                
        self.optimal_first_guesses = best_guesses
            
    def _initialize_candidates(self):
        """すべての可能な数字の組み合わせを生成"""
        self.candidates = list(itertools.product(range(self.digit_range), repeat=self.num_digits))
        # タプルをリストに変換
        self.candidates = [list(c) for c in self.candidates]
    
    def reset(self):
        """エージェントの状態をリセット"""
        self._initialize_candidates()
        self.history = []
        self.last_state = None
        self.last_action = None
    
    def _generate_state_representation(self, history):
        """
        現在の状態を表現する文字列を生成
        
        Parameters
        ----------
        history : list
            ゲームの履歴 [(guess, hits, blows), ...]
            
        Returns
        -------
        str
            状態の文字列表現
        """
        if not history:
            return "initial"
        
        # 履歴からコンパクトな状態を生成
        state_parts = []
        for guess, hits, blows in history[-min(3, len(history)):]:  # 直近の3ターンだけを使用
            state_parts.append(f"{guess}:{hits}{blows}")
            
        # 重要な情報として残りの候補数も状態に含める
        candidate_count = min(200, len(self.candidates))
        state_parts.append(f"cand:{candidate_count}")
            
        return "|".join(state_parts)
    
    def _update_candidates(self, history):
        """
        履歴に基づいて候補を絞り込む
        
        Parameters
        ----------
        history : list
            ゲームの履歴 [(guess, hits, blows), ...]
        """
        if not history:
            return
            
        # 候補リストを初期化
        if len(self.candidates) == 0:
            self._initialize_candidates()
            
        # デバッグ用
        old_count = len(self.candidates)
        
        # 履歴の各ターンに対して候補を絞り込む
        new_candidates = self.candidates.copy()
        for guess, hits, blows in history:
            # 新しい候補リスト
            filtered_candidates = []
            
            for candidate in new_candidates:
                h, b = self._calculate_hits_blows(guess, candidate)
                if h == hits and b == blows:
                    filtered_candidates.append(candidate)
            
            # 候補がなくならないようにチェック
            if filtered_candidates:
                new_candidates = filtered_candidates
                
        # 最終的な候補リストを更新
        self.candidates = new_candidates
        
        # デバッグ用
        print(f"候補数: {old_count} -> {len(self.candidates)}")
        
        # 候補がなくなった場合は初期化（エラー回避）
        if len(self.candidates) == 0:
            self._initialize_candidates()
            print("警告: 候補がなくなったため再初期化しました")
            
    def _calculate_hits_blows(self, guess, answer):
        """
        予測と正解の間のヒットとブローを計算
        
        Parameters
        ----------
        guess : list
            予測の数字リスト
        answer : list
            正解の数字リスト
            
        Returns
        -------
        tuple
            (ヒット数, ブロー数)
        """
        hits = 0
        blows = 0
        
        # 同じ位置・同じ数字をカウント（ヒット）
        for i in range(len(guess)):
            if guess[i] == answer[i]:
                hits += 1
        
        # 数字の出現回数をカウント
        guess_counts = Counter(guess)
        answer_counts = Counter(answer)
        
        # 共通の数字をカウント
        for digit, count in guess_counts.items():
            blows += min(count, answer_counts[digit])
        
        # ヒットの分を引く
        blows -= hits
        
        return hits, blows
    
    def _calculate_candidate_reduction(self, guess, candidates):
        """
        ある予測が候補をどれだけ減らせるかの期待値を計算（情報利得）
        
        Parameters
        ----------
        guess : list
            評価する予測
        candidates : list
            現在の候補リスト
            
        Returns
        -------
        float
            情報利得の値
        """
        # 候補が多い場合はサンプリングして計算を効率化
        if len(candidates) > 50:
            candidates_sample = random.sample(candidates, 50)
        else:
            candidates_sample = candidates
        
        # 可能な (hits, blows) の組み合わせと、その結果残る候補数のマップ
        outcome_partitions = {}
        
        # 各候補が正解だった場合のヒットとブローを計算
        for candidate in candidates_sample:
            h, b = self._calculate_hits_blows(guess, candidate)
            outcome = (h, b)
            
            if outcome not in outcome_partitions:
                outcome_partitions[outcome] = 0
            outcome_partitions[outcome] += 1
        
        # 情報エントロピーの計算（候補が均等に分かれるほど高い値）
        total = len(candidates_sample)
        entropy = 0
        
        for count in outcome_partitions.values():
            probability = count / total
            # Shannon情報量
            entropy -= probability * math.log2(probability)
        
        # 分割の多様性も評価（より多くの分割ができるほど良い）
        diversity = len(outcome_partitions) / ((self.num_digits + 1) * (self.num_digits + 1))
        
        # 総合的な情報利得スコア
        return 0.7 * entropy + 0.3 * diversity
    
    def _select_action_bandit(self, candidates, history):
        """
        バンディットアルゴリズムを使用して行動選択
        初期段階（情報収集フェーズ）で使用
        
        Parameters
        ----------
        candidates : list
            現在の候補リスト
        history : list
            ゲームの履歴
            
        Returns
        -------
        list
            選択された予測
        """
        if not history and self.optimal_first_guesses:
            # 初回の予測は事前計算された最適な初期予測から選択
            return random.choice(self.optimal_first_guesses)
        
        if len(candidates) <= 1:
            # 候補が1つ以下なら、それを選択
            return candidates[0] if candidates else [0] * self.num_digits
        
        # 情報利得に基づく選択
        # 少数のサンプルからコスト効率を高める
        sample_size = min(20, len(candidates))
        sampled_candidates = random.sample(candidates, sample_size)
        
        # 各候補の情報利得を計算
        best_guess = None
        best_info_gain = -float('inf')
        
        for candidate in sampled_candidates:
            info_gain = self._calculate_candidate_reduction(candidate, candidates)
            
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_guess = candidate
                
        return best_guess or random.choice(candidates)
    
    def _select_action_q_learning(self, state, candidates, use_bandit_exploration=True):
        """
        Q学習を使用して行動選択
        後半段階（解決フェーズ）で使用
        
        Parameters
        ----------
        state : str
            現在の状態の文字列表現
        candidates : list
            現在の候補リスト
        use_bandit_exploration : bool
            バンディットアルゴリズムを探索に使用するかどうか
            
        Returns
        -------
        list
            選択された予測
        """
        if len(candidates) <= 1:
            # 候補が1つ以下なら、それを選択
            return candidates[0] if candidates else [0] * self.num_digits

        # ε-greedy法：確率εでランダムに行動し、確率(1-ε)で最適な行動を選択
        if random.random() < self.exploration_rate:
            # 探索戦略：バンディットアルゴリズムかランダム選択
            if use_bandit_exploration:
                return self._select_action_bandit(candidates, self.history)
            else:
                return random.choice(candidates)
        else:
            # 現在の状態で最も価値の高い行動を選択（活用）
            # 各候補のQ値を取得
            q_values = {}
            
            # 計算効率化のため、候補が多い場合はサンプリング
            if len(candidates) > 20:
                evaluate_candidates = random.sample(candidates, 20)
            else:
                evaluate_candidates = candidates
                
            for candidate in evaluate_candidates:
                action = tuple(candidate)
                q_values[action] = self.q_values[state][action]
                
            # 最大のQ値を持つ行動を選択（同点の場合はランダム）
            max_q = max(q_values.values()) if q_values else 0
            best_actions = [action for action, q in q_values.items() if q == max_q]
            
            if best_actions:
                # 最大Q値が0の場合（未訪問状態の場合）、50%の確率で情報利得に基づく選択を行う
                if max_q == 0 and use_bandit_exploration and random.random() < 0.5:
                    return self._select_action_bandit(candidates, self.history)
                else:
                    return list(random.choice(best_actions))
            else:
                # Q値が未定義の場合は情報利得に基づく選択
                return self._select_action_bandit(candidates, self.history)
    
    def _preprocess_history(self, history):
        """
        環境から与えられる履歴を標準形式に変換
        
        Parameters
        ----------
        history : list
            環境から与えられる履歴
            
        Returns
        -------
        list
            標準形式に変換された履歴 [(guess, hits, blows), ...]
        """
        processed_history = []
        
        # 履歴の各要素をチェック
        for i, item in enumerate(history):
            # 履歴の形式をデバッグ
            print(f"履歴項目 {i}: タイプ={type(item)}, 値={item}")
            
            if isinstance(item, tuple) and len(item) >= 2:
                guess = item[0]
                
                # 環境のフォーマットによって処理を分岐
                if isinstance(item[1], tuple) and len(item[1]) == 2:
                    # タプル形式の結果 ([0, 1, 2], (1, 1))
                    hits, blows = item[1]
                    processed_history.append((guess, hits, blows))
                elif isinstance(item[1], dict) and 'hits' in item[1] and 'blows' in item[1]:
                    # 辞書形式の結果
                    hits = item[1]['hits']
                    blows = item[1]['blows']
                    processed_history.append((guess, hits, blows))
                elif len(item) == 3:
                    # 3要素タプル形式
                    hits = item[1]
                    blows = item[2]
                    processed_history.append((guess, hits, blows))
                else:
                    print(f"未対応の履歴形式: {item}")
        
        print(f"処理された履歴: {processed_history}")
        return processed_history
    
    def predict(self, history):
        """
        次の予測を行う
        
        Parameters
        ----------
        history : list
            ゲームの履歴 [(guess, hits, blows), ...]
            
        Returns
        -------
        list
            次の予測
        """
        # 履歴を標準形式に変換
        processed_history = self._preprocess_history(history)
        
        # デバッグ情報
        print(f"処理前の履歴: {history}")
        print(f"処理後の履歴: {processed_history}")
        print(f"処理前の候補数: {len(self.candidates)}")
        
        # 履歴に基づいて候補を更新
        self._update_candidates(processed_history)
        
        # 現在の状態を表現
        state = self._generate_state_representation(processed_history)
        
        # 現在のターン数
        current_turn = len(processed_history)
        
        # デバッグ情報
        print(f"現在の状態: {state}")
        print(f"現在のターン: {current_turn}")
        print(f"更新後の候補数: {len(self.candidates)}")
        if len(self.candidates) <= 5:
            print(f"残り候補: {self.candidates}")
        
        # 選択アルゴリズムの決定（ハイブリッドアプローチ）
        if len(self.candidates) == 1:
            # 候補が1つだけなら、それを選択
            action = self.candidates[0]
            print(f"候補が1つなので選択: {action}")
        elif current_turn == 0:
            # 最初のターンはバンディットアルゴリズムのみを使用
            action = self._select_action_bandit(self.candidates, processed_history)
            print(f"初回ターン、バンディットで選択: {action}")
        else:
            # それ以降は候補数に応じて戦略を変える
            # 候補が多い場合は情報収集（バンディット重視）
            # 候補が少ない場合は最適化（Q学習重視）
            if len(self.candidates) > 20:
                # 候補が多い場合、バンディットの要素を強く
                if random.random() < 0.6:  # 60%の確率でバンディット
                    action = self._select_action_bandit(self.candidates, processed_history)
                    print(f"多数候補、バンディットで選択: {action}")
                else:
                    # 40%の確率でQ学習（バンディット探索あり）
                    action = self._select_action_q_learning(state, self.candidates, True)
                    print(f"多数候補、Q学習で選択: {action}")
            else:
                # 候補が少ない場合、Q学習の要素を強く
                if random.random() < 0.2:  # 20%の確率でバンディット
                    action = self._select_action_bandit(self.candidates, processed_history)
                    print(f"少数候補、バンディットで選択: {action}")
                else:
                    # 80%の確率でQ学習（バンディット探索あり）
                    action = self._select_action_q_learning(state, self.candidates, True)
                    print(f"少数候補、Q学習で選択: {action}")
            
        # 状態と行動を記録
        self.last_state = state
        self.last_action = tuple(action)
        
        # 履歴を更新
        self.history = processed_history.copy()
        
        return action
    
    def update_q_values(self, final_reward):
        """
        Q値を更新
        
        Parameters
        ----------
        final_reward : float
            エピソード終了時の最終報酬
        """
        if self.last_state is None or self.last_action is None:
            return
            
        # 現在の状態とアクションのQ値を更新
        current_q = self.q_values[self.last_state][self.last_action]
        
        # 勝利時は大きく更新、それ以外は通常更新
        if final_reward >= 0.9:  # 勝利と見なせる報酬
            # 勝利した場合は大きく更新
            self.q_values[self.last_state][self.last_action] = (
                (1 - self.learning_rate * 2) * current_q +
                self.learning_rate * 2 * final_reward
            )
        else:
            # 通常の更新
            self.q_values[self.last_state][self.last_action] = (
                (1 - self.learning_rate) * current_q +
                self.learning_rate * final_reward
            )
        
        # 最適な初期予測の更新（成功した場合）
        if final_reward >= 0.9 and self.history:
            # 最初の予測
            first_guess = self.history[0][0]
            
            if first_guess not in self.optimal_first_guesses:
                self.optimal_first_guesses.append(first_guess)
                
            # 最適な初期予測は最大10個まで
            if len(self.optimal_first_guesses) > 10:
                self.optimal_first_guesses = self.optimal_first_guesses[-10:]
    
    def calculate_reward(self, history):
        """
        ゲームの履歴から報酬を計算
        
        Parameters
        ----------
        history : list
            ゲームの履歴 [(guess, hits, blows), ...]
            
        Returns
        -------
        float
            計算された報酬
        """
        if not history:
            return 0
            
        # 最後の予測の結果
        last_guess, hits, blows = history[-1]
        
        # 正解の場合は大きな報酬
        if hits == self.num_digits:
            # ターン数が少ないほど報酬を大きく
            turn_bonus = max(0, 1 - (len(history) - 1) / 10)
            return 1.0 + turn_bonus * 0.5
            
        # 情報収集の報酬（ヒットとブローの合計が多いほど高い）
        hit_blow_quality = (hits * 2 + blows) / (self.num_digits * 2)
        
        # 候補削減の報酬（候補が少ないほど高い）
        if len(self.candidates) < 5:
            reduction_quality = 0.9  # 候補が非常に少ない
        elif len(self.candidates) < 10:
            reduction_quality = 0.7  # 候補が少ない
        elif len(self.candidates) < 20:
            reduction_quality = 0.5  # 候補がやや少ない
        else:
            # 候補数に反比例した報酬
            remaining_ratio = len(self.candidates) / (self.digit_range ** self.num_digits)
            reduction_quality = 1 - remaining_ratio
            
        # ターン数ペナルティ（ターン数が多いほど報酬が下がる）
        turn_penalty = max(0, 1 - (len(history) / 10) * 0.2)
            
        # 総合的な報酬
        return 0.3 * hit_blow_quality + 0.5 * reduction_quality + 0.2 * turn_penalty
    
    def save_model(self, model_path):
        """
        モデルを保存
        
        Parameters
        ----------
        model_path : str
            保存先のパス
        """
        # defaultdictは通常のdictに変換して保存
        q_values_dict = {k: dict(v) for k, v in self.q_values.items()}
        
        model_data = {
            'q_values': q_values_dict,
            'optimal_first_guesses': self.optimal_first_guesses
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, model_path):
        """
        モデルを読み込み
        
        Parameters
        ----------
        model_path : str
            読み込むモデルのパス
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            # dictをdefaultdictに変換
            self.q_values = defaultdict(lambda: defaultdict(float))
            for state, actions in model_data.get('q_values', {}).items():
                for action, value in actions.items():
                    self.q_values[state][action] = value
                    
            self.optimal_first_guesses = model_data.get('optimal_first_guesses', [])
            
            # 候補の初期化
            self._initialize_candidates()
            
            print(f"モデルを読み込みました: {model_path}")
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            print(f"モデルの読み込みに失敗しました: {e}")
            self._initialize_candidates() 