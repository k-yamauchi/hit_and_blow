#!/usr/bin/env python3
"""
Hit and Blowゲームを解くためのハイブリッド多腕バンディットアルゴリズムを用いたエージェント
UCB（Upper Confidence Bound）アルゴリズムに情報理論と候補絞り込みを組み合わせたハイブリッドアプローチ
"""

import random
import pickle
import math
import numpy as np
from collections import defaultdict, Counter
import itertools
from hit_and_blow.agents.base import AgentBase


class BanditAgent(AgentBase):
    """
    ハイブリッド多腕バンディットエージェント
    
    効率的な情報獲得のための多腕バンディットアルゴリズムと
    候補リストの維持・絞り込みを組み合わせたハイブリッドアプローチ
    """
    
    def __init__(self, num_digits=3, digit_range=10, exploration_param=2.0, model_path=None, **kwargs):
        """
        初期化
        
        Parameters
        ----------
        num_digits : int
            数字の桁数
        digit_range : int
            数字の範囲 (0からdigit_range-1)
        exploration_param : float
            UCBアルゴリズムの探索パラメータ（大きいほど探索を重視）
        model_path : str
            保存されたモデルのパス
        """
        super().__init__(num_digits, digit_range)
        
        # UCBのパラメータ
        self.exploration_param = exploration_param
        
        # 統計情報
        self.action_counts = defaultdict(int)      # 各行動の選択回数
        self.action_rewards = defaultdict(float)   # 各行動の累積報酬
        
        # 最適な初期予測（訓練で学習）
        self.optimal_first_guesses = []
        
        # 最後のエピソードの状態と行動の履歴
        self.last_state = None
        self.last_action = None
        self.history = []
        
        # 候補の数字の組み合わせ
        self.candidates = []
        
        # モデルの読み込み
        if model_path:
            self.load_model(model_path)
        else:
            self._initialize_candidates()
            
        # ハイブリッド戦略の初期予測を設定
        if not self.optimal_first_guesses:
            # 情報理論に基づく初期予測
            # 多くの情報を得るための良いスタート予測
            if self.num_digits == 3 and self.digit_range == 6:
                self.optimal_first_guesses = [[0, 1, 2], [0, 2, 4], [1, 3, 5]]
            elif self.num_digits == 4 and self.digit_range == 10:
                self.optimal_first_guesses = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 2, 5, 8]]
            elif self.num_digits == 3 and self.digit_range == 10:
                self.optimal_first_guesses = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
            else:
                # その他のケースでは、ランダムな組み合わせを生成
                self._generate_optimal_first_guesses()
                
    def _generate_optimal_first_guesses(self):
        """オプティマルな初期予測を生成"""
        if self.optimal_first_guesses:
            return
            
        # 候補リストから適切な初期予測を探索
        best_guesses = []
        all_candidates = self.candidates.copy()
        
        # 少なくとも3つの初期予測を生成
        for _ in range(3):
            if not all_candidates:
                break
                
            # 情報利得が最大の候補を見つける
            best_guess = None
            best_info_gain = -float('inf')
            
            # 最大100個の候補をランダムに選んで評価
            sample_size = min(100, len(all_candidates))
            sample = random.sample(all_candidates, sample_size)
            
            for candidate in sample:
                info_gain = self._calculate_candidate_reduction(candidate, all_candidates)
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_guess = candidate
            
            if best_guess:
                best_guesses.append(best_guess)
                
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
        現在の状態を表現する
        
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
        
        # 履歴から状態を生成
        state_repr = []
        for item in history:
            # 履歴の形式をチェック
            if isinstance(item, tuple) and len(item) == 3:
                guess, hits, blows = item
            elif isinstance(item, tuple) and len(item) == 2:
                guess, result = item
                if isinstance(result, dict) and "hits" in result and "blows" in result:
                    hits, blows = result["hits"], result["blows"]
                else:
                    continue
            else:
                continue
            
            state_repr.append(f"{guess}:{hits}{blows}")
        
        return "|".join(state_repr)
    
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
        
        # 候補を初期化
        self._initialize_candidates()
        
        # 履歴全体に基づいて候補を絞り込む
        for item in history:
            # 履歴の形式をチェック
            if isinstance(item, tuple) and len(item) == 3:
                guess, hits, blows = item
            elif isinstance(item, tuple) and len(item) == 2:
                guess, result = item
                if isinstance(result, dict) and "hits" in result and "blows" in result:
                    hits, blows = result["hits"], result["blows"]
                else:
                    continue
            else:
                continue
            
            # 現在の予測に基づいて候補を絞り込む
            new_candidates = []
            for candidate in self.candidates:
                h, b = self._calculate_hits_blows(guess, candidate)
                if h == hits and b == blows:
                    new_candidates.append(candidate)
            
            self.candidates = new_candidates
            
            # デバッグ出力: 残りの候補数
            # print(f"残りの候補数: {len(self.candidates)}")
    
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
    
    def _calculate_candidate_reduction(self, guess, current_candidates):
        """
        ある予測が候補をどれだけ減らせるかの期待値を計算
        情報利得の推定値として使用
        
        Parameters
        ----------
        guess : list
            評価する予測
        current_candidates : list
            現在の候補リスト
            
        Returns
        -------
        float
            候補削減の期待値
        """
        # サンプル数が多すぎる場合は、サンプリングして計算を効率化
        if len(current_candidates) > 50:
            current_candidates = random.sample(current_candidates, 50)
        
        # 可能な (hits, blows) の組み合わせと、その結果残る候補数のマップ
        outcome_candidates = {}
        
        # 各候補が正解だった場合のヒットとブローを計算
        for candidate in current_candidates:
            h, b = self._calculate_hits_blows(guess, candidate)
            outcome = (h, b)
            
            if outcome not in outcome_candidates:
                outcome_candidates[outcome] = 0
            outcome_candidates[outcome] += 1
        
        # 情報エントロピー（情報利得）を計算
        # 候補数が多いほど、分割後の候補が均等に分かれるほど高くなる
        total_candidates = len(current_candidates)
        
        # 結果の分布の均一さを評価（分割の均一さは情報量の多さを意味する）
        distribution_evenness = 0
        for count in outcome_candidates.values():
            probability = count / total_candidates
            # Shannon情報量
            distribution_evenness -= probability * math.log2(probability)
        
        # 結果の多様さも評価（多くの異なる結果が得られるほど良い）
        outcome_diversity = len(outcome_candidates) / ((self.num_digits + 1) * (self.num_digits + 1))
        
        # 総合的な情報利得スコア
        info_gain = 0.7 * distribution_evenness + 0.3 * outcome_diversity
        
        return info_gain
    
    def _select_by_information_gain(self, candidates):
        """
        情報利得に基づいて予測を選択する
        
        Parameters
        ----------
        candidates : list
            候補リスト
            
        Returns
        -------
        list
            選択された予測
        """
        if len(candidates) <= 1:
            return candidates[0] if candidates else [0] * self.num_digits
            
        # 少数の候補からランダムにサンプリング
        sample_size = min(20, len(candidates))
        sampled_candidates = random.sample(candidates, sample_size) if len(candidates) > sample_size else candidates
        
        # 情報利得が最大の候補を見つける
        best_candidate = None
        best_info_gain = -float('inf')
        
        for candidate in sampled_candidates:
            info_gain = self._calculate_candidate_reduction(candidate, candidates)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_candidate = candidate
                
        return best_candidate or random.choice(candidates)
    
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
        # 実際の履歴形式に合わせて履歴を変換
        processed_history = []
        
        for item in history:
            if isinstance(item, tuple) and len(item) >= 2:
                guess = item[0]
                
                # 環境のフォーマットによっては結果が辞書形式
                if isinstance(item[1], dict) and 'hits' in item[1] and 'blows' in item[1]:
                    hits = item[1]['hits']
                    blows = item[1]['blows']
                # 3要素タプルの場合
                elif len(item) == 3:
                    hits = item[1]
                    blows = item[2]
                else:
                    continue
                
                processed_history.append((guess, hits, blows))
        
        # 履歴に基づいて候補を更新
        # 候補を初期化
        self._initialize_candidates()
        
        # 変換後の履歴全体に基づいて候補を絞り込む
        for guess, hits, blows in processed_history:
            # 現在の予測に基づいて候補を絞り込む
            new_candidates = []
            for candidate in self.candidates:
                h, b = self._calculate_hits_blows(guess, candidate)
                if h == hits and b == blows:
                    new_candidates.append(candidate)
            
            self.candidates = new_candidates
        
        # 候補が一つだけなら、それを選択
        if len(self.candidates) == 1:
            guess = self.candidates[0]
        # 初回予測の場合は最適な初期予測を使用
        elif not history and self.optimal_first_guesses:
            guess = random.choice(self.optimal_first_guesses)
        # それ以外は情報利得に基づいて予測を選択
        else:
            guess = self._select_by_information_gain(self.candidates)
        
        # 履歴を更新
        self.history = processed_history.copy()
        # 状態と行動を記録
        self.last_state = self._generate_state_representation(processed_history)
        self.last_action = tuple(guess)
        
        return guess
    
    def update_reward(self, reward):
        """
        最後の行動に対する報酬を更新
        
        Parameters
        ----------
        reward : float
            得られた報酬
        """
        if self.last_state is not None and self.last_action is not None:
            # 行動回数と累積報酬を更新
            self.action_counts[(self.last_state, self.last_action)] += 1
            self.action_rewards[(self.last_state, self.last_action)] += reward
    
    def calculate_reward(self, history):
        """
        与えられた履歴から報酬を計算
        
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
        
        # 最後の予測の結果を取得
        last_item = history[-1]
        
        # 履歴の形式をチェック
        if isinstance(last_item, tuple) and len(last_item) == 3:
            last_guess, hits, blows = last_item
        elif isinstance(last_item, tuple) and len(last_item) == 2:
            last_guess, result = last_item
            if isinstance(result, dict) and "hits" in result and "blows" in result:
                hits, blows = result["hits"], result["blows"]
            else:
                return 0
        else:
            return 0
        
        # 正解の場合は大きな報酬
        if hits == self.num_digits:
            return 1.0
        
        # ヒットとブローの合計が多いほど報酬が高い
        guess_quality = (hits * 2 + blows) / (self.num_digits * 2)
        
        # 候補の削減率
        reduction = 0
        if len(history) >= 2 and len(self.candidates) > 0:
            # 現在の候補数が10個未満の場合は高い報酬を与える
            if len(self.candidates) < 10:
                reduction = 0.8
            # それ以外は候補数に反比例した報酬
            else:
                remaining_ratio = len(self.candidates) / (self.digit_range ** self.num_digits)
                reduction = 1 - remaining_ratio
        
        # 総合的な報酬
        return 0.3 * guess_quality + 0.7 * reduction
    
    def update_final_reward(self, final_reward):
        """
        エピソード終了時に最終報酬を更新
        
        Parameters
        ----------
        final_reward : float
            エピソード全体での報酬
        """
        # 最後の行動に対する報酬を更新
        self.update_reward(final_reward)
        
        # 最適な初期予測の更新（成功した場合）
        if final_reward > 0 and self.history:
            # 最初の予測を取得
            first_item = self.history[0]
            first_guess = None
            
            # 履歴の形式をチェック
            if isinstance(first_item, tuple):
                if len(first_item) >= 1:
                    first_guess = first_item[0]
            
            if first_guess and first_guess not in self.optimal_first_guesses:
                self.optimal_first_guesses.append(first_guess)
            
            # 最適な初期予測は最大10個まで
            if len(self.optimal_first_guesses) > 10:
                self.optimal_first_guesses = self.optimal_first_guesses[-10:]
    
    def save_model(self, model_path):
        """
        モデルを保存
        
        Parameters
        ----------
        model_path : str
            保存先のパス
        """
        model_data = {
            'action_counts': dict(self.action_counts),
            'action_rewards': dict(self.action_rewards),
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
                
            self.action_counts = defaultdict(int, model_data.get('action_counts', {}))
            self.action_rewards = defaultdict(float, model_data.get('action_rewards', {}))
            self.optimal_first_guesses = model_data.get('optimal_first_guesses', [])
            
            # 候補の初期化
            self._initialize_candidates()
            
            print(f"モデルを読み込みました: {model_path}")
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            print(f"モデルの読み込みに失敗しました: {e}")
            self._initialize_candidates() 