"""エージェントの実装"""

from abc import ABC, abstractmethod
import random
import math
import time
import numpy as np
from collections import defaultdict

from .environment import Environment
from .utils import generate_all_possible_codes, calculate_hits_blows

class BaseAgent(ABC):
    """エージェントの基底クラス"""
    
    def __init__(self, digits=3, number_range=6, allow_repetition=False):
        """
        初期化
        
        Args:
            digits (int): 桁数
            number_range (int): 数字の範囲 (0からnumber_range-1)
            allow_repetition (bool): 数字の繰り返しを許可するか
        """
        self.digits = digits
        self.number_range = number_range
        self.allow_repetition = allow_repetition
        self.initialize()
        
    def initialize(self):
        """エージェントの初期化"""
        # 候補となる全ての数列を生成
        self.all_candidates = generate_all_possible_codes(
            self.digits, self.number_range, self.allow_repetition)
        
        # 現在の候補を全候補で初期化
        self.candidates = self.all_candidates.copy()
        
    def reset(self):
        """エージェントをリセット"""
        self.initialize()
        
    def update_candidates(self, guess, hits, blows):
        """
        予測結果に基づいて候補を更新
        
        Args:
            guess (list): 予測した数列
            hits (int): ヒット数
            blows (int): ブロー数
        """
        # 予測結果と矛盾しない候補のみを残す
        new_candidates = set()
        for candidate in self.candidates:
            candidate_hits, candidate_blows = calculate_hits_blows(guess, candidate)
            if candidate_hits == hits and candidate_blows == blows:
                new_candidates.add(candidate)
        
        self.candidates = new_candidates
        
    @abstractmethod
    def predict(self, previous_guesses=None, results=None, adaptation_info=None):
        """
        次の予測を返す
        
        Args:
            previous_guesses (list): これまでの予測のリスト
            results (list): これまでの結果のリスト（(hits, blows)のタプル）
            adaptation_info (dict): 相手の進捗に基づく戦略調整情報
            
        Returns:
            list: 予測した数列
        """
        pass
    
    def get_remaining_candidates(self):
        """
        残りの候補数を返す
        
        Returns:
            int: 残りの候補数
        """
        return len(self.candidates)
    
    def get_possible_answers(self, max_count=5):
        """
        可能性のある答えを返す
        
        Args:
            max_count (int): 最大表示数
            
        Returns:
            list: 可能性のある答えのリスト
        """
        candidates_list = list(self.candidates)
        return candidates_list[:min(max_count, len(candidates_list))]

class RuleBasedAgent(BaseAgent):
    """ルールベースのエージェント"""
    
    def predict(self, previous_guesses=None, results=None, adaptation_info=None):
        """
        次の予測を返す
        
        Args:
            previous_guesses (list): これまでの予測のリスト
            results (list): これまでの結果のリスト（(hits, blows)のタプル）
            adaptation_info (dict): 相手の進捗に基づく戦略調整情報
            
        Returns:
            list: 予測した数列
        """
        if previous_guesses is None or results is None:
            previous_guesses = []
            results = []
            
        # 過去の予測と結果を使って候補を更新
        for guess, result in zip(previous_guesses, results):
            hits, blows = result
            self.update_candidates(guess, hits, blows)
            
        # 候補がない場合は初期化
        if not self.candidates:
            self.initialize()
            
        # 候補から最初の要素を予測として選択
        candidates_list = list(self.candidates)
        
        # 相手の進捗に基づく戦略調整（リスクを取るかどうか）
        if adaptation_info and adaptation_info.get("strategy") == "risky":
            # リスクが高い戦略：残りの候補からランダムに選択
            return random.choice(candidates_list)
        
        return candidates_list[0]

class ProbabilisticAgent(BaseAgent):
    """確率的なエージェント（情報利得最大化）"""
    
    def predict(self, previous_guesses=None, results=None, adaptation_info=None):
        """
        次の予測を返す（エントロピー最大化）
        
        Args:
            previous_guesses (list): これまでの予測のリスト
            results (list): これまでの結果のリスト（(hits, blows)のタプル）
            adaptation_info (dict): 相手の進捗に基づく戦略調整情報
            
        Returns:
            list: 予測した数列
        """
        if previous_guesses is None or results is None:
            previous_guesses = []
            results = []
            
        # 過去の予測と結果を使って候補を更新
        for guess, result in zip(previous_guesses, results):
            hits, blows = result
            self.update_candidates(guess, hits, blows)
            
        # 候補がない場合は初期化
        if not self.candidates:
            self.initialize()
            
        # 候補が1つしかない場合はそれを返す
        if len(self.candidates) == 1:
            return list(self.candidates)[0]
            
        # エントロピーを最大化する予測を選択
        possible_guesses = self.all_candidates.copy()
        if len(self.candidates) <= 5:  # 候補が少ない場合は候補内から選ぶ
            possible_guesses = self.candidates.copy()
            
        best_guess = None
        best_entropy = -1
        
        # 相手の進捗に基づく戦略調整
        hit_bonus = 0.0
        if adaptation_info:
            if adaptation_info.get("strategy") == "risky":
                # リスクが高い戦略：候補内から直接選択する確率を上げる
                if random.random() < adaptation_info.get("risk_factor", 0.3):
                    return random.choice(list(self.candidates))
                hit_bonus = 0.3  # ヒット数のボーナスを増やしてより直接的な予測を促進
            elif adaptation_info.get("strategy") == "safe":
                # 安全な戦略：ヒットボーナスを下げて情報収集を優先
                hit_bonus = 0.05
        else:
            hit_bonus = 0.1  # デフォルト値
            
        # 各予測候補についてエントロピーを計算
        for guess in possible_guesses:
            # 各可能な結果の出現回数をカウント
            outcome_distribution = defaultdict(int)
            
            # 全ての候補についてこの予測をした場合の結果を計算
            for candidate in self.candidates:
                result = calculate_hits_blows(guess, candidate)
                outcome_distribution[result] += 1
                
            # エントロピー計算
            entropy = 0
            total = len(self.candidates)
            expected_hits = 0
            
            for result, count in outcome_distribution.items():
                p = count / total
                entropy -= p * math.log2(p)
                expected_hits += result[0] * p  # ヒット数の期待値
                
            # ヒット数にボーナスを与える（同じエントロピーならヒットが多い方を優先）
            adjusted_entropy = entropy + hit_bonus * expected_hits
                
            if adjusted_entropy > best_entropy:
                best_entropy = adjusted_entropy
                best_guess = guess
                
        return best_guess

class HybridQLearningAgent(BaseAgent):
    """ハイブリッドQ学習エージェント"""
    
    def __init__(self, digits=3, number_range=6, allow_repetition=False, 
                learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1,
                model_path=None):
        """
        初期化
        
        Args:
            digits (int): 桁数
            number_range (int): 数字の範囲 (0からnumber_range-1)
            allow_repetition (bool): 数字の繰り返しを許可するか
            learning_rate (float): 学習率
            discount_factor (float): 割引率
            exploration_rate (float): 探索率
            model_path (str): モデルファイルのパス
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_values = {}  # Q値を格納する辞書
        self.model_path = model_path
        
        super().__init__(digits, number_range, allow_repetition)
        
        # モデルファイルが指定されている場合は読み込む
        if self.model_path:
            self.load_model(self.model_path)
    
    def get_state_key(self, turn, candidates_count, last_hits=None, last_blows=None):
        """
        状態を表す文字列キーを生成
        
        Args:
            turn (int): 現在のターン数
            candidates_count (int): 候補数
            last_hits (int): 前回のヒット数
            last_blows (int): 前回のブロー数
            
        Returns:
            str: 状態を表す文字列キー
        """
        # 候補数をカテゴリ化
        if candidates_count >= 100:
            candidates_category = "many"
        elif candidates_count >= 10:
            candidates_category = "medium"
        else:
            candidates_category = "few"
            
        if last_hits is None or last_blows is None:
            return f"t{turn}_c{candidates_category}"
        
        return f"t{turn}_c{candidates_category}_h{last_hits}b{last_blows}"
    
    def get_action_key(self, action):
        """
        行動を表す文字列キーを生成
        
        Args:
            action (tuple): 行動（数列）
            
        Returns:
            str: 行動を表す文字列キー
        """
        return str(action)
    
    def get_q_value(self, state_key, action_key):
        """
        Q値を取得
        
        Args:
            state_key (str): 状態のキー
            action_key (str): 行動のキー
            
        Returns:
            float: Q値
        """
        if state_key not in self.q_values:
            self.q_values[state_key] = {}
        
        if action_key not in self.q_values[state_key]:
            self.q_values[state_key][action_key] = 0.0
            
        return self.q_values[state_key][action_key]
    
    def update_q_value(self, state_key, action_key, reward, next_state_key):
        """
        Q値を更新
        
        Args:
            state_key (str): 状態のキー
            action_key (str): 行動のキー
            reward (float): 報酬
            next_state_key (str): 次の状態のキー
        """
        # 次の状態で取りうる行動のQ値の最大値
        max_next_q = 0.0
        if next_state_key in self.q_values:
            next_q_values = self.q_values[next_state_key].values()
            if next_q_values:
                max_next_q = max(next_q_values)
                
        # Q値の更新
        current_q = self.get_q_value(state_key, action_key)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        self.q_values[state_key][action_key] = new_q
    
    def select_action(self, state_key, possible_actions):
        """
        行動を選択
        
        Args:
            state_key (str): 状態のキー
            possible_actions (list): 可能な行動のリスト
            
        Returns:
            tuple: 選択された行動
        """
        # 探索：ランダムに行動を選択
        if random.random() < self.exploration_rate:
            return random.choice(possible_actions)
            
        # 活用：Q値が最大の行動を選択
        best_action = None
        best_q_value = float('-inf')
        
        for action in possible_actions:
            action_key = self.get_action_key(action)
            q_value = self.get_q_value(state_key, action_key)
            
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
                
        # Q値が同じ行動が複数ある場合はランダムに選択
        if best_action is None:
            return random.choice(possible_actions)
            
        return best_action
    
    def predict(self, previous_guesses=None, results=None, adaptation_info=None):
        """
        次の予測を返す
        
        Args:
            previous_guesses (list): これまでの予測のリスト
            results (list): これまでの結果のリスト（(hits, blows)のタプル）
            adaptation_info (dict): 相手の進捗に基づく戦略調整情報
            
        Returns:
            list: 予測した数列
        """
        if previous_guesses is None or results is None:
            previous_guesses = []
            results = []
            
        # 過去の予測と結果を使って候補を更新
        for guess, result in zip(previous_guesses, results):
            hits, blows = result
            self.update_candidates(guess, hits, blows)
            
        # 候補がない場合は初期化
        if not self.candidates:
            self.initialize()
            
        # 候補が1つしかない場合はそれを返す
        if len(self.candidates) == 1:
            return list(self.candidates)[0]
            
        # 候補が少ない場合は候補の中からQ値が最大のものを選ぶ
        if len(self.candidates) <= 5:
            possible_actions = [tuple(candidate) for candidate in self.candidates]
            
            # 相手の進捗に基づく戦略調整
            if adaptation_info and adaptation_info.get("strategy") == "risky":
                # リスクが高い戦略：候補からランダムに選択
                risk_factor = adaptation_info.get("risk_factor", 0.3)
                if random.random() < risk_factor:
                    return list(random.choice(possible_actions))
            
            # 現在の状態を特定
            turn = len(previous_guesses) + 1
            candidates_count = len(self.candidates)
            last_hits = None
            last_blows = None
            
            if results:
                last_hits, last_blows = results[-1]
                
            state_key = self.get_state_key(turn, candidates_count, last_hits, last_blows)
            
            # Q値に基づいて行動を選択
            best_action = self.select_action(state_key, possible_actions)
            return list(best_action)
            
        # エントロピーを最大化する予測を選択
        probabilistic_agent = ProbabilisticAgent(
            digits=self.digits, 
            number_range=self.number_range,
            allow_repetition=self.allow_repetition
        )
        
        # 候補を共有
        probabilistic_agent.candidates = self.candidates.copy()
        
        # 確率的エージェントの予測を利用（相手の進捗情報も渡す）
        return probabilistic_agent.predict(previous_guesses, results, adaptation_info)
    
    def save_model(self, path):
        """
        モデル（Q値）を保存
        
        Args:
            path (str): 保存先のパス
        """
        import pickle
        
        with open(path, 'wb') as f:
            pickle.dump(self.q_values, f)
            
        print(f"モデルを{path}に保存しました")
    
    def load_model(self, path):
        """
        モデル（Q値）を読み込み
        
        Args:
            path (str): 読み込み元のパス
        """
        import pickle
        
        try:
            with open(path, 'rb') as f:
                self.q_values = pickle.load(f)
                
            print(f"モデルを{path}から読み込みました")
        except FileNotFoundError:
            print(f"モデルファイル{path}が見つかりません")
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")

class AgentRegistry:
    """エージェントを登録・取得するクラス"""
    
    _registry = {
        'rule_based': RuleBasedAgent,
        'probabilistic': ProbabilisticAgent,
        'hybrid_q_learning': HybridQLearningAgent,
    }
    
    @classmethod
    def register(cls, name, agent_class):
        """
        エージェントを登録
        
        Args:
            name (str): エージェント名
            agent_class (class): エージェントクラス
        """
        cls._registry[name] = agent_class
        
    @classmethod
    def get_class(cls, name):
        """
        エージェントクラスを取得
        
        Args:
            name (str): エージェント名
            
        Returns:
            class: エージェントクラス
            
        Raises:
            ValueError: 指定されたエージェントが見つからない場合
        """
        if name not in cls._registry:
            raise ValueError(f"エージェント '{name}' は登録されていません")
            
        return cls._registry[name]
        
    @classmethod
    def list_agents(cls):
        """
        登録されているエージェント名のリストを返す
        
        Returns:
            list: エージェント名のリスト
        """
        return list(cls._registry.keys()) 