from typing import List, Dict, Any, Tuple, Optional
import random
import numpy as np
import pickle
import os
import math
from collections import defaultdict, Counter
from hit_and_blow.agents.base import AgentBase
import itertools
import time

class HybridQLearningAgent(AgentBase):
    """
    ハイブリッドQ学習エージェント
    Q学習をベースに、ルールベース、情報理論、ヒューリスティックを組み合わせた手法
    """
    
    def __init__(self, num_digits: int = 4, digit_range: int = 10,
                 learning_rate: float = 0.2, discount_factor: float = 0.95,
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
        
        # 状態履歴
        self.state_history = []
        self.action_history = []
        
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
        self.candidates = list(itertools.permutations(range(self.digit_range), self.num_digits))
        
    def get_state_representation(self, history: List[Tuple[List[int], int, int]]) -> str:
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
            guess, hits, blows = entry
            state_parts.append(f"{hits}H{blows}B")
            
        # 残りの候補数も状態に加える（重要な情報）
        remaining = len(self.candidates)
        state_parts.append(f"R{remaining}")
        
        return "|".join(state_parts)
        
    def _update_candidates(self, last_entry: Tuple[List[int], int, int]):
        """
        ルールベースエージェントのロジックを用いて候補を絞り込む
        """
        guess, hits, blows = last_entry
        
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
        
    def get_valid_actions(self, history: List[Tuple[List[int], int, int]]) -> List[Tuple[int, ...]]:
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
        
    def predict(self, history: List[Tuple[List[int], int, int]]) -> List[int]:
        """
        次の予測を行う
        Q学習: 状態から最も価値の高い行動を選ぶ
        """
        # 初回リセット（候補リストの初期化）
        if not history and not self.candidates:
            self.reset()
            
        # 最初の予測は最適な初期パターンを使用
        if not history:
            action = self._get_optimal_first_guess()
            self.action_history.append(action)
            return list(action)
            
        # 状態表現を取得
        current_state = self.get_state_representation(history)
        self.state_history.append(current_state)
        
        # 有効な行動リストを取得
        valid_actions = self.get_valid_actions(history)
        
        # 探索と活用のバランス
        if random.random() < self.exploration_rate:
            # ランダムな行動（探索）
            action = random.choice(valid_actions)
        else:
            # 価値が最大の行動（活用）
            if self.q_table[current_state] and any(self.q_table[current_state][a] > 0 for a in valid_actions):
                # Q値が高い行動を選択
                action = max(valid_actions, key=lambda a: self.q_table[current_state][a])
            else:
                # Q値が学習されていない場合は情報量最大の行動を選択
                action = self._select_informative_action(valid_actions, history)
        
        self.action_history.append(action)
        return list(action)
    
    def _select_informative_action(self, actions: List[Tuple[int, ...]], history: List[Tuple[List[int], int, int]]) -> Tuple[int, ...]:
        """
        最も情報量の多い行動を選択する（情報理論的アプローチ）
        各行動の結果に対する可能性の分布を均等にするものを選ぶ
        """
        # 候補が少ない場合は最初の候補を返す（計算コスト削減）
        if len(actions) == 1 or len(self.candidates) <= 2:
            return actions[0]
            
        best_action = None
        best_score = float('-inf')
        
        # 各行動の情報量を計算
        for action in actions[:5]:  # 計算コスト削減のため最初の5つのみ評価
            outcomes = defaultdict(int)
            
            # 各候補に対する結果を計算
            for candidate in self.candidates:
                hits, blows = self._calculate_hits_blows(list(action), list(candidate))
                outcomes[(hits, blows)] += 1
                
            # エントロピー（情報量）の計算
            entropy = 0
            total = len(self.candidates)
            
            for count in outcomes.values():
                p = count / total
                entropy -= p * math.log2(p) if p > 0 else 0
                
            # 評価スコア: エントロピーと候補の削減期待値の組み合わせ
            expected_remaining = sum(count * count / total for count in outcomes.values())
            score = entropy - 0.1 * expected_remaining
            
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action if best_action else actions[0]
        
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
        
    def update_q_values(self, final_reward: float) -> None:
        """
        エピソード終了時にQ値を更新する
        
        Args:
            final_reward: 最終報酬（タスク達成までのターン数に基づく）
        """
        # 逆順に状態・行動履歴をたどりながらQ値を更新
        cum_reward = final_reward
        
        for t in range(len(self.state_history) - 1, -1, -1):
            state = self.state_history[t]
            action = self.action_history[t]
            
            # 現在のQ値
            current_q = self.q_table[state][action]
            
            # Q値の更新（TD学習）
            new_q = current_q + self.learning_rate * (cum_reward - current_q)
            self.q_table[state][action] = new_q
            
            # 次の更新のための報酬の割引
            cum_reward *= self.discount_factor
            
    def save_model(self, path: str) -> None:
        """
        モデルを保存する
        """
        # defaultdictを通常の辞書に変換
        q_table_dict = {}
        for state, actions in self.q_table.items():
            q_table_dict[state] = dict(actions)
            
        # 保存データにはQ値の他にエージェントの設定も含める
        save_data = {
            'q_table': q_table_dict,
            'digits': self.num_digits,
            'number_range': self.digit_range,
            'allow_repetition': False,  # 現時点では繰り返しは常に無効
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate,
            'optimal_first_guesses': dict(self._optimal_first_guesses),
            'training_stats': {
                'last_updated': time.time(),
                'description': f'Hybrid Q-Learning model for {self.num_digits} digits (range 0-{self.digit_range-1})'
            }
        }
        
        # ディレクトリがなければ作成
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # モデルの保存
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
            
        print(f"モデルを保存しました: {path}")
            
    def load_model(self, path: str) -> None:
        """
        保存されたモデルを読み込む
        """
        try:
            with open(path, 'rb') as f:
                save_data = pickle.load(f)
                
            # 新形式のモデルファイルの場合
            if isinstance(save_data, dict):
                if 'q_table' in save_data:
                    q_table_dict = save_data['q_table']
                    
                    # ハイパーパラメータの更新
                    if 'digits' in save_data and 'number_range' in save_data:
                        self.num_digits = save_data.get('digits', self.num_digits)
                        self.digit_range = save_data.get('number_range', self.digit_range)
                    
                    if 'learning_rate' in save_data:
                        self.learning_rate = save_data.get('learning_rate', self.learning_rate)
                    
                    if 'discount_factor' in save_data:
                        self.discount_factor = save_data.get('discount_factor', self.discount_factor)
                    
                    if 'exploration_rate' in save_data:
                        self.exploration_rate = save_data.get('exploration_rate', self.exploration_rate)
                    
                    # 最適な初期予測のロード
                    if 'optimal_first_guesses' in save_data:
                        self._optimal_first_guesses = dict(save_data.get('optimal_first_guesses', {}))
                else:
                    # q_tableのみの辞書の場合
                    q_table_dict = save_data
            else:
                # 型が不正な場合
                raise ValueError(f"モデルの形式が不正です: {type(save_data)}")
                
            # 辞書をdefaultdictに変換
            self.q_table = defaultdict(lambda: defaultdict(float))
            
            # q_table_dictの構造チェック
            if isinstance(q_table_dict, dict):
                for state, actions in q_table_dict.items():
                    if isinstance(actions, dict):
                        for action, q_value in actions.items():
                            self.q_table[state][action] = q_value
                    else:
                        # actionsが辞書でない場合はスキップ
                        print(f"警告: state={state}のactionsが辞書ではありません: {type(actions)}")
            else:
                raise ValueError(f"Q表の形式が不正です: {type(q_table_dict)}")
                    
            print(f"モデルを読み込みました: {path}")
            # ハイパーパラメータの表示
            print(f"設定: 桁数={self.num_digits}, 範囲=0-{self.digit_range-1}, 学習率={self.learning_rate:.3f}, 割引率={self.discount_factor:.3f}, 探索率={self.exploration_rate:.3f}")
            
        except Exception as e:
            print(f"モデル読み込み失敗: {e}")
            
    def calculate_reward(self, history: List[Tuple[List[int], int, int]]) -> float:
        """
        現在の状態に対する報酬を計算
        - ヒット数に応じた報酬
        - 候補数の減少に応じた報酬
        """
        if not history:
            return 0.0
            
        # 最新の結果
        _, hits, blows = history[-1]
        
        # ヒット数に応じた報酬
        reward = hits * 0.2 + blows * 0.1
        
        # 候補数の減少に応じた追加報酬（進捗に報酬を与える）
        if len(history) >= 2:
            old_state = self.get_state_representation(history[:-1])
            new_state = self.get_state_representation(history)
            
            # 状態から候補数を抽出
            old_candidates = int(old_state.split("|")[-1][1:])
            new_candidates = int(new_state.split("|")[-1][1:])
            
            # 候補数の減少率に応じた報酬
            if old_candidates > 0:  # ゼロ除算を防ぐ
                reduction_rate = (old_candidates - new_candidates) / old_candidates
                reward += reduction_rate * 0.3
        
        return reward 