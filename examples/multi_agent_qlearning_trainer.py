#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import time
import argparse
import random
import json
import pickle
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hit_and_blow.core.agent import HybridQLearningAgent
from hit_and_blow.core.utils import generate_all_possible_codes, calculate_hits_blows
from adversarial_game import AdversarialGame

def format_time(seconds):
    """秒数を時:分:秒形式にフォーマット"""
    return str(timedelta(seconds=int(seconds)))

class EnhancedQLearningAgent(HybridQLearningAgent):
    """拡張されたQ学習エージェント（保存・読み込み機能付き）"""
    
    def __init__(self, digits=3, number_range=6, allow_repetition=False, 
                 learning_rate=0.1, discount_factor=0.95, exploration_rate=0.2):
        super().__init__(digits, number_range, allow_repetition)
        
        # 学習パラメータ
        self.learning_rate = learning_rate      # 学習率
        self.discount_factor = discount_factor  # 割引率
        self.exploration_rate = exploration_rate  # 探索率
        
        # Q値テーブルの初期化
        self.q_table = defaultdict(float)
        
        # Q値のより詳細な記録（状態、アクション、報酬の履歴）
        self.episode_memory = deque(maxlen=100)  # 直近100エピソードのメモリ
        
        # エピソード内の履歴
        self.episode_guesses = []
        self.episode_results = []
        
        # 統計データ
        self.training_stats = {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'total_reward': 0,
            'episodes': 0,
            'avg_turns': []
        }
        
        # 経験再生用のメモリ
        self.replay_memory = deque(maxlen=10000)
        
    def predict(self, guesses, results, adaptation_info=None):
        """次の予測を行う（学習モードかどうかで挙動を変える）"""
        # 新しいゲームの開始を検出
        if not guesses:
            self.episode_guesses = []
            self.episode_results = []
        
        # 前回の結果を履歴に記録
        if len(guesses) > len(self.episode_guesses) and guesses:
            self.episode_guesses = guesses.copy()
            
        if len(results) > len(self.episode_results) and results:
            self.episode_results = results.copy()
            
        if hasattr(self, 'is_training') and self.is_training:
            return self._training_predict(guesses, results, adaptation_info)
        else:
            return super().predict(guesses, results, adaptation_info)
    
    def _get_candidates_from_history(self, guesses, results):
        """
        過去の予測と結果に基づいて候補を取得
        
        Args:
            guesses (list): これまでの予測のリスト
            results (list): これまでの結果のリスト（(hits, blows)のタプル）
            
        Returns:
            set: 候補の集合
        """
        # 全ての可能な候補を生成
        candidates = generate_all_possible_codes(
            self.digits, self.number_range, self.allow_repetition)
        
        # 過去の予測と結果を使って候補を絞り込む
        for guess, result in zip(guesses, results):
            hits, blows = result
            new_candidates = set()
            for candidate in candidates:
                candidate_hits, candidate_blows = calculate_hits_blows(guess, candidate)
                if candidate_hits == hits and candidate_blows == blows:
                    new_candidates.add(candidate)
            candidates = new_candidates
        
        return candidates
    
    def _get_state_key(self, guesses, results):
        """
        状態からキーを生成
        
        Args:
            guesses (list): これまでの予測のリスト
            results (list): これまでの結果のリスト（(hits, blows)のタプル）
            
        Returns:
            str: 状態を表すキー
        """
        if not guesses:
            return "initial"
        
        state_parts = []
        for i, (guess, result) in enumerate(zip(guesses, results)):
            state_parts.append(f"{i}:{','.join(map(str, guess))}:{result[0]},{result[1]}")
        
        return "|".join(state_parts)
    
    def _training_predict(self, guesses, results, adaptation_info=None):
        """学習モード用の予測（探索と活用のバランスを取る）"""
        # 残りの候補を取得
        candidates = self._get_candidates_from_history(guesses, results)
        
        # 候補がない場合はランダム予測
        if not candidates:
            if self.allow_repetition:
                return [random.randint(0, self.number_range - 1) for _ in range(self.digits)]
            else:
                # 重複なしのランダム予測
                numbers = list(range(self.number_range))
                random.shuffle(numbers)
                return numbers[:self.digits]
        
        # 候補リストを作成
        candidates_list = list(candidates)
        
        # 探索：ランダムな行動を選択
        if random.random() < self.exploration_rate:
            return list(random.choice(candidates_list))
        
        # 活用：Q値に基づいた最良の行動を選択
        state_key = self._get_state_key(guesses, results)
        best_action = None
        best_q_value = float('-inf')
        
        for candidate in candidates:
            candidate_key = str(candidate)
            q_value = self.q_table.get((state_key, candidate_key), 0.0)
            
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = candidate
        
        # Q値が同じ場合はランダム選択
        if best_action is None:
            best_action = random.choice(candidates_list)
        
        return list(best_action)
    
    def update_q_value(self, state, action, reward, next_state, next_candidates):
        """Q値を更新する"""
        state_key = self._get_state_key(state['guesses'], state['results'])
        action_key = str(tuple(action))
        
        # 現在のQ値
        current_q = self.q_table.get((state_key, action_key), 0.0)
        
        # 次の状態で可能な最大Q値を取得
        next_state_key = self._get_state_key(next_state['guesses'], next_state['results'])
        max_next_q = float('-inf')
        
        for next_candidate in next_candidates:
            next_candidate_key = str(next_candidate)
            next_q = self.q_table.get((next_state_key, next_candidate_key), 0.0)
            max_next_q = max(max_next_q, next_q)
        
        # 次の状態の候補がない場合は0とする
        if max_next_q == float('-inf'):
            max_next_q = 0.0
        
        # Q値の更新（Q学習の更新式）
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[(state_key, action_key)] = new_q
        
        return new_q
    
    def add_to_replay_memory(self, state, action, reward, next_state, done):
        """経験再生メモリに追加"""
        self.replay_memory.append((state, action, reward, next_state, done))
    
    def replay_experiences(self, batch_size=32):
        """経験再生による学習"""
        if len(self.replay_memory) < batch_size:
            return
        
        # バッチをランダムに選択
        batch = random.sample(self.replay_memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            # 次の状態での候補を取得
            if done:
                next_candidates = []
            else:
                next_candidates = self._get_candidates_from_history(
                    next_state['guesses'], next_state['results'])
            
            # Q値を更新
            self.update_q_value(state, action, reward, next_state, next_candidates)
    
    def save_model(self, path):
        """モデルを保存"""
        model_data = {
            'q_table': dict(self.q_table),
            'training_stats': self.training_stats,
            'digits': self.digits,
            'number_range': self.number_range,
            'allow_repetition': self.allow_repetition,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path):
        """モデルを読み込み"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = defaultdict(float, model_data['q_table'])
            self.training_stats = model_data['training_stats']
            self.digits = model_data['digits']
            self.number_range = model_data['number_range']
            self.allow_repetition = model_data['allow_repetition']
            self.learning_rate = model_data['learning_rate']
            self.discount_factor = model_data['discount_factor']
            self.exploration_rate = model_data['exploration_rate']
            
            return True
        except (FileNotFoundError, pickle.PickleError) as e:
            print(f"モデル読み込みエラー: {e}")
            return False

class MultiAgentQLearningTrainer:
    """複数エージェントを使用するQ学習エージェントのトレーナー"""
    
    def __init__(self, digits=3, number_range=6, allow_repetition=False):
        self.digits = digits
        self.number_range = number_range
        self.allow_repetition = allow_repetition
        
        # メインQ学習エージェント
        self.agent = EnhancedQLearningAgent(
            digits=digits, 
            number_range=number_range,
            allow_repetition=allow_repetition
        )
        
        # 対戦相手用のQ学習エージェント
        self.opponent_q_agent = EnhancedQLearningAgent(
            digits=digits, 
            number_range=number_range,
            allow_repetition=allow_repetition,
            exploration_rate=0.3  # メインエージェントと異なる探索率を設定
        )
        
        # 相手エージェント用のタイプ（q_learningを追加）
        self.opponent_types = ['rule_based', 'probabilistic', 'q_learning']
        
        # 学習統計
        self.training_stats = {
            'episodes': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'win_rate_history': [],
            'avg_turns_history': [],
            'opponent_stats': {
                'rule_based': {'wins': 0, 'losses': 0, 'draws': 0},
                'probabilistic': {'wins': 0, 'losses': 0, 'draws': 0},
                'q_learning': {'wins': 0, 'losses': 0, 'draws': 0}
            }
        }
    
    def train(self, num_episodes=1000, batch_size=32, decay_exploration=True):
        """
        Q学習エージェントの学習
        
        Args:
            num_episodes (int): 学習エピソード数
            batch_size (int): 経験再生のバッチサイズ
            decay_exploration (bool): 探索率を徐々に減少させるか
        """
        # 対戦ゲームの初期化
        game = AdversarialGame(
            self.digits, 
            self.number_range,
            allow_repetition=self.allow_repetition
        )
        
        # 学習モードに設定
        self.agent.is_training = True
        self.opponent_q_agent.is_training = True
        initial_exploration_rate = self.agent.exploration_rate
        
        # 統計用の変数
        total_wins = 0
        total_losses = 0
        total_draws = 0
        total_turns = 0
        win_rates = []
        
        # 対戦相手別の統計
        opponent_stats = {
            'rule_based': {'wins': 0, 'losses': 0, 'games': 0},
            'probabilistic': {'wins': 0, 'losses': 0, 'games': 0},
            'q_learning': {'wins': 0, 'losses': 0, 'games': 0}
        }
        
        print(f"学習開始: {num_episodes}エピソード")
        start_time = time.time()
        
        for episode in range(num_episodes):
            # 探索率の減衰（学習の進行に合わせて探索を減らす）
            if decay_exploration:
                decay_factor = 1 - episode / num_episodes
                self.agent.exploration_rate = initial_exploration_rate * decay_factor
                self.opponent_q_agent.exploration_rate = (initial_exploration_rate + 0.1) * decay_factor
            
            # 対戦相手のタイプをランダムに選択
            opponent_type = random.choice(self.opponent_types)
            
            # エピソードの実行（1ゲーム）
            episode_result = self._run_training_episode(game, opponent_type)
            
            # 統計の更新
            if episode_result['winner'] == 1:
                total_wins += 1
                opponent_stats[opponent_type]['wins'] += 1
            elif episode_result['winner'] == 2:
                total_losses += 1
                opponent_stats[opponent_type]['losses'] += 1
            else:
                total_draws += 1
            
            opponent_stats[opponent_type]['games'] += 1
            total_turns += episode_result['turns']
            
            # 経験再生による学習
            self.agent.replay_experiences(batch_size)
            if opponent_type == 'q_learning':
                self.opponent_q_agent.replay_experiences(batch_size)
            
            # 定期的な進捗報告
            if (episode + 1) % 100 == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_episode = elapsed_time / (episode + 1)
                remaining_time = avg_time_per_episode * (num_episodes - episode - 1)
                
                win_rate = total_wins / (episode + 1) * 100
                win_rates.append(win_rate)
                
                print(f"エピソード {episode + 1}/{num_episodes} "
                      f"勝率: {win_rate:.1f}% "
                      f"平均ターン: {total_turns / (episode + 1):.1f}")
                
                # 対戦相手別の統計
                for opp_type, stats in opponent_stats.items():
                    if stats['games'] > 0:
                        opp_win_rate = stats['wins'] / stats['games'] * 100
                        print(f"  vs {opp_type}: {opp_win_rate:.1f}% ({stats['wins']}/{stats['games']})")
                
                print(f"経過時間: {format_time(elapsed_time)}, "
                      f"残り時間: {format_time(remaining_time)}")
                print(f"探索率: {self.agent.exploration_rate:.3f}, "
                      f"Q値エントリ数: {len(self.agent.q_table)}")
        
        # 学習結果の更新
        self.training_stats['episodes'] += num_episodes
        self.training_stats['wins'] += total_wins
        self.training_stats['losses'] += total_losses
        self.training_stats['draws'] += total_draws
        self.training_stats['win_rate_history'].extend(win_rates)
        
        # 対戦相手別の統計を更新
        for opp_type, stats in opponent_stats.items():
            self.training_stats['opponent_stats'][opp_type]['wins'] += stats['wins']
            self.training_stats['opponent_stats'][opp_type]['losses'] += stats['losses']
            self.training_stats['opponent_stats'][opp_type]['draws'] += (
                stats['games'] - stats['wins'] - stats['losses'])
        
        final_win_rate = total_wins / num_episodes * 100
        avg_turns = total_turns / num_episodes
        
        self.training_stats['avg_turns_history'].append(avg_turns)
        
        # 学習結果の表示
        total_time = time.time() - start_time
        print(f"\n学習完了: {num_episodes}エピソード")
        print(f"勝率: {final_win_rate:.1f}% ({total_wins}/{num_episodes})")
        print(f"平均ターン数: {avg_turns:.1f}")
        print(f"総学習時間: {format_time(total_time)} ({total_time:.1f}秒)")
        print(f"Q値エントリ数: {len(self.agent.q_table)}")
        
        # 対戦相手別の最終統計
        print("\n対戦相手別の勝率:")
        for opp_type, stats in opponent_stats.items():
            if stats['games'] > 0:
                opp_win_rate = stats['wins'] / stats['games'] * 100
                print(f"  vs {opp_type}: {opp_win_rate:.1f}% ({stats['wins']}/{stats['games']})")
        
        # 学習モードを解除
        self.agent.is_training = False
        self.opponent_q_agent.is_training = False
        
        return self.training_stats
    
    def _run_training_episode(self, game, opponent_type):
        """
        学習用のエピソードを実行
        
        Args:
            game: AdversarialGameインスタンス
            opponent_type: 対戦相手のタイプ
            
        Returns:
            dict: エピソード結果
        """
        # 対戦相手がQ学習の場合は別のQ学習エージェントを使用
        p2_custom_agent = None
        if opponent_type == 'q_learning':
            p2_custom_agent = self.opponent_q_agent
        
        # ゲームの実行
        result = game.run_adversarial_game(
            p1_type='q_learning',
            p2_type=opponent_type,
            adaptation=True,
            p1_custom_agent=self.agent,
            p2_custom_agent=p2_custom_agent
        )
        
        # 学習用のデータ整理
        p1_guesses = []
        p1_results = []
        
        # p2（対戦相手）がQ学習の場合は相手も学習する
        if opponent_type == 'q_learning' and hasattr(self.opponent_q_agent, 'episode_guesses'):
            p2_guesses = self.opponent_q_agent.episode_guesses
            p2_results = self.opponent_q_agent.episode_results
        
        # 直接ゲーム実行時に保存するようオーバーライドする必要があるが、
        # 現在の実装では結果が返ってくるため、それを使って学習データを生成する
        
        # ターン数
        turns = result['turns']
        winner = result['winner']
        # 勝敗による報酬
        p1_final_reward = 10.0 if winner == 1 else -5.0 if winner == 2 else 0.0
        p2_final_reward = 10.0 if winner == 2 else -5.0 if winner == 1 else 0.0
        
        # 勝敗を記録
        is_p1_win = (winner == 1)
        is_p2_win = (winner == 2)
        
        # エージェントの直近の予測・結果履歴があれば使用
        if hasattr(self.agent, 'episode_guesses') and hasattr(self.agent, 'episode_results'):
            p1_guesses = self.agent.episode_guesses
            p1_results = self.agent.episode_results
        
        # 履歴がない場合は、基本的な報酬設計
        if not p1_guesses:
            # 簡単な疑似データを作成
            state = {'guesses': [], 'results': []}
            next_state = {'guesses': [], 'results': []}
            action = [0, 0, 0]  # ダミーアクション
            
            # 勝敗に応じた単純報酬
            if is_p1_win:
                reward = 10.0 - turns * 0.5  # 早く勝つほど報酬が高い
            elif is_p2_win:
                reward = -5.0
            else:
                reward = 0.0
                
            self.agent.add_to_replay_memory(state, action, reward, next_state, True)
            return result
        
        # 実際の履歴があれば、各ターンごとに報酬を与える
        history_length = min(len(p1_guesses), len(p1_results))
        
        # P1（メインエージェント）の学習
        for i in range(history_length):
            # 現在の状態
            state = {
                'guesses': p1_guesses[:i] if i > 0 else [],
                'results': p1_results[:i] if i > 0 else []
            }
            
            # アクション
            action = p1_guesses[i]
            
            # 次の状態
            next_state = {
                'guesses': p1_guesses[:i+1],
                'results': p1_results[:i+1]
            }
            
            # 報酬の計算
            hits, blows = p1_results[i]
            
            # 基本報酬：ヒット数に比例、ターンが少ない方が良い
            reward = hits * 1.0 + blows * 0.2
            
            # 最終ターンには追加報酬
            done = (i == history_length - 1)
            if done:
                reward += p1_final_reward
                
                # 早く解けるほど報酬を増やす
                if is_p1_win:
                    reward += max(0, 10 - i) * 0.5
                
            # 報酬を追加
            self.agent.add_to_replay_memory(state, action, reward, next_state, done)
        
        # P2がQ学習の場合、P2も学習する
        if opponent_type == 'q_learning' and hasattr(self.opponent_q_agent, 'episode_guesses'):
            p2_history_length = min(len(p2_guesses), len(p2_results))
            
            for i in range(p2_history_length):
                # 現在の状態
                state = {
                    'guesses': p2_guesses[:i] if i > 0 else [],
                    'results': p2_results[:i] if i > 0 else []
                }
                
                # アクション
                action = p2_guesses[i]
                
                # 次の状態
                next_state = {
                    'guesses': p2_guesses[:i+1],
                    'results': p2_results[:i+1]
                }
                
                # 報酬の計算
                hits, blows = p2_results[i]
                
                # 基本報酬：ヒット数に比例、ターンが少ない方が良い
                reward = hits * 1.0 + blows * 0.2
                
                # 最終ターンには追加報酬
                done = (i == p2_history_length - 1)
                if done:
                    reward += p2_final_reward
                    
                    # 早く解けるほど報酬を増やす
                    if is_p2_win:
                        reward += max(0, 10 - i) * 0.5
                    
                # 報酬を追加
                self.opponent_q_agent.add_to_replay_memory(state, action, reward, next_state, done)
        
        return result
    
    def evaluate(self, num_games=100, opponent_types=None):
        """
        学習したエージェントを評価
        
        Args:
            num_games (int): 評価ゲーム数
            opponent_types (list): 対戦相手のタイプのリスト（Noneの場合はデフォルト）
            
        Returns:
            dict: 評価結果
        """
        if opponent_types is None:
            opponent_types = ['rule_based', 'probabilistic', 'q_learning']
        
        # 対戦ゲームの初期化
        game = AdversarialGame(
            self.digits, 
            self.number_range,
            allow_repetition=self.allow_repetition
        )
        
        # 学習モードをオフに設定
        if hasattr(self.agent, 'is_training'):
            self.agent.is_training = False
        
        if hasattr(self.opponent_q_agent, 'is_training'):
            self.opponent_q_agent.is_training = False
        
        results = {}
        
        print(f"\n===== 評価開始: {num_games}ゲーム/対戦相手 =====")
        
        for opponent in opponent_types:
            print(f"\n対戦相手: {opponent}")
            
            wins = 0
            losses = 0
            draws = 0
            total_turns = 0
            
            for i in range(num_games):
                # 対戦相手がQ学習の場合は別のQ学習エージェントを使用
                p2_custom_agent = None
                if opponent == 'q_learning':
                    p2_custom_agent = self.opponent_q_agent
                
                # ゲームの実行
                result = game.run_adversarial_game(
                    p1_type='q_learning',
                    p2_type=opponent,
                    adaptation=True,
                    p1_custom_agent=self.agent,
                    p2_custom_agent=p2_custom_agent
                )
                
                # 結果の集計
                if result['winner'] == 1:
                    wins += 1
                elif result['winner'] == 2:
                    losses += 1
                else:
                    draws += 1
                
                total_turns += result['turns']
                
                # 定期的な進捗表示
                if (i + 1) % 10 == 0:
                    print(f"  ゲーム {i + 1}/{num_games} 完了")
            
            # 結果の保存
            win_rate = wins / num_games * 100
            avg_turns = total_turns / num_games
            
            results[opponent] = {
                'wins': wins,
                'losses': losses,
                'draws': draws,
                'win_rate': win_rate,
                'avg_turns': avg_turns
            }
            
            print(f"  勝率: {win_rate:.1f}% ({wins}/{num_games})")
            print(f"  平均ターン数: {avg_turns:.1f}")
        
        return results
    
    def save_model(self, main_path="models/multi_q_learning_model.pkl", 
                  opponent_path="models/opponent_q_learning_model.pkl"):
        """モデルを保存"""
        os.makedirs(os.path.dirname(main_path), exist_ok=True)
        
        # メインエージェントの保存
        self.agent.save_model(main_path)
        print(f"メインエージェントのモデルを保存しました: {main_path}")
        
        # 対戦相手エージェントの保存
        self.opponent_q_agent.save_model(opponent_path)
        print(f"対戦相手エージェントのモデルを保存しました: {opponent_path}")
        
        # トレーニング統計も保存
        stats_path = main_path + ".stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_stats, f, indent=2)
        
        return True
    
    def load_model(self, main_path="models/multi_q_learning_model.pkl", 
                 opponent_path="models/opponent_q_learning_model.pkl"):
        """モデルを読み込み"""
        # メインエージェントの読み込み
        main_success = self.agent.load_model(main_path)
        
        # 対戦相手エージェントの読み込み
        opponent_success = self.opponent_q_agent.load_model(opponent_path)
        
        if main_success:
            print(f"メインエージェントのモデルを読み込みました: {main_path}")
            
            # トレーニング統計も読み込み
            stats_path = main_path + ".stats.json"
            try:
                with open(stats_path, 'r', encoding='utf-8') as f:
                    self.training_stats = json.load(f)
            except FileNotFoundError:
                print("トレーニング統計が見つかりません")
        
        if opponent_success:
            print(f"対戦相手エージェントのモデルを読み込みました: {opponent_path}")
        
        return main_success and opponent_success

def main():
    parser = argparse.ArgumentParser(description='Hit and Blow マルチエージェントQ学習トレーナー')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'both'], 
                      default='both', help='実行モード')
    parser.add_argument('--episodes', type=int, default=3000, 
                      help='学習エピソード数')
    parser.add_argument('--eval-games', type=int, default=30, 
                      help='評価ゲーム数')
    parser.add_argument('--digits', type=int, default=3, 
                      help='桁数 (デフォルト: 3)')
    parser.add_argument('--range', type=int, default=6, 
                      help='数字の範囲 0-(range-1) (デフォルト: 6)')
    parser.add_argument('--save-model', type=str, default='models/multi_q_learning_model.pkl', 
                      help='メインモデルの保存パス')
    parser.add_argument('--save-opponent-model', type=str, default='models/opponent_q_learning_model.pkl', 
                      help='対戦相手モデルの保存パス')
    parser.add_argument('--load-model', type=str, 
                      help='読み込むメインモデルのパス（指定しない場合は新規学習）')
    parser.add_argument('--load-opponent-model', type=str, 
                      help='読み込む対戦相手モデルのパス（指定しない場合は新規学習）')
    parser.add_argument('--batch-size', type=int, default=32, 
                      help='経験再生のバッチサイズ')
    parser.add_argument('--opponent-ratio', type=str, default='33-33-34', 
                      help='対戦相手の割合（rule_based-probabilistic-q_learning）')
    
    args = parser.parse_args()
    
    # フォルダの作成
    os.makedirs('models', exist_ok=True)
    
    # トレーナーの初期化
    trainer = MultiAgentQLearningTrainer(
        digits=args.digits,
        number_range=args.range,
        allow_repetition=False
    )
    
    # モデルの読み込み（指定された場合）
    if args.load_model:
        opponent_model = args.load_opponent_model if args.load_opponent_model else args.load_model.replace('multi', 'opponent')
        trainer.load_model(args.load_model, opponent_model)
    
    # 学習モード
    if args.mode in ['train', 'both']:
        print(f"\n===== 学習モード: {args.episodes}エピソード =====")
        trainer.train(
            num_episodes=args.episodes,
            batch_size=args.batch_size,
            decay_exploration=True
        )
        
        # モデルの保存
        trainer.save_model(args.save_model, args.save_opponent_model)
    
    # 評価モード
    if args.mode in ['evaluate', 'both']:
        print(f"\n===== 評価モード: {args.eval_games}ゲーム/対戦相手 =====")
        trainer.evaluate(num_games=args.eval_games)

if __name__ == "__main__":
    main() 