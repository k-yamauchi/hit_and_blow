#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import time
import argparse
import random
import json
from datetime import datetime, timedelta
from collections import defaultdict

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hit_and_blow.core.agent import RuleBasedAgent, ProbabilisticAgent, HybridQLearningAgent
from hit_and_blow.core.utils import generate_all_possible_codes, calculate_hits_blows
from adversarial_game import AdversarialGame

def format_time(seconds):
    """秒数を時:分:秒形式にフォーマット"""
    return str(timedelta(seconds=int(seconds)))

class SimpleAdaptiveAgent:
    """他のエージェントの戦略を組み合わせた適応型エージェント"""
    
    def __init__(self, digits=3, number_range=6, allow_repetition=False):
        self.digits = digits
        self.number_range = number_range
        self.allow_repetition = allow_repetition
        
        # 基本エージェント
        self.rule_agent = RuleBasedAgent(digits, number_range, allow_repetition)
        self.prob_agent = ProbabilisticAgent(digits, number_range, allow_repetition)
        self.ql_agent = HybridQLearningAgent(digits, number_range, allow_repetition)
        
        # 学習した防御的な数列のリスト
        self.defensive_sequences = []
        
        # 各エージェントの成功率の追跡
        self.agent_performance = {
            'rule': {'wins': 0, 'games': 0},
            'prob': {'wins': 0, 'games': 0},
            'ql': {'wins': 0, 'games': 0}
        }
        
        # 全ての可能な数列
        self.all_candidates = list(generate_all_possible_codes(digits, number_range, allow_repetition))
    
    def predict(self, guesses, results, adaptation_info=None):
        """
        次の予測を行う（適応情報を利用）
        """
        # 相手の情報に基づいて戦略を調整
        if adaptation_info and len(guesses) > 0:
            opponent_progress = adaptation_info.get('progress', 0)
            my_progress = adaptation_info.get('my_progress', 0)
            
            # 相手が優位な場合はリスク大きめの戦略
            if opponent_progress > my_progress + 0.3:
                return self._risky_predict(guesses, results)
            
            # 相手が不利な場合は安全な戦略
            elif my_progress > opponent_progress + 0.3:
                return self._safe_predict(guesses, results)
        
        # デフォルトは性能の良いエージェントを使用
        return self._best_agent_predict(guesses, results)
    
    def _best_agent_predict(self, guesses, results):
        """最も成績の良いエージェントを使用"""
        win_rates = {}
        
        for agent_type, stats in self.agent_performance.items():
            games = stats['games']
            win_rate = stats['wins'] / games if games > 0 else 0.33
            win_rates[agent_type] = win_rate
        
        # 最も勝率の高いエージェントを選択（十分なデータがない場合はランダム）
        if sum(stats['games'] for stats in self.agent_performance.values()) < 10:
            best_agent = random.choice(['rule', 'prob', 'ql'])
        else:
            best_agent = max(win_rates, key=win_rates.get)
        
        # 選択したエージェントの予測を返す
        if best_agent == 'rule':
            return self.rule_agent.predict(guesses, results)
        elif best_agent == 'prob':
            return self.prob_agent.predict(guesses, results)
        else:
            return self.ql_agent.predict(guesses, results)
    
    def _risky_predict(self, guesses, results):
        """リスクの高い予測（確率的なアプローチ）"""
        # 残りの候補を取得
        candidates = self._get_remaining_candidates(guesses, results)
        
        if len(candidates) <= 1:
            return list(candidates[0]) if candidates else self.rule_agent.predict(guesses, results)
            
        # リスクの高い選択：最も可能性の高い候補を直接選ぶ
        return list(random.choice(candidates[:min(3, len(candidates))]))
    
    def _safe_predict(self, guesses, results):
        """安全な予測（ルールベースのアプローチ）"""
        # 基本的にはルールベースエージェントに従う
        return self.rule_agent.predict(guesses, results)
    
    def _get_remaining_candidates(self, guesses, results):
        """これまでの予測と結果から残りの候補を取得"""
        candidates = self.all_candidates.copy()
        
        for guess, result in zip(guesses, results):
            hits, blows = result
            
            filtered_candidates = []
            for candidate in candidates:
                candidate_hits, candidate_blows = calculate_hits_blows(guess, candidate)
                
                if candidate_hits == hits and candidate_blows == blows:
                    filtered_candidates.append(candidate)
                    
            candidates = filtered_candidates
            
        return candidates
    
    def select_secret(self):
        """防御的な数列を選択"""
        if self.defensive_sequences and random.random() < 0.8:
            # 学習した防御的な数列から選択
            return random.choice(self.defensive_sequences)
        else:
            # ランダムな選択
            return random.sample(range(self.number_range), self.digits)
    
    def update_performance(self, agent_type, won):
        """エージェントの性能を更新"""
        self.agent_performance[agent_type]['games'] += 1
        if won:
            self.agent_performance[agent_type]['wins'] += 1
    
    def add_defensive_sequence(self, sequence):
        """防御的な数列を追加"""
        if sequence not in self.defensive_sequences:
            self.defensive_sequences.append(sequence)
            # 数列が多すぎる場合は古いものを削除
            if len(self.defensive_sequences) > 20:
                self.defensive_sequences.pop(0)
    
    def save_model(self, path):
        """モデルを保存"""
        data = {
            'defensive_sequences': self.defensive_sequences,
            'agent_performance': self.agent_performance
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def load_model(self, path):
        """モデルを読み込み"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                self.defensive_sequences = data.get('defensive_sequences', [])
                self.agent_performance = data.get('agent_performance', {
                    'rule': {'wins': 0, 'games': 0},
                    'prob': {'wins': 0, 'games': 0},
                    'ql': {'wins': 0, 'games': 0}
                })
            
            return True
        except (FileNotFoundError, json.JSONDecodeError):
            return False

class SimpleSelfPlayTrainer:
    """シンプルな自己対戦学習"""
    
    def __init__(self, digits=3, number_range=6, allow_repetition=False):
        self.digits = digits
        self.number_range = number_range
        self.allow_repetition = allow_repetition
        
        # 適応型エージェント
        self.agent = SimpleAdaptiveAgent(digits, number_range, allow_repetition)
        
        # 他のエージェントタイプ
        self.agent_types = ['rule_based', 'probabilistic', 'hybrid_q_learning']
        
        # 数列評価のための勝敗記録
        self.sequence_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_turns': 0})
    
    def train(self, num_games=100, analyze_sequences=False):
        """
        自己対戦を通じた学習の実行
        
        Args:
            num_games (int): 実行するゲーム数
            analyze_sequences (bool): 数列分析を行うかどうか
        """
        game = AdversarialGame(
            self.digits, 
            self.number_range,
            allow_repetition=self.allow_repetition
        )
        
        if analyze_sequences:
            return self._analyze_defensive_sequences(game, num_games)
        
        # 自己対戦を実行
        for i in range(num_games):
            # 対戦相手のタイプをランダムに選択
            opponent_type = random.choice(self.agent_types)
            
            # 自分のエージェントタイプを選択（性能追跡のため）
            my_agent_type = random.choice(['rule', 'prob', 'ql'])
            
            # ゲームの実行
            result = game.run_adversarial_game(
                p1_type='self_play',
                p2_type=opponent_type,
                adaptation=True,
                p1_custom_agent=self.agent
            )
            
            # 結果の分析
            if result['winner'] == 1:  # 勝利
                self.agent.update_performance(my_agent_type, True)
                
                # 使用した数列の記録
                secret = result.get('p1_secret', [])
                if secret:
                    self.sequence_performance[str(secret)]['wins'] += 1
                    self.sequence_performance[str(secret)]['total_turns'] += result['turns']
                    
                    # 優秀な数列を防御的数列として記録
                    if self.sequence_performance[str(secret)]['wins'] >= 3:
                        self.agent.add_defensive_sequence(secret)
            
            else:  # 敗北または引き分け
                self.agent.update_performance(my_agent_type, False)
                
                # 使用した数列の記録
                secret = result.get('p1_secret', [])
                if secret:
                    self.sequence_performance[str(secret)]['losses'] += 1
            
            # 定期的な進捗表示
            if (i + 1) % 10 == 0:
                print(f"ゲーム {i + 1}/{num_games} 完了")
                
                # 最近の成績を表示
                win_rates = {
                    agent_type: stats['wins'] / stats['games'] if stats['games'] > 0 else 0
                    for agent_type, stats in self.agent.agent_performance.items()
                }
                print(f"エージェント勝率: Rule={win_rates['rule']:.2f}, "
                      f"Prob={win_rates['prob']:.2f}, QL={win_rates['ql']:.2f}")
        
        # 最終的な学習結果を表示
        self._print_training_results()
        
        return {
            'agent_performance': self.agent.agent_performance,
            'defensive_sequences': self.agent.defensive_sequences
        }
    
    def _print_training_results(self):
        """学習結果の表示"""
        print("\n===== 学習結果 =====")
        
        # エージェント別の性能
        print("各エージェントの勝率:")
        for agent_type, stats in self.agent.agent_performance.items():
            games = stats['games']
            wins = stats['wins']
            win_rate = wins / games if games > 0 else 0
            print(f"  {agent_type}: {win_rate:.2f} ({wins}/{games})")
        
        # 最も効果的な防御的数列
        print("\n最も効果的な防御的数列:")
        top_sequences = sorted(
            self.sequence_performance.items(),
            key=lambda x: (x[1]['wins'] / max(1, x[1]['wins'] + x[1]['losses']), -x[1]['total_turns'] / max(1, x[1]['wins'])),
            reverse=True
        )[:5]
        
        for seq_str, stats in top_sequences:
            total = stats['wins'] + stats['losses']
            win_rate = stats['wins'] / total if total > 0 else 0
            avg_turns = stats['total_turns'] / stats['wins'] if stats['wins'] > 0 else 0
            print(f"  {seq_str}: 勝率 {win_rate:.2f} ({stats['wins']}/{total}), 平均ターン数 {avg_turns:.1f}")
    
    def _analyze_defensive_sequences(self, game, num_samples=30):
        """
        防御的な数列の分析
        
        Args:
            game: 対戦ゲームインスタンス
            num_samples: 分析するサンプル数
        
        Returns:
            list: 最も効果的な防御的数列のリスト
        """
        print("===== 防御的な数列の分析開始 =====")
        
        # 分析対象のエージェント
        target_agents = self.agent_types
        print(f"対象エージェント: {target_agents}")
        
        # 全ての可能な数列
        all_sequences = list(generate_all_possible_codes(
            self.digits, 
            self.number_range, 
            self.allow_repetition
        ))
        
        # サンプリング対象の数列（全てを試すのは時間がかかるため）
        sample_sequences = random.sample(all_sequences, min(num_samples, len(all_sequences)))
        
        # 各数列の性能を分析
        sequence_scores = {}
        
        for i, sequence in enumerate(sample_sequences):
            # 進捗表示
            if (i + 1) % 10 == 0:
                print(f"  数列分析: {i + 1}/{len(sample_sequences)} 完了")
            
            # 各エージェントに対する性能
            agent_turns = {}
            
            for agent_type in target_agents:
                # テスト用のゲームを実行
                test_result = game.test_defensive_sequence(
                    secret=sequence,
                    agent_type=agent_type
                )
                
                # 解読に必要なターン数を記録
                agent_turns[agent_type] = test_result.get('turns', 0)
            
            # 平均ターン数を計算
            avg_turns = sum(agent_turns.values()) / len(agent_turns) if agent_turns else 0
            
            # 結果を記録
            sequence_scores[str(sequence)] = {
                'avg_turns': avg_turns,
                'agent_turns': agent_turns,
                'sequence': sequence
            }
        
        # 最も効果的な数列（平均ターン数が多いもの）
        top_sequences = sorted(
            sequence_scores.values(),
            key=lambda x: x['avg_turns'],
            reverse=True
        )[:5]
        
        # 結果の表示
        print("\n最も効果的な防御的数列:")
        for i, data in enumerate(top_sequences):
            seq = data['sequence']
            avg = data['avg_turns']
            turns = data['agent_turns']
            
            turn_details = ", ".join([f"{agent}:{t}" for agent, t in turns.items()])
            print(f"{i+1}. {seq} - 平均 {avg:.2f} ターン ({turn_details})")
            
            # 有効な数列を学習データに追加
            self.agent.add_defensive_sequence(seq)
        
        return [data['sequence'] for data in top_sequences]
    
    def save_model(self, path_prefix="models/simple_self_play"):
        """モデルの保存"""
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        self.agent.save_model(f"{path_prefix}.json")
        print(f"モデルを保存しました: {path_prefix}.json")
    
    def load_model(self, path_prefix="models/simple_self_play"):
        """モデルの読み込み"""
        success = self.agent.load_model(f"{path_prefix}.json")
        if success:
            print(f"モデルを読み込みました: {path_prefix}.json")
        else:
            print("モデルの読み込みに失敗しました。新しいモデルを初期化します。")
        
        return success

def main():
    parser = argparse.ArgumentParser(description='Hit and Blow簡易自己対戦学習')
    parser.add_argument('--digits', type=int, default=3, help='桁数 (デフォルト: 3)')
    parser.add_argument('--range', type=int, default=6, help='数字の範囲 0-(range-1) (デフォルト: 6)')
    parser.add_argument('--total-games', type=int, default=1000, help='総対戦回数 (デフォルト: 1000)')
    parser.add_argument('--analyze', action='store_true', help='防御的な数列の分析モードを有効化')
    parser.add_argument('--samples', type=int, default=30, help='分析サンプル数 (デフォルト: 30)')
    parser.add_argument('--save-interval', type=int, default=200, help='モデル保存間隔 (デフォルト: 200ゲームごと)')
    parser.add_argument('--load-model', action='store_true', help='保存されたモデルを読み込む')
    
    args = parser.parse_args()
    
    # フォルダの作成
    os.makedirs('models', exist_ok=True)
    
    # 開始時刻
    start_time = time.time()
    print(f"学習開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"設定: 桁数={args.digits}, 範囲=0-{args.range-1}, 総ゲーム数={args.total_games}")
    
    # トレーナーの初期化
    trainer = SimpleSelfPlayTrainer(
        digits=args.digits,
        number_range=args.range,
        allow_repetition=False  # 繰り返し数字を無効化
    )
    
    # モデルの読み込み（指定された場合）
    if args.load_model:
        trainer.load_model()
    
    # 分析モード
    if args.analyze:
        trainer.train(num_games=args.samples, analyze_sequences=True)
        trainer.save_model()
        return
    
    # ゲームをバッチに分割して実行
    batch_games = min(args.save_interval, 100)  # 1バッチあたりのゲーム数
    num_batches = args.total_games // batch_games
    remaining_games = args.total_games % batch_games
    
    game_count = 0
    
    for batch in range(num_batches):
        batch_start = time.time()
        
        print(f"\nバッチ {batch+1}/{num_batches} 開始 ({batch_games}ゲーム)")
        
        # バッチ実行
        trainer.train(num_games=batch_games)
        
        game_count += batch_games
        
        # 進捗状況の表示
        elapsed = time.time() - start_time
        batch_time = time.time() - batch_start
        avg_time_per_game = elapsed / game_count
        remaining = avg_time_per_game * (args.total_games - game_count)
        
        print(f"進捗: {game_count}/{args.total_games} ゲーム完了 ({game_count/args.total_games*100:.1f}%)")
        print(f"経過時間: {format_time(elapsed)}, バッチ実行時間: {format_time(batch_time)}")
        print(f"1ゲームあたり平均: {avg_time_per_game:.2f}秒")
        print(f"残り時間: 約 {format_time(remaining)}")
        
        # 定期的にモデルを保存
        trainer.save_model(f"models/simple_self_play_{game_count}")
    
    # 残りのゲームを実行
    if remaining_games > 0:
        print(f"\n残り {remaining_games}ゲームを実行中...")
        trainer.train(num_games=remaining_games)
        game_count += remaining_games
    
    # 最終モデルの保存
    trainer.save_model()
    
    # 終了時刻と総実行時間
    total_time = time.time() - start_time
    print(f"\n学習完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"総実行時間: {format_time(total_time)} ({total_time:.2f}秒)")
    print(f"平均: {total_time / args.total_games:.2f}秒/ゲーム")

if __name__ == "__main__":
    main() 