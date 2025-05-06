#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import random
import numpy as np
from collections import defaultdict
import time

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hit_and_blow.core.agent import AgentRegistry
from hit_and_blow.core.advisor import HitAndBlowAdvisor
from hit_and_blow.core.utils import generate_all_possible_codes, calculate_hits_blows

class AdversarialGame:
    """Hit and Blowの対戦型ゲームを実装するクラス"""
    
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
        
        # 候補となる全ての数列を生成
        self.all_candidates = generate_all_possible_codes(
            self.digits, self.number_range, self.allow_repetition)
        
        # エージェントの初期化
        self.agents = {}
        
    def initialize_agent(self, agent_name, agent_type, player_id):
        """指定されたタイプのエージェントを初期化"""
        try:
            # AgentRegistryからエージェントクラスを取得
            agent_class = AgentRegistry.get_class(agent_type)
            self.agents[player_id] = agent_class(
                digits=self.digits, 
                number_range=self.number_range,
                allow_repetition=self.allow_repetition
            )
            print(f"プレイヤー{player_id} ({agent_name}): {agent_type}エージェントを初期化しました")
        except Exception as e:
            print(f"エージェント初期化エラー: {e}")
            # フォールバックとしてルールベースエージェントを使用
            agent_class = AgentRegistry.get_class('rule_based')
            self.agents[player_id] = agent_class(
                digits=self.digits, 
                number_range=self.number_range,
                allow_repetition=self.allow_repetition
            )
            print(f"フォールバック: プレイヤー{player_id}にルールベースエージェントを使用します")
    
    def select_defensive_secret(self, strategy='entropy'):
        """
        防御力の高い秘密の数列を選択
        
        Args:
            strategy (str): 防御戦略 ('entropy', 'random', 'repetition')
            
        Returns:
            list: 選択された秘密の数列
        """
        if strategy == 'random':
            # 完全にランダムな数列
            return random.sample(list(self.all_candidates), 1)[0]
        
        elif strategy == 'repetition' and self.allow_repetition:
            # 繰り返し数字を含む防御的な数列（繰り返しが許可されている場合）
            possible_repeating = []
            for candidate in self.all_candidates:
                # 繰り返しを含む数列を抽出
                if len(set(candidate)) < len(candidate):
                    possible_repeating.append(candidate)
            
            if possible_repeating:
                return random.choice(possible_repeating)
        
        # エントロピー最大化の防御的な数列
        # 初期予測[0,1,2]に対して中間的な結果を返す数列を優先
        best_candidates = []
        initial_guess = list(range(min(3, self.digits)))
        
        for candidate in self.all_candidates:
            hits, blows = calculate_hits_blows(initial_guess, candidate)
            
            # 中間的な結果（1ヒット1ブローなど）を返す数列を優先
            if hits == 1 and blows == 1:
                best_candidates.append(candidate)
        
        # 適切な候補がなければ、一般的なパターンを避けた数列を選択
        if not best_candidates:
            non_sequential = []
            for candidate in self.all_candidates:
                # 連続した数字を含まない数列
                is_sequential = False
                for i in range(len(candidate) - 1):
                    if candidate[i] + 1 == candidate[i + 1]:
                        is_sequential = True
                        break
                
                if not is_sequential:
                    non_sequential.append(candidate)
            
            if non_sequential:
                best_candidates = non_sequential
            else:
                best_candidates = list(self.all_candidates)
        
        return random.choice(best_candidates)
    
    def run_adversarial_game(self, p1_type, p2_type, p1_secret=None, p2_secret=None, 
                            p1_defensive_strategy='entropy', p2_defensive_strategy='entropy',
                            adaptation=True, max_turns=20, p1_custom_agent=None, p2_custom_agent=None):
        """
        対戦型ゲームを実行
        
        Args:
            p1_type (str): プレイヤー1のエージェントタイプ
            p2_type (str): プレイヤー2のエージェントタイプ
            p1_secret (list): プレイヤー1の秘密の数列（Noneの場合は自動生成）
            p2_secret (list): プレイヤー2の秘密の数列（Noneの場合は自動生成）
            p1_defensive_strategy (str): プレイヤー1の防御戦略
            p2_defensive_strategy (str): プレイヤー2の防御戦略
            adaptation (bool): 相手の進捗に応じて戦略を調整するか
            max_turns (int): 最大ターン数
            p1_custom_agent (object): プレイヤー1のカスタムエージェント（指定された場合）
            p2_custom_agent (object): プレイヤー2のカスタムエージェント（指定された場合）
            
        Returns:
            dict: ゲーム結果の統計
        """
        # エージェントの初期化
        if p1_custom_agent:
            self.agents[1] = p1_custom_agent
            print(f"プレイヤー1 ({p1_type}): カスタムエージェントを使用")
        else:
            self.initialize_agent("プレイヤー1", p1_type, 1)
        
        if p2_custom_agent:
            self.agents[2] = p2_custom_agent
            print(f"プレイヤー2 ({p2_type}): カスタムエージェントを使用")
        else:
            self.initialize_agent("プレイヤー2", p2_type, 2)
        
        # 秘密の数列が指定されていない場合は生成
        if p1_secret is None:
            # カスタムエージェントでselect_secretメソッドが利用可能な場合はそれを使用
            if p1_custom_agent and hasattr(p1_custom_agent, 'select_secret'):
                p1_secret = p1_custom_agent.select_secret()
            else:
                p1_secret = self.select_defensive_secret(p1_defensive_strategy)
        
        if p2_secret is None:
            # カスタムエージェントでselect_secretメソッドが利用可能な場合はそれを使用
            if p2_custom_agent and hasattr(p2_custom_agent, 'select_secret'):
                p2_secret = p2_custom_agent.select_secret()
            else:
                p2_secret = self.select_defensive_secret(p2_defensive_strategy)
        
        print(f"\n===== 対戦開始 =====")
        print(f"プレイヤー1 ({p1_type}) の秘密: {p1_secret}")
        print(f"プレイヤー2 ({p2_type}) の秘密: {p2_secret}")
        
        # ゲーム状態の初期化
        p1_guesses = []
        p2_guesses = []
        p1_results = []
        p2_results = []
        
        p1_found = False
        p2_found = False
        winner = None
        turns = 0
        
        # メインゲームループ
        while not (p1_found or p2_found) and turns < max_turns:
            turns += 1
            print(f"\n----- ターン {turns} -----")
            
            # プレイヤー1のターン
            p1_adaptation_info = None
            if adaptation and turns > 1:
                # 相手の進捗に応じて戦略を調整
                p1_adaptation_info = self._get_adaptation_info(p2_results)
                # 追加情報を付与
                if p1_adaptation_info:
                    p1_adaptation_info['opponent_guesses'] = p2_guesses
                    p1_adaptation_info['opponent_results'] = p2_results
                    p1_adaptation_info['progress'] = self._calculate_progress(p2_results)
                    p1_adaptation_info['my_progress'] = self._calculate_progress(p1_results)
            
            p1_guess = self.agents[1].predict(p1_guesses, p1_results, p1_adaptation_info)
            hits, blows = calculate_hits_blows(p1_guess, p2_secret)
            p1_guesses.append(p1_guess)
            p1_results.append((hits, blows))
            
            print(f"プレイヤー1の予測: {p1_guess} → 結果: {hits}ヒット, {blows}ブロー")
            
            if hits == self.digits:
                p1_found = True
                winner = 1
                print(f"プレイヤー1が正解を見つけました！")
                break
            
            # プレイヤー2のターン
            p2_adaptation_info = None
            if adaptation:
                # 相手の進捗に応じて戦略を調整
                p2_adaptation_info = self._get_adaptation_info(p1_results)
                # 追加情報を付与
                if p2_adaptation_info:
                    p2_adaptation_info['opponent_guesses'] = p1_guesses
                    p2_adaptation_info['opponent_results'] = p1_results
                    p2_adaptation_info['progress'] = self._calculate_progress(p1_results)
                    p2_adaptation_info['my_progress'] = self._calculate_progress(p2_results)
            
            p2_guess = self.agents[2].predict(p2_guesses, p2_results, p2_adaptation_info)
            hits, blows = calculate_hits_blows(p2_guess, p1_secret)
            p2_guesses.append(p2_guess)
            p2_results.append((hits, blows))
            
            print(f"プレイヤー2の予測: {p2_guess} → 結果: {hits}ヒット, {blows}ブロー")
            
            if hits == self.digits:
                p2_found = True
                winner = 2
                print(f"プレイヤー2が正解を見つけました！")
                break
        
        # 結果の集計
        result = {
            "winner": winner,
            "turns": turns,
            "p1_found": p1_found,
            "p2_found": p2_found,
            "p1_guesses": len(p1_guesses),
            "p2_guesses": len(p2_guesses),
            "p1_last_result": p1_results[-1] if p1_results else None,
            "p2_last_result": p2_results[-1] if p2_results else None,
            "p1_secret": p1_secret,
            "p2_secret": p2_secret
        }
        
        print("\n===== ゲーム終了 =====")
        if winner:
            print(f"勝者: プレイヤー{winner} ({p1_type if winner == 1 else p2_type})")
        else:
            print(f"引き分け (最大ターン数に到達)")
        
        return result
    
    def _get_adaptation_info(self, opponent_results):
        """
        相手の進捗情報に基づく戦略適応情報を生成
        
        Args:
            opponent_results (list): 相手の結果のリスト [(hits, blows), ...]
            
        Returns:
            dict: 適応情報
        """
        if not opponent_results:
            return {"strategy": "normal"}
        
        # 最新の結果
        last_hits, last_blows = opponent_results[-1]
        
        # 相手が優位（2ヒット以上）の場合、リスクを取る戦略
        if last_hits >= 2:
            return {
                "strategy": "risky",
                "opponent_advantage": True,
                "risk_factor": min(1.0, 0.3 + (last_hits / self.digits) * 0.7)  # ヒット数に応じたリスク係数
            }
        
        # 相手が不利（0-1ヒット）の場合、安全な戦略
        return {
            "strategy": "safe",
            "opponent_advantage": False,
            "safety_factor": 0.8
        }
    
    def _calculate_progress(self, results):
        """
        結果からゲームの進捗を計算
        
        Args:
            results (list): [(hits, blows), ...] 形式の結果リスト
            
        Returns:
            float: 0.0～1.0の進捗値
        """
        if not results:
            return 0.0
        
        # 最新の結果を取得
        hits, blows = results[-1]
        
        # ヒット数に基づく進捗 (0.0 - 1.0)
        hit_progress = hits / self.digits
        
        # ヒット数とブロー数の合計（最大でdigits）
        total_feedback = hits + blows
        total_progress = total_feedback / self.digits * 0.5  # 最大で0.5
        
        # ヒット数重視の進捗計算
        progress = hit_progress * 0.7 + total_progress * 0.3
        
        return progress
    
    def analyze_defensive_codes(self, agent_types, num_samples=10, top_n=5):
        """
        防御力の高い数列を分析
        
        Args:
            agent_types (list): 分析に使用するエージェントタイプのリスト
            num_samples (int): 分析するサンプル数
            top_n (int): 表示する上位の数
            
        Returns:
            dict: 防御力の高い数列の辞書
        """
        print(f"\n===== 防御的な数列の分析開始 =====")
        print(f"エージェント: {agent_types}")
        print(f"サンプル数: {num_samples}")
        
        # サンプリングする候補
        if num_samples >= len(self.all_candidates):
            samples = list(self.all_candidates)
        else:
            samples = random.sample(list(self.all_candidates), num_samples)
        
        results = defaultdict(dict)
        
        for agent_type in agent_types:
            print(f"\n-- {agent_type}エージェントでの分析 --")
            
            try:
                # エージェントの初期化
                agent_class = AgentRegistry.get_class(agent_type)
                agent = agent_class(
                    digits=self.digits, 
                    number_range=self.number_range,
                    allow_repetition=self.allow_repetition
                )
                
                for i, secret in enumerate(samples):
                    guesses = []
                    feedback = []
                    turns = 0
                    found = False
                    
                    # このエージェントがこの秘密の数列を解くのに何ターン必要か
                    while not found and turns < 20:  # 最大20ターン
                        turns += 1
                        guess = agent.predict(guesses, feedback)
                        hits, blows = calculate_hits_blows(guess, secret)
                        guesses.append(guess)
                        feedback.append((hits, blows))
                        
                        if hits == self.digits:
                            found = True
                    
                    # 結果を保存
                    results[tuple(secret)][agent_type] = turns
                    
                    if (i + 1) % 10 == 0:
                        print(f"  {i+1}/{len(samples)} 完了")
                
            except Exception as e:
                print(f"エージェント実行エラー: {e}")
        
        # 防御力のスコアリング (より多くのターンを要求するほど防御力が高い)
        defense_scores = {}
        for secret, agent_results in results.items():
            # 全エージェントの平均ターン数をスコアとする
            avg_turns = sum(agent_results.values()) / len(agent_results)
            defense_scores[secret] = avg_turns
        
        # 防御力の高い順にソート
        top_defensive = sorted(defense_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        print(f"\n----- 防御力の高い上位{top_n}の数列 -----")
        for i, (secret, score) in enumerate(top_defensive):
            print(f"{i+1}. {list(secret)} - 平均 {score:.2f}ターン必要")
            # 各エージェントごとの詳細
            for agent_type in agent_types:
                if agent_type in results[secret]:
                    print(f"   - {agent_type}: {results[secret][agent_type]}ターン")
        
        # リストはハッシュ不可能なため、タプルのまま返す
        return {secret: score for secret, score in top_defensive}

def main():
    parser = argparse.ArgumentParser(description='Hit and Blow 対戦型ゲーム')
    parser.add_argument('--digits', type=int, default=3, help='桁数 (デフォルト: 3)')
    parser.add_argument('--range', type=int, default=6, help='数字の範囲 0-(range-1) (デフォルト: 6)')
    parser.add_argument('--mode', type=str, default='game', choices=['game', 'analyze'],
                      help='実行モード (game: 対戦ゲーム, analyze: 防御的数列分析) (デフォルト: game)')
    parser.add_argument('--p1', type=str, default='hybrid_q_learning', 
                      help='プレイヤー1のエージェントタイプ (デフォルト: hybrid_q_learning)')
    parser.add_argument('--p2', type=str, default='probabilistic', 
                      help='プレイヤー2のエージェントタイプ (デフォルト: probabilistic)')
    parser.add_argument('--repetition', action='store_true', help='数字の繰り返しを許可')
    parser.add_argument('--analyze-agents', type=str, default='rule_based,probabilistic,hybrid_q_learning',
                      help='分析に使用するエージェントタイプ（カンマ区切り）')
    parser.add_argument('--samples', type=int, default=50, 
                      help='分析するサンプル数 (デフォルト: 50)')
    parser.add_argument('--games', type=int, default=1,
                      help='実行するゲーム数 (デフォルト: 1)')
    parser.add_argument('--no-adapt', action='store_true', 
                      help='相手の進捗に応じた戦略調整を無効化')
    
    args = parser.parse_args()
    
    game = AdversarialGame(digits=args.digits, number_range=args.range, 
                          allow_repetition=args.repetition)
    
    if args.mode == 'analyze':
        # 防御的な数列の分析
        agent_types = args.analyze_agents.split(',')
        game.analyze_defensive_codes(agent_types, num_samples=args.samples)
    
    else:  # game mode
        # 対戦ゲームの実行
        wins = {1: 0, 2: 0, None: 0}
        avg_turns = []
        
        for i in range(args.games):
            print(f"\n====== ゲーム {i+1}/{args.games} ======")
            result = game.run_adversarial_game(
                p1_type=args.p1,
                p2_type=args.p2,
                adaptation=not args.no_adapt
            )
            
            wins[result["winner"]] += 1
            avg_turns.append(result["turns"])
        
        # 結果の集計
        if args.games > 1:
            print("\n====== 総合結果 ======")
            print(f"ゲーム数: {args.games}")
            print(f"プレイヤー1 ({args.p1}) の勝利: {wins[1]} ({wins[1]/args.games*100:.1f}%)")
            print(f"プレイヤー2 ({args.p2}) の勝利: {wins[2]} ({wins[2]/args.games*100:.1f}%)")
            print(f"引き分け: {wins[None]} ({wins[None]/args.games*100:.1f}%)")
            print(f"平均ターン数: {sum(avg_turns)/len(avg_turns):.2f}")

if __name__ == "__main__":
    main() 