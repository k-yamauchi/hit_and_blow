#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import itertools
import math
from typing import List, Tuple, Set, Optional

from .agent import AgentRegistry
from .utils import generate_all_possible_codes, calculate_hits_blows

class HitAndBlowAdvisor:
    """
    Hit and Blowゲームのアドバイザークラス
    ユーザーが友人とHit and Blowをプレイする際の補助として、
    次の予測や可能性のある答えを提案します。
    """
    
    def __init__(self, digits=3, number_range=6, allow_repetition=False, agent_type='probabilistic'):
        """
        初期化
        
        Args:
            digits (int): 桁数
            number_range (int): 数字の範囲 (0からnumber_range-1)
            allow_repetition (bool): 数字の繰り返しを許可するか
            agent_type (str): 使用するエージェントタイプ
        """
        self.digits = digits
        self.number_range = number_range
        self.allow_repetition = allow_repetition
        self.agent_type = agent_type
        
        # エージェントの初期化
        try:
            agent_class = AgentRegistry.get_class(agent_type)
            self.agent = agent_class(
                digits=digits,
                number_range=number_range,
                allow_repetition=allow_repetition
            )
        except Exception as e:
            print(f"エージェント初期化エラー: {e}")
            # フォールバックとしてルールベースエージェントを使用
            agent_class = AgentRegistry.get_class('rule_based')
            self.agent = agent_class(
                digits=digits,
                number_range=number_range,
                allow_repetition=allow_repetition
            )
            print(f"フォールバック: ルールベースエージェントを使用します")
        
        # 履歴の初期化
        self.guesses = []
        self.results = []
        
    def reset(self):
        """アドバイザーをリセット"""
        self.agent.reset()
        self.guesses = []
        self.results = []
    
    def suggest_next_guess(self) -> List[int]:
        """
        次の予測を提案
        
        Returns:
            List[int]: 提案された次の予測
        """
        guess = self.agent.predict(self.guesses, self.results)
        return guess
    
    def update_with_result(self, guess: List[int], hits: int, blows: int) -> None:
        """
        予測結果でアドバイザーを更新
        
        Args:
            guess (List[int]): 予測した数列
            hits (int): ヒット数
            blows (int): ブロー数
        """
        self.guesses.append(guess)
        self.results.append((hits, blows))
    
    def get_remaining_candidates_count(self) -> int:
        """
        残りの候補数を取得
        
        Returns:
            int: 残りの候補数
        """
        return self.agent.get_remaining_candidates()
    
    def get_possible_answers(self, max_count=5) -> List[List[int]]:
        """
        可能性のある答えのリストを取得
        
        Args:
            max_count (int): 最大表示数
            
        Returns:
            List[List[int]]: 可能性のある答えのリスト
        """
        return [list(candidate) for candidate in self.agent.get_possible_answers(max_count)]
    
    def provide_hint(self) -> str:
        """
        ヒントを提供
        
        Returns:
            str: ヒント文字列
        """
        # ランダムなヒントを選択
        if not self.guesses:
            return "最初の予測としては、[0, 1, 2]のような均等分布が情報を最大化します。"
            
        last_guess = self.guesses[-1]
        last_hits, last_blows = self.results[-1]
        
        if last_hits == self.digits:
            return "おめでとうございます！正解です！"
            
        # ヒントの種類
        hints = [
            f"現在の候補数は{self.get_remaining_candidates_count()}個です。",
            f"次の候補として{self.suggest_next_guess()}を提案します。",
            f"有力な可能性としては{self.get_possible_answers(3)}があります。"
        ]
        
        # ゲームの進行状況に応じた特別なヒント
        if last_hits > 0:
            hints.append(f"前回の予測では{last_hits}個の数字が正しい位置にあります。")
        if last_blows > 0:
            hints.append(f"前回の予測では{last_blows}個の数字は含まれていますが、位置が違います。")
        
        # 候補が少ない場合は具体的な候補を示す
        if self.get_remaining_candidates_count() <= 5:
            hints.append(f"残りの候補は{self.get_possible_answers()}だけです。")
            
        return random.choice(hints)
    
    def get_status_report(self):
        """
        現在の状態のレポートを取得
        
        Returns:
            dict: 状態レポート
        """
        return {
            "candidates_count": len(self.agent.get_possible_answers()),
            "history_length": len(self.guesses),
            "top_candidates": self.get_possible_answers(5) if self.get_possible_answers() else []
        } 