#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
from typing import List, Tuple, Set, Optional
import numpy as np

from hit_and_blow.core.utils import generate_all_possible_codes, calculate_hits_blows


class AdviceGenerator:
    """Q学習モデルを使用してプレイヤーにアドバイスを提供するクラス"""
    
    def __init__(self, model_path: str = "models/q_learning_model.pkl"):
        """
        アドバイス生成器の初期化
        
        Args:
            model_path: Q学習モデルのパス
        """
        self.model_path = model_path
        self.q_model = None
        self.digits = 3
        self.number_range = 6
        self.allow_repetition = False
        
        # モデルのロード
        self._load_model()
    
    def _load_model(self) -> bool:
        """モデルをロード"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = model_data.get('q_table', {})
            self.digits = model_data.get('digits', 3)
            self.number_range = model_data.get('number_range', 6)
            self.allow_repetition = model_data.get('allow_repetition', False)
            
            return True
        except (FileNotFoundError, pickle.PickleError) as e:
            print(f"モデル読み込みエラー: {e}")
            return False
    
    def _get_state_key(self, guesses: List[List[int]], results: List[Tuple[int, int]]) -> str:
        """状態キーの生成"""
        if not guesses:
            return "initial"
        
        state_parts = []
        for i, (guess, result) in enumerate(zip(guesses, results)):
            state_parts.append(f"{i}:{','.join(map(str, guess))}:{result[0]},{result[1]}")
        
        return "|".join(state_parts)
    
    def get_candidates(self, guesses: List[List[int]], results: List[Tuple[int, int]]) -> Set[Tuple[int, ...]]:
        """残りの候補を取得"""
        candidates = generate_all_possible_codes(
            self.digits, self.number_range, self.allow_repetition)
        
        for guess, result in zip(guesses, results):
            hits, blows = result
            new_candidates = set()
            for candidate in candidates:
                candidate_hits, candidate_blows = calculate_hits_blows(guess, candidate)
                if candidate_hits == hits and candidate_blows == blows:
                    new_candidates.add(candidate)
            candidates = new_candidates
        
        return candidates
    
    def get_advice(self, 
                  guesses: List[List[int]], 
                  results: List[Tuple[int, int]], 
                  top_n: int = 3) -> List[Tuple[Tuple[int, ...], float]]:
        """
        Q学習モデルに基づいてアドバイスを提供
        
        Args:
            guesses: これまでの予測リスト
            results: これまでの結果リスト（hits, blowsのタプル）
            top_n: 提案する候補の数
            
        Returns:
            上位n個の候補とそのスコア
        """
        # 残りの候補を取得
        candidates = self.get_candidates(guesses, results)
        
        if not candidates:
            return []
        
        # 状態キーの取得
        state_key = self._get_state_key(guesses, results)
        
        # 各候補のQ値を計算
        candidate_scores = []
        for candidate in candidates:
            candidate_key = str(candidate)
            q_value = self.q_table.get((state_key, candidate_key), 0.0)
            candidate_scores.append((candidate, q_value))
        
        # スコアの高い順に並べ替え
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 上位n個を返す
        return candidate_scores[:top_n]
    
    def explain_advice(self, 
                      guesses: List[List[int]], 
                      results: List[Tuple[int, int]]) -> str:
        """
        アドバイスの説明を生成
        
        Args:
            guesses: これまでの予測リスト
            results: これまでの結果リスト
            
        Returns:
            説明文字列
        """
        advice = self.get_advice(guesses, results)
        candidates = self.get_candidates(guesses, results)
        
        if not advice:
            return "候補が見つかりませんでした。入力した予測と結果が矛盾している可能性があります。"
        
        explanation = f"現在の候補数: {len(candidates)}個\n\n"
        explanation += "おすすめの予測:\n"
        
        for i, (candidate, score) in enumerate(advice, 1):
            explanation += f"{i}. {list(candidate)} (スコア: {score:.2f})\n"
        
        # 確率の説明を追加
        explanation += f"\nこれらの予測は、Q学習モデルが過去の経験から学習した最も報酬が高いと予測される行動です。"
        
        if len(guesses) == 0:
            explanation += "\n最初の予測では、多くの情報を得られる組み合わせを選ぶことをお勧めします。"
        
        return explanation


def main():
    """テスト用のメイン関数"""
    advice_generator = AdviceGenerator()
    
    # テスト用のデータ
    guesses = [[3, 1, 5], [0, 2, 4]]
    results = [(1, 1), (0, 2)]
    
    # アドバイスを取得
    advice = advice_generator.get_advice(guesses, results)
    explanation = advice_generator.explain_advice(guesses, results)
    
    print(explanation)

if __name__ == "__main__":
    main() 