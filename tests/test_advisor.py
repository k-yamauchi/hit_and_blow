#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import unittest

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hit_and_blow.core.advisor import HitAndBlowAdvisor

class TestHitAndBlowAdvisor(unittest.TestCase):
    """HitAndBlowAdvisorクラスのテスト"""
    
    def setUp(self):
        """各テストの前に実行されるセットアップ"""
        self.advisor = HitAndBlowAdvisor(digits=3, number_range=6)
    
    def test_initialization(self):
        """初期化のテスト"""
        self.assertEqual(self.advisor.digits, 3)
        self.assertEqual(self.advisor.number_range, 6)
        self.assertEqual(len(self.advisor.history), 0)
        
        # 候補数のチェック（3桁、範囲0-5の場合は6P3=120通り）
        expected_candidates_count = 120  # 6P3 = 6*5*4 = 120
        self.assertEqual(len(self.advisor.candidates), expected_candidates_count)
    
    def test_reset(self):
        """リセット機能のテスト"""
        # 一度予測と結果を更新
        self.advisor.update_with_result([0, 1, 2], 1, 1)
        
        # 候補が絞られていることを確認
        self.assertTrue(len(self.advisor.candidates) < 120)
        self.assertEqual(len(self.advisor.history), 1)
        
        # リセット
        self.advisor.reset()
        
        # 初期状態に戻っていることを確認
        self.assertEqual(len(self.advisor.history), 0)
        self.assertEqual(len(self.advisor.candidates), 120)
    
    def test_calculate_hits_blows(self):
        """ヒットとブローの計算テスト"""
        # テストケース: 候補 vs 予測 -> 期待されるヒット数とブロー数
        test_cases = [
            ([0, 1, 2], [0, 1, 2], (3, 0)),  # 完全一致
            ([0, 1, 2], [3, 4, 5], (0, 0)),  # 完全不一致
            ([0, 1, 2], [0, 2, 1], (1, 2)),  # 1つの数字と位置が一致、2つの数字が含まれるが位置が異なる
            ([0, 1, 2], [2, 1, 0], (0, 3)),  # すべての数字が含まれるが位置が異なる
            ([0, 1, 2], [0, 3, 4], (1, 0)),  # 1つの数字と位置が一致、他は不一致
            ([1, 1, 2], [1, 2, 2], (2, 1)),  # 重複数字の処理テスト
        ]
        
        for candidate, guess, expected in test_cases:
            result = self.advisor._calculate_hits_blows(candidate, guess)
            self.assertEqual(result, expected, f"Failed for candidate={candidate}, guess={guess}")
    
    def test_update_with_result(self):
        """結果に基づく候補更新のテスト"""
        # 予測 [0, 1, 2] で 1ヒット、1ブロー の場合
        self.advisor.update_with_result([0, 1, 2], 1, 1)
        
        # 候補が適切に絞られているか確認
        for candidate in self.advisor.candidates:
            # 各候補について、予測 [0, 1, 2] に対するヒットとブローを計算
            hits, blows = self.advisor._calculate_hits_blows(candidate, [0, 1, 2])
            
            # ヒットとブローが期待通りか確認
            self.assertEqual((hits, blows), (1, 1), 
                            f"Candidate {candidate} should give 1 hit and 1 blow for [0, 1, 2]")
    
    def test_suggest_next_guess(self):
        """次の予測の提案テスト"""
        # 初期状態での提案
        initial_suggestion = self.advisor.suggest_next_guess()
        
        # 提案が有効な候補であることを確認
        self.assertTrue(initial_suggestion in self.advisor.candidates)
        
        # ゲーム進行後の提案
        self.advisor.update_with_result([0, 1, 2], 0, 2)  # 0ヒット、2ブロー
        self.advisor.update_with_result([3, 2, 1], 1, 1)  # 1ヒット、1ブロー
        
        next_suggestion = self.advisor.suggest_next_guess()
        
        # 提案が現在の候補の中にあることを確認
        self.assertTrue(next_suggestion in self.advisor.candidates)
        
        # 候補が適切に絞られていることを確認
        for candidate in self.advisor.candidates:
            # 各候補について、過去の予測に対するヒットとブローを計算
            hits1, blows1 = self.advisor._calculate_hits_blows(candidate, [0, 1, 2])
            hits2, blows2 = self.advisor._calculate_hits_blows(candidate, [3, 2, 1])
            
            # 結果が期待通りか確認
            self.assertEqual((hits1, blows1), (0, 2))
            self.assertEqual((hits2, blows2), (1, 1))
    
    def test_full_game_scenario(self):
        """完全なゲームシナリオのテスト"""
        # 実際のゲームシナリオをシミュレート
        # 答えを [3, 1, 4] と仮定（但し範囲は0-5）
        
        # 予測1: [0, 1, 2] -> 0ヒット、1ブロー
        self.advisor.update_with_result([0, 1, 2], 0, 1)
        
        # 予測2: [3, 0, 5] -> 1ヒット、0ブロー
        self.advisor.update_with_result([3, 0, 5], 1, 0)
        
        # 予測3: [3, 1, 4] -> 3ヒット、0ブロー
        self.advisor.update_with_result([3, 1, 4], 3, 0)
        
        # 候補が1つだけ（正解）になっているか確認
        self.assertEqual(len(self.advisor.candidates), 1)
        self.assertEqual(self.advisor.candidates[0], [3, 1, 4])

if __name__ == '__main__':
    unittest.main() 